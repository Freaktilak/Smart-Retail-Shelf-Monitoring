# ============================================
# Smart Retail Shelf Monitoring System
# DATA 690 - Applied Artificial Intelligence
# UMBC Data Science | Spring 2025
# ============================================

import cv2
import numpy as np
import onnxruntime as ort
import time
import subprocess
from collections import deque

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "/home/tilakraj/SmartShelf/best_opset19.onnx"
CAMERA_INDEX = 0

# --- Detection thresholds ---
# Raised from 0.25 → 0.40 to reject weak/shadow false positives.
# If you find genuine products are being missed, lower to 0.35.
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45

INPUT_SIZE = 640
GRID_ROWS  = 4
GRID_COLS  = 6

RESTOCK_THRESHOLD = 3
SKIP_FRAMES       = 2
DASHBOARD_WIDTH   = 300
CAMERA_HEIGHT     = 480
CAMERA_WIDTH      = 640

# --- Temporal stabilization ---
# How many past frames to remember per zone (rolling window).
STABILITY_BUFFER  = 8
# Fraction of recent frames a zone must be occupied to count as STOCKED.
# 0.5 = majority vote (zone flips only when >50 % of last 8 frames agree).
# Raise to 0.625 for even less flicker; lower to 0.4 for faster response.
STABILITY_MAJORITY = 0.5

# --- Visual (texture) empty-zone detection ---
# A zone whose pixel standard deviation is BELOW this value is considered
# visually empty — it is a uniform dark/black/solid area with no product.
#
# Tuning guide:
#   < 18  → only catches pitch-black zones (too strict for laptop demos)
#   30–40 → catches dark backgrounds, black gaps, laptop screen dark areas ✓
#   > 55  → starts flagging lightly-coloured but empty shelves (too loose)
#
# Start at 35. If real products are being wrongly flagged as empty, lower it.
# If dark empty gaps are still missed, raise it toward 45.
VISUAL_STD_THRESHOLD = 35.0

# How many consecutive frames the visual check must agree before
# a zone is confirmed visually empty (prevents a single dark frame flipping it).
VISUAL_BUFFER = 5


# ============================================
# ZONE STABILIZER
# ============================================
class ZoneStabilizer:
    """
    Smooths per-zone occupancy over a rolling window of recent frames.

    Instead of flipping stocked/empty on every single frame, a zone is
    declared EMPTY only when the majority of the last STABILITY_BUFFER
    frames agree it is empty.  This eliminates the "flickering" that
    happens when the model is uncertain on low-stock shelves.

    Usage (in app.py):
        stabilizer = ZoneStabilizer()
        ...
        raw_grid  = _raw_grid_from_detections(detections, frame)
        stable_grid = stabilizer.update(raw_grid)
        # use stable_grid instead of raw_grid
    """

    def __init__(self,
                 rows=GRID_ROWS,
                 cols=GRID_COLS,
                 buffer_size=STABILITY_BUFFER,
                 majority=STABILITY_MAJORITY):
        self.rows    = rows
        self.cols    = cols
        self.buf_sz  = buffer_size
        self.maj     = majority
        # One deque per zone cell; each entry is True (occupied) or False
        self._buf = [
            [deque(maxlen=buffer_size) for _ in range(cols)]
            for _ in range(rows)
        ]
        # Pre-fill with True so the system starts "fully stocked"
        # and requires evidence of emptiness before alerting
        for r in range(rows):
            for c in range(cols):
                for _ in range(buffer_size):
                    self._buf[r][c].append(True)

    def update(self, raw_grid):
        """
        Feed in a fresh GRID_ROWS×GRID_COLS boolean numpy array.
        Returns a smoothed boolean numpy array of the same shape.
        """
        stable = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                self._buf[r][c].append(bool(raw_grid[r][c]))
                occupied_frames = sum(self._buf[r][c])
                stable[r][c] = (occupied_frames / self.buf_sz) >= self.maj
        return stable

    def reset(self):
        """Hard-reset all buffers (e.g. after a camera reconnect)."""
        for r in range(self.rows):
            for c in range(self.cols):
                self._buf[r][c].clear()
                for _ in range(self.buf_sz):
                    self._buf[r][c].append(True)


# ============================================
# VISUAL STABILIZER
# ============================================
class VisualStabilizer:
    """
    Tracks per-zone pixel standard deviation over time.

    A zone is declared VISUALLY EMPTY when its std_dev has been
    below VISUAL_STD_THRESHOLD for VISUAL_BUFFER consecutive frames.

    This is the second opinion alongside YOLO detections.
    It catches uniform dark / black areas that YOLO misses because
    there is no product-shaped object to detect — only empty shelf.

    Design decision: consecutive frames (not majority vote) because
    a truly dark empty zone will ALWAYS be dark.  If a zone is
    sometimes dark and sometimes textured, it probably has a product.
    """

    def __init__(self,
                 rows=GRID_ROWS,
                 cols=GRID_COLS,
                 buffer_size=VISUAL_BUFFER,
                 threshold=VISUAL_STD_THRESHOLD):
        self.rows      = rows
        self.cols      = cols
        self.buf_sz    = buffer_size
        self.threshold = threshold
        # Rolling std_dev deque per zone
        self._buf = [
            [deque(maxlen=buffer_size) for _ in range(cols)]
            for _ in range(rows)
        ]

    def update(self, frame):
        """
        Compute per-zone std_dev from current frame.
        Returns a bool ndarray: True = visually empty.
        """
        h, w   = frame.shape[:2]
        cell_h = h // self.rows
        cell_w = w // self.cols
        result = np.zeros((self.rows, self.cols), dtype=bool)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for r in range(self.rows):
            for c in range(self.cols):
                y1 = r * cell_h
                y2 = min(y1 + cell_h, h)
                x1 = c * cell_w
                x2 = min(x1 + cell_w, w)

                cell   = gray[y1:y2, x1:x2]
                stddev = float(np.std(cell))
                self._buf[r][c].append(stddev)

                # Only call it visually empty if ALL recent frames agree
                buf = self._buf[r][c]
                if len(buf) == self.buf_sz:
                    result[r][c] = all(v < self.threshold for v in buf)
                # If buffer not full yet, default to not-empty (safe)

        return result

    def reset(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self._buf[r][c].clear()


# ============================================
# VISUAL EMPTY CHECK  (single zone, single frame)
# Used for debug / tuning printouts
# ============================================
def _is_visually_empty(frame, row, col,
                       threshold=VISUAL_STD_THRESHOLD):
    """
    Returns (is_empty: bool, std_dev: float) for one zone in one frame.

    Use this in a test script to calibrate VISUAL_STD_THRESHOLD:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                empty, std = _is_visually_empty(frame, r, c)
                print(f"R{r+1}C{c+1}: std={std:.1f}  {'EMPTY' if empty else 'ok'}")
    """
    h, w   = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    y1 = row * cell_h
    y2 = min(y1 + cell_h, h)
    x1 = col * cell_w
    x2 = min(x1 + cell_w, w)

    cell   = frame[y1:y2, x1:x2]
    gray   = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    stddev = float(np.std(gray))
    return stddev < threshold, stddev


# ============================================
# LOAD ONNX MODEL
# ============================================
def load_model(model_path):
    print("Loading Smart Shelf model...")
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    print("Model loaded successfully!")
    input_name = session.get_inputs()[0].name
    return session, input_name


# ============================================
# GET CPU TEMPERATURE
# ============================================
def get_cpu_temp():
    try:
        result = subprocess.run(
            ['cat', '/sys/devices/virtual/thermal/thermal_zone0/temp'],
            capture_output=True, text=True
        )
        temp = int(result.stdout.strip()) / 1000
        return f"{temp:.1f}C"
    except Exception:
        return "N/A"


# ============================================
# APPLY ZOOM AND CROP
# ============================================
def apply_zoom(frame, zoom, offset_x, offset_y):
    h, w = frame.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    cx = w // 2 + offset_x
    cy = h // 2 + offset_y
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))


# ============================================
# PREPROCESS FRAME FOR YOLO
# ============================================
def preprocess(frame, input_size=640):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# ============================================
# NON-MAXIMUM SUPPRESSION
# ============================================
def nms(boxes, scores, iou_threshold):
    x1     = boxes[:, 0]
    y1     = boxes[:, 1]
    x2     = boxes[:, 2]
    y2     = boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_   = np.maximum(0, xx2 - xx1)
        h_   = np.maximum(0, yy2 - yy1)
        inter = w_ * h_
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ============================================
# RUN YOLO DETECTION
# ============================================
def detect(session, input_name, frame):
    orig_h, orig_w = frame.shape[:2]
    inp     = preprocess(frame, INPUT_SIZE)
    outputs = session.run(None, {input_name: inp})
    predictions = outputs[0][0].T          # (num_preds, 5+)

    boxes  = []
    scores = []
    for pred in predictions:
        x_c, y_c, w, h = pred[0], pred[1], pred[2], pred[3]
        score = pred[4]
        if score < CONFIDENCE_THRESHOLD:
            continue
        x1 = int((x_c - w / 2) * orig_w / INPUT_SIZE)
        y1 = int((y_c - h / 2) * orig_h / INPUT_SIZE)
        x2 = int((x_c + w / 2) * orig_w / INPUT_SIZE)
        y2 = int((y_c + h / 2) * orig_h / INPUT_SIZE)
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))

    if not boxes:
        return []
    keep = nms(np.array(boxes), np.array(scores), IOU_THRESHOLD)
    return [boxes[i] for i in keep]


# ============================================
# RAW GRID FROM DETECTIONS  (single frame)
# ============================================
def _raw_grid(frame, detections):
    """
    Returns a raw (unsmoothed) GRID_ROWS×GRID_COLS bool array.
    True = at least one detection centre falls in that cell.
    """
    h, w   = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    grid   = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for box in detections:
        x1, y1, x2, y2 = box
        cx  = (x1 + x2) // 2
        cy  = (y1 + y2) // 2
        row = min(cy // cell_h, GRID_ROWS - 1)
        col = min(cx // cell_w, GRID_COLS - 1)
        grid[row][col] = True
    return grid


# ============================================
# EMPTY ZONE DETECTION LOGIC
# ============================================
def detect_empty_zones(frame, detections, stabilizer=None):
    """
    Draw grid overlay and compute empty zone count.

    A zone is marked EMPTY when the YOLO stabilizer confirms no
    detections have landed in it across recent frames.

    Parameters
    ----------
    frame       : BGR image (annotated in place)
    detections  : list of [x1,y1,x2,y2] boxes from detect()
    stabilizer  : ZoneStabilizer | None

    Returns
    -------
    frame         : annotated BGR image
    empty_count   : int
    grid_occupied : GRID_ROWS×GRID_COLS bool ndarray
    """
    h, w   = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    # ── YOLO-based zone occupancy ─────────────────────────────
    raw_grid = _raw_grid(frame, detections)

    if stabilizer is not None:
        grid_occupied = stabilizer.update(raw_grid)
    else:
        grid_occupied = raw_grid

    # ── Draw overlay ──────────────────────────────────────────
    empty_count = 0
    overlay     = frame.copy()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            if not grid_occupied[row][col]:
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                              (0, 0, 255), -1)
                empty_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (50, 50, 50), 1)

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    return frame, empty_count, grid_occupied


# ============================================
# DRAW DETECTIONS
# ============================================
def draw_detections(frame, detections):
    for box in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


# ============================================
# DASHBOARD HELPERS  (unchanged — kept for
# standalone main() usage)
# ============================================
def dash_text(panel, text, x, y,
              scale=0.45, color=(150, 150, 150), bold=1):
    cv2.putText(panel, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold)

def dash_line(panel, y):
    cv2.line(panel, (0, y), (DASHBOARD_WIDTH, y), (70, 70, 100), 1)

def dash_bar(panel, x, y, w, h, pct, color):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * pct / 100)
    if filled > 0:
        cv2.rectangle(panel, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (100, 100, 100), 1)


# ============================================
# BUILD DASHBOARD PANEL
# ============================================
def build_dashboard(height, item_count, empty_count,
                    fps, zoom, grid_occupied):
    total_zones   = GRID_ROWS * GRID_COLS
    stocked_zones = total_zones - empty_count
    stock_pct     = (stocked_zones / total_zones) * 100

    if stock_pct >= 75:
        bar_color    = (0, 200, 0)
        status_label = "FULLY STOCKED"
    elif stock_pct >= 40:
        bar_color    = (0, 165, 255)
        status_label = "LOW STOCK"
    else:
        bar_color    = (0, 0, 220)
        status_label = "CRITICAL"

    panel      = np.zeros((height, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    panel[:]   = (25, 25, 35)
    y          = 0

    cv2.rectangle(panel, (0, 0), (DASHBOARD_WIDTH, 40), (40, 40, 70), -1)
    dash_text(panel, "DASHBOARD", 10, 28, scale=0.75,
              color=(200, 200, 255), bold=2)
    dash_line(panel, 40)
    y = 40

    y += 14
    dash_text(panel, "STOCK LEVEL", 10, y, color=(160, 160, 160))
    y += 14
    dash_bar(panel, 10, y, 220, 18, stock_pct, bar_color)
    dash_text(panel, f"{stock_pct:.0f}%", 238, y + 14,
              scale=0.5, color=(255, 255, 255), bold=1)
    y += 28
    dash_text(panel, status_label, 10, y, scale=0.52,
              color=bar_color, bold=2)
    y += 14

    y += 12
    dash_text(panel, "STATISTICS", 10, y, color=(160, 160, 160))
    stats = [
        ("Items Detected", str(item_count),    (0, 220, 0)),
        ("Empty Zones",    str(empty_count),   (80, 80, 255)),
        ("Stocked Zones",  str(stocked_zones), (0, 180, 0)),
        ("Total Zones",    str(total_zones),   (200, 200, 200)),
    ]
    for label, value, color in stats:
        y += 22
        dash_text(panel, label, 10, y, color=(150, 150, 150))
        dash_text(panel, value, 230, y, scale=0.55, color=color, bold=2)
    y += 12
    dash_line(panel, y)

    y += 12
    dash_text(panel, "ROW STATUS", 10, y, color=(160, 160, 160))
    for row in range(GRID_ROWS):
        y += 22
        row_pct = (sum(grid_occupied[row]) / GRID_COLS) * 100
        if row_pct >= 75:
            rc, rs = (0, 220, 0),   "OK"
        elif row_pct >= 40:
            rc, rs = (0, 165, 255), "LOW"
        else:
            rc, rs = (0, 0, 255),   "EMPTY"
        dash_text(panel, f"Row {row + 1}", 10, y, color=(160, 160, 160))
        dash_bar(panel, 70, y - 12, 150, 14, row_pct, rc)
        dash_text(panel, rs, 228, y, scale=0.45, color=rc, bold=2)
    y += 12
    dash_line(panel, y)

    y += 12
    dash_text(panel, "SYSTEM INFO", 10, y, color=(160, 160, 160))
    sys_info = [
        ("FPS",    f"{fps:.1f}",   (255, 220, 0)),
        ("Zoom",   f"{zoom:.1f}x", (200, 200, 200)),
        ("Temp",   get_cpu_temp(), (100, 200, 255)),
        ("Device", "Jetson Nano",  (150, 150, 255)),
    ]
    for label, value, color in sys_info:
        y += 22
        dash_text(panel, label, 10, y, color=(150, 150, 150))
        dash_text(panel, value, 110, y, scale=0.48, color=color)

    return panel


# ============================================
# DRAW CAMERA UI
# ============================================
def draw_camera_ui(frame, item_count, empty_count, fps):
    h, w = frame.shape[:2]
    total_zones = GRID_ROWS * GRID_COLS
    stock_pct = ((total_zones - empty_count) / total_zones) * 100

    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 30), -1)
    cv2.putText(frame, "SMART RETAIL SHELF MONITOR",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    if empty_count > RESTOCK_THRESHOLD:
        cv2.rectangle(frame, (0, h - 38), (w, h), (0, 0, 180), -1)
        cv2.putText(frame,
                    f"WARNING: {empty_count} empty zones - RESTOCK NEEDED!",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, h - 38), (w, h), (0, 100, 0), -1)
        cv2.putText(frame,
                    f"STATUS: Shelf {stock_pct:.0f}% stocked - All good!",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2)
    return frame


# ============================================
# MAIN APPLICATION  (standalone OpenCV mode)
# ============================================
def main():
    print("=" * 50)
    print("Smart Retail Shelf Monitoring System")
    print("DATA 690 - UMBC Data Science")
    print("=" * 50)
    print("CONTROLS:")
    print("  Z / X  -> Zoom In / Out")
    print("  W / V  -> Pan Up / Down")
    print("  A / D  -> Pan Left / Right")
    print("  R      -> Reset view")
    print("  S      -> Save screenshot")
    print("  Q      -> Quit")
    print("=" * 50)

    session, input_name = load_model(MODEL_PATH)
    stabilizer = ZoneStabilizer()

    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    print("Camera opened successfully!")
    print("=" * 50)

    cv2.namedWindow("Smart Retail Shelf Monitor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Smart Retail Shelf Monitor",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    zoom = 1.0; offset_x = 0; offset_y = 0
    zoom_step = 0.1; pan_step = 20

    fps = 0; frame_count = 0
    start_time = time.time()
    frame_skip_counter = 0
    last_detections = []
    last_grid  = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    last_empty = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if zoom != 1.0 or offset_x != 0 or offset_y != 0:
            frame = apply_zoom(frame, zoom, offset_x, offset_y)

        frame_skip_counter += 1
        if frame_skip_counter % SKIP_FRAMES == 0:
            last_detections = detect(session, input_name, frame)

        frame = draw_detections(frame, last_detections)
        frame, last_empty, last_grid = detect_empty_zones(
            frame, last_detections, stabilizer=stabilizer)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            start_time  = time.time()

        frame = draw_camera_ui(frame, len(last_detections), last_empty, fps)
        dashboard = build_dashboard(
            frame.shape[0], len(last_detections),
            last_empty, fps, zoom, last_grid)

        combined = np.hstack([frame, dashboard])
        cv2.imshow("Smart Retail Shelf Monitor", combined)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): print("Quitting..."); break
        elif key == ord('s'):
            fn = f"/home/tilakraj/SmartShelf/capture_{int(time.time())}.jpg"
            cv2.imwrite(fn, combined)
            print(f"Screenshot saved: {fn}")
        elif key == ord('z'):
            zoom = min(zoom + zoom_step, 4.0)
            print(f"Zoom: {zoom:.1f}x")
        elif key == ord('x'):
            zoom = max(zoom - zoom_step, 1.0)
            if zoom == 1.0: offset_x = offset_y = 0
            print(f"Zoom: {zoom:.1f}x")
        elif key == ord('w'): offset_y -= pan_step
        elif key == ord('a'): offset_x -= pan_step
        elif key == ord('d'): offset_x += pan_step
        elif key == ord('v'): offset_y += pan_step
        elif key == ord('r'):
            zoom = 1.0; offset_x = offset_y = 0
            print("View reset!")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
