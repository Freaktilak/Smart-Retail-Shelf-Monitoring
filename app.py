"""
Smart Retail Shelf Monitoring System
Flask Web Server — app.py
DATA 690 - UMBC Data Science | Spring 2025
===========================================
Run:  python3 app.py
Open: http://localhost:5000  (or http://<jetson-ip>:5000)
"""

import os
import sys
import cv2
import json
import time
import threading
import logging
import subprocess
from datetime import datetime
from typing import Optional
from flask import Flask, Response, render_template, jsonify, request

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import functions + constants from your existing backend
from smart_shelf import (
    load_model,
    detect,
    detect_empty_zones,
    draw_detections,
    ZoneStabilizer,
    GRID_ROWS,
    GRID_COLS,
    RESTOCK_THRESHOLD,
    SKIP_FRAMES,
)

# ─── App Configuration ────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_opset19.onnx")
CAMERA_INDEX   = 0
JPEG_QUALITY   = 70       # 60-75 recommended for Jetson Nano bandwidth
STREAM_FPS_CAP = 25       # Max FPS pushed to browser (reduce if laggy)
STATS_INTERVAL = 0.5      # Seconds between SSE pushes
MAX_ALERT_LOG  = 10       # Alert entries kept in memory

# ─── Flask App ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
app = Flask(__name__)

# ─── Shared State (thread-safe) ───────────────────────────────────────────────
_frame_lock  = threading.Lock()
_stats_lock  = threading.Lock()
_latest_frame: Optional[bytes] = None

_stats = {
    "item_count":    0,
    "empty_count":   0,
    "stocked_count": GRID_ROWS * GRID_COLS,
    "total_zones":   GRID_ROWS * GRID_COLS,
    "stock_pct":     100.0,
    "fps":           0.0,
    "status":        "INITIALIZING",
    "status_level":  "warning",      # "ok" | "warning" | "critical"
    "grid":          [[False] * GRID_COLS for _ in range(GRID_ROWS)],
    "cpu_temp":      "N/A",
    "uptime_s":      0,
    "alert_log":     [],
    "ts":            0.0,
}

_app_start   = time.time()
_frame_count = 0          # total frames encoded (for stats)

# ─── Helper: CPU Temperature ──────────────────────────────────────────────────
def _cpu_temp() -> str:
    """Read Jetson Nano CPU temp from sysfs."""
    try:
        r = subprocess.run(
            ["cat", "/sys/devices/virtual/thermal/thermal_zone0/temp"],
            capture_output=True, text=True, timeout=1,
        )
        return f"{int(r.stdout.strip()) / 1000:.1f}°C"
    except Exception:
        return "N/A"

# ─── Helper: Alert Log ────────────────────────────────────────────────────────
def _push_alert(msg: str, level: str = "warning") -> None:
    """Prepend an entry to the shared alert log (thread-safe)."""
    entry = {
        "ts":    datetime.now().strftime("%H:%M:%S"),
        "msg":   msg,
        "level": level,
    }
    with _stats_lock:
        _stats["alert_log"].insert(0, entry)
        _stats["alert_log"] = _stats["alert_log"][:MAX_ALERT_LOG]

# ─── Detection Thread ─────────────────────────────────────────────────────────
def _detection_loop() -> None:
    """
    Background thread: grabs frames, runs YOLO, updates shared state.
    Runs forever until the process exits.
    """
    global _latest_frame

    # ── Load Model ────────────────────────────────────────────
    log.info("Loading YOLO model from: %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        log.error("Model file not found at %s", MODEL_PATH)
        log.error("Copy your .onnx file to: %s", MODEL_PATH)
        with _stats_lock:
            _stats["status"]       = "MODEL NOT FOUND"
            _stats["status_level"] = "critical"
        _push_alert("Model file not found — check models/ folder", "critical")
        return

    try:
        session, input_name = load_model(MODEL_PATH)
        log.info("Model loaded successfully.")
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        with _stats_lock:
            _stats["status"]       = "MODEL ERROR"
            _stats["status_level"] = "critical"
        _push_alert(f"Model error: {exc}", "critical")
        return

    # ── Open Camera ───────────────────────────────────────────
    log.info("Opening camera index %d ...", CAMERA_INDEX)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # Minimize latency

    if not cap.isOpened():
        log.error("Cannot open camera %d", CAMERA_INDEX)
        with _stats_lock:
            _stats["status"]       = "CAMERA ERROR"
            _stats["status_level"] = "critical"
        _push_alert("Camera failed to open", "critical")
        return

    log.info("Camera opened. Starting detection loop...")
    _push_alert("System online — detection active", "ok")

    # ── Loop Variables ────────────────────────────────────────
    total_zones       = GRID_ROWS * GRID_COLS
    skip_ctr          = 0
    fps_count         = 0
    fps_timer         = time.time()
    fps               = 0.0
    last_detections   = []
    last_empty        = 0
    last_grid         = [[False] * GRID_COLS for _ in range(GRID_ROWS)]
    prev_level        = "ok"
    temp_update_ctr   = 0
    cached_temp       = _cpu_temp()
    encode_params     = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # ── Temporal stabilizer — smooths flicker on low-stock shelves ──
    stabilizer = ZoneStabilizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed — retrying in 100 ms")
            time.sleep(0.1)
            continue

        # ── Inference (every SKIP_FRAMES) ─────────────────────
        skip_ctr += 1
        if skip_ctr >= SKIP_FRAMES:
            skip_ctr = 0
            try:
                last_detections = detect(session, input_name, frame)
            except Exception as exc:
                log.error("Detect error: %s", exc)

        # ── Draw Overlays ──────────────────────────────────────
        frame = draw_detections(frame, last_detections)
        frame, last_empty, grid_np = detect_empty_zones(
            frame, last_detections,
            stabilizer=stabilizer)
        last_grid = grid_np.tolist()          # numpy → list for JSON

        # ── FPS Calculation ────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps       = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        # ── Classify Status ────────────────────────────────────
        stocked   = total_zones - last_empty
        stock_pct = round((stocked / total_zones) * 100, 1)

        if stock_pct >= 75:
            status_label = "FULLY STOCKED"
            status_level = "ok"
        elif stock_pct >= 40:
            status_label = "LOW STOCK"
            status_level = "warning"
        else:
            status_label = "CRITICAL — RESTOCK NOW"
            status_level = "critical"

        # Push alert only on level transition
        if status_level != prev_level:
            _push_alert(f"Status → {status_label}", status_level)
            prev_level = status_level

        # ── CPU Temp (every ~120 frames to save CPU) ───────────
        temp_update_ctr += 1
        if temp_update_ctr >= 120:
            cached_temp     = _cpu_temp()
            temp_update_ctr = 0

        # ── Encode JPEG ────────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _frame_lock:
                _latest_frame = buf.tobytes()

        # ── Update Shared Stats ────────────────────────────────
        with _stats_lock:
            _stats.update({
                "item_count":    len(last_detections),
                "empty_count":   last_empty,
                "stocked_count": stocked,
                "total_zones":   total_zones,
                "stock_pct":     stock_pct,
                "fps":           round(fps, 1),
                "status":        status_label,
                "status_level":  status_level,
                "grid":          last_grid,
                "cpu_temp":      cached_temp,
                "uptime_s":      int(time.time() - _app_start),
                "ts":            time.time(),
            })

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template(
        "dashboard.html",
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming endpoint — opens in <img> tag."""
    frame_interval = 1.0 / STREAM_FPS_CAP

    def _generate():
        while True:
            t0 = time.time()
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            # Rate-limit the stream to STREAM_FPS_CAP
            sleep_t = frame_interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats_stream")
def stats_stream():
    """
    Server-Sent Events endpoint.
    The browser connects once; stats are pushed every STATS_INTERVAL seconds.
    """
    def _generate():
        while True:
            with _stats_lock:
                payload = json.dumps(_stats)
            yield f"data: {payload}\n\n"
            time.sleep(STATS_INTERVAL)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # Disable Nginx buffering if used
        },
    )


@app.route("/api/stats")
def api_stats():
    """One-shot JSON snapshot (called on initial page load)."""
    with _stats_lock:
        return jsonify(_stats)


@app.route("/api/snapshot", methods=["POST"])
def snapshot():
    """Save the current annotated frame as a JPEG in captures/."""
    captures_dir = os.path.join(BASE_DIR, "captures")
    os.makedirs(captures_dir, exist_ok=True)

    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"error": "No frame available yet"}), 503

    fname = f"capture_{int(time.time())}.jpg"
    path  = os.path.join(captures_dir, fname)
    with open(path, "wb") as fh:
        fh.write(frame)

    log.info("Snapshot saved: %s", path)
    _push_alert(f"Snapshot saved: {fname}", "ok")
    return jsonify({"file": fname, "path": path})


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start detection in background thread
    t = threading.Thread(target=_detection_loop, name="DetectionThread", daemon=True)
    t.start()

    log.info("=" * 55)
    log.info("  SmartShelf Monitor — Flask Server")
    log.info("  Dashboard: http://0.0.0.0:5000")
    log.info("  Press Ctrl+C to stop")
    log.info("=" * 55)

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False,   # Must be False — reloader breaks threads
    )
