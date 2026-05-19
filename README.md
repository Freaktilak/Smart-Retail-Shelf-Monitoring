# SmartShelf — AI-Powered Retail Shelf Monitoring System

> Real-time shelf occupancy detection using YOLOv8, ONNX Runtime, and a live web dashboard. Deployed on NVIDIA Jetson Nano.

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?style=flat-square&logo=flask)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange?style=flat-square)
![ONNX](https://img.shields.io/badge/Runtime-ONNX-purple?style=flat-square)
![Jetson](https://img.shields.io/badge/Device-Jetson%20Nano-green?style=flat-square&logo=nvidia)
![Course](https://img.shields.io/badge/UMBC-DATA%20690-red?style=flat-square)

---

## Overview

ShelfIQ is a real-time retail shelf monitoring system built for **DATA 690 — Applied Artificial Intelligence** at UMBC. It uses a YOLOv8 object detection model to identify products on retail shelves through a live camera feed, maps detections onto a 4×6 grid, and classifies each zone as **stocked** or **empty**.

The system runs entirely on a **Jetson Nano edge device** — no cloud, no internet connection required during operation. A web-based dashboard displays live video, occupancy metrics, zone availability maps, and automatic restock recommendations.

---

## Live Dashboard Preview

| Fully Stocked | Partially Stocked |
|---|---|
| 100% occupancy · 24/24 zones | 91.7% occupancy · 2 empty zones flagged |

---

## Key Features

- **Real-time YOLO detection** — YOLOv8 model exported to ONNX opset 19, running at ~1.3 fps on Jetson Nano CPU
- **Grid-based zone analysis** — 4×6 grid (24 zones) mapped over the camera frame
- **Temporal stabilization** — ZoneStabilizer uses an 8-frame majority vote to eliminate flickering detections
- **Live web dashboard** — MJPEG video stream + Server-Sent Events for real-time stats without page refresh
- **Auto restock recommendations** — pinpoints exactly which rows and zones are empty
- **Occupancy trend chart** — 60-second rolling history using Chart.js
- **One-click snapshot** — saves annotated frames to the captures/ folder
- **Edge deployed** — no cloud, no GPU required, runs on $99 hardware

---

## System Architecture

```
Camera (USB)
     │
     ▼
Jetson Nano
     │
     ├── smart_shelf.py
     │     ├── ONNX Runtime inference (YOLOv8)
     │     ├── Custom NMS
     │     ├── 4×6 Grid Zone Mapper
     │     └── ZoneStabilizer (8-frame buffer)
     │
     └── app.py (Flask)
           ├── /video_feed     → MJPEG stream
           ├── /stats_stream   → Server-Sent Events
           └── /api/snapshot   → Save frame to disk
                    │
                    ▼
            Browser Dashboard
            (dashboard.html + Chart.js)
```

---

## Project Structure

```
SmartShelf/
├── app.py                   # Flask server — MJPEG stream, SSE, snapshot API
├── smart_shelf.py           # YOLO inference, NMS, grid logic, ZoneStabilizer
├── requirements.txt         # Python dependencies
├── templates/
│   └── dashboard.html       # Web dashboard UI
├── static/
│   ├── style.css            # Enterprise retail theme
│   └── dashboard.js         # Live stats, Chart.js, zone grid
├── models/
│   └── .gitkeep             # Place your best_opset19.onnx file here
├── captures/                # Snapshots saved here (auto-created)
└── README.md
```

> **Note:** The trained ONNX model (`best_opset19.onnx`) is not included in this repository due to file size. Place your model file inside the `models/` folder before running.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Inference Runtime | ONNX Runtime — CPUExecutionProvider |
| Image Processing | OpenCV, NumPy |
| Backend | Python 3.8, Flask |
| Streaming | MJPEG over HTTP, Server-Sent Events |
| Frontend | HTML, CSS, JavaScript (no frameworks) |
| Charts | Chart.js |
| Hardware | NVIDIA Jetson Nano, USB Camera |
| OS | Ubuntu 18.04 |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Freaktilak/Smart-Retail-Shelf-Monitoring.git
cd Smart-Retail-Shelf-Monitoring
```

### 2. Place your ONNX model

```bash
cp /path/to/best_opset19.onnx models/best_opset19.onnx
```

### 3. Create virtual environment

> Use `--system-site-packages` so the venv can access Jetson's pre-installed `cv2` and `onnxruntime`.

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python3 -c "import cv2, onnxruntime, flask, numpy; print('ALL OK')"
```

### 6. Run the application

```bash
python3 app.py
```

### 7. Open the dashboard

Open a browser and go to:

```
http://localhost:5000
```

Or from another device on the same network:

```
http://<jetson-ip>:5000
```

---

## Configuration

All key parameters are at the top of `smart_shelf.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum detection confidence. Raise if false positives appear. |
| `IOU_THRESHOLD` | `0.45` | NMS overlap threshold for duplicate suppression. |
| `GRID_ROWS` | `4` | Number of horizontal shelf zones. |
| `GRID_COLS` | `6` | Number of vertical shelf zones. |
| `STABILITY_BUFFER` | `8` | Frames remembered per zone for majority vote. |
| `STABILITY_MAJORITY` | `0.50` | Fraction of frames required to confirm a zone state change. |
| `SKIP_FRAMES` | `2` | Run inference every N frames to reduce CPU load. |
| `RESTOCK_THRESHOLD` | `3` | Number of empty zones before a restock warning triggers. |

---

## How It Works

### Detection Pipeline

1. Camera captures a 640×480 frame
2. Frame is resized to 640×640, converted to RGB, normalized to 0–1
3. YOLOv8 ONNX model runs inference and returns bounding boxes with confidence scores
4. Custom NMS filters duplicate detections using IOU threshold
5. Detection centres are mapped onto a 4×6 grid
6. ZoneStabilizer applies an 8-frame majority vote per zone
7. Zones with no confirmed detection are marked as empty
8. Results stream to the web dashboard via SSE every 0.5 seconds

### ZoneStabilizer

The stabilizer solves the flickering problem. Without it, a single missed detection in one frame immediately flips a zone from stocked to empty. With it, a zone only changes state when the majority of the last 8 frames agree — this matches how a human operator would judge the shelf.

```python
# A zone is EMPTY when:
occupied_frames / buffer_size < STABILITY_MAJORITY
# i.e. fewer than 50% of the last 8 frames had a detection in that zone
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| Live Feed | MJPEG stream with YOLO bounding boxes |
| Status Hero | Overall shelf status with occupancy % and progress bar |
| Stat Strip | Occupancy, items detected, empty zones, FPS |
| Zone Map | 4×6 heatmap — green = stocked, red = empty |
| Row Performance | Per-row fill rate with status tags |
| Occupancy Trend | 60-second rolling line chart |
| Restock Panel | Lists exact rows and zone numbers that need attention |
| Event Log | Timestamped status change history |
| Snapshot | Saves current annotated frame to captures/ folder |

---

## Performance (Jetson Nano)

| Metric | Value |
|---|---|
| Inference speed | ~1.3 fps (CPU, SKIP_FRAMES=2) |
| Camera resolution | 640 × 480 |
| CPU temperature | ~62–67°C under load |
| Detection accuracy | 100% on fully stocked shelf, 91.7% on partially stocked |
| Zone stability | 88–96% consistency across rows |

---

## Known Limitations

- **Binary zone classification** — a zone with 1 product and a zone with 6 products both show as stocked. Fill-level scoring (detection count / expected capacity) is planned as a next step.
- **Testing environment** — development and testing were done using retail shelf images displayed on a laptop screen. On a real physical shelf, accuracy and stability are higher.
- **FPS** — 1.3 fps on Jetson Nano CPU. Upgrading to Jetson AGX Orin with TensorRT would increase this to 60+ fps.

---

## Future Work

- Fill-level scoring per zone (0–100%) instead of binary stocked/empty
- SKU-level product identification
- Multi-camera support
- Inventory management API integration
- Historical occupancy analytics
- Predictive restocking using trend data
- Planogram compliance checking

---

## Course Information

**Course:** DATA 690 — Applied Artificial Intelligence  
**Institution:** University of Maryland, Baltimore County (UMBC)  
**Program:** Data Science  
**Semester:** Spring 2025  
**Student:** Tilak Raj  

---

## License

This project is built for academic purposes as part of the UMBC DATA 690 course.
