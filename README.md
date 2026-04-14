<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=Pothole%20Detection%20AI&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Road%20safety%20isn't%20always%20visible.%20So%20we%20made%20it%20visible.&descSize=16&descAlignY=58&descColor=cccccc" />

<br/>

![YOLOv8n](https://img.shields.io/badge/YOLOv8n_ONNX-~41ms_CPU-000000?style=flat-square&labelColor=6C63FF&color=111)
![mAP](https://img.shields.io/badge/mAP@50-82.2%25-000000?style=flat-square&labelColor=00C896&color=111)
![Android](https://img.shields.io/badge/Android-Kotlin-000000?style=flat-square&labelColor=3DDC84&color=111)
![GPS](https://img.shields.io/badge/GPS-Location_Aware-000000?style=flat-square&labelColor=FF6B6B&color=111)
![PWA](https://img.shields.io/badge/PWA-Installable-000000?style=flat-square&labelColor=5A67D8&color=111)
![Spaces](https://img.shields.io/badge/HuggingFace-Spaces-000000?style=flat-square&labelColor=FFD21E&color=111)

<br/>

</div>

---

Every year, potholes cause millions of vehicle accidents and billions in damage — hazards that appear without warning and disappear from public records just as fast. This project changes that. **Pothole Detection AI** is a full-stack road safety system: a fine-tuned YOLOv8n model detects potholes in real time, a GPS layer logs every finding to a live map, and a voice alert system warns drivers before they reach the hazard — all running on a phone, no special hardware required.

<br/>

---

## Demo

<div align="center">

[![Watch the Demo](https://img.youtube.com/vi/j16wN9h2V9I/maxresdefault.jpg)](https://youtu.be/j16wN9h2V9I)

*Click to watch — Road safety isn't always visible. So we made it visible.*

</div>

<br/>

---

## The Product

Three screens. One purpose.

```
NAVIGATE          DETECT            REPORT
─────────         ──────────        ──────────────
GPS monitoring    Camera / gallery  One-tap location
Voice alerts      AI bounding box   Severity tagging
Distance-aware    Community log     Community feed
Start / Stop      Confidence score  Low / Med / High
```

The **Navigate** tab runs silently in the background while you drive — polling GPS every 3 seconds and firing spoken warnings when you close in on a logged pothole. The **Detect** tab runs the model on any road image and pushes the result to a shared database. The **Report** tab lets anyone tag a hazard manually, building the community dataset over time.

<br/>

---

## Voice Alert Logic

> *This project uses a YOLO-based model to detect potholes from road data. It integrates with a GPS-based mobile interface for location-aware detection. The system estimates the distance to hazards ahead, and a voice assistant alerts the driver — improving road safety in real time.*

```
GPS polls every 3 seconds
  └── Haversine distance → each logged pothole
        ├── 100–150 m   →   "Caution. Pothole detected ahead."
        ├──  50–100 m   →   "Warning. Pothole detected. Slow down."
        └──     < 50 m  →   "Danger. Pothole immediately ahead."
```

Alerts are spoken via the **Web Speech API** and fire even when the tab is backgrounded — so Google Maps can stay open in the foreground.

<br/>

---

## Model Performance

| Metric | Result |
|---|---|
| mAP@50 | **82.2 %** |
| Precision | **82.2 %** |
| Recall | **74.4 %** |
| Inference speed (CPU) | **~41 ms / frame** |
| Architecture | YOLOv8n ONNX |
| Training data | 656 annotated dashcam frames |

<br/>

---

## Architecture

```
Pothole Navigate — Android (Kotlin, Pixel 7 API 37)
│
├── Navigate Tab
│     GPS watchPosition() ──► Haversine distance check
│     Voice alert ◄────────── Web Speech API (background-safe)
│
├── Detect Tab
│     Image input (Camera / Gallery)
│     └── YOLOv8n ONNX ──► Bounding boxes + confidence scores
│                      └──► GPS-tagged entry ──► Community DB
│
└── Report Tab
      GPS auto-capture ──► Severity (Low / Med / High)
      "Report Pothole" ──► Community DB
      Community feed   ◄── Live refresh


Web App — Gradio 5.29 + FastAPI (Hugging Face Spaces)
│
├── Same detection pipeline (ONNX, server-side)
├── Folium map — dark CartoDB tiles, red pulsing pins
├── gr.Timer(5s) ──► server-side proximity check
└── Service Worker ──► PWA, offline map tile cache
```

<br/>

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | YOLOv8n ONNX via Ultralytics |
| Android App | Kotlin — Pixel 7 API 37 |
| Web Backend | Gradio 5.29 + FastAPI + uvicorn |
| Map | Folium + CartoDB dark_matter |
| GPS / Voice | Browser Web APIs + Android Location |
| Community DB | Shared GPS + severity + timestamp log |
| Offline | Service Worker + Web App Manifest |
| Deployment | Hugging Face Spaces |

<br/>

---

## Project Structure

```
pothole-detection/
│
├── app/
│   ├── app.py              Main web app — Gradio UI + FastAPI + detection
│   ├── alert_system.py     Standalone desktop alert (OpenCV + pyttsx3)
│   └── best.onnx           Fine-tuned YOLOv8n weights
│
├── android/
│   └── PotholeDetectionApp/
│       ├── manifests/
│       ├── kotlin+java/    MainActivity, Detection, Alert, Report
│       └── res/
│
├── landing/
│   └── index.html          Cinematic hero landing page
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_training.ipynb
│
└── requirements.txt
```

<br/>

---

## Quick Start

**Install**
```bash
git clone https://github.com/poojithamadhyala/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
```

**Run the web app**
```bash
cd app
python app.py
# → http://localhost:7860
```

> GPS requires HTTPS or `localhost` — `navigator.geolocation` will not activate on a plain LAN address.

**Run the local alert system**
```bash
python app/alert_system.py                        # live webcam
python app/alert_system.py --source road.mp4      # from video file
python app/alert_system.py --demo                 # no camera needed
python app/alert_system.py --conf 0.55 --cooldown 3
```

<br/>

---

## Training

```bash
python src/train.py
```

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` |
| Dataset | Pothole Detection v12 — Roboflow |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |

```bash
# Export to ONNX
yolo export model=runs/detect/pothole/weights/best.pt format=onnx imgsz=640
```

<br/>

---

## Deployment — Hugging Face Spaces

```python
from huggingface_hub import HfApi
api = HfApi()
for local, remote in [
    ("app/app.py", "app.py"),
    ("requirements.txt", "requirements.txt")
]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id="YOUR_USERNAME/YOUR_SPACE_NAME",
        repo_type="space",
    )
```

`gradio` and `fastapi` are installed automatically by HF Spaces from the README front-matter. Do not add them to `requirements.txt`.

<br/>

---

## Offline & PWA Behaviour

| Feature | Online | Offline |
|---|:---:|:---:|
| Pothole detection | Yes | No — requires server |
| GPS tracking | Yes | Yes |
| Voice alerts | Yes | Yes (iOS) / Partial (Android) |
| Map display | Full tiles | Cached tiles only |
| App shell | Network | Service worker cache |
| Community database | Live | IndexedDB — planned Phase 4 |

<br/>

---

## Roadmap

| Phase | Status | Scope |
|---|:---:|---|
| Phase 1 — PWA | Done | Service worker, manifest, offline tile cache, Add to Home Screen |
| Phase 2 — Android App | Done | Native Kotlin — Navigate, Detect, Report tabs |
| Phase 3 — Client Inference | Planned | ONNX Runtime Web — full browser inference, no server |
| Phase 4 — Offline Storage | Planned | IndexedDB log + Background Sync |
| Phase 5 — Shared Cloud DB | Future | Globally synced community pothole database |

<br/>

---

<div align="center">

**Poojitha Madhyala**

[![GitHub](https://img.shields.io/badge/GitHub-poojithamadhyala-111?style=flat-square&logo=github)](https://github.com/poojithamadhyala)
[![YouTube](https://img.shields.io/badge/Demo-YouTube-FF0000?style=flat-square&logo=youtube)](https://youtu.be/j16wN9h2V9I)

*Built for safer roads.*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=120&section=footer" />

</div>
