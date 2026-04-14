<div align="center">

<br/>

# Road safety isn't always visible.
## *So we made it visible.*

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=22&pause=2000&color=FF4444&center=true&vCenter=true&width=700&lines=🚗+Pothole+Detection+AI;Live+Detection.+GPS+Mapping.+Voice+Alerts.;YOLOv8n+%7C+Android+%7C+Community+Reporting" alt="Typing SVG" />

<br/><br/>

![Model](https://img.shields.io/badge/Model-YOLOv8n_ONNX-6C63FF?style=for-the-badge&logo=pytorch&logoColor=white)
![mAP](https://img.shields.io/badge/mAP%4050-82.2%25-00C896?style=for-the-badge)
![Platform](https://img.shields.io/badge/Android-Native_App-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![GPS](https://img.shields.io/badge/GPS-Location_Aware-FF6B6B?style=for-the-badge&logo=googlemaps&logoColor=white)
![Voice](https://img.shields.io/badge/Voice-Driver_Alerts-FFD21E?style=for-the-badge)
![PWA](https://img.shields.io/badge/PWA-Web_Ready-5A67D8?style=for-the-badge&logo=pwa&logoColor=white)

<br/>

> Every year, millions of vehicle accidents are caused by potholes — hazards that are **easy to miss and hard to avoid**.  
> **Pothole Detection AI** uses real-time computer vision, GPS tracking, and voice warnings to keep drivers safe.

<br/>

---

</div>

## 🎬 Demo Video

> *"Road safety isn't always visible. So we made it visible."*

The system detects potholes in real-time, pins them on a live GPS map, and speaks driver alerts — even when your navigation app is open in the foreground.

<!-- Replace with your actual video link -->
<!-- [![Watch the Demo](https://img.shields.io/badge/▶_Watch_Demo-FF0000?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK) -->

---

## 📱 App Screens — Pothole Navigate

<div align="center">

| 🗺️ Navigate | 📸 Detect | 📋 Report |
|:---:|:---:|:---:|
| GPS monitoring + voice alerts | AI detection + community logging | Manual reporting + severity tagging |
| Start / Stop monitoring | Camera or gallery upload | Low / Medium / High severity |
| Distance-aware spoken warnings | Bounding boxes + confidence scores | Live community database feed |

</div>

---

## 🌟 Key Features

| | Feature | Description |
|--|--------|-------------|
| 🎯 | **AI Detection** | YOLOv8n detects potholes with **82.2% mAP@50** on real dashcam footage |
| 📍 | **Live GPS Logging** | Every detection auto-pins to a GPS map — no manual input needed |
| 🔊 | **Voice Driver Alerts** | Spoken distance warnings: *"Pothole ahead — 80 metres"* |
| 🗺️ | **Community Database** | Detections from all users feed a shared pothole map |
| 📊 | **Severity Tagging** | AI and community reports tagged Low / Medium / High |
| 📲 | **Installable PWA** | Add to home screen like a native app — no App Store needed |

---

## 🔊 How the Voice Alert System Works

> *This project uses a YOLO-based model to detect potholes from road data. It integrates with a GPS-based mobile interface for location-aware detection. The system estimates the distance to hazards ahead, and a voice assistant alerts the driver — improving road safety in real time.*

```
Driver is moving →
  GPS polls location every 3 seconds →
    Haversine formula calculates distance to each logged pothole →
      🟡 150–100m  →  "Caution! Pothole detected ahead."
      🟠 100–50m   →  "Warning! Pothole detected. Slow down!"
      🔴  < 50m    →  "Danger! Pothole immediately ahead. Slow down!"
```

Alerts fire via **Web Speech API** — works even when the tab is backgrounded with Google Maps open.

---

## 📊 Model Performance

<div align="center">

| Metric | Value |
|--------|-------|
| 🎯 mAP@50 | **82.2%** |
| 🔍 Precision | **82.2%** |
| 📡 Recall | **74.4%** |
| ⚡ Inference Speed (CPU) | **~41 ms / frame** |
| 🏗️ Architecture | YOLOv8n ONNX |
| 🖼️ Training Data | 656 annotated dashcam frames |

</div>

---

## 🏗️ System Architecture

```
📱 Pothole Navigate App
│
├── 🗺️  Navigate Tab
│   ├── GPS watchPosition() → monitors location every 3s
│   ├── Haversine distance calculation to known potholes
│   ├── Voice alerts via Web Speech API (background-safe)
│   └── Start / Stop monitoring toggle
│
├── 📸  Detect Tab
│   ├── Upload via Camera or Gallery
│   ├── YOLOv8n ONNX inference (~41ms, CPU)
│   ├── Bounding boxes + confidence scores rendered
│   ├── GPS coordinates captured automatically
│   └── Detection logged to community database
│
└── 📋  Report Tab
    ├── One-tap GPS location capture
    ├── Severity selection: Low / Medium / High
    ├── "Report Pothole Here" → writes to shared DB
    └── Community reports feed (live, refreshable)
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **AI Model** | YOLOv8n ONNX via Ultralytics |
| **Mobile App** | Android (Kotlin) — Pixel 7 API 37 |
| **Web App** | Gradio 5.29 + FastAPI + uvicorn |
| **Map** | Folium + CartoDB dark_matter tiles |
| **GPS + Voice** | Browser Web APIs / Android Location APIs |
| **Community DB** | Shared pothole log (GPS + severity + timestamp) |
| **Offline** | Service Worker + PWA manifest |
| **Deployment** | Hugging Face Spaces |

</div>

---

## 📁 Project Structure

```
pothole-detection/
├── 📂 app/
│   ├── 🐍 app.py              # Gradio UI + FastAPI + detection logic
│   ├── 🔔 alert_system.py     # Standalone desktop alert (OpenCV + pyttsx3)
│   └── 🤖 best.onnx           # Fine-tuned YOLOv8n weights
├── 📂 android/                # Native Android app (Kotlin)
│   └── PotholeDetectionApp/
│       ├── manifests/
│       ├── kotlin+java/       # MainActivity, Detection, Alert logic
│       └── res/
├── 📂 landing/
│   └── 🌐 index.html          # Cinematic hero landing page
├── 📂 src/
│   ├── train.py               # YOLOv8 training script
│   ├── predict.py             # Batch prediction
│   └── evaluate.py            # Model evaluation
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_training.ipynb
└── 📄 requirements.txt
```

---

## ⚡ Quick Start — Web App

### 1️⃣ Clone & Install

```bash
git clone https://github.com/poojithamadhyala/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
```

### 2️⃣ Run

```bash
cd app
python app.py
# → Open http://localhost:7860
```

> 💡 GPS requires **HTTPS or localhost** — `navigator.geolocation` won't activate on plain `http://192.168.x.x`

### 3️⃣ Local Alert System (Webcam / Dashcam)

```bash
python app/alert_system.py                         # Live webcam
python app/alert_system.py --source road.mp4       # From video file
python app/alert_system.py --demo                  # Demo mode
python app/alert_system.py --conf 0.55 --cooldown 3
```

---

## 📱 Using the Android App

### 🗺️ Navigate Tab — Driver Alert Mode
1. Tap **Start Monitoring** — GPS activates, green dot confirms *"GPS active — monitoring..."*
2. Toggle **Voice Alerts** on
3. Drive normally — spoken warnings fire automatically as you approach logged potholes
4. Keep Google Maps open separately; alerts still fire in background

### 📸 Detect Tab — AI Detection
1. Tap **Camera** or **Gallery** to load a road image
2. Hit **Detect & Log Potholes** — model annotates with bounding boxes instantly
3. Detections are logged with GPS coordinates to the community database
4. Confidence scores and total pothole count shown

### 📋 Report Tab — Community Reporting
1. GPS coordinates are auto-captured (green dot confirms location)
2. Select severity: **Low / Medium / High**
3. Tap **Report Pothole Here** — pins it to the shared community map
4. Scroll the **Community Reports** feed to see all logged hazards from all users

---

## 🚨 Alert Severity Levels

| Level | Distance | Voice Message |
|-------|----------|---------------|
| 🟡 Caution | 100–150 m | *"Caution! Pothole detected ahead."* |
| 🟠 Warning | 50–100 m | *"Warning! Pothole detected. Slow down!"* |
| 🔴 Danger | < 50 m | *"Danger! Pothole immediately ahead. Slow down!"* |

---

## 🌐 PWA / Offline Support

| Feature | Online | Offline |
|---------|:------:|:-------:|
| Pothole detection | ✅ | ❌ Needs server |
| GPS tracking | ✅ | ✅ |
| Voice alerts | ✅ | ✅ iOS / ⚠️ Android |
| Map display | ✅ Full tiles | ✅ Cached tiles |
| App shell | ✅ Network | ✅ Service worker |
| Community database | ✅ Live | 🔄 IndexedDB planned |

---

## 🏋️ Training Your Own Model

```bash
python src/train.py
```

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n.pt` |
| Dataset | Pothole Detection v12 (Roboflow) |
| Epochs | 100 |
| Image size | 640 |
| Batch | 16 |
| Optimizer | AdamW |

```bash
# Export to ONNX after training
yolo export model=runs/detect/pothole/weights/best.pt format=onnx imgsz=640
```

---

## 🚢 Deploy to Hugging Face Spaces

```python
from huggingface_hub import HfApi
api = HfApi()
for local, remote in [("app/app.py", "app.py"), ("requirements.txt", "requirements.txt")]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id="YOUR_USERNAME/YOUR_SPACE_NAME",
        repo_type="space",
    )
```

> `gradio` and `fastapi` are auto-installed by HF Spaces — don't add them to `requirements.txt`.

---

## 🗺️ Roadmap

| Phase | Status | What |
|-------|:------:|------|
| **Phase 1 — PWA** | ✅ Done | Service worker, manifest, offline map cache, Add to Home Screen |
| **Phase 2 — Android App** | ✅ Done | Native Kotlin app — Navigate, Detect, Report tabs |
| **Phase 3 — Client Inference** | 🔄 Planned | ONNX Runtime Web — detect entirely in browser, no server |
| **Phase 4 — Offline Storage** | 🔄 Planned | IndexedDB + Background Sync for offline-first |
| **Phase 5 — Shared Cloud DB** | 🔮 Future | Synced community pothole database across all users globally |

---

## 👩‍💻 Author

<div align="center">

**Poojitha Madhyala**

[![GitHub](https://img.shields.io/badge/GitHub-poojithamadhyala-181717?style=for-the-badge&logo=github)](https://github.com/poojithamadhyala)

*Built for safer roads — one detection at a time* 🚗

<br/>

⭐ **If this project helped you, please give it a star!** ⭐

</div>
