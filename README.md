<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=🚗+Pothole+Detection+AI;Real-Time+Road+Safety+System;YOLOv8n+%2B+GPS+%2B+Voice+Alerts" alt="Typing SVG" />

<br/>

![Model](https://img.shields.io/badge/Model-YOLOv8n_ONNX-6C63FF?style=for-the-badge&logo=pytorch&logoColor=white)
![mAP](https://img.shields.io/badge/mAP%4050-82.2%25-00C896?style=for-the-badge)
![Inference](https://img.shields.io/badge/Inference-~41ms_CPU-FF6B6B?style=for-the-badge)
![PWA](https://img.shields.io/badge/PWA-Ready-5A67D8?style=for-the-badge&logo=pwa&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-5.29-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces-FFD21E?style=for-the-badge)

<br/>

> **A full-stack road safety system** — detect potholes in real-time, pin them on a live GPS map, and get spoken voice warnings before you drive into one.

<br/>

---

</div>

## 🌟 What Makes This Special?

| Feature | Description |
|--------|-------------|
| 🎯 **AI Detection** | Fine-tuned YOLOv8n model with **82.2% mAP@50** on real dashcam footage |
| 📍 **Auto GPS Logging** | Browser captures your location — every pothole is pinned on a live map |
| 🔊 **Voice Alerts** | Spoken warnings at **150m → 100m → 50m** — works even with tab backgrounded |
| 📲 **Installable PWA** | Add to home screen like a native app — no App Store needed |
| 🌐 **100% Browser APIs** | GPS, speech synthesis, service worker — all native, no third-party SDKs |

---

## 🚀 Live Demo

<div align="center">

[![Open in HuggingFace Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/poojithamadhyala/pothole-detection)

</div>

---

## 🗺️ How It Works

```
📱 Your Phone
│
├── 📸 Upload Road Photo
│   └── YOLOv8n ONNX → Bounding boxes + confidence scores
│
├── 📍 GPS Auto-Capture
│   └── navigator.geolocation → coordinates pinned to detection log
│
├── 🗺️ Live Folium Map
│   └── Red pulsing circles = potholes  |  Blue dot = you
│
└── 🔊 Driver Alert Mode
    ├── 🟡 150–100m  →  "Caution! Pothole detected ahead."
    ├── 🟠 100–50m   →  "Warning! Slow down!"
    └── 🔴 < 50m     →  "Danger! Pothole immediately ahead!"
```

---

## 📊 Model Performance

<div align="center">

| Metric | Value |
|--------|-------|
| 🎯 mAP@50 | **82.2%** |
| 🔍 Precision | **82.2%** |
| 📡 Recall | **74.4%** |
| ⚡ Inference (CPU) | **~41 ms/frame** |
| 🏗️ Architecture | YOLOv8n ONNX |
| 🖼️ Training Images | 656 annotated dashcam frames |

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **Backend** | Gradio 5.29 + FastAPI + uvicorn |
| **Model** | YOLOv8n ONNX via Ultralytics |
| **Map** | Folium + CartoDB dark_matter tiles |
| **GPS / Voice** | Browser Web APIs (no API key needed) |
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
├── 📂 landing/
│   └── 🌐 index.html          # Hero landing page
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

## ⚡ Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/poojithamadhyala/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
```

### 2️⃣ Run the Web App

```bash
cd app
python app.py
# → Open http://localhost:7860
```

> 💡 **GPS requires HTTPS or localhost** — `navigator.geolocation` won't work on plain `http://192.168.x.x`

### 3️⃣ Local Alert System (Webcam / Dashcam)

```bash
# Webcam (live)
python app/alert_system.py

# From a video file
python app/alert_system.py --source road.mp4

# Demo mode (no camera)
python app/alert_system.py --demo

# Custom threshold
python app/alert_system.py --conf 0.55 --cooldown 3
```

---

## 📱 Using the Web App

### 🔍 Detect Potholes Tab
1. Open the app on your phone (Chrome on Android / Safari on iOS)
2. **Allow location access** — GPS coordinates appear in the status bar
3. Upload a road photo (camera, gallery, screenshot — all formats supported)
4. Hit **Detect Potholes** → model annotates the image, logs the GPS location, and pins it on the map

### 🔊 Driver Alert Mode Tab
1. First build your pothole database by detecting on your usual routes
2. Tap **"🔊 Tap here to activate voice alerts"** (required browser gesture)
3. Open Google Maps in another tab and start driving
4. Alerts fire automatically as you approach logged potholes

### 📲 Install as PWA
**Android (Chrome):** Menu (⋮) → Add to Home Screen → Install  
**iPhone (Safari):** Share (↑) → Add to Home Screen → Add

---

## 🚨 Alert Severity Levels

| Severity | Distance | Bbox / Frame | Message |
|----------|----------|:---:|---------|
| 🟡 Caution | 100–150 m | < 4% | *"Caution! Pothole detected X metres ahead."* |
| 🟠 Warning | 50–100 m | 4–12% | *"Warning! Pothole detected ahead. Slow down!"* |
| 🔴 Danger | < 50 m | > 12% | *"Danger! Pothole immediately ahead. Slow down!"* |

---

## 🌐 PWA / Offline Support

| Feature | Online | Offline |
|---------|:------:|:-------:|
| Pothole detection | ✅ | ❌ Needs server |
| GPS tracking | ✅ | ✅ |
| Voice alerts | ✅ | ✅ iOS / ⚠️ Android |
| Map display | ✅ Full | ✅ Cached tiles |
| App shell load | ✅ Network | ✅ Service worker |
| Pothole database | ✅ In-memory | 🔄 IndexedDB planned |

---

## 🏋️ Training Your Own Model

```bash
# Train
python src/train.py
```

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n.pt` |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Dataset | Pothole Detection v12 (Roboflow) |

```bash
# Export to ONNX
yolo export model=runs/detect/pothole/yolov8n-v1/weights/best.pt format=onnx imgsz=640
```

---

## 🚢 Deploy to Hugging Face Spaces

### Files required:
```
app.py  |  best.onnx  |  requirements.txt  |  README.md
```

### Upload via Python:
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

> `gradio` and `fastapi` are auto-installed by HF Spaces from the README front-matter — don't add them to `requirements.txt`.

---

## 🗺️ Roadmap

| Phase | Status | What |
|-------|:------:|------|
| **Phase 1 — PWA** | ✅ Done | Service worker, manifest, offline map cache, "Add to Home Screen" |
| **Phase 2 — Client Inference** | 🔄 Planned | ONNX Runtime Web — detect entirely in browser, no server needed |
| **Phase 3 — Offline Storage** | 🔄 Planned | IndexedDB local log + Background Sync to upload when online |
| **Phase 4 — Native App** | 🔮 Future | React Native + ONNX Runtime native — background GPS + hardware acceleration |

---

## 🏗️ Architecture Deep Dive

<details>
<summary><b>Click to expand full architecture diagram</b></summary>

```
Browser (your phone)
│
├── GPS            navigator.geolocation.watchPosition()
│                  → writes "lat,lon" into a hidden Gradio textbox
│                  → Python reads it on every detection
│
├── Detection      Upload image → Gradio → Python
│                  → YOLOv8n ONNX (cv2 + ultralytics)
│                  → Returns annotated image + detection summary
│                  → Logs {lat, lon, timestamp, conf} to pothole_log
│
├── Map            Folium map (dark CartoDB tiles) → HTML string → <iframe>
│                  → Red pulsing circles = potholes
│                  → Blue dot = driver position
│
├── Driver Alerts  Two parallel systems:
│   ├── Server     gr.Timer(5s) → Python Haversine → alert panel HTML
│   └── Client     JS setInterval(3s) → reads pothole DB → Haversine in JS
│                  → Web Speech API (works in background tab)
│
└── PWA            sw.js → caches app shell + map tiles
                   manifest.json → "Add to Home Screen"
```

</details>

---

## 👩‍💻 Author

<div align="center">

**Poojitha Madhyala**

[![GitHub](https://img.shields.io/badge/GitHub-poojithamadhyala-181717?style=for-the-badge&logo=github)](https://github.com/poojithamadhyala)

*Built with ❤️ for safer roads*

</div>

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
