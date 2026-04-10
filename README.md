---
title: Pothole Detection AI
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
short_description: YOLOv8n pothole detection with GPS driver alerts
---

# Pothole Detection AI

A full-stack road safety system that detects potholes in real-time using a fine-tuned YOLOv8n model, logs them on a live GPS map, alerts drivers with voice warnings before they reach a hazard, and works as an installable PWA on your phone.

---

## What This App Does — Plain English

1. **You upload a road photo** → the AI finds potholes and draws boxes around them
2. **Your phone's GPS is captured automatically** in the browser — no typing needed
3. **Every detection is pinned on a live map** with the exact coordinates
4. **Driver Alert Mode** monitors your location every 3 seconds while you drive. When you come within 150 m of a logged pothole, your phone speaks a warning:
   - 🟡 100–150 m → *"Caution! Pothole detected X metres ahead."*
   - 🟠 50–100 m → *"Warning! Pothole detected X metres ahead. Slow down!"*
   - 🔴 < 50 m → *"Danger! Pothole detected X metres ahead. Slow down!"*
5. **You can install it to your phone's home screen** like a native app (no app store needed)

---

## How the App is Built — Architecture

```
Browser (your phone)
│
├── GPS            navigator.geolocation.watchPosition()
│                  → writes "lat,lon" into a hidden Gradio textbox
│                  → Python reads it on every detection
│
├── Detection      You upload an image
│                  → Gradio sends it to Python
│                  → Python runs YOLOv8n ONNX model (cv2 + ultralytics)
│                  → Returns annotated image + detection summary
│                  → If GPS is active, logs {lat, lon, timestamp, conf} to pothole_log
│
├── Map            Python builds a Folium map (dark CartoDB tiles)
│                  → Renders it as an HTML string inside an <iframe>
│                  → Pothole pins are red pulsing circles, driver dot is blue
│
├── Driver Alerts  Two parallel systems:
│   ├── Server-side  gr.Timer(5s) → Python checks proximity → updates alert panel HTML
│   └── Client-side  JS setInterval(3s) → reads pothole DB from hidden textbox
│                                        → runs Haversine formula in JS
│                                        → speaks via Web Speech API
│                  Client-side runs even when tab is backgrounded (Google Maps open)
│
└── PWA            Service Worker (sw.js) caches app shell + map tiles
                   Web App Manifest (manifest.json) enables "Add to Home Screen"
```

### Server stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Python backend | Gradio 5.29 + FastAPI | Gradio for UI; FastAPI to serve `/sw.js`, `/manifest.json`, icons |
| Model | YOLOv8n ONNX via Ultralytics | Fast CPU inference (~41 ms/frame) |
| Map | Folium + CartoDB dark_matter | No API key, offline tiles via SW cache |
| GPS/Voice/SW | Browser Web APIs | GPS, speech synthesis, PWA — all native browser features |
| Deployment | Hugging Face Spaces + uvicorn | HF detects the `app` FastAPI variable and runs uvicorn |

---

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | **0.822** |
| Precision | **82.2 %** |
| Recall | **74.4 %** |
| Inference speed (CPU) | **~41 ms / frame** |
| Architecture | YOLOv8n ONNX |
| Training images | 656 annotated dashcam frames |

---

## Project Structure

```
pothole-detection/
├── app/
│   ├── app.py              # Main app — Gradio UI + FastAPI PWA routes + detection logic
│   ├── alert_system.py     # Standalone desktop alert (OpenCV + pyttsx3)
│   └── best.onnx           # Fine-tuned YOLOv8n model weights
├── landing/
│   └── index.html          # Web3-style hero landing page (standalone HTML)
├── src/
│   ├── train.py            # YOLOv8 training script
│   ├── predict.py          # Batch prediction script
│   └── evaluate.py         # Model evaluation script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_training.ipynb
├── runs/
│   ├── detect/             # Training outputs (weights, metrics, plots)
│   └── predictions/        # Sample output images
├── requirements.txt
└── README.md
```

---

## Key Files Explained

### `app/app.py` — What's inside

The entire web app lives in one file. Here's a map of the important sections:

```
app.py
│
├── Imports + model loading
│   └── Tries 3 candidate paths for best.onnx (local dev + HF Spaces)
│
├── _make_icon_png(size)
│   └── Generates 192×192 and 512×512 PNG icons using cv2 at startup
│       (Chrome requires PNG icons to trigger the PWA install prompt)
│
├── GPS helpers
│   ├── haversine_m()     great-circle distance in metres
│   ├── parse_gps()       "lat,lon" string → (float, float)
│   └── nearby_potholes() filters log entries within 150 m
│
├── Map helpers
│   ├── build_map_html()  Folium dark map → HTML string for <iframe>
│   └── build_log_md()    pothole log → Markdown table
│
├── Alert HTML
│   └── build_alert_html()  returns color-coded driver alert panel HTML
│
├── Core functions
│   ├── detect_potholes()   main detection function — handles ALL image formats:
│   │                       uint8 / float32, RGB / RGBA / grayscale
│   ├── check_proximity()   called by gr.Timer every 5 s (server-side)
│   └── clear_log()         resets pothole database
│
├── PWA assets (Python strings served as files)
│   ├── ICON_SVG            SVG icon (fallback)
│   ├── SW_CODE             service worker JavaScript
│   ├── MANIFEST            web app manifest dict
│   └── PWA_HEAD            HTML injected into <head>:
│                           manifest link, Apple meta tags,
│                           SW registration, install prompt handler
│
├── GPS_JS                  large JS block injected into <head>:
│                           GPS watchPosition, voice alerts (Web Speech API),
│                           Haversine in JS, pothole DB reader,
│                           beep sounds via AudioContext
│
├── CSS                     ~300 lines of dark indigo-navy theme
│
├── Gradio UI
│   ├── Tab 1: Detect Potholes   image upload → detect → map + log
│   └── Tab 2: Driver Alert Mode  voice activate + proximity panel
│
└── FastAPI wrapper
    ├── GET /manifest.json   → web app manifest (JSON)
    ├── GET /sw.js           → service worker (JS, Service-Worker-Allowed: /)
    ├── GET /icon-192.png    → 192×192 PNG (Chrome installability requirement)
    ├── GET /icon-512.png    → 512×512 PNG (maskable, Android home screen)
    ├── GET /icon.svg        → SVG icon (fallback)
    ├── gr.mount_gradio_app(fapp, demo, path="/")
    └── app = fapp           ← HF Spaces detects this and runs uvicorn
```

---

## Quick Start — Local Development

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
cd app
python app.py
```

This starts a uvicorn server at [http://localhost:7860](http://localhost:7860).

> **GPS note:** `navigator.geolocation` requires **HTTPS** or `localhost`. It will not work on plain `http://192.168.x.x` — use `localhost` or deploy to HF Spaces.

### 3. Local alert system (webcam / dashcam)

```bash
# Real-time webcam
python app/alert_system.py

# From a video file
python app/alert_system.py --source road.mp4

# Demo mode (no webcam)
python app/alert_system.py --demo

# Custom threshold
python app/alert_system.py --conf 0.55 --cooldown 3
```

---

## How to Use the Web App

### Detect Potholes tab

1. Open the app on your phone (Chrome on Android, or Safari on iOS)
2. Allow location access when prompted — GPS coordinates appear in the status bar
3. Upload a road photo from any source: camera, gallery, clipboard, or web
4. Click **Detect potholes** — the model annotates the image and:
   - Shows bounding boxes with confidence scores
   - Logs the location to the pothole database
   - Pins the location on the live map

> The app handles all image formats automatically:
> RGB, RGBA (screenshots), grayscale, float32 — all normalised before inference.

### Driver Alert Mode tab

1. First detect potholes on your usual routes to build the database
2. Switch to this tab and tap **"🔊 Tap here first to activate voice alerts"** (browser requires a user gesture to unlock audio)
3. Open Google Maps (or any navigation app) in a new tab and start driving
4. This tab keeps running in the background and speaks when you approach a pothole
5. The server also checks proximity every 5 seconds and updates the alert panel

### Installing to Home Screen

The app works as a Progressive Web App (PWA). Tap the **"📲 Add to Home Screen"** button in the hero section:

- **If Chrome shows the native install dialog** → tap Install
- **If no dialog appears** (Chrome 121+ or iOS) → the button shows manual instructions:

**Android (Chrome):**
1. Tap the three-dot menu (⋮) in Chrome
2. Tap **"Add to Home Screen"** or **"Install app"**
3. Tap Add

**iPhone / iPad (Safari):**
1. Tap the Share button (↑) at the bottom
2. Scroll down and tap **"Add to Home Screen"**
3. Tap Add

Once installed, the app opens fullscreen with no browser chrome, just like a native app.

---

## Local Alert System — Severity Levels

`alert_system.py` estimates severity from bounding-box area relative to frame size:

| Severity | Bbox / Frame area | Meaning |
|----------|:-----------------:|---------|
| Low | < 4 % | Pothole is far ahead |
| Medium | 4 – 12 % | Approaching |
| High | > 12 % | Immediately ahead — danger |

Alerts are logged to `alert_log.csv`: `timestamp`, `severity`, `confidence`, `x1`, `y1`, `x2`, `y2`.

---

## PWA / Offline Behaviour

| Feature | Online | Offline |
|---------|--------|---------|
| Pothole detection | ✅ Full | ❌ Requires Python server |
| GPS tracking | ✅ | ✅ Native browser API |
| Voice alerts | ✅ | ✅ (iOS) / ⚠️ needs net (Android Chrome) |
| Map display | ✅ Full tiles | ✅ Cached tiles only (previously visited areas) |
| App shell load | ✅ Network | ✅ From service worker cache |
| Pothole database | ✅ In-memory | ❌ Lost on page reload (Phase 3: IndexedDB) |

> **Planned (Phase 3):** Store detections in IndexedDB so they survive offline/reload, and sync to a shared cloud database when connectivity returns.

---

## Deployment — Hugging Face Spaces

### How it works on HF

The README front-matter sets `sdk: gradio`. HF Spaces runs `python app.py`. Because the file exposes `app = fapp` (a FastAPI instance), HF detects it and serves the app with uvicorn instead of Gradio's built-in launcher. The Gradio UI is mounted onto FastAPI at `/` via `gr.mount_gradio_app()`.

### Files required in the Space

```
app.py
best.onnx
requirements.txt
README.md
```

### `requirements.txt`

```
ultralytics>=8.2.0
opencv-python-headless>=4.9.0
folium>=0.17.0
PyYAML>=6.0.1
uvicorn>=0.29.0
```

> `gradio` and `fastapi` are **not** listed — they are installed automatically by HF Spaces based on `sdk_version: "5.29.0"` in the README front-matter.

### Upload via Python

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

> **HTTPS required:** `navigator.geolocation`, `SpeechSynthesisUtterance`, and Service Workers all require HTTPS. HF Spaces provides this automatically.

---

## Training

Model trained on the [Pothole Detection v12 dataset](https://universe.roboflow.com/) (YOLOv8 format).

```bash
python src/train.py
```

| Parameter | Value |
|-----------|-------|
| Base model | yolov8n.pt |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |

Export to ONNX after training:

```bash
yolo export model=runs/detect/runs/pothole/yolov8n-v1/weights/best.pt format=onnx imgsz=640
```

---

## Roadmap

| Phase | Status | What |
|-------|--------|------|
| **Phase 1 — PWA** | ✅ Done | Service worker, manifest, offline map tile cache, "Add to Home Screen" |
| **Phase 2 — Client inference** | Planned | ONNX Runtime Web — detect potholes entirely in the browser, no server needed |
| **Phase 3 — Offline storage** | Planned | IndexedDB for local pothole log + Background Sync to upload when online |
| **Phase 4 — Native app** | Future | React Native + ONNX Runtime native — hardware-accelerated, background GPS |

---

## Author

**Poojitha Madhyala** — [GitHub](https://github.com/poojithamadhyala)
