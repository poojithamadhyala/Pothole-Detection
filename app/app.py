import cv2
import numpy as np
import gradio as gr
import folium
import html as html_lib
import math
import json
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response as FastAPIResponse

# HEIC/HEIF support for images from iOS devices
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    _os.path.join(_HERE, "best.onnx"),
    _os.path.join(_HERE, "..", "best.onnx"),
    _os.path.join(_HERE, "..", "runs", "detect", "runs",
                  "pothole", "yolov8n-v1", "weights", "best.onnx"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if _os.path.exists(p)), _MODEL_CANDIDATES[0])

CONF_THRESH    = 0.25
IOU_THRESH     = 0.45
IMAGE_SIZE     = 640
BOX_COLOR      = (59, 130, 246)
TEXT_COLOR     = (255, 255, 255)
BOX_THICKNESS  = 2
FONT_SCALE     = 0.55
ALERT_RADIUS_M = 150   # warn driver when within this many metres of a logged pothole

print("[+] Loading ONNX model...")
model = YOLO(MODEL_PATH)
print("[+] Model ready")


def _make_icon_png(size: int) -> bytes:
    """Generate a 192/512-px PNG icon using cv2 — required by Chrome for PWA installability."""
    s   = size
    img = np.zeros((s, s, 3), dtype=np.uint8)
    img[:] = (46, 29, 26)                                          # #1a1d2e background

    # Road band
    img[int(s * 0.43):int(s * 0.57), :] = (85, 65, 51)            # #334155

    # Pothole ellipse
    cx, cy = s // 2, s // 2
    cv2.ellipse(img, (cx, cy), (s // 9, s // 14), 0, 0, 360, (15, 12, 10), -1)
    cv2.ellipse(img, (cx, cy), (s // 9, s // 14), 0, 0, 360,
                (241, 102, 99), max(2, s // 64))                    # #6366f1 ring

    # Warning triangle
    tri = np.array([[cx, int(s * 0.08)],
                    [int(cx + s * 0.29), int(s * 0.43)],
                    [int(cx - s * 0.29), int(s * 0.43)]], np.int32)
    cv2.fillPoly(img, [tri], (241, 102, 99))

    # "!" inside triangle
    fs  = s / 160.0
    thk = max(1, s // 96)
    tw, th = cv2.getTextSize("!", cv2.FONT_HERSHEY_SIMPLEX, fs, thk)[0]
    cv2.putText(img, "!", (cx - tw // 2, int(s * 0.43) - th // 2 - max(2, s // 80)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thk, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", img)
    return bytes(buf)


_ICON_192 = _make_icon_png(192)
_ICON_512 = _make_icon_png(512)


# ── GPS helpers ───────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_gps(gps_str: str):
    """Parse 'lat,lon' string → (float, float) or (None, None)."""
    if not gps_str or "," not in gps_str:
        return None, None
    try:
        lat, lon = map(float, gps_str.split(",", 1))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except ValueError:
        pass
    return None, None


def nearby_potholes(clat: float, clon: float, log: list,
                    radius_m: float = ALERT_RADIUS_M) -> list:
    """Return log entries within radius_m of (clat, clon), sorted by distance."""
    results = []
    for entry in log:
        dist = haversine_m(clat, clon, entry["lat"], entry["lon"])
        if dist <= radius_m:
            results.append({**entry, "dist_m": dist})
    results.sort(key=lambda x: x["dist_m"])
    return results


# ── Map helpers ───────────────────────────────────────────────────────────────

def build_map_html(log: list, curr_lat=None, curr_lon=None) -> str:
    """Dark folium map with pothole pins and optional driver position."""
    if not log and curr_lat is None:
        m = folium.Map(location=[20.0, 0.0], zoom_start=2, tiles="CartoDB dark_matter")
    else:
        centre = ([curr_lat, curr_lon] if curr_lat is not None
                  else [log[-1]["lat"], log[-1]["lon"]])
        m = folium.Map(location=centre, zoom_start=16, tiles="CartoDB dark_matter")

        for i, entry in enumerate(log, 1):
            popup_html = (
                f"<div style='font-family:monospace;font-size:13px;color:#e2e8f0;"
                f"background:#1a1a2e;padding:10px 12px;border-radius:8px;"
                f"border:1px solid #ef4444;min-width:180px;'>"
                f"<b style='color:#ff4444;font-size:14px;'>#{i} — {entry['count']} pothole(s)</b><br><br>"
                f"🕒 {entry['timestamp']}<br>"
                f"📍 {entry['lat']:.6f}, {entry['lon']:.6f}<br>"
                f"🎯 Conf: {entry['max_conf']:.2f}"
                f"</div>"
            )
            pin_html = (
                f'<div style="position:relative;width:44px;height:44px;">'
                f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:44px;height:44px;border-radius:50%;background:rgba(255,60,60,0.25);border:2px solid rgba(255,60,60,0.6);animation:pulse 1.6s ease-out infinite;"></div>'
                f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:28px;height:28px;border-radius:50%;background:#ff3c3c;border:3px solid #ffffff;box-shadow:0 0 8px rgba(255,60,60,0.9);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#fff;">{i}</div>'
                f'</div>'
                f'<style>@keyframes pulse{{0%{{transform:translate(-50%,-50%) scale(1);opacity:1;}}100%{{transform:translate(-50%,-50%) scale(2.2);opacity:0;}}}}</style>'
            )
            folium.Marker(
                location=[entry["lat"], entry["lon"]],
                icon=folium.DivIcon(html=pin_html, icon_size=(44, 44), icon_anchor=(22, 22)),
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"#{i}: {entry['count']} pothole(s) · {entry['timestamp']}",
            ).add_to(m)

        if curr_lat is not None:
            driver_pin = (
                '<div style="position:relative;width:50px;height:50px;">'
                '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:50px;height:50px;border-radius:50%;background:rgba(59,130,246,0.25);border:2px solid rgba(59,130,246,0.7);animation:driverpulse 1.2s ease-out infinite;"></div>'
                '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:22px;height:22px;border-radius:50%;background:#3b82f6;border:3px solid #fff;box-shadow:0 0 10px rgba(59,130,246,0.9);"></div>'
                '</div>'
                '<style>@keyframes driverpulse{0%{transform:translate(-50%,-50%) scale(1);opacity:1;}100%{transform:translate(-50%,-50%) scale(2.5);opacity:0;}}</style>'
            )
            folium.Marker(
                location=[curr_lat, curr_lon],
                icon=folium.DivIcon(html=driver_pin, icon_size=(50, 50), icon_anchor=(25, 25)),
                tooltip="📍 Your current location",
            ).add_to(m)

    raw_html = m.get_root().render()
    escaped  = html_lib.escape(raw_html, quote=True)
    return (
        f'<div class="map-wrap">'
        f'<iframe srcdoc="{escaped}" width="100%" height="460px"></iframe>'
        f'</div>'
    )


def build_log_md(log: list) -> str:
    if not log:
        return "*No detections logged yet. Detect potholes with GPS active to populate the database.*"
    lines  = "### Pothole Database\n\n"
    lines += "| # | Timestamp | Lat | Lon | Potholes | Max Conf |\n"
    lines += "|---|-----------|-----|-----|:--------:|:--------:|\n"
    for i, e in enumerate(log, 1):
        lines += (
            f"| {i} | {e['timestamp']} | {e['lat']:.5f} | {e['lon']:.5f} "
            f"| {e['count']} | {e['max_conf']:.2f} |\n"
        )
    return lines


# ── Alert HTML ────────────────────────────────────────────────────────────────

def build_alert_html(nearby: list, curr_lat=None) -> str:
    if curr_lat is None:
        return """
        <div class="alert-waiting">
          <div class="alert-icon">📡</div>
          <div>
            <div class="alert-title">Waiting for GPS signal…</div>
            <div class="alert-sub">Allow location access in your browser to activate driver alerts</div>
          </div>
        </div>"""

    if not nearby:
        return """
        <div class="alert-clear" id="driver-alert">
          <div class="alert-icon">✅</div>
          <div>
            <div class="alert-title">All Clear — No potholes nearby</div>
            <div class="alert-sub">Actively monitoring within 150 m of your location</div>
          </div>
        </div>"""

    closest  = nearby[0]
    count    = len(nearby)
    dist_m   = closest["dist_m"]
    severity = "HIGH" if dist_m < 50 else "MEDIUM" if dist_m < 100 else "LOW"
    cls      = {"HIGH": "alert-high", "MEDIUM": "alert-medium", "LOW": "alert-low"}[severity]

    return f"""
    <div class="{cls}" id="driver-alert" data-alert="true">
      <div class="alert-icon alert-flash">⚠️</div>
      <div style="flex:1">
        <div class="alert-title">POTHOLE AHEAD — {int(dist_m)} m away</div>
        <div class="alert-sub">
          {count} pothole location{'s' if count > 1 else ''} within 150 m &nbsp;·&nbsp; Severity: {severity}
        </div>
      </div>
      <div class="alert-dist-badge">{int(dist_m)}<span>m</span></div>
    </div>"""


# ── Core functions ────────────────────────────────────────────────────────────

def detect_potholes(image, conf_threshold, gps_str, log):
    lat, lon = parse_gps(gps_str)

    if image is None:
        return None, "Upload an image to get started.", build_map_html(log, lat, lon), build_log_md(log), log, json.dumps(log)

    # Normalise to uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Grayscale → RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    # RGBA → RGB (clipboard / web screenshots)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    img_bgr  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results  = model.predict(
        source=img_bgr, conf=conf_threshold,
        iou=IOU_THRESH, imgsz=IMAGE_SIZE, verbose=False,
    )

    result     = results[0]
    detections = []
    annotated  = img_bgr.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf       = float(box.conf[0])
        class_name = model.names[int(box.cls[0])]
        detections.append({"bbox": (x1, y1, x2, y2), "confidence": round(conf, 3),
                            "class_name": class_name})

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        cv2.rectangle(annotated, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), 1)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    n = len(detections)
    if n == 0:
        summary = "### No potholes detected\nTry lowering the confidence threshold slider."
    else:
        summary  = f"### {n} pothole{'s' if n > 1 else ''} detected\n\n"
        summary += "| # | Confidence | Location |\n"
        summary += "|---|:----------:|----------|\n"
        for i, d in enumerate(detections, 1):
            x1, y1, x2, y2 = d["bbox"]
            bar = "█" * int(d["confidence"] * 10)
            summary += f"| {i} | {d['confidence']:.2f} {bar} | ({x1},{y1})→({x2},{y2}) |\n"
        summary += f"\n*YOLOv8n ONNX · {n} detection{'s' if n > 1 else ''} · CPU inference*"

    if n > 0 and lat is not None:
        log = log + [{
            "lat"       : lat,
            "lon"       : lon,
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count"     : n,
            "max_conf"  : max(d["confidence"] for d in detections),
        }]
        summary += f"\n\n📍 *Logged to pothole database — ({lat:.5f}, {lon:.5f})*"
    elif n > 0:
        summary += "\n\n*GPS unavailable — detection not logged to map*"

    db_json = json.dumps(log)
    return annotated_rgb, summary, build_map_html(log, lat, lon), build_log_md(log), log, db_json


def check_proximity(gps_str, log):
    """Called by gr.Timer every 5 s — returns updated alert + map."""
    lat, lon = parse_gps(gps_str)
    nearby   = nearby_potholes(lat, lon, log) if lat is not None else []
    return build_alert_html(nearby, lat), build_map_html(log, lat, lon)


def clear_log(_):
    empty = []
    return build_map_html(empty), build_log_md(empty), empty, build_alert_html([], None), "[]"


# ── PWA assets ────────────────────────────────────────────────────────────────

ICON_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <rect width="192" height="192" rx="40" fill="#1a1d2e"/>
  <rect x="0" y="84" width="192" height="28" fill="#334155"/>
  <line x1="96" y1="84" x2="96" y2="112" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4 4"/>
  <ellipse cx="96" cy="98" rx="20" ry="12" fill="#0f172a" stroke="#6366f1" stroke-width="2.5"/>
  <polygon points="96,18 132,62 60,62" fill="#6366f1"/>
  <text x="96" y="56" font-size="28" text-anchor="middle" fill="white"
        font-family="Arial" font-weight="700">!</text>
</svg>"""

SW_CODE = """\
const CACHE       = 'pothole-ai-v1';
const TILE_CACHE  = 'pothole-tiles-v1';
const MODEL_CACHE = 'pothole-model-v1';   // reserved for Phase 2 client-side inference

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(['/', '/manifest.json', '/icon.svg']).catch(() => {}))
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(
        ks.filter(k => k !== CACHE && k !== TILE_CACHE && k !== MODEL_CACHE)
          .map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Gradio live endpoints must always hit the network
  if (['/queue/', '/run/', '/api/', '/upload', '/info']
        .some(p => url.pathname.startsWith(p))) return;

  // Map tiles — stale-while-revalidate (accumulates as user pans map)
  if (url.hostname.endsWith('.cartocdn.com') ||
      url.hostname.endsWith('tile.openstreetmap.org')) {
    e.respondWith(staleWhileRevalidate(req, TILE_CACHE));
    return;
  }

  // ONNX model — cache-first (large; set aside for Phase 2)
  if (url.pathname.endsWith('.onnx')) {
    e.respondWith(cacheFirst(req, MODEL_CACHE));
    return;
  }

  // Gradio versioned static assets — cache-first
  if (url.pathname.startsWith('/assets/') ||
      /\\.(js|css|woff2|png|svg|ico)$/.test(url.pathname)) {
    e.respondWith(cacheFirst(req, CACHE));
    return;
  }

  // Everything else — network-first, cache as fallback
  e.respondWith(fetch(req).catch(() => caches.match(req)));
});

async function cacheFirst(req, name) {
  const cache  = await caches.open(name);
  const cached = await cache.match(req);
  if (cached) return cached;
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch { return cached || new Response('Offline', { status: 503 }); }
}

async function staleWhileRevalidate(req, name) {
  const cache = await caches.open(name);
  const cached = await cache.match(req);
  const fresh = fetch(req).then(res => {
    if (res.ok) cache.put(req, res.clone());
    return res;
  }).catch(() => null);
  return cached || await fresh || new Response('', { status: 503 });
}

// Phase 3 hook: upload queued detections when back online
self.addEventListener('sync', e => {
  if (e.tag === 'sync-potholes')
    e.waitUntil(Promise.resolve()
      .then(() => console.log('[SW] sync-potholes — Phase 3 pending')));
});
"""

MANIFEST = {
    "name": "Pothole Detection AI",
    "short_name": "PotholeAI",
    "description": "YOLOv8n pothole detection with GPS driver alerts",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#1a1d2e",
    "theme_color": "#6366f1",
    "orientation": "portrait-primary",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        {"src": "/icon.svg",     "sizes": "any",     "type": "image/svg+xml"},
    ],
    "categories": ["utilities", "navigation"],
    "shortcuts": [{"name": "Detect Potholes", "url": "/",
                   "description": "Open the pothole detection camera"}],
}

# ── PWA meta tags + service-worker bootstrap (injected before GPS_JS) ─────────

PWA_HEAD = """\
<link rel="manifest" href="/manifest.json" />
<meta name="mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
<meta name="apple-mobile-web-app-title" content="PotholeAI" />
<meta name="theme-color" content="#1a1d2e" />
<script>
  /* ── Service Worker registration ── */
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js', { scope: '/' })
        .then(reg => {
          console.log('[PWA] SW registered — scope:', reg.scope);
          reg.addEventListener('updatefound', () => {
            reg.installing?.addEventListener('statechange', function () {
              if (this.state === 'installed' && navigator.serviceWorker.controller)
                console.log('[PWA] Update ready — refresh to activate');
            });
          });
        })
        .catch(err => console.warn('[PWA] SW registration failed:', err));
    });
  }

  /* ── Install-to-home-screen prompt ── */
  let _pwaPrompt = null;

  // Capture the native Chrome prompt when available
  window.addEventListener('beforeinstallprompt', e => {
    e.preventDefault();
    _pwaPrompt = e;
    // Update button text to show native prompt is ready
    const lbl = document.getElementById('pwa_install_label');
    if (lbl) lbl.textContent = '📲 Install App';
  });

  window.installPWA = async () => {
    if (_pwaPrompt) {
      // Chrome native install flow
      _pwaPrompt.prompt();
      const { outcome } = await _pwaPrompt.userChoice;
      _pwaPrompt = null;
      if (outcome === 'accepted') {
        const btn = document.getElementById('pwa_install_btn');
        if (btn) btn.innerHTML =
          '<span style="color:#4ade80;font-weight:600">✅ App installed!</span>';
      }
    } else {
      // Fallback: show manual instructions for the current platform
      const isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent);
      const msg = isIOS
        ? `📱 iPhone / iPad:\n1. Tap the Share button (↑) in Safari\n2. Scroll down and tap "Add to Home Screen"\n3. Tap "Add" — done!`
        : `📱 Android (Chrome):\n1. Tap the three-dot menu (⋮) in Chrome\n2. Tap "Add to Home Screen" or "Install app"\n3. Tap "Add" — done!\n\nIf you don't see the option, make sure you're using Chrome (not an in-app browser).`;
      alert(msg);
    }
  };

  window.addEventListener('appinstalled', () => {
    _pwaPrompt = null;
    const btn = document.getElementById('pwa_install_btn');
    if (btn) btn.innerHTML =
      '<span style="color:#4ade80;font-weight:600">✅ App installed!</span>';
    console.log('[PWA] Installed to home screen');
  });
</script>
"""

# ── JavaScript injected into <head> ──────────────────────────────────────────

GPS_JS = """
<script>
(function () {
  'use strict';

  /* ── Shared GPS state (module-level so runProximityCheck can read it) ── */
  let currentLat = null;
  let currentLon = null;

  /* ── Audio helpers ── */
  let audioCtx = null;
  let lastBeep  = 0;

  function initAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  function beep(freq, dur, vol) {
    if (!audioCtx) return;
    const now  = audioCtx.currentTime;
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.type = 'sine'; osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
    osc.start(now); osc.stop(now + dur);
  }

  function playAlertBeeps() {
    const t = Date.now();
    if (t - lastBeep < 8000) return;
    lastBeep = t;
    beep(880, 0.18, 0.45);
    setTimeout(() => beep(660, 0.22, 0.45), 210);
    setTimeout(() => beep(880, 0.20, 0.45), 430);
  }

  /* ── Voice / Web Speech API ── */
  let lastVoice = 0;

  function speak(text) {
    if (!window.speechSynthesis) return;
    const now = Date.now();
    if (now - lastVoice < 10000) return;   // 10 s cooldown between voice alerts
    lastVoice = now;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.92; utt.pitch = 1.0; utt.volume = 1.0;
    window.speechSynthesis.speak(utt);
  }

  /* ── Pothole DB reader (synced from Python → hidden JSON textbox) ── */
  function getPotholeDB() {
    const wrapper = document.getElementById('pothole_db_json');
    if (!wrapper) return [];
    const el = wrapper.querySelector('textarea') || wrapper.querySelector('input');
    if (!el || !el.value) return [];
    try { return JSON.parse(el.value); } catch (e) { return []; }
  }

  /* ── Haversine distance in metres (pure JS, no network) ── */
  function haversineM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const toRad = x => x * Math.PI / 180;
    const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2)**2 +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /* ── Client-side proximity check — runs every 3 s, works in background tab ── */
  function runProximityCheck() {
    if (currentLat === null) return;
    const db = getPotholeDB();
    let closest = null;
    for (const entry of db) {
      const dist = haversineM(currentLat, currentLon, entry.lat, entry.lon);
      if (dist <= 150 && (!closest || dist < closest.dist)) {
        closest = { ...entry, dist };
      }
    }

    const statusEl = document.getElementById('voice_status');
    if (closest) {
      const d     = Math.round(closest.dist);
      const label = d < 50  ? 'Danger! Pothole'   :
                    d < 100 ? 'Warning! Pothole'   : 'Caution! Pothole';
      speak(`${label} detected ${d} meters ahead. Please slow down!`);
      playAlertBeeps();
      if (statusEl) statusEl.innerHTML =
        `<span style="color:#fca5a5;font-weight:700">🚨 POTHOLE ${d} m AHEAD — voice alert fired</span>`;
    } else {
      if (statusEl) {
        if (db.length > 0)
          statusEl.innerHTML =
            `<span style="color:#4ade80">✅ All clear — monitoring ${db.length} logged location(s)</span>`;
        else
          statusEl.innerHTML =
            `<span style="color:#64748b">No pothole database yet — detect potholes first.</span>`;
      }
    }
  }

  setInterval(runProximityCheck, 3000);

  /* ── "Activate Voice" button handler (called from onclick in HTML) ── */
  window.activateVoice = function () {
    initAudio();
    speak('Voice monitoring activated. I will warn you when approaching a pothole.');
    const btn = document.getElementById('voice_activate_btn');
    if (btn) {
      btn.textContent = '🔊 Voice Active — tap to re-activate';
      btn.style.background   = 'rgba(34,197,94,0.15)';
      btn.style.borderColor  = 'rgba(34,197,94,0.5)';
      btn.style.color        = '#4ade80';
    }
    const statusEl = document.getElementById('voice_status');
    if (statusEl)
      statusEl.innerHTML = '<span style="color:#4ade80">🔊 Voice monitoring active</span>';
  };

  /* ── Watch server-side alert panel for DOM-driven beeps ── */
  function watchAlertPanel() {
    const panel = document.getElementById('driver_alert_panel');
    if (!panel) { setTimeout(watchAlertPanel, 1200); return; }
    new MutationObserver(() => {
      if (panel.querySelector('[data-alert="true"]')) playAlertBeeps();
    }).observe(panel, { childList: true, subtree: true, attributes: true });
  }

  /* ── Update Gradio hidden textbox from JS ── */
  function setGradioTextbox(elemId, value) {
    const wrapper = document.getElementById(elemId);
    if (!wrapper) return;
    const el = wrapper.querySelector('textarea') ||
               wrapper.querySelector('input[type="text"]') ||
               wrapper.querySelector('input');
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  /* ── GPS status display ── */
  function setGpsStatus(html) {
    const el = document.getElementById('gps_status_text');
    if (el) el.innerHTML = html;
  }

  /* ── Geolocation watcher ── */
  function startGPS() {
    if (!navigator.geolocation) {
      setGpsStatus('<span style="color:#f87171">✕</span> Geolocation not supported');
      return;
    }
    setGpsStatus('<span style="color:#facc15">◌</span> Requesting GPS permission…');
    navigator.geolocation.watchPosition(
      function (pos) {
        currentLat = pos.coords.latitude;
        currentLon = pos.coords.longitude;
        const acc  = Math.round(pos.coords.accuracy);
        setGpsStatus(
          '<span style="color:#4ade80">●</span> GPS Active &nbsp;·&nbsp; ' +
          currentLat.toFixed(5) + ', ' + currentLon.toFixed(5) +
          ' <span style="opacity:0.5;font-size:0.75em">(±' + acc + ' m)</span>'
        );
        setGradioTextbox('gps_hidden', currentLat.toFixed(6) + ',' + currentLon.toFixed(6));
      },
      function (err) {
        setGpsStatus('<span style="color:#f87171">⚠️</span> GPS denied — ' + err.message);
      },
      { enableHighAccuracy: true, maximumAge: 3000, timeout: 15000 }
    );
  }

  /* ── Boot ── */
  function boot() {
    startGPS();
    watchAlertPanel();
    document.body.addEventListener('click', initAudio, { once: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else { boot(); }
  setTimeout(boot, 2500);
})();
</script>
"""

# ── IndexedDB: offline pothole cache ─────────────────────────────────────────

IDB_JS = """
<script>
(function () {
  'use strict';

  /* ── Schema ── */
  const DB_NAME    = 'pothole-ai-db';
  const DB_VERSION = 1;
  const STORE      = 'potholes';
  const DEDUP_M    = 30;   // skip new entry if an existing one is within 30 m

  let _db = null;

  function openDB() {
    return new Promise((resolve, reject) => {
      if (_db) { resolve(_db); return; }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('synced',    'synced',    { unique: false });
        }
      };
      req.onsuccess = e => { _db = e.target.result; resolve(_db); };
      req.onerror   = e => reject(e.target.error);
    });
  }

  /* ── CRUD helpers ── */
  window.idbGetAll = function () {
    return openDB().then(db => new Promise((resolve, reject) => {
      const tx  = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).getAll();
      req.onsuccess = () => resolve(req.result);
      req.onerror   = () => reject(req.error);
    }));
  };

  window.idbAdd = function (entry) {
    return openDB().then(db => new Promise((resolve, reject) => {
      const tx  = db.transaction(STORE, 'readwrite');
      const req = tx.objectStore(STORE).add(entry);
      req.onsuccess = () => resolve(req.result);
      req.onerror   = () => reject(req.error);
    }));
  };

  window.idbClear = function () {
    return openDB().then(db => new Promise((resolve, reject) => {
      const tx  = db.transaction(STORE, 'readwrite');
      const req = tx.objectStore(STORE).clear();
      req.onsuccess = () => resolve();
      req.onerror   = () => reject(req.error);
    }));
  };

  /* ── Haversine (IDB_JS loads before GPS_JS, so define locally) ── */
  function haversineM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const toRad = x => x * Math.PI / 180;
    const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2)**2 +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /* ── Badge ── */
  function updateBadge(n) {
    const el = document.getElementById('idb_count_badge');
    if (!el) return;
    if (n === 0) { el.textContent = ''; el.style.display = 'none'; }
    else         { el.textContent = '💾 ' + n + ' saved offline'; el.style.display = 'inline'; }
  }

  /* ── "Saved locally" toast (fires on each new save) ── */
  function showSavedToast(total) {
    const toast = document.createElement('div');
    toast.style.cssText = [
      'position:fixed;bottom:18px;right:18px',
      'background:rgba(34,197,94,0.92);color:#fff;font-size:0.8rem;font-weight:600',
      'padding:8px 16px;border-radius:12px;z-index:9999',
      'box-shadow:0 4px 16px rgba(34,197,94,0.35)',
      'animation:idb-fadein 0.3s ease',
    ].join(';');
    toast.textContent = '💾 Saved locally (' + total + ' total)';
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.transition = 'opacity 0.5s';
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 600);
    }, 2500);
  }

  /* ── Save with 30 m dedup ── */
  window.idbSave = function (entry) {
    window.idbGetAll().then(existing => {
      for (const e of existing) {
        if (e.lat != null && entry.lat != null &&
            haversineM(e.lat, e.lon, entry.lat, entry.lon) < DEDUP_M) {
          return;   // too close — skip
        }
      }
      window.idbAdd({ ...entry, synced: false }).then(() => {
        window.idbGetAll().then(all => {
          updateBadge(all.length);
          showSavedToast(all.length);
        });
      });
    });
  };

  /* ── Mirror Python detections → IDB ── */
  let _lastMirrorJson = '[]';

  function watchGradioTextbox() {
    const wrapper = document.getElementById('pothole_db_json');
    if (!wrapper) { setTimeout(watchGradioTextbox, 1200); return; }

    function onChange() {
      const el = wrapper.querySelector('textarea') || wrapper.querySelector('input');
      if (!el) return;
      const raw = el.value;
      if (!raw || raw === _lastMirrorJson) return;
      _lastMirrorJson = raw;

      let entries;
      try { entries = JSON.parse(raw); } catch (e) { return; }
      if (!Array.isArray(entries) || entries.length === 0) return;

      // Only save the latest entry (last one appended by Python)
      const latest = entries[entries.length - 1];
      if (latest && latest.lat != null) {
        window.idbSave({
          lat:       latest.lat,
          lon:       latest.lon,
          timestamp: latest.timestamp || new Date().toISOString(),
          conf:      latest.conf      || null,
        });
      }
    }

    new MutationObserver(onChange).observe(wrapper,
      { subtree: true, childList: true, characterData: true, attributes: true });
    setInterval(onChange, 2000);   // fallback poll
  }

  /* ── Restore banner ── */
  function showRestoreBanner(n) {
    // Remove any existing banner first
    const old = document.getElementById('idb_restore_banner');
    if (old) old.remove();

    const banner = document.createElement('div');
    banner.id = 'idb_restore_banner';
    banner.style.cssText = [
      'position:fixed;bottom:18px;left:50%;transform:translateX(-50%)',
      'background:rgba(99,102,241,0.92);color:#fff;font-size:0.82rem;font-weight:600',
      'padding:9px 20px;border-radius:999px;z-index:9999',
      'box-shadow:0 4px 20px rgba(99,102,241,0.4)',
      'animation:idb-fadein 0.4s ease',
      'white-space:nowrap',
    ].join(';');
    banner.textContent = '💾 Restored ' + n + ' pothole' + (n === 1 ? '' : 's') + ' from offline storage';

    // Inject keyframe once
    if (!document.getElementById('idb-anim-style')) {
      const style = document.createElement('style');
      style.id = 'idb-anim-style';
      style.textContent = '@keyframes idb-fadein{from{opacity:0;transform:translateX(-50%) translateY(12px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}';
      document.head.appendChild(style);
    }

    document.body.appendChild(banner);
    setTimeout(() => {
      banner.style.transition = 'opacity 0.6s';
      banner.style.opacity = '0';
      setTimeout(() => banner.remove(), 700);
    }, 4000);
  }

  /* ── Restore IDB → proximity engine on page load ── */
  function restoreIntoProximity() {
    window.idbGetAll().then(entries => {
      if (!entries || entries.length === 0) return;
      updateBadge(entries.length);
      showRestoreBanner(entries.length);

      const formatted = entries.map(e => ({
        lat:       e.lat,
        lon:       e.lon,
        timestamp: e.timestamp || '',
        conf:      e.conf      || 0,
      }));

      function tryWrite() {
        const wrapper = document.getElementById('pothole_db_json');
        if (!wrapper) { setTimeout(tryWrite, 800); return; }
        const el = wrapper.querySelector('textarea') || wrapper.querySelector('input');
        if (!el)      { setTimeout(tryWrite, 800); return; }
        el.value = JSON.stringify(formatted);
        el.dispatchEvent(new Event('input',  { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        console.log('[IDB] Restored', formatted.length, 'potholes into proximity engine');
      }
      tryWrite();
    }).catch(err => console.warn('[IDB] restore error:', err));
  }

  /* ── Clear hook — called by Python clear_btn via js= ── */
  window.idbClearAll = function () {
    window.idbClear().then(() => {
      updateBadge(0);
      _lastMirrorJson = '[]';
      console.log('[IDB] Cleared');
    });
  };

  /* ── Boot ── */
  function idbBoot() {
    openDB()
      .then(() => {
        restoreIntoProximity();
        watchGradioTextbox();
        console.log('[IDB] Ready');
      })
      .catch(err => console.warn('[IDB] Not available:', err));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', idbBoot);
  } else { idbBoot(); }
  setTimeout(idbBoot, 2500);
})();
</script>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    background: linear-gradient(160deg, #1a1d2e 0%, #1e2235 50%, #1a1f35 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 1.6rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 45%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 1.2rem 0 1.5rem;
}
.stat-badge {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 7px;
    backdrop-filter: blur(8px);
}
.stat-badge span.val { color: #e2e8f0; font-weight: 600; }
.stat-badge span.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #6366f1;
    display: inline-block;
    box-shadow: 0 0 6px rgba(99,102,241,0.8);
    animation: dotpulse 2s ease-in-out infinite;
}
@keyframes dotpulse {
    0%,100% { box-shadow: 0 0 4px rgba(99,102,241,0.7); }
    50%      { box-shadow: 0 0 12px rgba(99,102,241,1.0), 0 0 22px rgba(99,102,241,0.4); }
}

/* ── GPS status bar ── */
.gps-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 1.2rem;
    font-size: 0.84rem;
    color: #94a3b8;
}

/* ── Driver alert panels ── */
.alert-waiting, .alert-clear, .alert-low, .alert-medium, .alert-high {
    display: flex;
    align-items: center;
    gap: 16px;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.4s ease;
}
.alert-waiting {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
}
.alert-clear {
    background: rgba(34,197,94,0.08);
    border: 2px solid rgba(34,197,94,0.4);
}
.alert-low {
    background: rgba(234,179,8,0.08);
    border: 2px solid rgba(234,179,8,0.5);
    animation: glow-low 2s ease-in-out infinite;
}
.alert-medium {
    background: rgba(249,115,22,0.10);
    border: 2px solid rgba(249,115,22,0.6);
    animation: glow-med 1.4s ease-in-out infinite;
}
.alert-high {
    background: rgba(239,68,68,0.13);
    border: 2px solid rgba(239,68,68,0.7);
    box-shadow: 0 0 32px rgba(239,68,68,0.2);
    animation: glow-high 0.9s ease-in-out infinite;
}
@keyframes glow-low  { 0%,100%{box-shadow:none} 50%{box-shadow:0 0 16px rgba(234,179,8,0.3)} }
@keyframes glow-med  { 0%,100%{box-shadow:none} 50%{box-shadow:0 0 22px rgba(249,115,22,0.4)} }
@keyframes glow-high { 0%,100%{box-shadow:0 0 20px rgba(239,68,68,0.2)} 50%{box-shadow:0 0 44px rgba(239,68,68,0.5)} }

.alert-icon { font-size: 2.2rem; line-height: 1; }
.alert-flash { animation: flashicon 0.8s ease-in-out infinite; }
@keyframes flashicon { 0%,100%{opacity:1} 50%{opacity:0.35} }

.alert-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
}
.alert-sub { font-size: 0.82rem; color: #94a3b8; }

.alert-dist-badge {
    background: rgba(239,68,68,0.18);
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 12px;
    padding: 8px 14px;
    text-align: center;
    font-size: 1.7rem;
    font-weight: 800;
    color: #fca5a5;
    line-height: 1;
    min-width: 72px;
}
.alert-dist-badge span {
    display: block;
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 2px;
}

/* ── Panel cards ── */
.panel {
    background: linear-gradient(145deg, #252945 0%, #1f2340 100%) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 18px !important;
    padding: 1.4rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

/* ── Upload area ── */
.upload-area {
    border: 2px dashed rgba(99,102,241,0.5) !important;
    border-radius: 14px !important;
    background: rgba(99,102,241,0.04) !important;
    transition: border-color 0.25s, background 0.25s !important;
}
.upload-area:hover {
    border-color: rgba(167,139,250,0.8) !important;
    background: rgba(99,102,241,0.08) !important;
}

/* ── Detect button ── */
.detect-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.78rem !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
    cursor: pointer !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
}
.detect-btn:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.55), 0 2px 8px rgba(168,85,247,0.35) !important;
    transform: translateY(-2px) !important;
}
.detect-btn:active { transform: scale(0.97) !important; }

/* ── Clear button ── */
.clear-btn {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.45) !important;
    border-radius: 12px !important;
    color: #f87171 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem !important;
    transition: background 0.2s, box-shadow 0.2s !important;
    cursor: pointer !important;
}
.clear-btn:hover {
    background: rgba(239,68,68,0.15) !important;
    box-shadow: 0 0 16px rgba(239,68,68,0.25) !important;
}

/* ── Slider + labels ── */
.conf-slider input[type=range] { accent-color: #6366f1; }
label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Result box ── */
.result-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 2px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    min-height: 80px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.result-box h3 { color:#a5b4fc !important; font-size:1rem !important; margin:0 0 0.5rem !important; font-weight:600 !important; }
.result-box table { width:100%; border-collapse:collapse; font-size:0.82rem; color:#cbd5e1; }
.result-box th { color:#64748b; font-weight:500; padding:5px 8px; border-bottom:1px solid rgba(255,255,255,0.08); text-align:left; }
.result-box td { padding:5px 8px; border-bottom:1px solid rgba(255,255,255,0.04); }

/* ── Map ── */
.map-wrap {
    background: linear-gradient(145deg, #252945, #1f2340);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 18px;
    overflow: hidden;
    margin-top: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.map-wrap iframe { display:block; border:none; }

/* ── Log box ── */
.log-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 2px solid rgba(99,102,241,0.5);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    min-height: 120px;
    max-height: 280px;
    overflow-y: auto;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.log-box::-webkit-scrollbar { width: 5px; }
.log-box::-webkit-scrollbar-track { background: transparent; }
.log-box::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 99px; }
.log-box::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }

/* ── Voice activate button ── */
.start-btn {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15)) !important;
    border: 1.5px solid rgba(99,102,241,0.5) !important;
    border-radius: 12px !important;
    color: #a5b4fc !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100%;
    margin-bottom: 0.75rem;
}
.start-btn:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.25)) !important;
    box-shadow: 0 0 22px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem 1rem;
    color: #64748b;
    font-size: 0.78rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 2rem;
}
.footer a { color: #818cf8; text-decoration: none; }
.footer a:hover { color: #a5b4fc; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Pothole Detection AI", css=CSS, head=PWA_HEAD + IDB_JS + GPS_JS) as demo:

    pothole_log = gr.State([])

    # Hidden textbox — JS watchPosition writes "lat,lon" here
    gps_hidden = gr.Textbox(value="", visible=False, elem_id="gps_hidden")

    # Hidden textbox — Python writes JSON pothole DB here; JS reads it for client-side alerts
    pothole_db_json = gr.Textbox(value="[]", visible=False, elem_id="pothole_db_json")

    # Hero
    gr.HTML("""
    <div class="hero">
        <div class="hero-tag">&#x26A1; AI-Powered Road Safety</div>
        <h1>Pothole Detection AI</h1>
        <p>Detects potholes in real time and alerts drivers before they reach them — powered by YOLOv8n + GPS</p>
    </div>
    <div class="stats-row">
        <div class="stat-badge"><span class="dot"></span>mAP@50 <span class="val">0.822</span></div>
        <div class="stat-badge"><span class="dot"></span>Precision <span class="val">82.2%</span></div>
        <div class="stat-badge"><span class="dot"></span>Recall <span class="val">74.4%</span></div>
        <div class="stat-badge"><span class="dot"></span>Inference <span class="val">~41ms CPU</span></div>
        <div class="stat-badge"><span class="dot"></span>Alert Radius <span class="val">150 m</span></div>
    </div>
    <!-- PWA install button — always visible; uses native prompt if available, otherwise shows instructions -->
    <div style="display:flex;justify-content:center;margin-top:0.6rem;">
      <div id="pwa_install_btn">
        <button onclick="installPWA()" style="
          background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.45);
          border-radius:999px;color:#a5b4fc;font-size:0.78rem;font-weight:600;
          padding:6px 18px;cursor:pointer;display:flex;align-items:center;gap:7px;
          transition:background 0.2s;" onmouseover="this.style.background='rgba(99,102,241,0.22)'"
          onmouseout="this.style.background='rgba(99,102,241,0.12)'">
          <span id="pwa_install_label">📲 Add to Home Screen</span>
        </button>
      </div>
    </div>
    """)

    # GPS status bar (text updated live by JS)
    gr.HTML("""
    <div class="gps-bar">
        <span>📡</span>
        <span id="gps_status_text" style="flex:1">Requesting GPS location from your browser…</span>
        <span id="idb_count_badge" style="display:none;background:rgba(99,102,241,0.15);
          border:1px solid rgba(99,102,241,0.4);border-radius:999px;padding:2px 10px;
          font-size:0.75rem;font-weight:600;color:#a5b4fc;white-space:nowrap;"></span>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Detect Potholes ─────────────────────────────────────────────
        with gr.Tab("📷  Detect Potholes"):
            gr.HTML("""
            <div style="color:#94a3b8;font-size:0.84rem;margin-bottom:1rem;">
                Upload a road photo. YOLOv8n detects potholes and
                <b style="color:#a5b4fc">automatically logs them with your GPS location</b>
                into the pothole database used by the Driver Alert system.
            </div>""")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="panel"):
                    input_image = gr.Image(
                        label="Upload road image",
                        type="numpy",
                        sources=["upload", "clipboard"],
                        elem_classes="upload-area",
                        height=300,
                    )
                    conf_slider = gr.Slider(
                        minimum=0.10, maximum=0.90, value=0.25, step=0.05,
                        label="Confidence threshold — lower detects more, higher is more selective",
                        elem_classes="conf-slider",
                    )
                    with gr.Row():
                        run_btn   = gr.Button("Detect potholes", variant="primary",
                                              elem_classes="detect-btn", scale=3)
                        clear_btn = gr.Button("Clear database",  variant="secondary",
                                              elem_classes="clear-btn",  scale=1)

                with gr.Column(scale=1, elem_classes="panel"):
                    output_image = gr.Image(label="Detection result", type="numpy", height=300)
                    output_text  = gr.Markdown(
                        value="Results will appear here after detection.",
                        elem_classes="result-box",
                    )

        # ── Tab 2: Driver Mode ─────────────────────────────────────────────────
        with gr.Tab("🚗  Driver Alert Mode"):
            gr.HTML("""
            <div style="color:#94a3b8;font-size:0.84rem;margin-bottom:1.2rem;line-height:1.7;">
                Open this tab on your phone while driving.
                You can keep <b style="color:#e2e8f0">Google Maps or any navigation app</b> open
                in another tab — the voice alert runs entirely in the browser background.<br>
                <span style="color:#64748b;font-size:0.8rem;">
                  Proximity is checked every 3 seconds client-side (no internet needed after load).
                </span>
            </div>

            <!-- Step 1: activate voice -->
            <button id="voice_activate_btn" class="start-btn" onclick="activateVoice()">
              🔊 Tap here first to activate voice alerts
            </button>

            <!-- Live voice status -->
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
              border-radius:12px;padding:12px 16px;margin-bottom:1rem;font-size:0.85rem;">
              <span style="color:#64748b;">Voice status: </span>
              <span id="voice_status" style="color:#64748b;">
                Tap the button above to enable voice alerts
              </span>
            </div>

            <!-- How it works steps -->
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:1.2rem;">
              <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:12px 14px;font-size:0.8rem;color:#a5b4fc;">
                <b style="display:block;margin-bottom:4px;">Step 1</b>
                Detect road potholes using the Detect tab — they get logged with your GPS location.
              </div>
              <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:12px 14px;font-size:0.8rem;color:#a5b4fc;">
                <b style="display:block;margin-bottom:4px;">Step 2</b>
                Come back here, tap "Activate Voice", then open Google Maps in another tab.
              </div>
              <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:12px 14px;font-size:0.8rem;color:#a5b4fc;">
                <b style="display:block;margin-bottom:4px;">Step 3</b>
                Drive normally. The app speaks <i>"Warning! Pothole 80 metres ahead. Slow down!"</i>
              </div>
            </div>
            """)

            alert_panel = gr.HTML(
                value=build_alert_html([], None),
                elem_id="driver_alert_panel",
            )

            gr.HTML("""
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:0.75rem;">
              <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.25);
                border-radius:10px;padding:8px 14px;font-size:0.78rem;color:#86efac;">
                ✅ <b>Green</b> — All clear
              </div>
              <div style="background:rgba(234,179,8,0.07);border:1px solid rgba(234,179,8,0.25);
                border-radius:10px;padding:8px 14px;font-size:0.78rem;color:#fde68a;">
                ⚠️ <b>Yellow</b> — 100–150 m
              </div>
              <div style="background:rgba(249,115,22,0.07);border:1px solid rgba(249,115,22,0.25);
                border-radius:10px;padding:8px 14px;font-size:0.78rem;color:#fed7aa;">
                ⚠️ <b>Orange</b> — 50–100 m
              </div>
              <div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);
                border-radius:10px;padding:8px 14px;font-size:0.78rem;color:#fca5a5;">
                🚨 <b>Red</b> — &lt; 50 m danger
              </div>
            </div>""")

    # Shared full-width live map
    map_display = gr.HTML(value=build_map_html([]))

    # Pothole database log
    log_display = gr.Markdown(
        value="*No detections logged yet. Detect potholes with GPS active to populate the database.*",
        elem_classes="log-box",
    )

    # ── Auto proximity check every 5 s ────────────────────────────────────────
    timer = gr.Timer(value=5)
    timer.tick(
        fn      = check_proximity,
        inputs  = [gps_hidden, pothole_log],
        outputs = [alert_panel, map_display],
    )

    # ── Event wiring ──────────────────────────────────────────────────────────
    detect_outputs = [output_image, output_text, map_display, log_display,
                      pothole_log, pothole_db_json]

    run_btn.click(
        fn=detect_potholes,
        inputs=[input_image, conf_slider, gps_hidden, pothole_log],
        outputs=detect_outputs,
    )
    input_image.change(
        fn=detect_potholes,
        inputs=[input_image, conf_slider, gps_hidden, pothole_log],
        outputs=detect_outputs,
    )
    clear_btn.click(
        fn=clear_log,
        inputs=[pothole_log],
        outputs=[map_display, log_display, pothole_log, alert_panel, pothole_db_json],
        js="() => { window.idbClearAll?.(); }",
    )

    gr.HTML("""
    <div class="footer">
        Built by <a href="https://github.com/poojithamadhyala">poojithamadhyala</a> &nbsp;·&nbsp;
        YOLOv8n fine-tuned on 656 annotated dashcam images &nbsp;·&nbsp;
        <a href="https://github.com/poojithamadhyala/pothole-detection">GitHub</a>
    </div>""")


# ── FastAPI wrapper — serves PWA assets + mounts Gradio ──────────────────────
# HF Spaces detects the `app` variable and runs it with uvicorn automatically.

fapp = FastAPI(title="Pothole Detection AI")


@fapp.get("/manifest.json", include_in_schema=False)
async def get_manifest():
    return JSONResponse(MANIFEST, headers={"Cache-Control": "max-age=86400"})


@fapp.get("/sw.js", include_in_schema=False)
async def get_sw():
    return FastAPIResponse(
        SW_CODE,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@fapp.get("/icon-192.png", include_in_schema=False)
async def get_icon_192():
    return FastAPIResponse(
        _ICON_192,
        media_type="image/png",
        headers={"Cache-Control": "max-age=604800"},
    )


@fapp.get("/icon-512.png", include_in_schema=False)
async def get_icon_512():
    return FastAPIResponse(
        _ICON_512,
        media_type="image/png",
        headers={"Cache-Control": "max-age=604800"},
    )


@fapp.get("/icon.svg", include_in_schema=False)
async def get_icon():
    return FastAPIResponse(
        ICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "max-age=604800"},
    )


gr.mount_gradio_app(fapp, demo, path="/")

# Exposed for HF Spaces uvicorn auto-detection
app = fapp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fapp, host="0.0.0.0", port=7860)
