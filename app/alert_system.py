"""
Real-Time Pothole Detection & Driver Alert System
──────────────────────────────────────────────────
Run:
    python alert_system.py                  # webcam
    python alert_system.py --source road.mp4  # video file
"""

import cv2
import time
import csv
import threading
import argparse
import os
from datetime import datetime
from ultralytics import YOLO

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[!] pyttsx3 not found — audio alerts disabled. pip install pyttsx3")


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH       = "/Users/poojithamadhyala/pothole-detection/runs/detect/runs/pothole/yolov8n-v1/weights/best.onnx"
CONF_THRESHOLD   = 0.60   # minimum confidence to trigger alert
IOU_THRESHOLD    = 0.45
IMAGE_SIZE       = 640
COOLDOWN_SECONDS = 4      # seconds between consecutive audio/visual alerts
ALERT_LOG_PATH   = "alert_log.csv"

# Severity: fraction of frame area occupied by bounding box
SEV_LOW_MAX  = 0.04   # < 4%  → Low    (far away)
SEV_MED_MAX  = 0.12   # 4–12% → Medium
                       # > 12% → High   (close / dangerous)

# BGR colors
CLR = {
    "low"    : (0,   200,  0  ),   # green
    "medium" : (0,   165,  255),   # orange
    "high"   : (0,   0,    255),   # red
    "info"   : (59,  130,  246),   # blue (non-alert boxes)
    "white"  : (255, 255,  255),
    "black"  : (0,   0,    0  ),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Severity estimation ────────────────────────────────────────────────────────

def estimate_severity(x1: int, y1: int, x2: int, y2: int,
                      frame_w: int, frame_h: int) -> str:
    """Return 'low', 'medium', or 'high' based on bbox area vs frame area."""
    bbox_area  = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio = bbox_area / max(frame_area, 1)
    if ratio < SEV_LOW_MAX:
        return "low"
    elif ratio < SEV_MED_MAX:
        return "medium"
    else:
        return "high"


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_potholes(model: YOLO, frame, conf_thresh: float) -> list:
    """
    Run inference on a single BGR frame.
    Returns list of dicts: {x1, y1, x2, y2, confidence, class_name}
    """
    results    = model.predict(source=frame, conf=conf_thresh,
                               iou=IOU_THRESHOLD, imgsz=IMAGE_SIZE,
                               verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "x1"         : x1,
            "y1"         : y1,
            "x2"         : x2,
            "y2"         : y2,
            "confidence" : float(box.conf[0]),
            "class_name" : model.names[int(box.cls[0])],
        })
    return detections


# ── Audio alert ───────────────────────────────────────────────────────────────

def _speak_worker(text: str) -> None:
    """Run pyttsx3 in a daemon thread (engine must be created per-thread on macOS)."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[audio] {e}")


def trigger_audio_alert(severity: str) -> None:
    """Fire-and-forget speech in a background thread."""
    if not TTS_AVAILABLE:
        return
    messages = {
        "low"    : "Pothole detected ahead.",
        "medium" : "Warning: Pothole ahead!",
        "high"   : "Danger! Large pothole immediately ahead!",
    }
    msg = messages.get(severity, "Warning: Pothole ahead!")
    t = threading.Thread(target=_speak_worker, args=(msg,), daemon=True)
    t.start()


# ── Alert rendering ────────────────────────────────────────────────────────────

def trigger_visual_alert(frame, severity: str) -> None:
    """Draw the ⚠️ alert banner at the top of the frame (in-place)."""
    h, w = frame.shape[:2]
    banner_h = 52

    # Translucent red/orange overlay
    overlay = frame.copy()
    banner_color = CLR["high"] if severity == "high" else (0, 120, 200)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Text
    severity_label = severity.upper()
    msg = f"  POTHOLE AHEAD!  [{severity_label} SEVERITY]"
    cv2.putText(frame, msg, (10, 34),
                FONT_BOLD, 0.85, CLR["white"], 2, cv2.LINE_AA)


# ── Alert logging ─────────────────────────────────────────────────────────────

def _init_log(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "severity", "confidence",
                             "x1", "y1", "x2", "y2"])


def log_alert(path: str, severity: str, conf: float,
              x1: int, y1: int, x2: int, y2: int) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         severity, f"{conf:.3f}", x1, y1, x2, y2])


# ── Frame annotation ─────────────────────────────────────────────────────────

def annotate_frame(frame, detections: list,
                   frame_w: int, frame_h: int,
                   alert_active: bool) -> str | None:
    """
    Draw bounding boxes, labels, severity on the frame.
    Returns the highest severity found among alert-worthy detections, or None.
    """
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    worst_severity = None

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        conf   = d["confidence"]
        is_alert = conf >= CONF_THRESHOLD

        if is_alert:
            sev   = estimate_severity(x1, y1, x2, y2, frame_w, frame_h)
            color = CLR[sev]
            if (worst_severity is None or
                    severity_rank[sev] > severity_rank[worst_severity]):
                worst_severity = sev
        else:
            sev   = None
            color = CLR["info"]

        # Bounding box (double-border for visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), CLR["white"], 1)

        # Label background + text
        label = f"pothole {conf:.2f}"
        if sev:
            label += f" [{sev}]"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    FONT, 0.52, CLR["white"], 1, cv2.LINE_AA)

    return worst_severity


def draw_hud(frame, fps: float, total_alerts: int, cooldown_left: float) -> None:
    """Draw HUD overlay: FPS, alert count, cooldown indicator."""
    h, w = frame.shape[:2]

    # Bottom-left HUD
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Alerts: {total_alerts}",
    ]
    if cooldown_left > 0:
        hud_lines.append(f"Cooldown: {cooldown_left:.1f}s")

    y0 = h - 12 - (len(hud_lines) - 1) * 22
    for i, line in enumerate(hud_lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (10, y), FONT, 0.55, CLR["black"], 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), FONT, 0.55, CLR["white"], 1, cv2.LINE_AA)

    # Top-right: press Q to quit
    msg = "Q: quit"
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.45, 1)
    cv2.putText(frame, msg, (w - tw - 10, 20),
                FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


# ── Demo mode (image slideshow) ───────────────────────────────────────────────

DEMO_IMAGE_DIR = "/Users/poojithamadhyala/pothole-detection/data/Pothole detection.v12i.yolov8/test/images"

def _load_demo_frames(img_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    print(f"[+] Demo mode — {len(paths)} images loaded from {img_dir}")
    return paths


def run_demo(model: YOLO) -> None:
    """Loop over test images as a simulated video stream."""
    paths = _load_demo_frames(DEMO_IMAGE_DIR)
    _init_log(ALERT_LOG_PATH)

    last_alert_time = 0.0
    total_alerts    = 0
    idx             = 0

    while True:
        path  = paths[idx % len(paths)]
        frame = cv2.imread(path)
        if frame is None:
            idx += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        detections        = detect_potholes(model, frame, CONF_THRESHOLD)
        cooldown_left     = max(0.0, COOLDOWN_SECONDS - (now - last_alert_time))
        alert_on_cooldown = cooldown_left > 0

        worst_sev = annotate_frame(frame, detections, frame_w, frame_h,
                                   alert_active=not alert_on_cooldown)

        if worst_sev is not None and not alert_on_cooldown:
            trigger_visual_alert(frame, worst_sev)
            trigger_audio_alert(worst_sev)
            top = max(detections, key=lambda d: d["confidence"])
            log_alert(ALERT_LOG_PATH, worst_sev, top["confidence"],
                      top["x1"], top["y1"], top["x2"], top["y2"])
            last_alert_time = now
            total_alerts   += 1
            print(f"[ALERT] {datetime.now().strftime('%H:%M:%S')} | "
                  f"{worst_sev.upper()} | conf={top['confidence']:.2f} | "
                  f"{os.path.basename(path)}")
        elif worst_sev is not None and alert_on_cooldown:
            trigger_visual_alert(frame, worst_sev)

        # HUD — show image index instead of FPS
        draw_hud(frame, fps=0, total_alerts=total_alerts, cooldown_left=cooldown_left)
        cv2.putText(frame, f"Image {idx % len(paths) + 1}/{len(paths)}",
                    (10, frame_h - 60), FONT, 0.5, CLR["black"], 3, cv2.LINE_AA)
        cv2.putText(frame, f"Image {idx % len(paths) + 1}/{len(paths)}",
                    (10, frame_h - 60), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Pothole Alert — Demo Mode (SPACE=next  Q=quit)", frame)
        key = cv2.waitKey(800) & 0xFF   # auto-advance every 0.8s
        if key == ord("q"):
            break
        if key == ord(" "):             # manual skip
            idx += 1
            continue
        idx += 1

    cv2.destroyAllWindows()
    print(f"[+] Done. {total_alerts} alert(s) logged to {ALERT_LOG_PATH}")


# ── Main loop (webcam / video file) ──────────────────────────────────────────

def run(source=0) -> None:
    print(f"[+] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[+] Model ready")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n[!] Cannot open source: {source}")
        if source == 0:
            print("    macOS camera permission likely blocked.")
            print("    Fix: System Settings → Privacy & Security → Camera → enable Terminal")
            print("\n    OR run in demo mode with existing test images:")
            print("    python alert_system.py --demo\n")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Stream opened — {frame_w}x{frame_h}")

    _init_log(ALERT_LOG_PATH)

    last_alert_time = 0.0
    total_alerts    = 0
    prev_time       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[+] Stream ended.")
            break

        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        detections        = detect_potholes(model, frame, CONF_THRESHOLD)
        cooldown_left     = max(0.0, COOLDOWN_SECONDS - (now - last_alert_time))
        alert_on_cooldown = cooldown_left > 0

        worst_sev = annotate_frame(frame, detections, frame_w, frame_h,
                                   alert_active=not alert_on_cooldown)

        if worst_sev is not None and not alert_on_cooldown:
            trigger_visual_alert(frame, worst_sev)
            trigger_audio_alert(worst_sev)
            top = max(detections, key=lambda d: d["confidence"])
            log_alert(ALERT_LOG_PATH, worst_sev, top["confidence"],
                      top["x1"], top["y1"], top["x2"], top["y2"])
            last_alert_time = now
            total_alerts   += 1
            print(f"[ALERT] {datetime.now().strftime('%H:%M:%S')} | "
                  f"{worst_sev.upper()} | conf={top['confidence']:.2f}")
        elif worst_sev is not None and alert_on_cooldown:
            trigger_visual_alert(frame, worst_sev)

        draw_hud(frame, fps, total_alerts, cooldown_left)

        cv2.imshow("Pothole Detection — Real-Time Alert", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] Done. {total_alerts} alert(s) logged to {ALERT_LOG_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time pothole alert system")
    parser.add_argument("--source", default=0,
                        help="Video source: 0=webcam or path to video file")
    parser.add_argument("--demo",   action="store_true",
                        help="Demo mode: slideshow of test images (no webcam needed)")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS,
                        help=f"Alert cooldown seconds (default: {COOLDOWN_SECONDS})")
    args = parser.parse_args()

    CONF_THRESHOLD   = args.conf
    COOLDOWN_SECONDS = args.cooldown

    print(f"[+] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[+] Model ready")

    if args.demo:
        run_demo(model)
    else:
        source = args.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        run(source)
