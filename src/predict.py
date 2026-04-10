# =============================================================================
# src/predict.py
# Pothole Detection — Inference Script
# =============================================================================
# Usage:
#   Single image  : python src/predict.py --source data/sample_images/your.jpg
#   Folder        : python src/predict.py --source data/sample_images/
#   Webcam        : python src/predict.py --source 0
#   Save results  : python src/predict.py --source data/sample_images/ --save
# =============================================================================

import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

MODEL_PATH   = "runs/detect/runs/pothole/yolov8n-v1/weights/best.pt"
CONF_THRESH  = 0.25          # minimum confidence to show a detection
IOU_THRESH   = 0.45          # NMS IoU threshold (suppress overlapping boxes)
IMAGE_SIZE   = 640           # must match training size
OUTPUT_DIR   = "runs/predictions"

# Bounding box style
BOX_COLOR    = (80, 80, 255)   # BGR — red-ish
TEXT_COLOR   = (255, 255, 255) # white
BOX_THICKNESS= 2
FONT_SCALE   = 0.6


# -----------------------------------------------------------------------------
# 2. LOAD MODEL
# -----------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    Loads the trained YOLOv8 model from the given path.
    Raises a clear error if the weights file doesn't exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\nModel weights not found at: {model_path}\n"
            f"Make sure training completed and the path is correct.\n"
            f"Run: python src/train.py   to train first."
        )
    print(f"[+] Loading model from {model_path}")
    model = YOLO(model_path)
    print(f"[+] Model loaded — {sum(p.numel() for p in model.model.parameters()):,} parameters\n")
    return model


# -----------------------------------------------------------------------------
# 3. PREDICT ON A SINGLE IMAGE
# -----------------------------------------------------------------------------

def predict_image(model: YOLO, img_path: str, save: bool = False, show: bool = True):
    """
    Runs inference on a single image.
    Draws bounding boxes and confidence scores.
    Optionally saves the annotated image to OUTPUT_DIR.

    Returns:
        annotated_img : np.ndarray  (BGR, same size as input)
        detections    : list of dicts with keys: bbox, confidence, class_name
    """
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"  [!] Image not found: {img_path}")
        return None, []

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [!] Could not read image: {img_path}")
        return None, []

    # Run inference and time it
    t0 = time.time()
    results = model.predict(
        source     = str(img_path),
        conf       = CONF_THRESH,
        iou        = IOU_THRESH,
        imgsz      = IMAGE_SIZE,
        verbose    = False,
    )
    inference_ms = (time.time() - t0) * 1000

    # Parse detections
    detections   = []
    annotated    = img.copy()
    result       = results[0]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf            = float(box.conf[0])
        cls_id          = int(box.cls[0])
        class_name      = model.names[cls_id]

        detections.append({
            "bbox"       : (x1, y1, x2, y2),
            "confidence" : round(conf, 3),
            "class_name" : class_name,
        })

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Draw label background
        label     = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)

        # Draw label text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA
        )

    # Print summary
    print(f"  Image   : {img_path.name}")
    print(f"  Found   : {len(detections)} pothole(s)")
    print(f"  Time    : {inference_ms:.1f}ms")
    for d in detections:
        print(f"    → {d['class_name']} @ {d['confidence']:.2f}  bbox={d['bbox']}")

    # Optionally save
    if save:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved   : {out_path}")

    # Optionally display
    if show:
        cv2.imshow("Pothole Detection", annotated)
        print("  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print()
    return annotated, detections


# -----------------------------------------------------------------------------
# 4. PREDICT ON A FOLDER OF IMAGES
# -----------------------------------------------------------------------------

def predict_folder(model: YOLO, folder_path: str, save: bool = True):
    """
    Runs inference on every .jpg and .png in a folder.
    Saves all annotated images to OUTPUT_DIR.
    Prints a summary table at the end.
    """
    folder = Path(folder_path)
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    images = [i for i in images if not i.name.startswith("pred_")]  # skip already predicted

    if not images:
        print(f"  [!] No images found in {folder_path}")
        return

    print(f"[+] Running inference on {len(images)} images in {folder_path}\n")

    all_detections = []
    total_time     = 0

    for img_path in sorted(images):
        t0 = time.time()
        _, dets = predict_image(model, str(img_path), save=save, show=False)
        total_time += (time.time() - t0) * 1000
        all_detections.append((img_path.name, len(dets)))

    # Summary
    total_potholes = sum(d[1] for d in all_detections)
    avg_time       = total_time / len(images)

    print("=" * 50)
    print(f"  SUMMARY")
    print("=" * 50)
    print(f"  Images processed : {len(images)}")
    print(f"  Total potholes   : {total_potholes}")
    print(f"  Avg inference    : {avg_time:.1f}ms per image")
    print(f"  Results saved to : {OUTPUT_DIR}/")
    print("=" * 50)


# -----------------------------------------------------------------------------
# 5. BENCHMARK INFERENCE SPEED
# -----------------------------------------------------------------------------

def benchmark(model: YOLO, img_path: str, runs: int = 20):
    """
    Runs inference N times on the same image and reports average speed.
    Put the result in your README: 'Xms average inference on Apple M4 CPU'
    """
    print(f"[+] Benchmarking over {runs} runs...\n")

    times = []
    for i in range(runs):
        t0 = time.time()
        model.predict(source=img_path, conf=CONF_THRESH, imgsz=IMAGE_SIZE, verbose=False)
        times.append((time.time() - t0) * 1000)

    times_np = np.array(times)
    print(f"  Mean   : {times_np.mean():.1f}ms")
    print(f"  Median : {np.median(times_np):.1f}ms")
    print(f"  Min    : {times_np.min():.1f}ms")
    print(f"  Max    : {times_np.max():.1f}ms")
    print(f"\n  → Use this in your README:")
    print(f"    'Average inference: {times_np.mean():.0f}ms per image on CPU'")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pothole Detection — Inference")
    parser.add_argument("--source",    type=str, default="data/sample_images",
                        help="Image path, folder path, or 0 for webcam")
    parser.add_argument("--model",     type=str, default=MODEL_PATH,
                        help="Path to trained weights (.pt)")
    parser.add_argument("--conf",      type=float, default=CONF_THRESH,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--save",      action="store_true",
                        help="Save annotated images to runs/predictions/")
    parser.add_argument("--show",      action="store_true",
                        help="Display each image in a window")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference speed")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args  = parse_args()
    model = load_model(args.model)

    source = Path(args.source)

    # Benchmark mode
    if args.benchmark:
        sample = next(source.glob("*.jpg")) if source.is_dir() else source
        benchmark(model, str(sample))

    # Folder mode
    elif source.is_dir():
        predict_folder(model, str(source), save=True)

    # Single image mode
    else:
        predict_image(model, str(source), save=args.save, show=args.show)