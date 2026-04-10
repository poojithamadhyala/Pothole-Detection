# =============================================================================
# src/train.py
# Pothole Detection — YOLOv8 Training Script
# =============================================================================
# Usage (local):  python src/train.py
# Usage (Colab):  paste entire file into a cell and run
# =============================================================================

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# 1. CONFIGURATION — change these values if needed
# -----------------------------------------------------------------------------

# Path to your unzipped dataset folder (contains train/, valid/, test/, data.yaml)
DATASET_DIR  = "data/Pothole detection.v12i.yolov8"

MODEL_SIZE   = "yolov8n.pt"     # n=nano (fastest). Options: n, s, m, l, x
EPOCHS       = 50               # early stopping (PATIENCE) will cut this short
IMAGE_SIZE   = 640              # standard YOLO input size
BATCH_SIZE   = 16               # reduce to 8 if you get OOM errors
PATIENCE     = 10               # reduced: stop sooner when val loss plateaus
PROJECT_DIR  = "runs/detect/runs/pothole"   # where results are saved
RUN_NAME     = "yolov8n-v1"    # name your experiment

# Speed-up settings
DEVICE       = "mps"            # Apple Silicon GPU — change to 0 for NVIDIA, "cpu" as fallback
WORKERS      = 4                # parallel data-loading threads
CACHE        = True             # load images into RAM — eliminates disk I/O each epoch
AMP          = True             # mixed precision (float16) — ~2x faster on GPU/MPS
CLOSE_MOSAIC = 5                # disable mosaic in last N epochs to stabilise faster


# -----------------------------------------------------------------------------
# 2. VERIFY DATASET
# -----------------------------------------------------------------------------

def verify_dataset():
    """
    Checks that the dataset folder exists and has the expected structure.
    Prints a summary of image counts per split.
    """
    print("\n[1/3] Verifying dataset...")

    dataset_path = Path(DATASET_DIR)
    data_yaml    = dataset_path / "data.yaml"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\nDataset folder not found: {DATASET_DIR}\n"
            f"Make sure you unzipped the dataset into data/pothole_dataset/\n"
            f"Run: unzip ~/Downloads/Pothole_detection_v12i_yolov8.zip -d data/pothole_dataset"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"\ndata.yaml not found inside {DATASET_DIR}\n"
            f"Check that your zip extracted correctly."
        )

    # Read class info
    with open(data_yaml) as f:
        info = yaml.safe_load(f)

    # Count images per split
    splits = {}
    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            splits[split] = count
        else:
            splits[split] = 0

    print(f"  Classes : {info.get('names', 'unknown')}")
    print(f"  Train   : {splits['train']} images")
    print(f"  Val     : {splits['valid']} images")
    print(f"  Test    : {splits['test']} images")
    print(f"  YAML    : {data_yaml}\n")

    return str(data_yaml)


# -----------------------------------------------------------------------------
# 3. TRAIN
# -----------------------------------------------------------------------------

def train(data_yaml_path: str):
    """
    Fine-tunes YOLOv8n on the pothole dataset.

    Key augmentations enabled by default in ultralytics:
      - Mosaic (4-image composite)     — improves small object detection
      - Random horizontal flip         — handles both driving sides
      - HSV colour jitter              — handles day/night/weather variation
      - Scale + translate              — handles varying camera distances

    These are all on by default — no extra config needed.
    """
    print("[2/3] Starting training...")
    print(f"  Model   : {MODEL_SIZE}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  ImgSz   : {IMAGE_SIZE}\n")

    # Load pretrained YOLOv8 nano backbone (pretrained on COCO)
    # Fine-tuning from COCO weights is much faster than training from scratch
    model = YOLO(MODEL_SIZE)

    results = model.train(
        data         = data_yaml_path,
        epochs       = EPOCHS,
        imgsz        = IMAGE_SIZE,
        batch        = BATCH_SIZE,
        patience     = PATIENCE,        # stops early if val loss plateaus
        project      = PROJECT_DIR,
        name         = RUN_NAME,
        exist_ok     = True,            # overwrite previous run with same name
        plots        = True,            # saves training curves, confusion matrix
        verbose      = True,
        # --- speed-ups ---
        device       = DEVICE,          # use GPU/MPS instead of CPU
        workers      = WORKERS,         # parallel data loading
        cache        = CACHE,           # images stay in RAM after first epoch
        amp          = AMP,             # float16 mixed precision
        close_mosaic = CLOSE_MOSAIC,    # stabilise faster at end of training
    )

    return results


# -----------------------------------------------------------------------------
# 4. EVALUATE ON TEST SET
# -----------------------------------------------------------------------------

def evaluate(data_yaml_path: str):
    """
    Loads the best checkpoint and runs it on the held-out test set.
    Reports mAP@50, mAP@50-95, precision, recall.
    These are the numbers you put in your README.
    """
    print("\n[3/3] Evaluating best model on test set...")

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights/best.pt"

    if not best_weights.exists():
        print(f"  ERROR: could not find weights at {best_weights}")
        print("  Make sure training completed successfully.")
        return

    model   = YOLO(str(best_weights))
    metrics = model.val(
        data    = data_yaml_path,
        split   = "test",           # explicitly evaluate on test split
        plots   = True,
        verbose = True,
    )

    # Print the numbers you'll copy into your README
    print("\n" + "="*50)
    print("  RESULTS — copy these into your README")
    print("="*50)
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("="*50)
    print(f"\n  Plots saved to: {PROJECT_DIR}/{RUN_NAME}/\n")


# -----------------------------------------------------------------------------
# 5. EXPORT TO ONNX
# -----------------------------------------------------------------------------

def export_onnx():
    """
    Exports the best PyTorch checkpoint to ONNX format.
    ONNX models run on any hardware without PyTorch installed.
    Mention this in your README: 'Exported to ONNX — Xms inference on CPU'
    """
    print("\n[Bonus] Exporting to ONNX...")

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights/best.pt"
    model        = YOLO(str(best_weights))

    model.export(
        format   = "onnx",
        imgsz    = IMAGE_SIZE,
        dynamic  = True,            # allows variable batch sizes
        simplify = True,            # smaller file, faster inference
    )

    onnx_path = best_weights.with_suffix(".onnx")
    print(f"  Saved : {onnx_path}")
    print("  Tip   : benchmark with `yolo benchmark model=best.onnx`\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_yaml = verify_dataset()
    evaluate(data_yaml)
    export_onnx()
    print("Done! Your trained model is at:")
    print(f"  {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print("\nNext step: run  python src/predict.py  to test on sample images.")