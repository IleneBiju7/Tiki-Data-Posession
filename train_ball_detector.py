"""Fine-tune the ball detector (COCO "sports ball" class kept) on 2x-enlarged 320 px crops.

Gentle fine-tuning settings: SGD, low learning rate, no warm-up bias boost -- the library defaults are
meant for training from scratch and wipe out the pretrained ball class. Seeded.

Usage: python train_ball_detector.py <data.yaml> <run_name> [epochs] [base weights, default yolo11n.pt] [batch]
Weights end up in models/weights/<run_name>.pt
"""
import os
import shutil
import sys
from pathlib import Path

import atd_cache
import ball_detect as bd
import devices

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
RUNS = atd_cache.DATA_DIR / "detector_runs"


def main():
    from ultralytics import YOLO
    data, name = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    base = sys.argv[4] if len(sys.argv) > 4 else "yolo11n.pt"
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    model = YOLO(str(bd.WEIGHTS.parent / base))
    model.train(data=str(Path(data).resolve()), imgsz=640, epochs=epochs, batch=batch, device=devices.yolo_device(), workers=2, seed=7,
                deterministic=True, project=str(RUNS), name=name, exist_ok=True, plots=False, verbose=False,
                optimizer="SGD", lr0=0.002, momentum=0.9, warmup_epochs=0, warmup_bias_lr=0.0,
                mosaic=0.5, degrees=0, flipud=0.0, fliplr=0.5, val=False)
    out = bd.WEIGHTS.parent / f"{name}.pt"
    shutil.copy(RUNS / name / "weights" / "last.pt", out)
    print(f"weights -> {out}")


if __name__ == "__main__":
    main()
