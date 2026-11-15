"""Independent check of ball detectors on human-labelled footage from OTHER matches (SoccerNet tracking
test set). Used only to compare detectors, never for training.

Mirrors how our pipeline uses the detector: 320x320 crops enlarged 2x. For each labelled frame, one
crop contains the ball (random position) and one crop does not (to count false alarms).

  recall@t       share of ball crops where a detection >= t lies within max(8 px, ball size) of the label
  false_alarm@t  share of no-ball crops with any detection >= t

Usage: python eval_soccernet.py weights1.pt [weights2.pt ...]
"""
import configparser
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import devices

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent.parent / "data" / "open" / "soccernet" / "test"
CROP, SCALE = 320, 2
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
SEED = 7


def ball_boxes(seq):
    info = configparser.ConfigParser()
    info.read(seq / "gameinfo.ini")
    ids = {int(k.split("_")[1]) for k, v in info["Sequence"].items() if k.startswith("trackletid_") and "ball" in v.lower()}
    boxes = {}
    for line in open(seq / "gt" / "gt.txt"):
        f, tid, x, y, w, h = map(float, line.split(",")[:6])
        if int(tid) in ids:
            boxes[int(f)] = (x + w / 2, y + h / 2, max(w, h))
    return boxes


def crops(rng, every=2):
    """Yield (crop, ball_x, ball_y, size) one at a time -- streaming keeps memory small."""
    for seq in sorted(ROOT.iterdir()):
        boxes = ball_boxes(seq)
        for k, img_path in enumerate(sorted((seq / "img1").glob("*.jpg"))):
            f = int(img_path.stem)
            if f not in boxes or k % every:
                continue
            cx, cy, size = boxes[f]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            x0 = int(np.clip(cx - rng.uniform(0.15, 0.85) * CROP, 0, w - CROP))
            y0 = int(np.clip(cy - rng.uniform(0.15, 0.85) * CROP, 0, h - CROP))
            yield img[y0:y0 + CROP, x0:x0 + CROP], (cx - x0) * SCALE, (cy - y0) * SCALE, size * SCALE
            for _ in range(10):                                  # a crop well away from the ball
                nx, ny = int(rng.integers(0, w - CROP)), int(rng.integers(0, h - CROP))
                if not (nx - 20 < cx < nx + CROP + 20 and ny - 20 < cy < ny + CROP + 20):
                    yield img[ny:ny + CROP, nx:nx + CROP], None, None, None
                    break


def main():
    from ultralytics import YOLO
    rng = np.random.default_rng(SEED)
    models = {Path(w).name: YOLO(w) for w in sys.argv[1:]}
    hit = {m: {t: 0 for t in THRESHOLDS} for m in models}
    alarm = {m: {t: 0 for t in THRESHOLDS} for m in models}
    n_ball = n_none = 0
    for crop, bx, by, size in crops(rng):
        big = cv2.resize(crop, (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC)
        n_ball += bx is not None
        n_none += bx is None
        for name, model in models.items():
            r = model.predict(big, imgsz=CROP * SCALE, conf=min(THRESHOLDS), classes=[32], device=devices.yolo_device(), verbose=False)[0]
            xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            for t in THRESHOLDS:
                keep = conf >= t
                if bx is None:
                    alarm[name][t] += int(keep.any())
                elif keep.any():
                    c = (xyxy[keep, :2] + xyxy[keep, 2:]) / 2
                    if np.hypot(c[:, 0] - bx, c[:, 1] - by).min() <= max(16, size):
                        hit[name][t] += 1
    for name in models:
        print(f"\n{name}  ({n_ball} ball crops, {n_none} no-ball crops; other matches, human labels)")
        for t in THRESHOLDS:
            print(f"  conf >= {t:<4}: recall {hit[name][t] / n_ball:.1%}   false alarms {alarm[name][t] / n_none:.1%}")


if __name__ == "__main__":
    main()
