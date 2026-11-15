"""Training crops from SoccerNet-Tracking TRAIN split (other matches, human-annotated ball boxes).

320x320 crops enlarged 2x, COCO class 32 ("sports ball"), plus
no-ball crops. Reads straight from train.zip (no full unzip). Every 3rd frame per clip.

Usage: python build_soccernet_crops.py
"""
import configparser
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "data" / "open" / "soccernet" / "train.zip"
OUT = SRC.parent / "crops"
CROP, SCALE, EVERY, NEG_SHARE = 320, 2, 3, 0.3
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(SRC)
    names = z.namelist()
    seqs = sorted({n.split("/")[1] for n in names if n.count("/") >= 2 and n.split("/")[1]})
    n_pos = n_neg = 0
    for seq in seqs:
        info = configparser.ConfigParser()
        info.read_string(z.read(f"train/{seq}/gameinfo.ini").decode())
        ids = {int(k.split("_")[1]) for k, v in info["Sequence"].items() if k.startswith("trackletid_") and "ball" in v.lower()}
        boxes = {}
        for line in z.read(f"train/{seq}/gt/gt.txt").decode().splitlines():
            f, tid, x, y, w, h = map(float, line.split(",")[:6])
            if int(tid) in ids:
                boxes[int(f)] = (x + w / 2, y + h / 2, max(w, h))
        for f in sorted(boxes)[::EVERY]:
            path = f"train/{seq}/img1/{f:06d}.jpg"
            if path not in names:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(path), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cx, cy, size = boxes[f]
            if rng.random() < NEG_SHARE:
                for _ in range(10):
                    x0, y0 = int(rng.integers(0, w - CROP)), int(rng.integers(0, h - CROP))
                    if not (x0 - 20 < cx < x0 + CROP + 20 and y0 - 20 < cy < y0 + CROP + 20):
                        break
                label = ""
                n_neg += 1
            else:
                x0 = int(np.clip(cx - rng.uniform(0.15, 0.85) * CROP, 0, w - CROP))
                y0 = int(np.clip(cy - rng.uniform(0.15, 0.85) * CROP, 0, h - CROP))
                side = max(size * SCALE, 10) / (CROP * SCALE)
                label = f"32 {(cx - x0) / CROP:.5f} {(cy - y0) / CROP:.5f} {side:.5f} {side:.5f}\n"
                n_pos += 1
            crop = cv2.resize(img[y0:y0 + CROP, x0:x0 + CROP], (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC)
            name = f"{seq}_{f:06d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / f"{name}.txt").write_text(label)
    print(f"SoccerNet train crops: {n_pos} with ball, {n_neg} without, from {len(seqs)} clips -> {OUT}")


if __name__ == "__main__":
    main()
