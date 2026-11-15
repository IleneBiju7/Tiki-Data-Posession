"""Training images at wide-camera scales from the SoccerNet TRAIN crops (build_soccernet_crops.py output: human ball
boxes, other matches; the test set is never used). Each crop (a 320 px TV region enlarged 2x) is brought to a
wide-camera scale: the TV region x0.4 (INTER_AREA), then enlarged 2x (INTER_CUBIC),
a 256 px tile. 3x3 tiles from random crops make one 640x640 image (a random 640 window of the 768 px grid); crops
without a ball (30%) give tiles without one. Ball boxes follow (side at least MIN_SIDE px).
Usage: python build_shrunk_mosaics.py [n images, default 3000]   -> ../data/open/soccernet/crops_shrunk

--scales=lo,hi: each image gets its own shrink, log-uniform between lo and hi (a range of wide cameras, not one
match's scale), and as many tiles as fill it -> ../data/open/soccernet/crops_multiscale. For the SoccerNet-only
detector: python build_shrunk_mosaics.py 3000 --scales=0.25,0.75, then train_ball_detector.py
../data/open/soccernet/ball_soccernet.yaml ball_soccernet_s 6 yolo11s.pt 4 (SoccerNet crops + these, nothing else).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "data" / "open" / "soccernet" / "crops"
SCALES = next((tuple(float(v) for v in a.split("=", 1)[1].split(",")) for a in sys.argv if a.startswith("--scales=")),
              None)
OUT = SRC.with_name("crops_shrunk" if SCALES is None else "crops_multiscale")
SHRINK, CROP, SCALE, GRID, OUT_SIZE, MIN_SIDE, SEED = 0.4, 320, 2, 3, 640, 6.0, 7


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_out = int(args[0]) if args else 3000
    rng = np.random.default_rng(SEED)
    names = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))
    native = int(round(CROP * SHRINK))   # 128 px: the TV region at this scale
    tile = native * SCALE                # 256 px after the 2x enlargement
    grid = GRID
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    order, k, n_balls = rng.permutation(len(names)), 0, 0
    for i in range(n_out):
        if SCALES is not None:   # this image's camera: shrink log-uniform in the range, enough tiles to fill it
            native = int(round(CROP * math.exp(rng.uniform(math.log(SCALES[0]), math.log(SCALES[1])))))
            tile = native * SCALE
            grid = math.ceil(OUT_SIZE / tile) + 1
        canvas = np.zeros((grid * tile, grid * tile, 3), np.uint8)
        boxes = []
        for gy in range(grid):
            for gx in range(grid):
                if k == len(order):
                    order, k = rng.permutation(len(names)), 0
                name = names[order[k]]
                k += 1
                small = cv2.resize(cv2.imread(str(SRC / "images" / f"{name}.jpg")), (native, native),
                                   interpolation=cv2.INTER_AREA)
                canvas[gy * tile:(gy + 1) * tile, gx * tile:(gx + 1) * tile] = cv2.resize(
                    small, (tile, tile), interpolation=cv2.INTER_CUBIC)
                for line in (SRC / "labels" / f"{name}.txt").read_text().splitlines():
                    if line.strip():
                        _, cx, cy, w, _ = line.split()
                        boxes.append((gx * tile + float(cx) * tile, gy * tile + float(cy) * tile,
                                      max(float(w) * tile, MIN_SIDE)))
        ox, oy = (int(v) for v in rng.integers(0, grid * tile - OUT_SIZE + 1, 2))
        lines = [f"32 {(x - ox) / OUT_SIZE:.5f} {(y - oy) / OUT_SIZE:.5f} {s / OUT_SIZE:.5f} {s / OUT_SIZE:.5f}"
                 for x, y, s in boxes if 0 <= x - ox < OUT_SIZE and 0 <= y - oy < OUT_SIZE]
        n_balls += len(lines)
        cv2.imwrite(str(OUT / "images" / f"shrunk_{i:05d}.jpg"), canvas[oy:oy + OUT_SIZE, ox:ox + OUT_SIZE],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / f"shrunk_{i:05d}.txt").write_text("".join(l + "\n" for l in lines))
    print(f"{n_out} images {f'at x{SHRINK}' if SCALES is None else f'at scales {SCALES}'}, {n_balls} balls "
          f"({n_balls / n_out:.1f} per image) -> {OUT}")


if __name__ == "__main__":
    main()
