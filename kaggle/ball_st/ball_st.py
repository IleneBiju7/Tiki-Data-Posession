"""The ball detector fine-tuned for wide fixed cameras on a Kaggle GPU: ball_soccernet_k.pt (SoccerNet only, see
kaggle/ball_soccernet) trained 3 more epochs on the same SoccerNet crops and mosaics plus SoccerTrack v2 crops (CC BY
4.0, panoramic cameras like ours): the ball at the acting players' feet at kick and dribble events, boxed by our own
detector where its sightings agree over neighbouring frames (ball_pseudo_st.py, ball_pseudo_select.py). The two
benchmark test matches (M7, M9) are not in it. No competition data. Output: ball_st_k3.pt
"""

import configparser
import glob
import math
import os
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4"], check=False)
import cv2
import numpy as np

OUT = "/kaggle/working"
CROPS, MOSAICS = f"{OUT}/crops", f"{OUT}/crops_multiscale"
CROP, SCALE, EVERY, NEG_SHARE, SEED = 320, 2, 3, 0.3, 7
SCALES, GRID_TILE, OUT_SIZE, MIN_SIDE, N_MOSAICS = (0.25, 0.75), 2, 640, 6.0, 3000


def crops():
    rng = np.random.default_rng(SEED)
    for d in ("images", "labels"):
        os.makedirs(f"{CROPS}/{d}", exist_ok=True)
    seqs = sorted({p for p in glob.glob("/kaggle/input/**/train/SNMOT-*", recursive=True)
                   if os.path.exists(os.path.join(p, "gt", "gt.txt"))}, key=os.path.basename)
    n_pos = n_neg = 0
    for seq in seqs:
        info = configparser.ConfigParser()
        info.read(os.path.join(seq, "gameinfo.ini"))
        ids = {int(k.split("_")[1]) for k, v in info["Sequence"].items() if k.startswith("trackletid_") and "ball" in v.lower()}
        boxes = {}
        for line in open(os.path.join(seq, "gt", "gt.txt")):
            f, tid, x, y, w, h = map(float, line.split(",")[:6])
            if int(tid) in ids:
                boxes[int(f)] = (x + w / 2, y + h / 2, max(w, h))
        for f in sorted(boxes)[::EVERY]:
            path = os.path.join(seq, "img1", f"{f:06d}.jpg")
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
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
            name = f"{os.path.basename(seq)}_{f:06d}"
            cv2.imwrite(f"{CROPS}/images/{name}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(f"{CROPS}/labels/{name}.txt", "w").write(label)
    print(f"crops: {n_pos} with ball, {n_neg} without, from {len(seqs)} clips", flush=True)


def mosaics():
    rng = np.random.default_rng(SEED)
    names = sorted(os.path.splitext(p)[0] for p in os.listdir(f"{CROPS}/images"))
    for d in ("images", "labels"):
        os.makedirs(f"{MOSAICS}/{d}", exist_ok=True)
    order, k, n_balls = rng.permutation(len(names)), 0, 0
    for i in range(N_MOSAICS):
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
                small = cv2.resize(cv2.imread(f"{CROPS}/images/{name}.jpg"), (native, native), interpolation=cv2.INTER_AREA)
                canvas[gy * tile:(gy + 1) * tile, gx * tile:(gx + 1) * tile] = cv2.resize(small, (tile, tile), interpolation=cv2.INTER_CUBIC)
                for line in open(f"{CROPS}/labels/{name}.txt").read().splitlines():
                    if line.strip():
                        _, cx, cy, w, _ = line.split()
                        boxes.append((gx * tile + float(cx) * tile, gy * tile + float(cy) * tile, max(float(w) * tile, MIN_SIDE)))
        ox, oy = (int(v) for v in rng.integers(0, grid * tile - OUT_SIZE + 1, 2))
        lines = [f"32 {(x - ox) / OUT_SIZE:.5f} {(y - oy) / OUT_SIZE:.5f} {s / OUT_SIZE:.5f} {s / OUT_SIZE:.5f}"
                 for x, y, s in boxes if 0 <= x - ox < OUT_SIZE and 0 <= y - oy < OUT_SIZE]
        n_balls += len(lines)
        cv2.imwrite(f"{MOSAICS}/images/shrunk_{i:05d}.jpg", canvas[oy:oy + OUT_SIZE, ox:ox + OUT_SIZE], [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{MOSAICS}/labels/shrunk_{i:05d}.txt", "w").write("".join(l + "\n" for l in lines))
    print(f"mosaics: {N_MOSAICS}, {n_balls} balls", flush=True)



ST = f"{OUT}/crops_st"


def soccertrack():
    src = sorted(glob.glob("/kaggle/input/**/crops_st_all/images", recursive=True))[0]   # Kaggle unpacks the upload
    for d in ("images", "labels"):
        os.makedirs(f"{ST}/{d}", exist_ok=True)
    n = 0
    for p in glob.glob(f"{src}/*.jpg"):
        name = os.path.basename(p)[:-4]
        if name.startswith("._"):   # macOS archive metadata, not images
            continue
        img = cv2.imread(p)   # 320 px crop, enlarged 2x as every other training crop
        if img is None:
            continue
        cv2.imwrite(f"{ST}/images/{name}.jpg", cv2.resize(img, (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy(f"{os.path.dirname(src)}/labels/{name}.txt", f"{ST}/labels/{name}.txt")
        n += 1
    print(f"soccertrack: {n} crops", flush=True)


def train():
    from ultralytics import YOLO
    import yaml
    base = sorted(glob.glob("/kaggle/input/**/ball_soccernet_k.pt", recursive=True))[0]
    names = YOLO(base).names
    data = f"{OUT}/ball_st.yaml"
    yaml.safe_dump({"path": OUT, "train": [f"{CROPS}/images", f"{MOSAICS}/images", f"{ST}/images"],
                    "val": f"{CROPS}/images", "names": names}, open(data, "w"), sort_keys=False)
    model = YOLO(base)
    model.train(data=data, imgsz=640, epochs=3, batch=8, device=0, workers=2, seed=7, deterministic=True,
                project=f"{OUT}/runs", name="ball_st_k3", exist_ok=True, plots=False, verbose=False,
                optimizer="SGD", lr0=0.002, momentum=0.9, warmup_epochs=0, warmup_bias_lr=0.0,
                mosaic=0.5, degrees=0, flipud=0.0, fliplr=0.5, val=False)
    shutil.copy(f"{OUT}/runs/ball_st_k3/weights/last.pt", f"{OUT}/ball_st_k3.pt")
    shutil.rmtree(CROPS); shutil.rmtree(MOSAICS); shutil.rmtree(ST); shutil.rmtree(f"{OUT}/runs")
    print("weights -> ball_st_k3.pt", flush=True)


if __name__ == "__main__":
    crops()
    mosaics()
    soccertrack()
    train()
