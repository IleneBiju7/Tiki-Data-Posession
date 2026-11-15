"""How often is a chain of weak ball sightings really the ball? Checked on human-labelled footage from OTHER
matches (SoccerNet tracking test set), never on BAR-ATH.

Mirrors ball_detect_v2.py: a 320x320 crop enlarged 2x, centred on a prior ball position. Per sequence two crop
streams run over every frame:
  ball   centred on the labelled ball plus a fixed random offset (a prior that is a few metres wrong)
  empty  a fixed spot well away from the ball (moved when the ball comes close)
Detections are linked into chains image to image (one or two missed images allowed, at most MAX_STEP px per
frame, a hard shot). A detection is "right" if it lies within max(8 px, ball size) of the label.

For each (min confidence, min chain duration) it prints the share of chained detections that are right, next to
the rule fuse_ball.py already trusts (conf >= 0.5, >= 3 sightings).

Results are cached in cache/soccernet_chain_dets.npz; --redetect runs the detector again.
Usage: python eval_chains.py [weights]   (default models/weights/ball_combined_s.pt)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import devices
import eval_soccernet as es

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "soccernet_chain_dets.npz"
MAX_STEP = 30.0        # px per frame on 1920x1080 broadcast footage (~35 m/s)
EVERY = 5              # the SoccerNet test images are every 5th video frame
OFFSET = 100           # px: largest prior error for the ball stream
SEED = 11


def detect(weights):
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO
    model = YOLO(weights)
    rng = np.random.default_rng(SEED)
    labelled = 0
    rows = []   # (sequence, stream, frame, conf, x, y, ball_x, ball_y, size)
    for k, seq in enumerate(sorted(es.ROOT.iterdir())):
        boxes = es.ball_boxes(seq)
        if not boxes:
            continue
        off = rng.uniform(-OFFSET, OFFSET, 2)
        spot = None
        for img_path in sorted((seq / "img1").glob("*.jpg")):
            f = int(img_path.stem)
            if f not in boxes:
                continue
            bx, by, size = boxes[f]
            labelled += 1
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            if spot is None or np.hypot(spot[0] + 160 - bx, spot[1] + 160 - by) < 400:
                while True:
                    spot = (int(rng.integers(0, w - es.CROP)), int(rng.integers(0, h - es.CROP)))
                    if np.hypot(spot[0] + 160 - bx, spot[1] + 160 - by) > 500:
                        break
            centre = (bx + off[0] - 160, by + off[1] - 160)
            for stream, (x0, y0) in ((0, centre), (1, spot)):
                x0, y0 = int(np.clip(x0, 0, w - es.CROP)), int(np.clip(y0, 0, h - es.CROP))
                crop = cv2.resize(img[y0:y0 + es.CROP, x0:x0 + es.CROP], (es.CROP * es.SCALE,) * 2,
                                  interpolation=cv2.INTER_CUBIC)
                r = model.predict(crop, imgsz=es.CROP * es.SCALE, conf=0.05, classes=[32], device=devices.yolo_device(), verbose=False)[0]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    rows.append((k, stream, f, c, x0 + (x1 + x2) / 4, y0 + (y1 + y2) / 4, bx, by, size))
        print(f"{seq.name}: {len(rows)} detections so far", flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, rows=np.array(rows, dtype=np.float64), labelled=labelled)


def chains(d, min_conf):
    """Chain id per detection (-1 = below min_conf), linking nearest detections across <= 3 frames."""
    cid = np.full(len(d), -1)
    nxt = 0
    for (seq, stream) in {(int(a), int(b)) for a, b in d[:, :2]}:
        idx = np.flatnonzero((d[:, 0] == seq) & (d[:, 1] == stream) & (d[:, 3] >= min_conf))
        idx = idx[np.argsort(d[idx, 2], kind="stable")]
        open_ = []   # (chain id, last frame, x, y)
        for i in idx:
            f, x, y = d[i, 2], d[i, 4], d[i, 5]
            best, bd = None, None
            for j, (c, lf, lx, ly) in enumerate(open_):
                gap = f - lf
                if 1 <= gap <= 3 * EVERY:
                    dist = np.hypot(x - lx, y - ly)
                    if dist <= MAX_STEP * gap and (bd is None or dist < bd):
                        best, bd = j, dist
            if best is None:
                open_.append((nxt, f, x, y))
                cid[i] = nxt
                nxt += 1
            else:
                c = open_[best][0]
                open_[best] = (c, f, x, y)
                cid[i] = c
            open_ = [o for o in open_ if f - o[1] <= 3 * EVERY]
    return cid


def main():
    if "--redetect" in sys.argv or not CACHE.exists():
        weights = next((a for a in sys.argv[1:] if not a.startswith("--")), str(HERE / "models" / "weights" / "ball_combined_s.pt"))
        detect(weights)
    z = np.load(CACHE)
    d, n_frames = z["rows"], int(z["labelled"])
    right = np.hypot(d[:, 4] - d[:, 6], d[:, 5] - d[:, 7]) <= np.maximum(8, d[:, 8])
    print(f"{len(d)} detections on {n_frames} labelled frames; {right.mean():.1%} of all detections are the ball")
    print("min conf  min chain  chained dets  right   (ball-stream recall: labelled frames with a right chained det)")
    for min_conf in (0.05, 0.1, 0.2, 0.3, 0.5):
        cid = chains(d, min_conf)
        for min_len in (3, 11, 13, 25, 50):
            ok = cid >= 0
            if not ok.any():
                continue
            ids, inv = np.unique(cid[ok], return_inverse=True)
            fr = d[ok, 2]
            span = np.zeros(len(ids)); cnt = np.bincount(inv)
            np.maximum.at(span, inv, fr); lo = np.full(len(ids), np.inf); np.minimum.at(lo, inv, fr)
            span = span - lo + 1
            # the existing rule counts sightings (>= 3); the new ones need a duration
            keep_chain = (cnt >= 3) if min_len == 3 else (span >= min_len) & (cnt >= 0.5 * span / EVERY)
            sel = np.zeros(len(d), bool)
            sel[np.flatnonzero(ok)] = keep_chain[inv]
            ball = sel & (d[:, 1] == 0) & right
            recall = len({(a, b) for a, b in d[ball][:, [0, 2]]}) / n_frames
            print(f"  {min_conf:<6}  {min_len:>3} fr     {sel.sum():>8}    {right[sel].mean():6.1%}   {recall:6.1%}")


if __name__ == "__main__":
    main()
