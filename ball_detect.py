"""Detect the ball in the tactical video with a pretrained public detector (COCO "sports ball").

For each processed frame, every ball detection is mapped from pixels to the pitch (0-1 normalised,
the ATD convention) through that frame's provided homography. Output: one row per detection.

  python ball_detect.py bench 13700 200        # speed / sanity check on a short clip
  python ball_detect.py run [step]             # graded frames, every `step`-th frame (default 1)
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import atd_cache
import devices

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # NMS is not implemented on Apple GPUs in torch 2.2

HERE = Path(__file__).resolve().parent
VIDEO = atd_cache.TACTICAL / "BAR-ATH_tactical.mp4"
OUT = atd_cache.DATA_DIR / "cache" / "ball_detections.csv.gz"
WEIGHTS = HERE / "models" / "weights" / "yolo11s.pt"
HALVES = [(11185, 90651), (110808, 190197)]
SPORTS_BALL = 32
IMGSZ = 1920
CONF = 0.05


def homographies():
    h = np.loadtxt(atd_cache.TACTICAL / "BAR-ATH_tactical_homography.csv", delimiter=",", skiprows=1,
                   usecols=range(1, 10))
    return h.reshape(-1, 3, 3)


def unproject(H, px, py):
    """Image pixels -> normalised pitch (0-1)."""
    inv = np.linalg.inv(H)
    d = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    return (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / d, (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / d


def load_model():
    from ultralytics import YOLO
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(WEIGHTS) if WEIGHTS.exists() else "yolo11s.pt")


def detect(frames_wanted, step_log=2000):
    model = load_model()
    H = homographies()
    cap = cv2.VideoCapture(str(VIDEO))
    rows, wanted = [], set(frames_wanted)
    first, last = min(frames_wanted), max(frames_wanted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    t0, done = time.time(), 0
    for f in range(first, last + 1):
        ok = cap.grab()
        if not ok:
            break
        if f not in wanted:
            continue
        _, img = cap.retrieve()
        res = model.predict(img, imgsz=IMGSZ, conf=CONF, classes=[SPORTS_BALL], device=devices.yolo_device(), verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            px, py = (x1 + x2) / 2, y2 - (y2 - y1) * 0.5
            nx, ny = unproject(H[f], px, py)
            rows.append((f, float(c), float(px), float(py), float(nx), float(ny), float(x2 - x1)))
        done += 1
        if done % step_log == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(wanted)} frames, {rate:.1f} fps, eta {(len(wanted) - done) / rate / 60:.0f} min",
                  flush=True)
    cap.release()
    return pd.DataFrame(rows, columns=["frame", "conf", "px", "py", "x", "y", "size"]), done / (time.time() - t0)


def select_track(det, n_frames, max_jump=6.0, max_gap=12):
    """One ball per frame from all detections: on the pitch only, then follow it through time.

    Spare balls and objects outside the touchlines are dropped. Where the previous pick is recent,
    the detection closest to it (within max_jump metres) wins; otherwise the most confident one.
    Returns (n_frames, 3): normalised x, y and confidence, NaN where nothing was picked.
    """
    on = det[(det.x > -0.01) & (det.x < 1.01) & (det.y > -0.01) & (det.y < 1.01)]
    out = np.full((n_frames, 3), np.nan, np.float32)
    last_f, last_xy = -10**9, None
    for f, g in on.groupby("frame", sort=True):
        xy = g[["x", "y"]].to_numpy() * (105.3, 68.0)
        conf = g["conf"].to_numpy()
        if last_xy is not None and f - last_f <= max_gap:
            dist = np.hypot(*(xy - last_xy).T)
            k = int(np.argmin(dist - conf))          # close to the last pick, confidence breaks ties
            if dist[k] > max_jump * max(1, (f - last_f) / 5):
                k = int(np.argmax(conf)) if conf.max() > 0.3 else -1
        else:
            k = int(np.argmax(conf)) if conf.max() > 0.15 else -1
        if k >= 0:
            out[f] = (*(xy[k] / (105.3, 68.0)), conf[k])
            last_f, last_xy = f, xy[k]
    return out


def main():
    mode = sys.argv[1]
    if mode == "bench":
        start, n = int(sys.argv[2]), int(sys.argv[3])
        det, fps = detect(list(range(start, start + n)), step_log=10**9)
        atd = atd_cache.load()["ball"]
        track = select_track(det, len(atd))
        frames = np.arange(start, start + n)
        found = np.isfinite(track[frames, 0])
        print(f"speed {fps:.1f} frames/s; ball found on {found.mean():.0%} of frames "
              f"(tracker has a ball on {np.isfinite(atd[frames, 0]).mean():.0%})")
        both = [f for f in frames if np.isfinite(atd[f, 0]) and np.isfinite(track[f, 0])]
        if both:
            gap = np.hypot((track[both, 0] - atd[both, 0]) * 105.3, (track[both, 1] - atd[both, 1]) * 68.0)
            print(f"where both have a ball: median distance {np.median(gap):.1f} m, "
                  f"agree within 2 m on {np.mean(gap < 2):.0%}, differ by >5 m on {np.mean(gap > 5):.0%}")
        miss = [f for f in frames if not np.isfinite(atd[f, 0]) and np.isfinite(track[f, 0])]
        print(f"frames where the tracker has NO ball but the detector does: {len(miss)}")
    else:
        step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        frames = [f for a, b in HALVES for f in range(a, b + 1, step)]
        print(f"detecting on {len(frames)} frames ...", flush=True)
        det, fps = detect(frames)
        det.to_csv(OUT, index=False)
        print(f"wrote {len(det)} detections for {det.frame.nunique()} frames -> {OUT} ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
