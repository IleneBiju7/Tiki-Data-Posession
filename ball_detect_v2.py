"""Run a ball detector (models/weights) over the match.

The detector was trained on 320 px crops enlarged 2x, so it is run the same way: on every 2nd graded
frame, one crop centred on the best prior guess of the ball (our fused ball carried over gaps, else
the tracker ball, else the players' centroid), and a second crop on the players' centroid when that
is far from the prior (the ball is usually where the players are).

Usage: python ball_detect_v2.py [step] [weights file in models/weights] [output tag] [prior]
       prior: "tracker" (the provided tracker ball, carried over gaps) or the tag of a fused ball from an earlier
       pass (cache/BAR-ATH_ball_fused_<tag>.npy). run_final.sh runs two passes: tracker prior, then the first
       pass's fused ball, so the pipeline needs nothing that it does not build itself.
Output: cache/ball_detections_v2.csv.gz, or ball_detections_<tag>.csv.gz (same columns as ball_detect.py)
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

import atd_cache
import devices
import ball_detect as bd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
WEIGHTS = bd.WEIGHTS.parent / "ball_combined.pt"
OUT = atd_cache.DATA_DIR / "cache" / "ball_detections_v2.csv.gz"
PITCH = np.array((105.3, 68.0))
CROP, SCALE = 320, 2
FAR = 15.0  # m


def project(H, x, y):
    d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d


def main():
    from ultralytics import YOLO
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    weights = bd.WEIGHTS.parent / sys.argv[2] if len(sys.argv) > 2 else WEIGHTS
    out = OUT.with_name(f"ball_detections_{sys.argv[3]}.csv.gz") if len(sys.argv) > 3 else OUT
    atd = atd_cache.load()
    if len(sys.argv) > 4 and sys.argv[4] == "tracker":
        prior = pd.DataFrame(atd["ball"].astype(np.float64)).ffill().to_numpy()
    else:
        prior_name = (f"BAR-ATH_ball_fused_{sys.argv[4]}.npy" if len(sys.argv) > 4
                      else "BAR-ATH_ball_fused_v6.npy" if len(sys.argv) > 3 else "BAR-ATH_ball_fused_v3.npy")
        prior = pd.DataFrame(np.load(atd_cache.DATA_DIR / "cache" / prior_name)).ffill().to_numpy()
    frames = [f for a, b in bd.HALVES for f in range(a, b + 1, step)]
    model = YOLO(str(weights))
    H = bd.homographies()
    cap = cv2.VideoCapture(str(bd.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    wanted, rows, t0, done = set(frames), [], time.time(), 0
    for f in range(frames[0], frames[-1] + 1):
        if not cap.grab():
            break
        if f not in wanted:
            continue
        _, img = cap.retrieve()
        centroid = np.nanmean(atd["players"][f], 0)
        guess = prior[f] if np.isfinite(prior[f, 0]) else (atd["ball"][f] if np.isfinite(atd["ball"][f, 0]) else centroid)
        centres = [guess]
        if np.isfinite(centroid[0]) and np.hypot(*((guess - centroid) * PITCH)) > FAR:
            centres.append(centroid)
        for cx_n, cy_n in centres:
            cx, cy = project(H[f], cx_n, cy_n)
            x0 = int(np.clip(cx - CROP / 2, 0, 1920 - CROP))
            y0 = int(np.clip(cy - CROP / 2, 0, 1080 - CROP))
            crop = cv2.resize(img[y0:y0 + CROP, x0:x0 + CROP], (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC)
            r = model.predict(crop, imgsz=CROP * SCALE, conf=0.05, classes=[32], device=devices.yolo_device(), verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                px, py = x0 + (x1 + x2) / 2 / SCALE, y0 + (y1 + y2) / 2 / SCALE
                nx, ny = bd.unproject(H[f], px, py)
                rows.append((f, float(c), float(px), float(py), float(nx), float(ny), float((x2 - x1) / SCALE)))
        done += 1
        if done % 5000 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(frames)} frames, {rate:.1f} fps, eta {(len(frames) - done) / rate / 60:.0f} min", flush=True)
    det = pd.DataFrame(rows, columns=["frame", "conf", "px", "py", "x", "y", "size"])
    det.to_csv(out, index=False)
    print(f"wrote {len(det)} detections on {det.frame.nunique()} frames -> {out}")


if __name__ == "__main__":
    main()
