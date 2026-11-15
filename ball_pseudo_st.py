"""Our ball detector at the feet of SoccerTrack v2's acting players (soccertrack_events.py), on the panoramic video.

At every kick or dribble event the ball is at the actor's feet. Reading the half's video once, in order, around each
event frame (NEAR frames either side) this cuts a CROP x CROP square centred on the actor's feet at the event (bottom
centre of the box, the same square in every frame), keeps it as a JPEG for the training set, and runs the detector on
it as ball_detect_v2.py does on our match (enlarged 2x, conf >= 0.05). Every detection is written with its frame
offset from the event and its distance to the feet; ball_pseudo_select.py turns them into ball boxes.

Usage: python ball_pseudo_st.py <match> <half> [--weights=ball_soccernet_k.pt] [--preview=N]
       reads data/open/soccertrack/videos/<match>/<match>_panorama_<half>_half.mp4 and events_feet.csv
       writes data/open/soccertrack/pseudo/<match>_<half>/ (crops/<frame>_<player>.jpg, dets.csv)
       --preview=N: also N crops with the actor's box and the detections drawn (preview/), to look at
"""
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import devices

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent.parent / "data" / "open" / "soccertrack"
WEIGHTS = Path(__file__).resolve().parent / "models" / "weights"
CROP, SCALE = 320, 2
NEAR = 2   # frames either side of the event


def arg(key, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), default)


def main():
    from ultralytics import YOLO
    match, half = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    preview = int(arg("preview", 0))
    model = YOLO(str(WEIGHTS / arg("weights", "ball_soccernet_k.pt")))
    rows = [r for r in csv.DictReader(open(ROOT / "events_feet.csv")) if r["match"] == match and r["half"] == half]
    by_frame = {}   # video frame -> the events whose window covers it, with the frame's offset from the event
    for r in rows:
        for d in range(-NEAR, NEAR + 1):
            by_frame.setdefault(int(r["frame"]) + d, []).append((d, r))
    out = ROOT / "pseudo" / f"{match}_{half}"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    if preview:
        (out / "preview").mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "videos" / match / f"{match}_panorama_{half}_half.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dets, frame, t0, last = [], 0, time.time(), max(by_frame)
    shown = 0
    while frame < last:
        if not cap.grab():
            break
        frame += 1   # 1-based, as the annotations
        if frame not in by_frame:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        for d, r in by_frame[frame]:
            bx, by, bw, bh = (float(r[k]) for k in ("x", "y", "w", "h"))
            fx, fy = bx + bw / 2, by + bh                       # the actor's feet at the event
            x0 = int(np.clip(fx - CROP / 2, 0, W - CROP))
            y0 = int(np.clip(fy - CROP / 2, 0, H - CROP))
            crop = img[y0:y0 + CROP, x0:x0 + CROP]
            name = f"{int(r['frame']):06d}_{r['player_id']}_{d:+d}"
            cv2.imwrite(str(out / "crops" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            big = cv2.resize(crop, (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC)
            res = model.predict(big, imgsz=CROP * SCALE, conf=0.05, classes=[32], device=devices.yolo_device(), verbose=False)[0]
            found = []
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                px, py = x0 + (x1 + x2) / 2 / SCALE, y0 + (y1 + y2) / 2 / SCALE
                size = max(x2 - x1, y2 - y1) / SCALE
                found.append((float(c), px, py, size))
                dets.append([int(r["frame"]), d, r["player_id"], r["label"], name, x0, y0, fx, fy, bw, bh, round(float(c), 4),
                             round(px, 1), round(py, 1), round(size, 1), round(float(np.hypot(px - fx, py - fy)), 1)])
            if d == 0 and shown < preview:
                vis = cv2.resize(crop, (CROP * 2, CROP * 2), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(vis, (int((bx - x0) * 2), int((by - y0) * 2)), (int((bx + bw - x0) * 2), int((by + bh - y0) * 2)),
                              (255, 255, 0), 1)
                cv2.circle(vis, (int((fx - x0) * 2), int((fy - y0) * 2)), 4, (255, 255, 0), -1)
                for c, px, py, size in found:
                    cv2.circle(vis, (int((px - x0) * 2), int((py - y0) * 2)), int(size + 6), (0, 0, 255), 1)
                    cv2.putText(vis, f"{c:.2f}", (int((px - x0) * 2) + 8, int((py - y0) * 2) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                cv2.putText(vis, f"{r['label']} f{frame}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imwrite(str(out / "preview" / f"{name}.jpg"), vis)
                shown += 1
    with open(out / "dets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "offset", "player_id", "label", "crop", "x0", "y0", "feet_x", "feet_y", "box_w", "box_h", "conf",
                    "px", "py", "size", "dist"])
        w.writerows(dets)
    n = len(rows)
    print(f"{match} {half}: {n} events, {len(dets)} detections, video {W}x{H}, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
