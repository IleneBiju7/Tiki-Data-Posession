"""A ball detector on the SoccerTrack v2 benchmark test matches (M7 128057, M9 132831), never trained on: every saved
crop around a kick or dribble event (ball_pseudo_st.py, the event frame and NEAR frames either side) goes through the
detector as on our match (enlarged 2x, conf >= 0.05). Per detector:
  found at the feet   events whose event frame has a sighting at conf >= 0.5 within R_FEET box heights of the feet
  found by the rule   events where ball_pseudo_select.py's rule (sightings agreeing over the neighbouring frames) finds it
  far confident       sightings at conf >= 0.5 more than FAR box heights from the feet, per 1,000 crops (the ball is at
                      the feet in these frames, so nearly all of them are false)
Usage: python ball_eval_st.py ball_soccernet_k.pt [other weights in models/weights ...]
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import ball_pseudo_select as sel
import devices

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", Path(__file__).resolve().parent / "models" / "weights"))
DEVICE = os.environ.get("DEVICE") or devices.yolo_device()
FAR = 4.0


def crops():
    ev = pd.read_csv(sel.ROOT / "events_feet.csv", dtype={"match": str, "player_id": str})
    ev = ev[ev.match.isin(sel.TEST)]
    rows = []
    for _, r in ev.iterrows():
        d = sel.ROOT / "pseudo" / f"{r.match}_{r.half}" / "crops"
        if not d.exists():
            continue
        fx, fy = r.x + r.w / 2, r.y + r.h
        x0 = int(np.clip(fx - sel.CROP / 2, 0, r.width - sel.CROP))
        y0 = int(np.clip(fy - sel.CROP / 2, 0, r.height - sel.CROP))
        for o in range(-2, 3):
            p = d / f"{int(r.frame):06d}_{r.player_id}_{o:+d}.jpg"
            if p.exists():
                rows.append((f"{r.match}_{r.half}", int(r.frame), o, r.player_id, str(p), x0, y0, fx, fy, r.w, r.h))
    return pd.DataFrame(rows, columns=["half", "frame", "offset", "player_id", "path", "x0", "y0", "feet_x", "feet_y",
                                       "box_w", "box_h"])


def detect(model, table):
    out = []
    for _, c in table.iterrows():
        big = cv2.resize(cv2.imread(c.path), (sel.CROP * sel.SCALE,) * 2, interpolation=cv2.INTER_CUBIC)
        r = model.predict(big, imgsz=sel.CROP * sel.SCALE, conf=0.05, classes=[32], device=DEVICE, verbose=False)[0]
        for (x1, y1, x2, y2), conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            px, py = c.x0 + (x1 + x2) / 2 / sel.SCALE, c.y0 + (y1 + y2) / 2 / sel.SCALE
            out.append((c.half, c.frame, c.offset, c.player_id, float(conf), px, py, max(x2 - x1, y2 - y1) / sel.SCALE,
                        c.feet_x, c.feet_y, c.box_h, float(np.hypot(px - c.feet_x, py - c.feet_y))))
    return pd.DataFrame(out, columns=["half", "frame", "offset", "player_id", "conf", "px", "py", "size", "feet_x",
                                      "feet_y", "box_h", "dist"])


def main():
    from ultralytics import YOLO
    table = crops()
    events = table[table.offset == 0]
    print(f"test matches: {len(events)} events, {len(table)} crops", flush=True)
    for w in sys.argv[1:]:
        d = detect(YOLO(str(WEIGHTS / w)), table)
        ev0 = d[d.offset == 0]
        feet = ev0[(ev0.conf >= 0.5) & (ev0.dist <= sel.R_FEET * ev0.box_h)].groupby(["half", "frame", "player_id"]).size()
        found_rule = 0
        for _, g in d.groupby("half"):
            s = sel.select(g)
            found_rule += int((s.offset == 0).sum()) if len(s) else 0
        far = d[(d.conf >= 0.5) & (d.dist > FAR * d.box_h)]
        print(f"{w:28s} found at the feet {len(feet) / len(events):6.1%}   found by the rule {found_rule / len(events):6.1%}   "
              f"far confident per 1000 crops {1000 * len(far) / len(table):5.1f}", flush=True)


if __name__ == "__main__":
    main()
