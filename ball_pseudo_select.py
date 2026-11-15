"""Ball boxes on SoccerTrack v2 from our detector's sightings at the acting players' feet (ball_pseudo_st.py), and the
training images made from them. No person decides any box.

Per kick or dribble event: a sighting in the event frame within R_FEET actor-box heights of the actor's feet is the
ball if it is confident (conf >= CONF_ALONE) or if, in at least SUPPORT of the NEAR frames either side, there is a
sighting no farther from it than STEP box heights per frame of offset (the ball moves at most about 30 m/s, a box
height is about 1.8 m) with that sighting at conf >= CONF_CHAIN. The best such sighting (most support, then highest
confidence) and its supporting sightings in the other frames become boxes. Each labelled crop becomes a training image as
build_soccernet_crops.py makes them: the 320 px crop enlarged 2x, the box as COCO class 32 ("sports ball").

The benchmark test matches (M7 128057, M9 132831) are never written as training images (ball_eval_st.py scores on
them). Usage: python ball_pseudo_select.py [--out=crops_st] [--native]   -> data/open/soccertrack/<out>/{images,labels}
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ST_ROOT", Path(__file__).resolve().parent.parent / "data" / "open" / "soccertrack"))
TEST = ("128057", "132831")
R_FEET, STEP, SUPPORT = 2.0, 0.7, 2
CONF_ALONE, CONF_CHAIN = 0.5, 0.1
CROP, SCALE = 320, 2
NATIVE = "--native" in sys.argv   # keep the 320 px crops as they are (enlarged 2x where they are trained)


def select(d):
    """Rows of d (one half's sightings) that are the ball, one per (event, frame offset) at most."""
    keep = []
    for (frame, pid), ev in d.groupby(["frame", "player_id"]):
        h = float(ev.box_h.iloc[0])
        at = ev[(ev.offset == 0) & (ev.dist <= R_FEET * h)]
        best = None
        for _, c in at.iterrows():
            sup = []
            for o in sorted(set(ev.offset) - {0}):
                nb = ev[ev.offset == o]
                if not len(nb):
                    continue
                gap = np.hypot(nb.px - c.px, nb.py - c.py)
                ok = nb[(gap <= STEP * h * abs(o)) & (nb.conf >= CONF_CHAIN)]
                if len(ok):
                    sup.append(ok.loc[np.hypot(ok.px - c.px, ok.py - c.py).idxmin()])
            if c.conf >= CONF_ALONE or (len(sup) >= SUPPORT and c.conf >= CONF_CHAIN):
                key = (len(sup), c.conf)
                if best is None or key > best[0]:
                    best = (key, c, sup)
        if best is not None:
            keep.append(best[1])
            keep.extend(best[2])
    return pd.DataFrame(keep)


def write(rows, half_dir, out, prefix):
    for _, r in rows.iterrows():
        img = cv2.imread(str(half_dir / "crops" / f"{r.crop}.jpg"))
        if img is None:
            continue
        big = img if NATIVE else cv2.resize(img, (CROP * SCALE, CROP * SCALE), interpolation=cv2.INTER_CUBIC)
        name = f"{prefix}_{r.crop}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cx, cy, s = (r.px - r.x0) / CROP, (r.py - r.y0) / CROP, max(float(r["size"]), 4.0) / CROP
        (out / "labels" / f"{name}.txt").write_text(f"32 {cx:.5f} {cy:.5f} {s:.5f} {s:.5f}\n")


def main():
    out = ROOT / next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "crops_st")
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    total = {"events": 0, "labelled events": 0, "images": 0}
    for half_dir in sorted((ROOT / "pseudo").glob("*_*")):
        if not (half_dir / "dets.csv").exists():
            continue
        match, half = half_dir.name.split("_")
        d = pd.read_csv(half_dir / "dets.csv", dtype={"player_id": str})
        n_events = len(list((half_dir / "crops").glob("*_+0.jpg")))
        rows = select(d)
        labelled = rows[rows.offset == 0] if len(rows) else rows
        print(f"{match} {half}: {n_events} events, ball found at {len(labelled)} ({len(labelled) / max(n_events, 1):.0%}), "
              f"{len(rows)} boxes with the neighbouring frames" + ("  [test match: not written]" if match in TEST else ""),
              flush=True)
        if match in TEST:
            continue
        write(rows, half_dir, out, half_dir.name)
        total["events"] += n_events
        total["labelled events"] += len(labelled)
        total["images"] += len(rows)
    print(f"training images: {total}")


if __name__ == "__main__":
    main()
