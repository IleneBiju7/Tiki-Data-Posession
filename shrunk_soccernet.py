"""The ball detector on SoccerNet test shrunk to wide-camera scales (other matches, human boxes, never trained on).

SoccerNet test: players ~107 px, the ball ~17 px. So every image is first shrunk by SHRINK, then handled as eval_chains.py does: a
320x320 crop enlarged 2x, two streams per sequence (ball: centred on the labelled ball plus a fixed random
offset; empty: a crop that does not contain the ball, as when the ball is hidden or the prior is wrong).
Sightings are put on the human person boxes (scaled) as in false_on_players.py and compared with the same
numbers at TV scale (cache/soccernet_chain_dets.npz). Cache: cache/soccernet_chain_dets_shrunk.npz (rows as in
eval_chains.py, in shrunk pixels); --weights=<name> for another detector (its own cache file); --shrink=<s> for
another scale than SHRINK (cache name ends _x<s>). Run from mtah-tools.
"""
import os
import sys

import cv2
import numpy as np

import devices
import eval_chains as ec
import eval_soccernet as es
import false_on_players as fp

SHRINK, OFFSET, MARGIN, SEED = 0.4, 100, 20, 11
SCALE_ARG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shrink=")), None)
if SCALE_ARG:   # other camera scales, so thresholds are not set at one match's scale only
    SHRINK = float(SCALE_ARG)
NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--weights=")), "ball_combined_s")
CACHE = ec.CACHE.with_name(("soccernet_chain_dets_shrunk.npz" if NAME == "ball_combined_s"
                            else f"soccernet_chain_dets_shrunk_{NAME}.npz").replace(".npz", f"_x{SCALE_ARG}.npz" if SCALE_ARG else ".npz"))
WEIGHTS = ec.HERE / "models" / "weights" / f"{NAME}.pt"


def detect():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    rng = np.random.default_rng(SEED)
    labelled, rows, C = 0, [], es.CROP
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
            bx, by, size = (v * SHRINK for v in boxes[f])
            labelled += 1
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, None, fx=SHRINK, fy=SHRINK, interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            inside = lambda s: (s[0] - MARGIN <= bx <= s[0] + C + MARGIN) and (s[1] - MARGIN <= by <= s[1] + C + MARGIN)  # noqa: E731
            if spot is None or inside(spot):
                spot = None
                for _ in range(200):
                    s = (int(rng.integers(0, w - C + 1)), int(rng.integers(0, h - C + 1)))
                    if not inside(s):
                        spot = s
                        break
            streams = [(0, (bx + off[0] - C / 2, by + off[1] - C / 2))] + ([(1, spot)] if spot else [])
            for stream, (x0, y0) in streams:
                x0, y0 = int(np.clip(x0, 0, w - C)), int(np.clip(y0, 0, h - C))
                crop = cv2.resize(img[y0:y0 + C, x0:x0 + C], (C * es.SCALE,) * 2, interpolation=cv2.INTER_CUBIC)
                r = model.predict(crop, imgsz=C * es.SCALE, conf=0.05, classes=[32], device=devices.yolo_device(), verbose=False)[0]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    rows.append((k, stream, f, c, x0 + (x1 + x2) / 4, y0 + (y1 + y2) / 4, bx, by, size))
        print(f"{seq.name}: {labelled} frames, {len(rows)} detections so far", flush=True)
    np.savez(CACHE, rows=np.array(rows, dtype=np.float64), labelled=labelled)


def report(name, d, n_frames, scale):
    ec.MAX_STEP = 30.0 * scale
    seqs = sorted(es.ROOT.iterdir())
    right = np.hypot(d[:, 4] - d[:, 6], d[:, 5] - d[:, 7]) <= np.maximum(8, d[:, 8])
    zone, on_keeper = np.full(len(d), 4), np.zeros(len(d), bool)
    for k in np.unique(d[:, 0]).astype(int):
        boxes = {f: [(x * scale, y * scale, bw * scale, bh * scale, kind) for x, y, bw, bh, kind in b]
                 for f, b in fp.person_boxes(seqs[k]).items()}
        for i in np.flatnonzero(d[:, 0] == k):
            zone[i], kind = fp.zone_of(d[i, 4], d[i, 5], boxes.get(int(d[i, 2]), []))
            on_keeper[i] = "goalkeeper" in kind
    tr = fp.trusted(d)
    ball, empty = d[:, 1] == 0, d[:, 1] == 1
    n_empty = len({(a, b) for a, b in d[empty][:, [0, 2]]})
    recall = len({(a, b) for a, b in d[tr & ball & right][:, [0, 2]]}) / n_frames
    print(f"\n{name}: {n_frames} labelled frames, {len(d)} sightings, trusted by our merge {tr.sum()} "
          f"({right[tr].mean():.1%} right); ball found in {recall:.1%} of frames (ball crop)")
    print(f"  ball crop: trusted {np.sum(tr & ball)}, {right[tr & ball].mean():.1%} right")
    print(f"  empty crop (no ball in it): trusted false sightings {np.sum(tr & empty)} "
          f"(on {n_frames} crops: {1000 * np.sum(tr & empty) / n_frames:.1f} per 1000)")
    for label, sel in (("trusted WRONG", tr & ~right), ("  of them in the empty crop", tr & ~right & empty),
                       ("trusted right", tr & right)):
        n = sel.sum()
        parts = ", ".join(f"{z_} {100 * np.mean(zone[sel] == j):.0f}%" for j, z_ in enumerate(fp.ZONES)) if n else ""
        print(f"  {label:28s} {n:6d}: {parts}")
    for cut, label in ((1, "head"), (2, "head or body")):
        drop = tr & (zone < cut)
        kept = tr & ~drop
        print(f"  drop trusted sightings on a player's {label}: removes {np.sum(drop & ~right)} of "
              f"{np.sum(tr & ~right)} wrong, {np.sum(drop & right)} of {np.sum(tr & right)} right -> "
              f"{right[kept].mean():.1%} right")
    print(f"  trusted wrong sightings on a goalkeeper: {np.sum(tr & ~right & on_keeper)}")


def main():
    ec.fb_conf, ec.fb_len = 0.3, 11
    if "--redetect" in sys.argv or not CACHE.exists():
        detect()
    if NAME != "ball_combined_s":
        return
    z0, z1 = np.load(ec.CACHE), np.load(CACHE)
    report("TV scale (as before)", z0["rows"], int(z0["labelled"]), 1.0)
    report(f"shrunk x{SHRINK}", z1["rows"], int(z1["labelled"]), SHRINK)


if __name__ == "__main__":
    main()
