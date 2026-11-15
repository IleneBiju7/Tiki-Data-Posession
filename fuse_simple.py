"""Our ball on the competition match, with every threshold set on external data (organisers, 3 Nov).

The plain rule, per graded frame:
  1. a trusted sighting of our detectors: in a confident chain (conf >= --anchor-conf, >= 3 sightings) or a weak chain
     (conf >= --chain-conf over --chain-seconds), both rules from fuse_ball.py with the values tune_chains.py and
     ball_merge_eval.py set on SoccerNet test (other matches, human labels); of several, the one nearest the previous ball
  2. otherwise the provided tracker's ball
  3. otherwise no ball (ball_sticky.py later keeps a hidden ball with the player who had it)
The only other number is physical: chains move at most fuse_ball.MAX_SPEED (35 m/s).

Usage: python fuse_simple.py <detections tag> <output tag> [--anchor-conf=0.5] [--chain-conf=0.3] [--chain-seconds=0.44]
       reads cache/ball_detections_<tag>.csv.gz (every frame), writes cache/BAR-ATH_ball_fused_<output tag>.npy
       --also=<tag>,<anchor conf>,<chain conf>,<chain seconds>: also trust a second detector's chains under its own rule
       (ball_merge_eval.py sets both rules together on SoccerNet test)
"""
import sys

import numpy as np
import pandas as pd

import atd_cache
import fuse_ball as fb


def arg(key, default):
    return float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), default))


def trusted_sightings(tag, anchor_conf, chain_conf, chain_s):
    fb.ANCHOR_CONF = anchor_conf
    det = pd.read_csv(atd_cache.DATA_DIR / "cache" / f"ball_detections_{tag}.csv.gz")
    by_frame = {int(f): g for f, g in det.groupby("frame")}
    return fb.anchored_set(by_frame, 1, 1 / 25) | fb.weak_chain_set(by_frame, chain_conf, int(round(chain_s * 25)))[0]


def main():
    tag, out_tag = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
    anchor_conf = arg("anchor-conf", fb.ANCHOR_CONF)
    chain_conf, chain_s = arg("chain-conf", fb.CHAIN_CONF), arg("chain-seconds", fb.CHAIN_SECONDS)
    trusted = trusted_sightings(tag, anchor_conf, chain_conf, chain_s)
    also = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--also=")), None)
    if also:
        trusted |= trusted_sightings(also[0], *map(float, also[1:]))
    pts = {}
    for f, x, y in trusted:
        pts.setdefault(f, []).append((x, y))
    trk = atd_cache.load()["ball"].astype(np.float64) * fb.PITCH
    n = len(trk)
    ours, source = np.full((n, 2), np.nan), np.zeros(n, np.int8)   # source: 1 detector, 2 tracker
    for a, b in fb.HALVES:
        prev = None
        for f in range(a, b + 1):
            if f in pts:
                cand = np.array(pts[f])
                ref = prev if prev is not None else (trk[f] if np.isfinite(trk[f, 0]) else None)
                ours[f] = cand[np.argmin(np.hypot(*(cand - ref).T))] if ref is not None else cand[0]
                source[f] = 1
            elif np.isfinite(trk[f, 0]):
                ours[f], source[f] = trk[f], 2
            if np.isfinite(ours[f, 0]):
                prev = ours[f]
    out = atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_fused_{out_tag}.npy"
    np.save(out, (ours / fb.PITCH).astype(np.float32))
    graded = np.zeros(n, bool)
    for a, b in fb.HALVES:
        graded[a:b + 1] = True
    print(f"rules: confident >= {anchor_conf}, weak >= {chain_conf} over {chain_s:.2f} s{f', also {also}' if also else ''}; {len(trusted)} trusted sightings")
    print(f"ball on {np.isfinite(ours[graded, 0]).mean():.1%} of graded frames: detector {np.mean(source[graded] == 1):.1%}, "
          f"tracker {np.mean(source[graded] == 2):.1%} -> {out.name}")


if __name__ == "__main__":
    main()
