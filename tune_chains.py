"""Set fuse_ball.py's sighting rules for a ball detector on SoccerNet test only (other matches, human labels), at TV
scale and shrunk to several wide-camera scales (organisers, 3 Nov: tune on external data).
Rule fixed in advance, as in fuse_ball.py: among confident chains (conf >= a, >= 3 sightings) plus weak chains
(conf >= w over >= n frames), keep the setting that finds the most balls (mean over the scales) while the trusted
sightings are right >= 95% of the time and there are at most 10 false sightings per 1000 crops without a ball (a hidden
ball must not come back as a phantom) at every scale. Caches from tv_eval.py and shrunk_soccernet.py --weights=<name>
[--shrink=<s>]. --wide: a wider grid, for a detector where nothing in the usual one passes (ball_st_k3).
Usage: python tune_chains.py <weights name> [--wide]   (run from mtah-tools)
"""
import itertools
import sys

import numpy as np

import eval_chains as ec
from shrunk_rules import kept

MIN_RIGHT, MAX_FALSE = 0.95, 10.0
GRID = ((0.3, 0.4, 0.5, 0.6), (0.15, 0.2, 0.25, 0.3, 0.35), (11, 17, 25))
WIDE = ((0.3, 0.4, 0.5, 0.6), (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), (11, 17, 25, 37))


def caches(name):
    c = ec.CACHE
    out = [("TV", 1.0, c.with_name(f"soccernet_chain_dets_{name}.npz"))]
    for s in ("0.3", "0.4", "0.6"):
        p = c.with_name(f"soccernet_chain_dets_shrunk_{name}.npz" if s == "0.4" else f"soccernet_chain_dets_shrunk_{name}_x{s}.npz")
        if p.exists():
            out.append((f"x{s}", float(s), p))
    return out


def main():
    name = sys.argv[1]
    data = []
    for label, scale, path in caches(name):
        z = np.load(path)
        d, n = z["rows"], int(z["labelled"])
        right = np.hypot(d[:, 4] - d[:, 6], d[:, 5] - d[:, 7]) <= np.maximum(8, d[:, 8])
        data.append((label, scale, d, n, right, d[:, 1] == 0, d[:, 1] == 1))
    print("scales:", [x[0] for x in data])
    rows = []
    for a, w, m in itertools.product(*(WIDE if "--wide" in sys.argv else GRID)):
        res = []
        for label, scale, d, n, right, ball, empty in data:
            ec.MAX_STEP = 30.0 * scale
            s = kept(d, a, 3, True) | kept(d, w, m, False)
            found = len({(x, y) for x, y in d[s & ball & right][:, [0, 2]]}) / n
            res.append((right[s & ball].mean(), found, 1000 * np.sum(s & empty) / n))
        res = np.array(res)
        rows.append((a, w, m, res))
    ok = [r for r in rows if r[3][:, 0].min() >= MIN_RIGHT and r[3][:, 2].max() <= MAX_FALSE]
    ok.sort(key=lambda r: -r[3][:, 1].mean())
    print(f"settings with trusted sightings >= {MIN_RIGHT:.0%} right and <= {MAX_FALSE:.0f} false per 1000 at every scale: "
          f"{len(ok)} of {len(rows)}")
    for a, w, m, res in ok[:8]:
        print(f"  confident >= {a}, weak >= {w} over {m:2d} frames: found " +
              "  ".join(f"{lab} {f:5.1%}" for (lab, *_), f in zip(data, res[:, 1])) +
              f"  | right min {res[:, 0].min():.1%}, false per 1000 max {res[:, 2].max():.1f}")
    if ok:
        a, w, m, _ = ok[0]
        print(f"chosen: ANCHOR_CONF {a}, CHAIN_CONF {w}, CHAIN_SECONDS {m / 25:.2f} ({m} frames)")


if __name__ == "__main__":
    main()
