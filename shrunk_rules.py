"""Which chain rule lets false sightings through at a wide-camera scale? SoccerNet test shrunk x0.4 (shrunk_soccernet.py
cache) next to TV scale: per rule, trusted sightings in the ball crop (share right), ball found (share of labelled
frames), and false sightings in the empty crop (no ball in it) per 1000 crops. Rules as fuse_ball.py: confident
chains (conf >= c, >= n sightings) and weak chains (conf >= c over >= n frames, half the images seen).
Run from mtah-tools.
"""
import numpy as np

import eval_chains as ec


def kept(d, min_conf, min_len, by_count):
    cid = ec.chains(d, min_conf)
    ok = cid >= 0
    sel = np.zeros(len(d), bool)
    if not ok.any():
        return sel
    ids, inv = np.unique(cid[ok], return_inverse=True)
    fr = d[ok, 2]
    hi = np.zeros(len(ids)); np.maximum.at(hi, inv, fr)
    lo = np.full(len(ids), np.inf); np.minimum.at(lo, inv, fr)
    span, cnt = hi - lo + 1, np.bincount(inv)
    keep = (cnt >= min_len) if by_count else (span >= min_len) & (cnt >= 0.5 * span / ec.EVERY)
    sel[np.flatnonzero(ok)] = keep[inv]
    return sel


def table(name, path, scale):
    ec.MAX_STEP = 30.0 * scale
    z = np.load(path)
    d, n = z["rows"], int(z["labelled"])
    right = np.hypot(d[:, 4] - d[:, 6], d[:, 5] - d[:, 7]) <= np.maximum(8, d[:, 8])
    ball, empty = d[:, 1] == 0, d[:, 1] == 1
    print(f"\n{name}: {n} labelled frames")
    print("  rule                          ball crop: trusted  right   ball found | empty crop: false per 1000")
    rules = [("confident", 0.5, 3, True), ("confident", 0.5, 4, True), ("confident", 0.6, 3, True),
             ("weak", 0.3, 11, False), ("weak", 0.3, 13, False), ("weak", 0.3, 25, False),
             ("weak", 0.4, 11, False), ("weak", 0.4, 13, False), ("weak", 0.2, 11, False)]
    sels = {}
    for kind, c, m, by_count in rules:
        s = kept(d, c, m, by_count)
        sels[(kind, c, m)] = s
        found = len({(a, b) for a, b in d[s & ball & right][:, [0, 2]]}) / n
        unit = "sightings" if by_count else "frames"
        print(f"  {kind:9s} conf>={c} >= {m:2d} {unit:9s}  {np.sum(s & ball):6d}  {right[s & ball].mean():6.1%}  "
              f"{found:6.1%}    | {1000 * np.sum(s & empty) / n:5.1f}")
    for a, b in ((("confident", 0.5, 3), ("weak", 0.3, 11)), (("confident", 0.5, 3), ("weak", 0.3, 25)),
                 (("confident", 0.5, 3), ("weak", 0.4, 13)), (("confident", 0.5, 4), ("weak", 0.4, 13))):
        s = sels[a] | sels[b]
        found = len({(x, y) for x, y in d[s & ball & right][:, [0, 2]]}) / n
        print(f"  both: {a[1]}/{a[2]} + weak {b[1]}/{b[2]:2d}          {np.sum(s & ball):6d}  "
              f"{right[s & ball].mean():6.1%}  {found:6.1%}    | {1000 * np.sum(s & empty) / n:5.1f}")


def main():
    table("TV scale", ec.CACHE, 1.0)
    table("shrunk x0.4", ec.CACHE.with_name("soccernet_chain_dets_shrunk.npz"), 0.4)


if __name__ == "__main__":
    main()
