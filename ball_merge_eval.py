"""Do two ball detectors together find more balls than one? SoccerNet test only (other matches, human labels), from the
sightings tv_eval.py and shrunk_soccernet.py saved, at TV scale and shrunk to the wide-camera scales. Nothing is
detected again and nothing comes from the competition match.

The first detector keeps the rule its own check set (tune_chains.py). Merges tried:
  union      also trust the second detector's chains (confident a over >= 3 sightings, weak w over m frames)
  agree      also trust a sighting both detectors make on one frame, within AGREE px, each with conf >= c, in a chain of
             >= m frames (m = 1: no chain needed)
  union+agree  both, on the best union rule
Per frame one sighting is used, the one nearest the last used, as fuse_simple.py does; "right" and "found" count those.
Rule from tune_chains.py: right >= 95% and <= 10 false per 1000 crops at every scale, most balls found on the mean over
the scales. --joint: search both detectors' rules together, since the first detector's rule alone uses nearly all of the
false-sighting limit. Usage: python ball_merge_eval.py ball_soccernet_k ball_st_k3 [--joint]   (run from mtah-tools)
"""
import functools
import itertools
import sys

import numpy as np

import eval_chains as ec
from shrunk_rules import kept
from tune_chains import MAX_FALSE, MIN_RIGHT, caches

FIRST = (0.6, 0.25, 25)   # ball_soccernet_k's rule, tune_chains.py
AGREE = 8.0               # px, the smallest distance eval_chains.py calls right
GRID_UNION = list(itertools.product((0.5, 0.6, 0.7, 0.8), (0.3, 0.35, 0.4, 0.45, 0.5, 0.6), (17, 25, 37, 50)))
GRID_AGREE = list(itertools.product((0.05, 0.1, 0.15, 0.2, 0.25, 0.3), (1, 11, 17, 25)))
GRID_FIRST = list(itertools.product((0.6, 0.7, 0.8), (0.25, 0.3, 0.35, 0.4), (25, 37)))
GRID_SECOND = list(itertools.product((0.6, 0.7, 0.8), (0.45, 0.5, 0.6), (37, 50)))


@functools.lru_cache(maxsize=None)
def rule_mask(key, a, w, m):
    d = DATA[key]
    return kept(d, a, 3, True) | kept(d, w, m, False)


@functools.lru_cache(maxsize=None)
def weak_mask(key, conf, m):
    return kept(DATA[key], conf, m, False)


def agreed(d1, d2, c):
    """Rows of d1 with a sighting of d2 on the same crop and frame, within AGREE px, both conf >= c."""
    by = {}
    for x in d2[d2[:, 3] >= c]:
        by.setdefault((x[0], x[1], x[2]), []).append(x[4:6])
    keep = [i for i in np.flatnonzero(d1[:, 3] >= c)
            if (pts := by.get((d1[i, 0], d1[i, 1], d1[i, 2]))) is not None
            and np.hypot(*(np.array(pts) - d1[i, 4:6]).T).min() <= AGREE]
    return d1[keep]


def score(rows, n):
    """Share right and share found among the sightings used per frame (nearest the last used), false per 1000 crops."""
    ball = rows[rows[:, 1] == 0]
    used = right = 0
    for seq in np.unique(ball[:, 0]):
        s = ball[ball[:, 0] == seq]
        s = s[np.argsort(s[:, 2], kind="stable")]
        prev = None
        for f in np.unique(s[:, 2]):
            c = s[s[:, 2] == f]
            pick = c[np.argmax(c[:, 3])] if prev is None else c[np.argmin(np.hypot(*(c[:, 4:6] - prev).T))]
            prev = pick[4:6]
            used += 1
            right += np.hypot(*(pick[4:6] - pick[6:8])) <= max(8, pick[8])
    empty = rows[rows[:, 1] == 1]
    false = len({(a, b) for a, b in empty[:, [0, 2]]})
    return right / max(used, 1), right / n, 1000 * false / n


def report(label, res, scales):
    res = np.array(res)
    print(f"{label:34s} found " + "  ".join(f"{s} {f:5.1%}" for s, f in zip(scales, res[:, 1])) +
          f"  mean {res[:, 1].mean():5.1%} | right min {res[:, 0].min():.1%}, false per 1000 max {res[:, 2].max():.1f}", flush=True)
    return res


def passes(res):
    return res[:, 0].min() >= MIN_RIGHT and res[:, 2].max() <= MAX_FALSE


DATA = {}


def main():
    n1, n2 = sys.argv[1:3]
    c1, c2 = caches(n1), caches(n2)
    scales = [x[0] for x in c1]
    if scales != [x[0] for x in c2]:
        sys.exit(f"scales differ: {scales} {[x[0] for x in c2]}")
    z = [(s, np.load(p1), np.load(p2), float(sc)) for (s, sc, p1), (_, _, p2) in zip(c1, c2)]
    for s, a, b, _ in z:
        if int(a["labelled"]) != int(b["labelled"]):
            sys.exit(f"{s}: not the same labelled frames")
        DATA[(s, 1)], DATA[(s, 2)] = a["rows"], b["rows"]
    N = {s: int(a["labelled"]) for s, a, _, _ in z}
    step = {s: 30.0 * sc for s, _, _, sc in z}

    def merged(union, ag, first=FIRST):
        res = []
        for s in scales:
            ec.MAX_STEP = step[s]
            parts = [DATA[(s, 1)][rule_mask((s, 1), *first)]]
            if union:
                parts.append(DATA[(s, 2)][rule_mask((s, 2), *union)])
            if ag:
                r = agreed(DATA[(s, 1)], DATA[(s, 2)], ag[0])
                if ag[1] > 1 and len(r):
                    r = r[kept(r, 0.0, ag[1], False)]
                parts.append(r)
            res.append(score(np.vstack(parts), N[s]))
        return res

    print(f"labelled frames per scale: {N}")
    base = report(f"{n1} alone {FIRST}", merged(None, None), scales)
    if "--joint" in sys.argv:
        out = [(np.array(merged(u, None, f)), f, u) for f, u in itertools.product(GRID_FIRST, GRID_SECOND)]
        good = sorted((x for x in out if passes(x[0])), key=lambda x: -x[0][:, 1].mean())
        print(f"\njoint: {len(good)} of {len(out)} settings pass")
        for r, f, u in good[:5]:
            report(f"first {f} second {u}", r, scales)
        r, f, u = max(out, key=lambda x: x[0][:, 1].mean())
        report(f"most found, limits ignored: {f} {u}", r, scales)
        if good:
            gain = good[0][0][:, 1].mean() - base[:, 1].mean()
            print(f"\nchosen: first {good[0][1]}, second {good[0][2]}, gain {100 * gain:+.1f} points; "
                  f"{'PASSES' if gain >= 0.02 else 'misses'} the +2 points set before this test")
        else:
            print("\nno merge passes the rule")
        return
    out = []
    for u in GRID_UNION:
        out.append((np.array(merged(u, None)), u, None))
    good = sorted((x for x in out if passes(x[0])), key=lambda x: -x[0][:, 1].mean())
    print(f"\nunion: {len(good)} of {len(out)} settings pass")
    for r, u, _ in good[:5]:
        report(f"union {u}", r, scales)
    best_u = good[0][1] if good else None

    out2 = []
    for g in GRID_AGREE:
        out2.append((np.array(merged(None, g)), None, g))
    good2 = sorted((x for x in out2 if passes(x[0])), key=lambda x: -x[0][:, 1].mean())
    print(f"\nagree: {len(good2)} of {len(out2)} settings pass")
    for r, _, g in good2[:5]:
        report(f"agree c,m {g}", r, scales)

    out3 = []
    if best_u:
        for g in GRID_AGREE:
            out3.append((np.array(merged(best_u, g)), best_u, g))
    good3 = sorted((x for x in out3 if passes(x[0])), key=lambda x: -x[0][:, 1].mean())
    print(f"\nunion {best_u} + agree: {len(good3)} of {len(out3)} settings pass")
    for r, u, g in good3[:5]:
        report(f"union+agree c,m {g}", r, scales)

    all_good = good + good2 + good3
    if not all_good:
        print("\nno merge passes the rule")
        return
    r, u, g = max(all_good, key=lambda x: x[0][:, 1].mean())
    gain = r[:, 1].mean() - base[:, 1].mean()
    print(f"\nchosen: union {u}, agree {g}: found mean {r[:, 1].mean():.1%} vs {base[:, 1].mean():.1%}, gain {100 * gain:+.1f} points; "
          f"{'PASSES' if gain >= 0.02 else 'misses'} the +2 points set before this test")


if __name__ == "__main__":
    main()
