"""Search richer decodings on saved held-out probabilities (no retraining, no competition data):
separate costs for live->dead and dead->live, and a log-prior bias on D.

Usage: python decode_search.py tagA[+tagB...] [--select=hard|normal]
  tagA+tagB averages the probabilities of several CV runs (ensemble). Scores are reported on the normal
  test (cv_probs_<tag>.npz) and the realistic test (cv_probs_<tag>_hard.npz); --select picks which one chooses.
"""
import itertools
import sys

import numpy as np
from numba import njit

import train

FPS = 25


@njit(cache=True)
def _viterbi(logp, cost):
    n = logp.shape[0]
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    new = np.empty(3)
    for i in range(1, n):
        for t in range(3):
            bs, bk = -1e300, 0
            for f in range(3):
                v = score[f] - cost[f, t]
                if v > bs:
                    bs, bk = v, f
            new[t] = bs + logp[i, t]
            back[i, t] = bk
        score[:] = new
    path = np.zeros(n, np.int8)
    path[-1] = np.argmax(score)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def viterbi(prob, cost, bias):
    return _viterbi(np.log(np.clip(prob, 1e-6, 1.0)).astype(np.float64) + bias, cost)


def decode(prob, period, team, to_dead, from_dead, bias_d, min_flip):
    cost = np.array([[0, team, to_dead], [team, 0, to_dead], [from_dead, from_dead, 0]], np.float64)
    bias = np.array([0, 0, bias_d])
    out = np.full(len(prob), train.X_CODE, np.int8)
    for p in (1, 2):
        sel = np.flatnonzero(period == p)
        if len(sel):
            out[sel] = train.remove_flips(viterbi(prob[sel], cost, bias), min_flip)
    return out


def load(tags, suffix):
    """Held-out probabilities of the first-stage models, averaged; "seqE@4" weighs one 4 times (predict_stack.py
    --seq-weight)."""
    parts = [t.split("@") for t in tags]
    ds = [np.load(train.MODELS / f"cv_probs_{p[0]}{suffix}.npz") for p in parts]
    w = [float(p[1]) if len(p) > 1 else 1.0 for p in parts]
    names = list(ds[0]["names"])
    return {n: (np.average([d[f"prob_{n}"] for d in ds], 0, weights=w), train.gt_codes(ds[0][f"lab_{n}"], ds[0][f"period_{n}"]),
                ds[0][f"period_{n}"]) for n in names}


def evaluate(data, params):
    return np.mean([train.score(gt, decode(p, per, *params))["spellF1"] for p, gt, per in data.values()])


def main():
    tags = sys.argv[1].split("," if "," in sys.argv[1] else "+")   # "," separates names that contain "+"
    select = next((a.split("=")[1] for a in sys.argv if a.startswith("--select=")), "hard")
    hard = load(tags, "_hard")
    normal = load(tags, "") if all((train.MODELS / f"cv_probs_{t}.npz").exists() for t in tags) else hard
    pick = hard if select == "hard" else normal
    # coarse search on a few matches' worth is too noisy; use all, but a staged grid
    best = None
    for team, td, fd, b, mf in itertools.product((5, 10, 20), (10, 20, 40, 80), (10, 20, 40, 80), (-0.3, 0.0, 0.3), (0, 25)):
        s = evaluate(pick, (team, td, fd, b, mf))
        if best is None or s > best[0]:
            best = (s, (team, td, fd, b, mf))
            print(f"  {select} {s:.4f}  team {team} to_dead {td} from_dead {fd} bias_D {b} min_flip {mf}", flush=True)
    s, params = best
    other = normal if select == "hard" else hard
    print(f"BEST {'+'.join(tags)}: {select} {s:.4f}, other test {evaluate(other, params):.4f}, params {params}")


if __name__ == "__main__":
    main()
