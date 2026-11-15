"""Decoding that knows how long spells last (an explicit-duration Viterbi, as in hidden semi-Markov models).

decode_search.py charges a fixed cost per change of state. Here each state also carries how long its spell has lasted
(in BLOCK-frame blocks, up to K blocks), and ending a spell of age a costs -log h(a), staying costs -log(1 - h(a)),
where h is the hazard of a possession spell (H or A) or a dead-ball spell (D) ending at that age. h comes from the
spell lengths in the labels of the training matches only (open data; the held-out match is never used for its own
decoding). W scales these duration terms against the frame probabilities, which are not independent frame to frame.
The fixed switch costs of decode_search.py stay on top. Blocks: log-probabilities summed over BLOCK frames; each
block's label is copied back to its frames.

Usage (from another script): hazards(label arrays) -> h; decode(prob, period, h, team, to_dead, from_dead, w)
"""
import numpy as np
from numba import njit

import train

BLOCK = 5          # frames per block (0.2 s at 25 fps)
K = 150            # ages tracked, in blocks (30 s); older spells share the last age


def spell_lengths(label, period):
    """Lengths in blocks of the possession (H/A) and dead-ball (D) spells of one match's labels."""
    out = {0: [], 2: []}
    for p in (1, 2):
        lab = label[period == p]
        lab = lab[lab < 3]
        if not len(lab):
            continue
        cut = np.flatnonzero(np.diff(lab) != 0) + 1
        for seg in np.split(lab, cut):
            out[0 if seg[0] < 2 else 2].append(max(1, int(round(len(seg) / BLOCK))))
    return out


def hazards(labels_periods, prior=2.0):
    """Per class (0: H or A, 2: D) the probability that a spell of age a (1..K blocks) ends at that age, from the
    given (label, period) pairs; counts smoothed with `prior` pseudo-spells spread like a geometric tail."""
    h = np.zeros((3, K + 1))
    for c in (0, 2):
        lens = np.concatenate([np.array(spell_lengths(l, p)[c]) for l, p in labels_periods])
        counts = np.bincount(np.minimum(lens, K), minlength=K + 1)[1:].astype(float)
        at_risk = counts[::-1].cumsum()[::-1]
        tail = lens[lens >= K]
        mean_tail = (tail - K + 1).mean() if len(tail) else 50.0
        base = 1.0 / max(np.mean(lens), 1.0)
        hz = (counts + prior * base) / (at_risk + prior)
        hz[-1] = 1.0 / max(mean_tail, 1.0)
        h[c, 1:] = np.clip(hz, 1e-4, 1 - 1e-4)
    h[1] = h[0]
    return h


@njit(cache=True)
def _viterbi(e, h, sw, w):
    n, nk = e.shape[0], h.shape[1] - 1
    log_end = w * np.log(h[:, 1:])
    log_stay = w * np.log(1.0 - h[:, 1:])
    score = np.full((3, nk), -1e300)
    for l in range(3):
        score[l, 0] = e[0, l]
    back_new = np.zeros((n, 3, 2), np.int16)   # for age 1 at step i: (previous label, previous age index)
    back_old = np.zeros((n, 3), np.int8)        # for the last age at step i: 1 if it came from the last age itself
    new = np.empty((3, nk))
    for i in range(1, n):
        for l in range(3):
            bs, bl, ba = -1e300, 0, 0
            for f in range(3):
                if f == l:
                    continue
                for a in range(nk):
                    v = score[f, a] + log_end[f, a] - sw[f, l]
                    if v > bs:
                        bs, bl, ba = v, f, a
            new[l, 0] = bs + e[i, l]
            back_new[i, l, 0], back_new[i, l, 1] = bl, ba
            for a in range(1, nk - 1):
                new[l, a] = score[l, a - 1] + log_stay[l, a - 1] + e[i, l]
            v1 = score[l, nk - 2] + log_stay[l, nk - 2]
            v2 = score[l, nk - 1] + log_stay[l, nk - 1]
            if v2 > v1:
                new[l, nk - 1], back_old[i, l] = v2 + e[i, l], 1
            else:
                new[l, nk - 1], back_old[i, l] = v1 + e[i, l], 0
        score[:, :] = new
    path = np.zeros(n, np.int8)
    l, a = 0, 0
    bs = -1e300
    for ll in range(3):
        for aa in range(nk):
            if score[ll, aa] > bs:
                bs, l, a = score[ll, aa], ll, aa
    for i in range(n - 1, -1, -1):
        path[i] = l
        if i == 0:
            break
        if a == 0:
            l, a = back_new[i, l, 0], back_new[i, l, 1]
        elif a == nk - 1:
            a = nk - 1 if back_old[i, l] == 1 else nk - 2
        else:
            a = a - 1
    return path


def decode(prob, period, h, team, to_dead, from_dead, w):
    sw = np.array([[0, team, to_dead], [team, 0, to_dead], [from_dead, from_dead, 0]], np.float64)
    out = np.full(len(prob), train.X_CODE, np.int8)
    lp = np.log(np.clip(prob, 1e-6, 1.0))
    for p in (1, 2):
        sel = np.flatnonzero(period == p)
        if not len(sel):
            continue
        nb = int(np.ceil(len(sel) / BLOCK))
        pad = np.zeros((nb * BLOCK, 3))
        pad[:len(sel)] = lp[sel]
        e = pad.reshape(nb, BLOCK, 3).sum(1)
        path = _viterbi(e, h, sw, w)
        out[sel] = np.repeat(path, BLOCK)[:len(sel)]
    return out
