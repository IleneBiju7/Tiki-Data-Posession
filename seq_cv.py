"""Held-out probabilities of the sequence model (seq_train.py cv_seq.npz, 5 per second) as a first-stage CV file like
train.py's (models/cv_probs_<tag>_hard.npz, every frame, linear between steps), so stack.py and decode_search.py can
use it; several runs are averaged (one ensemble member). Prints its decoded score alone and averaged with a
reference model on the same test copies.

Usage: python seq_cv.py <tag> <reference tag> <cv_seq.npz>...   e.g.  seqE v43 models/seq_kaggle_*/cv_seq.npz
"""
import sys

import numpy as np

import decode_search
import seq_export
import train

GRID = [(t, td, fd, 0.0, 0) for t in (2, 5, 10, 20) for td in (10, 20, 40, 80) for fd in (10, 20, 40, 80)]


def best(data):
    return max((decode_search.evaluate(data, p), p) for p in GRID)


def main():
    tag, ref, seqs = sys.argv[1], sys.argv[2], [np.load(f) for f in sys.argv[3:]]
    r = np.load(train.MODELS / f"cv_probs_{ref}_hard.npz")
    out = {}
    for n in r["names"]:
        p, n_frames = np.mean([s[f"prob_{n}"] for s in seqs], 0), len(r[f"lab_{n}"])
        at = np.arange(len(p)) * seq_export.STEP
        out[n] = np.stack([np.interp(np.arange(n_frames), at, p[:, c]) for c in range(3)], 1).astype(np.float32)
    np.savez_compressed(train.MODELS / f"cv_probs_{tag}_hard.npz", names=r["names"],
                        **{f"prob_{n}": out[n] for n in r["names"]},
                        **{f"lab_{n}": r[f"lab_{n}"] for n in r["names"]},
                        **{f"period_{n}": r[f"period_{n}"] for n in r["names"]})
    for tags in ([tag], [ref], [ref, tag]):
        s, p = best(decode_search.load(tags, "_hard"))
        print(f"{'+'.join(tags):12s} decoded realistic spellF1 {s:.4f}  {p}", flush=True)


if __name__ == "__main__":
    main()
