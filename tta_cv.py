"""Team-swap test-time averaging in the CV files the second stage learns from: each first-stage member's held-out
probabilities averaged with those of the same test copy with the two teams' identities swapped (H and A swapped
back), as predict_stack.py --swap-tta does on the competition match.

Usage: python tta_cv.py lgbm <tag>...                         -> models/cv_probs_<tag>T_hard.npz, from
                                                                 cv_probs_<tag>_hard.npz and cv_probs_<tag>_hardsw.npz
                                                                 (train.py --swap-tta, already swapped back)
       python tta_cv.py seq <tag> <reference tag> <run dir>...  -> models/cv_probs_<tag>_hard.npz, the runs' cv_seq.npz
                                                                 and cv_seq_xhs.npz (seq_train.py --test=xh,xhs)
                                                                 averaged, every frame, as seq_cv.py does without them
Prints each new file's decoded realistic spellF1 (best of seq_cv.py's grid).
"""
import sys
from pathlib import Path

import numpy as np

import decode_search
import seq_export
import train
from seq_cv import best


def save(tag, ref, probs):
    names = ref["names"]
    np.savez_compressed(train.MODELS / f"cv_probs_{tag}_hard.npz", names=names,
                        **{f"prob_{n}": probs[n].astype(np.float32) for n in names},
                        **{f"lab_{n}": ref[f"lab_{n}"] for n in names},
                        **{f"period_{n}": ref[f"period_{n}"] for n in names})
    s, p = best(decode_search.load([tag], "_hard"))
    print(f"{tag:12s} decoded realistic spellF1 {s:.4f}  {p}", flush=True)


def lgbm(tags):
    for t in tags:
        a = np.load(train.MODELS / f"cv_probs_{t}_hard.npz")
        b = np.load(train.MODELS / f"cv_probs_{t}_hardsw.npz")
        save(f"{t}T", a, {n: 0.5 * (a[f"prob_{n}"] + b[f"prob_{n}"]) for n in a["names"]})


def seq(tag, ref_tag, dirs):
    ref = np.load(train.MODELS / f"cv_probs_{ref_tag}_hard.npz")
    runs = [(np.load(Path(d) / "cv_seq.npz"), np.load(Path(d) / "cv_seq_xhs.npz")) for d in dirs]
    out = {}
    for n in ref["names"]:
        p = np.mean([0.5 * (a[f"prob_{n}"] + b[f"prob_{n}"][:, [1, 0, 2]]) for a, b in runs], 0)
        at, m = np.arange(len(p)) * seq_export.STEP, len(ref[f"lab_{n}"])
        out[n] = np.stack([np.interp(np.arange(m), at, p[:, c]) for c in range(3)], 1)
    save(tag, ref, out)


if __name__ == "__main__":
    if sys.argv[1] == "lgbm":
        lgbm(sys.argv[2:])
    else:
        seq(sys.argv[2], sys.argv[3], sys.argv[4:])
