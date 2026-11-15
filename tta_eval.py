"""A/B of two ideas on the open matches only: team-swap test-time averaging (TTA) and SkillCorner as extra training data.
Decoded realistic spellF1 (best of seq_cv.py's grid) of:
  seq <run dir>...   each sequence run's cv_seq.npz alone, then averaged with its team-swapped copy (cv_seq_xhs.npz,
                     H and A swapped back); the runs given together are also averaged as one group
  lgbm <tag>...      models/cv_probs_<tag>_hard.npz alone, then averaged with cv_probs_<tag>_hardsw.npz (already
                     swapped back by train.py)
Usage: python tta_eval.py seq models/seq_kaggle_wm7 ...  |  python tta_eval.py lgbm <tag>..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import seq_export  # noqa: E402
import train  # noqa: E402
from seq_cv import best  # noqa: E402

REF = np.load(train.MODELS / "cv_probs_v43x_hard.npz")
NAMES = list(REF["names"])


def frames(p, n):
    at = np.arange(len(p)) * seq_export.STEP
    return np.stack([np.interp(np.arange(n), at, p[:, c]) for c in range(3)], 1).astype(np.float32)


def data(probs):
    return {n: (probs[n], train.gt_codes(REF[f"lab_{n}"], REF[f"period_{n}"]), REF[f"period_{n}"]) for n in NAMES}


def show(title, probs):
    s, p = best(data(probs))
    print(f"{title:40s} {s:.4f}  {p}", flush=True)
    return s


def seq_runs(dirs):
    plain, tta = {n: [] for n in NAMES}, {n: [] for n in NAMES}
    for d in dirs:
        a = np.load(Path(d) / "cv_seq.npz")
        sw = Path(d) / "cv_seq_xhs.npz"
        b = np.load(sw) if sw.exists() else None
        one, one_tta = {}, {}
        for n in NAMES:
            m = len(REF[f"lab_{n}"])
            one[n] = frames(a[f"prob_{n}"], m)
            if b is not None:
                one_tta[n] = 0.5 * (one[n] + frames(b[f"prob_{n}"][:, [1, 0, 2]], m))
            plain[n].append(one[n])
            tta[n].append(one_tta.get(n, one[n]))
        show(f"{Path(d).name}", one)
        if b is not None:
            show(f"{Path(d).name} + swapped copy", one_tta)
    if len(dirs) > 1:
        show("group", {n: np.mean(plain[n], 0) for n in NAMES})
        show("group + swapped copies", {n: np.mean(tta[n], 0) for n in NAMES})


def lgbm(tags):
    for t in tags:
        a = np.load(train.MODELS / f"cv_probs_{t}_hard.npz")
        show(t, {n: a[f"prob_{n}"] for n in NAMES})
        sw = train.MODELS / f"cv_probs_{t}_hardsw.npz"
        if sw.exists():
            b = np.load(sw)
            show(f"{t} + swapped copy", {n: 0.5 * (a[f"prob_{n}"] + b[f"prob_{n}"]) for n in NAMES})


if __name__ == "__main__":
    (seq_runs if sys.argv[1] == "seq" else lgbm)(sys.argv[2:])
