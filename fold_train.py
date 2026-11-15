"""Degradation test, step 1: v43x's leave-one-match-out CV (the compliant flags), saving the 10 fold models so
any test copy (clean tracking, more ball missing, ...) can be scored later without retraining. Open matches only.
Writes models/folds_v43x/fold_<i>.txt; stops before the final fit. Then degradation_test.py."""
import sys
from pathlib import Path
sys.argv = ["train.py"] + ("--mixed-tags --sticky --hard-test --big --smooth --pff --trk --keeper --sp-calib --contacts "
                           "--height --det-noise --external --tag=v43xf").split()
sys.path.insert(0, ".")
import train  # noqa: E402

OUT = Path(__file__).parent / "models" / "folds_v43x"
OUT.mkdir(parents=True, exist_ok=True)
_train, calls = train.lgb.train, []


def train_and_save(*a, **k):
    if len(calls) == 10:
        raise SystemExit("10 folds saved; skipping the final fit")
    m = _train(*a, **k)
    m.save_model(str(OUT / f"fold_{len(calls)}.txt"))
    calls.append(1)
    return m


train.lgb.train = train_and_save
train.main()
