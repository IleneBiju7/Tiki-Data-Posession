"""Apply the trained model to BAR-ATH and write a submission CSV.

Reads only the provided automatic tracking (via atd_cache). Labels outside the graded halves are X.

Usage: python predict.py [name] [--ball=fused|fused_v2]   -> predictions/<name>.csv  (default: ours_v1)
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

import atd_cache
import features
import smart_tags
import train

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PITCH = (105.3, 68.0)                         # games.py, measured from the corrected tracking
HALVES = [(11185, 90651), (110808, 190197)]   # graded video frames, DATA.md
HALF_SPLIT = 100_730                          # middle of the halftime break
LETTERS = np.array(list("HADX"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "ours_v1"
    ball_name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ball=")), None)
    cfg = json.loads((HERE / "models" / "config.json").read_text())
    model = lgb.Booster(model_file=str(HERE / "models" / "possession_lgbm.txt"))

    atd = atd_cache.load()
    players = atd["players"].astype(np.float32) * np.array(PITCH, np.float32)
    ball = atd["ball"].astype(np.float32)
    if ball_name:  # our own video detections fused with the provided tracker ball (fuse_ball.py)
        ball = np.load(atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_{ball_name}.npy").astype(np.float32)
    ball = ball * np.array(PITCH, np.float32)
    n = len(ball)
    period = np.where(np.arange(n) < HALF_SPLIT, 1, 2).astype(np.int8)

    tags = None
    if cfg.get("use_tags"):
        tags = smart_tags.real_events(atd_cache.TACTICAL / "BAR-ATH_tactical_smart_tagging.xml", "Barcelona")
    df = features.build(players, atd["team"], ball, period, *PITCH, 25, fix_gk=cfg.get("fix_gk", False), tags=tags,
                        sticky=cfg.get("sticky", False))
    prob = model.predict(df[cfg["columns"]].to_numpy(np.float32))

    graded_period = np.zeros(n, np.int8)
    for p, (a, b) in enumerate(HALVES, start=1):
        graded_period[a:b + 1] = p
    codes = train.decode(prob, graded_period, cfg["switch_team"], cfg["switch_dead"], cfg.get("min_flip", 0))

    out = HERE / "predictions" / f"{name}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{c}\n" for i, c in enumerate(LETTERS[codes]))
    alive = codes[graded_period > 0]
    print(f"wrote {out}: H {np.mean(alive == 0):.3f} A {np.mean(alive == 1):.3f} D {np.mean(alive == 2):.3f}")


if __name__ == "__main__":
    main()
