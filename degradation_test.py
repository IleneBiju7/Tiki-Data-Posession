"""Degradation test, step 2 (open matches only): score v43x's 10 fold models (fold_train.py) and the organisers'
baseline (starter/baseline.py: nearest player to the tracker's ball, D where there is no ball, their smooth() with a
+/- 50 frame majority vote as in baseline_smooth50) on test copies from clean tracking to more and more ball missing,
plus the usual realistic copy.

Levels: clean = the open match's own tracking, nothing removed; ball-X% = the clean match with the ball (ours and the
tracker's alike) removed in gaps (median 6 s, as in SkillCorner) until X% of played frames have none; realistic = the
standard --external test copy every CV score uses. Needs the organisers' repository next to this one as official-repo/.
Usage: python fold_train.py, then python degradation_test.py -> models/degradation_results.json"""
import json
import sys
from pathlib import Path
sys.argv = ["x"] + ("--mixed-tags --sticky --hard-test --big --smooth --pff --trk --keeper --sp-calib --contacts "
                    "--height --det-noise --external").split()
sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "official-repo" / "starter"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import ball_height  # noqa: E402
import train  # noqa: E402
from baseline import smooth as organisers_smooth  # noqa: E402  (official-repo/starter/baseline.py)

FOLDS = Path(__file__).parent / "models" / "folds_v43x"
LEVEL = {"x": None}
_degrade = train.degrade


def clean_degrade(z, rng, p_lose=None, det_noise=False, shift=False):
    """The match's own tracking (players, teams, ball, the tracker's ball = the same ball), with LEVEL['x'] of the
    played frames' ball removed in gaps; the true ball height where the ball is seen."""
    x = LEVEL["x"]
    if x is None:
        return _degrade(z, rng, p_lose, det_noise, shift)
    g = np.random.default_rng([int(rng.bit_generator.state["state"]["state"] % (2 ** 32)), 5])
    ball, lab, fps = z["ball"].astype(np.float32).copy(), z["label"], int(z["fps"])
    played = np.flatnonzero(lab < 3)
    drop = np.zeros(len(ball), bool)
    while x > 0 and drop[played].mean() < x:
        s = int(g.choice(played))
        drop[s:s + int(np.clip(g.lognormal(np.log(6.0), 0.9), 0.5, 90) * fps)] = True
    ball[drop] = np.nan
    pitch = (float(z["length"]), float(z["width"]))
    height = np.zeros(len(ball), np.float32)
    if "ball_z" in z.files:
        height = np.where(np.isfinite(ball[:, 0]), np.nan_to_num(z["ball_z"]), 0.0).astype(np.float32)
    centre = ball_height.centre(ball_height.generic_camera(g, pitch))
    return z["players"].astype(np.float32), ball, z["team"], ball.copy(), height, centre


def baseline(name, level):
    """The organisers' baseline on the same copy: each frame, the team of the player nearest the tracker's ball (D where
    the tracker has no ball), then their smooth() with a +/- 50 frame majority vote (baseline_smooth50)."""
    z = np.load(train._path(name))
    rng = np.random.default_rng(train.TEST_SEED + 2)
    players, ball, team, trk, height, centre = train.degrade(z, rng, train.P_LOSE_TEST, True, False)
    dist = np.linalg.norm(players.astype(np.float64) - trk.astype(np.float64)[:, None, :], axis=-1)
    seen = np.isfinite(trk).all(axis=1) & np.isfinite(dist).any(axis=1)
    codes = np.full(len(trk), -1, np.int8)
    codes[seen] = np.asarray(team)[np.nanargmin(dist[seen], axis=1)]
    lab = organisers_smooth(codes, 50)
    return np.select([lab == "H", lab == "A"], [0, 1], 2).astype(np.int8)


def main():
    cfg = json.loads((train.MODELS / "config_v43x.json").read_text())
    cv = json.loads((train.MODELS / "cv_v43xf_hard.json").read_text())
    names = [n for n in train.load_matches(False) if not n.startswith(train.TRAIN_ONLY)]
    models = [lgb.Booster(model_file=str(FOLDS / f"fold_{i}.txt")) for i in range(len(names))]
    train.degrade = clean_degrade
    rows = []
    for title, x in (("clean tracking", 0.0), ("ball missing 10%", 0.1), ("ball missing 20%", 0.2),
                     ("ball missing 30%", 0.3), ("ball missing 40%", 0.4), ("realistic copy (--external)", None)):
        LEVEL["x"] = x
        ours, base = [], []
        for name, m in zip(names, models):
            df, label, _ = train.match_features(name, np.random.default_rng(train.TEST_SEED + 2), fix_gk=cfg.get("fix_gk", True),
                                                use_tags=cfg.get("use_tags", True), p_lose=train.P_LOSE_TEST, det_noise=True, test=True)
            period = np.load(train._path(name))["period"]
            gt = train.gt_codes(label, period)
            prob = m.predict(df[cfg["columns"]].to_numpy(np.float32))
            pred = train.decode(prob, period, cv["switch_team"], cv["switch_dead"], cv["min_flip"])
            ours.append(train.score(gt, pred)["spellF1"])
            bl = baseline(name, x)
            bl[period == 0] = train.X_CODE
            base.append(train.score(gt, bl)["spellF1"])
        rows.append((title, np.mean(ours), np.mean(base)))
        print(f"{title:30s} v43x {np.mean(ours):.4f}   organisers' baseline {np.mean(base):.4f}", flush=True)
    (train.MODELS / "degradation_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
