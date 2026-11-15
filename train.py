"""Train and cross-validate the possession model on open matches (other games only).

  1. degrade each clean match so it looks like automatic tracking (long ball gaps, especially
     over dead balls; dropped players; jitter)
  2. features.build() on the original and on the team-swapped copy (augmentation)
  3. LightGBM frame classifier -> P(H), P(A), P(D)
  4. Viterbi decoding with switching penalties -> whole possessions
  5. leave-one-match-out score with the official metric code (spellF1, macroF1)

Usage: python train.py [--skillcorner]   (SkillCorner broadcast matches as extra training data)
       python train.py --pff            (PFF World Cup 2022 matches as extra training data)
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

import ball_height
import features
import smart_tags
import tracks

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "official-repo" / "evaluation"))
import possession_metrics as pm  # noqa: E402

CONVERTED = ROOT / "data" / "open" / "converted"
CONVERTED_PFF = ROOT / "data" / "open" / "converted_pff"
TRAIN_ONLY = ("skillcorner", "pff")   # extra training matches, never held out
PFF_STEP = 10   # every 10th frame of each PFF match: 64 matches would not fit in memory at every 3rd
MODELS = HERE / "models"
X_CODE = 3
SEED = 7
TEST_SEED = 7   # test copies never change, so every version is scored on identical inputs
_seed_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seed=")), None)
if _seed_arg:   # training randomness only (degradation draws, LightGBM): for seed-bagged ensembles
    SEED = int(_seed_arg)
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=SEED)
N_ROUNDS = 400
if "--big" in sys.argv:   # higher-capacity variant (ensemble diversity)
    LGB_PARAMS.update(learning_rate=0.03, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.6)
    N_ROUNDS = 900
if "--huge" in sys.argv:
    LGB_PARAMS.update(learning_rate=0.02, num_leaves=255, min_data_in_leaf=60, feature_fraction=0.5)
    N_ROUNDS = 1500
SUBSAMPLE = 3
SMOOTH = "--smooth" in sys.argv   # no per-frame jitter on the degraded copies
TRK = "--trk" in sys.argv         # a separate provided-tracker ball channel (features.build trk=)
HEIGHT = "--height" in sys.argv       # airborne balls seen through a generic wide camera (ball_height.py)
SHIFT_TEST = "--shift-test" in sys.argv   # baseline: only the test copies see airborne balls that way, no height features
CONTACTS = "--contacts" in sys.argv   # touches of the ball and whether they were controlled (passes.py)
KEEPER = "--keeper" in sys.argv   # a hidden ball stays with the goalkeeper (ball_sticky keeper hold)
SP_CALIB = "--sp-calib" in sys.argv   # simulated set-piece tags timed like the real ones
# Organisers, 3 Nov: tune on external data only. The provider-
# like tracker's ball follows SkillCorner and PFF tracking (other matches: ball missing on 6-7% of live and 70-79% of
# dead frames, gaps median 4-6 s, 14-23% of off-pitch dead frames still with a ball); our detector's noise spans its
# SoccerNet test (0.4-8.9 false sightings per 1000 crops without a ball); numbers nothing outside can measure
# (keeper duplicates, tracker slips, airborne losses, Smart Tag timing) are drawn per copy from wide ranges; the
# airborne ball is seen through a generic wide camera (ball_height.generic_camera).
EXTERNAL = "--external" in sys.argv   # names the context cache of these copies (ctx_cache.py)
SWAP_TTA = "--swap-tta" in sys.argv   # also score the team-swapped realistic copy (test-time averaging): cv_probs_<tag>_hardsw.npz
P_LOSE_TEST = 0.3   # realistic test copy: tracker loses 70-79% of dead frames, the detector
#                   (~55-65% found on SoccerNet) about half of those -> our ball ~30%


# -- degradation -------------------------------------------------------------------------------

def degrade(z, rng, p_lose=None, det_noise=False, shift=False):
    """Make a clean match look like automatic broadcast-style tracking."""
    players, ball, label = z["players"].copy(), z["ball"].copy(), z["label"]
    fps = int(z["fps"])
    n = len(ball)
    drop = np.zeros(n, bool)
    height, cam = None, None
    if HEIGHT or shift:   # an airborne ball shows up where the camera ray meets the grass; its height is only estimated
        height = np.zeros(n, np.float32)
        if "ball_z" in z.files:
            # seeded from the main generator's current state without drawing from it, so everything else is unchanged
            hrng = np.random.default_rng([rng.bit_generator.state["state"]["state"], 7])
            pitch = (float(z["length"]), float(z["width"]))
            cam = ball_height.generic_camera(hrng, pitch)
            ball = ball_height.simulate(ball, z["ball_z"], cam, hrng, pitch=pitch,
                                        q_hi=hrng.uniform(0.2, 0.8))

    # dead-ball stretches: each match draws its own loss rate
    dead = np.r_[False, label == 2, False]
    starts, stops = np.flatnonzero(np.diff(dead.astype(int)) == 1), np.flatnonzero(np.diff(dead.astype(int)) == -1)
    p_lose = rng.uniform(0.2, 0.9) if p_lose is None else p_lose
    for a, b in zip(starts, stops):
        if rng.random() < p_lose:
            drop[max(0, a - int(rng.uniform(0, 3) * fps)):min(n, b + int(rng.uniform(0, 3) * fps))] = True

    # live play: long gaps until a target share of live frames is missing
    live = np.flatnonzero(label < 2)
    target = rng.uniform(0.01, 0.06) * len(live)   # under the tracker's 6-7% (SkillCorner, PFF) plus our detector's own misses
    while len(live) and (drop[live]).sum() < target:
        start = int(rng.choice(live))
        length = int(np.clip(rng.lognormal(np.log(6.0), 0.9), 0.5, 90) * fps)
        drop[start:start + length] = True
    # the tracker places an airborne ball under it, so it keeps the true ground position
    trk = tracker_channel(z, drop, starts, stops, rng.spawn(1)[0], errors=height is not None) if TRK else None
    if str(z["source"]) not in ("skillcorner", "pff"):   # broadcast tracking: its ball gaps are already real
        ball[drop] = np.nan
    if det_noise:
        ball = detector_noise(ball, players, float(z["length"]), float(z["width"]), fps, rng)

    # players: each slot loses random blocks (about 7% of its frames)
    per_min = rng.uniform(0.5, 1.5)
    for s in range(players.shape[1]):
        k = rng.poisson(n / (fps * 60) * per_min)
        for start in rng.integers(0, n, k):
            players[start:start + int(rng.uniform(1, 8) * fps), s] = np.nan

    players, team = add_keeper_duplicates(players, z["team"], ball, z["period"], float(z["length"]), fps, rng,
                                          share=rng.uniform(0.0, 0.2))
    # same clean-up as at prediction time, before jitter (jitter would hide frozen / drifting tracks)
    players = tracks.drop_drifting(tracks.drop_frozen(players), fps).astype(np.float32)
    if not SMOOTH:   # per-frame jitter: adds ~1 m/s to every measured speed
        players += rng.normal(0, 0.15, players.shape).astype(np.float32)
        ball += rng.normal(0, 0.25, ball.shape).astype(np.float32)
    if height is not None:
        if HEIGHT and cam is not None:   # estimated from the finished ball (gaps, false sightings), as on the video
            height = ball_height.from_pitch(cam, ball)
        height[~np.isfinite(ball[:, 0])] = 0.0
    centre = ball_height.centre(cam) if HEIGHT and cam is not None else None
    return players, ball, team, trk, height if HEIGHT else None, centre


def tracker_channel(z, drop, starts, stops, rng, errors=False):
    """A tracker's ball on its own: it misses everything our fused ball misses, plus most of the stoppages our
    detector recovers, plus some long gaps in play. A broadcast source's own ball (PFF, SkillCorner) already behaves
    like this and is kept."""
    ball = z["ball"].astype(np.float32).copy()
    if errors:
        _tracker_errors(ball, z["label"], int(z["fps"]), rng)
    if str(z["source"]) in ("skillcorner", "pff"):
        return ball
    fps, n, label = int(z["fps"]), len(ball), z["label"]
    tdrop = drop.copy()
    q = rng.uniform(0.3, 0.6)
    for a, b in zip(starts, stops):
        if not drop[a:b].all() and rng.random() < q:
            tdrop[max(0, a - int(rng.uniform(0, 3) * fps)):min(n, b + int(rng.uniform(0, 3) * fps))] = True
    live = np.flatnonzero(label < 2)
    extra = rng.uniform(0.01, 0.05)
    target = (tdrop[live].mean() + extra) * len(live) if len(live) else 0
    gap_s, gap_min = 6.0, 2   # SkillCorner / PFF gaps: median 4-6 s
    while len(live) and tdrop[live].sum() < target:
        start = int(rng.choice(live))
        tdrop[start:start + int(np.clip(rng.lognormal(np.log(gap_s), 0.6), gap_min, 60) * fps)] = True
    length, width = float(z["length"]), float(z["width"])
    with np.errstate(invalid="ignore"):
        out = (ball[:, 0] < -0.5) | (ball[:, 0] > length + 0.5) | (ball[:, 1] < -0.5) | (ball[:, 1] > width + 0.5)
    d = np.diff(np.r_[0, out.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if rng.random() < 0.8:                   # the tracker also drops most balls off the pitch
            tdrop[a:b] = True
    ball[tdrop] = np.nan
    return ball


def _tracker_errors(ball, label, fps, rng):
    """In place: short runs (median 0.25 s) where the tracker's ball is a few metres off (median 4 m, any direction),
    at 2-10 runs per minute of play."""
    live = np.flatnonzero(label < 2)
    if not len(live):
        return
    n = len(ball)
    rate = rng.uniform(2.0, 10.0)
    for start in rng.choice(live, rng.poisson(len(live) / fps / 60 * rate)):
        seg = slice(start, min(n, start + int(np.clip(rng.lognormal(np.log(0.25), 0.9), 0.08, 3.0) * fps)))
        a = rng.uniform(0, 2 * np.pi)
        ball[seg] += (rng.lognormal(np.log(4.0), 0.5) * np.array([np.cos(a), np.sin(a)])).astype(ball.dtype)


def detector_noise(ball, players, length, width, fps, rng):
    """Make a clean ball behave like a video detector's ball, with rates drawn per copy from wide ranges:
      - a ball off the pitch is mostly lost, sometimes left stuck on the line where the camera lost it
      - short false sightings far from the ball, often at a player's feet
      - more positional jitter, correlated over a few frames"""
    ball = ball.copy()
    n = len(ball)
    with np.errstate(invalid="ignore"):
        out = (ball[:, 0] < -0.5) | (ball[:, 0] > length + 0.5) | (ball[:, 1] < -0.5) | (ball[:, 1] > width + 0.5)
    d = np.diff(np.r_[0, out.astype(np.int8), 0])
    keep_out = rng.uniform(0.05, 0.25)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        r = rng.random()
        if r < keep_out:
            continue
        if r < keep_out + 0.2:                       # stuck on the line for a while, then lost
            k = min(b, a + int(rng.uniform(0.2, 1.5) * fps))
            ball[a:k, 0] = np.clip(ball[a:k, 0], 0, length)
            ball[a:k, 1] = np.clip(ball[a:k, 1], 0, width)
            ball[k:b] = np.nan
        else:
            ball[a:b] = np.nan
    # false sightings
    n_false = rng.poisson(n * rng.uniform(0.002, 0.010) / 5)
    for a in rng.integers(0, n, n_false):
        k = min(n, a + int(rng.integers(1, 16)))
        if rng.random() < 0.6:                       # at a random player's feet
            s = int(rng.integers(0, players.shape[1]))
            ball[a:k] = players[a:k, s] + rng.normal(0, 0.3, 2)
        else:
            ball[a:k] = (rng.uniform(0, length), rng.uniform(0, width))
    # correlated jitter
    e = rng.normal(0, 0.25, (n, 2))
    e = np.cumsum(e, 0)
    e = e - np.r_[np.zeros((5, 2)), e[:-5]]         # random walk over the last 5 frames, std ~0.55 m
    ball += (rng.uniform(0.08, 0.25) * e).astype(np.float32)
    return ball.astype(np.float32)


def add_keeper_duplicates(players, team, ball, period, length, fps, rng, share=0.10):
    """Mimic the automatic tracker's goalkeeper errors: a lagging duplicate of each keeper, on the
    wrong team, for about `share` of the frames."""
    _, _, flip = features.orient(players, team, ball, period, length)
    n = len(players)
    extra = np.full((n, 2, 2), np.nan, np.float32)
    for k, t in enumerate((0, 1)):
        goal_x = np.where(flip ^ (t == 1), length, 0.0)     # where team t defends, per frame
        depth = np.abs(players[:, team == t, 0] - goal_x[:, None])
        depth = np.where(np.isfinite(depth), depth, np.inf)
        keeper = np.flatnonzero(team == t)[depth.argmin(1)]
        lag = int(fps)
        src = np.clip(np.arange(n) - lag, 0, n - 1)
        pos = players[src, keeper[src]] + rng.normal(0, 1.5, 2).astype(np.float32)
        active = np.zeros(n, bool)
        while active.mean() < share * rng.uniform(0.5, 1.5):
            a = int(rng.integers(0, n))
            active[a:a + int(rng.uniform(5, 60) * fps)] = True
        extra[active, k] = pos[active]
    return np.concatenate([players, extra], axis=1), np.concatenate([team, np.array([1, 0], np.int8)])


def _path(name):
    return (CONVERTED_PFF if name.startswith("pff") else CONVERTED) / f"{name}.npz"


def load_matches(use_skillcorner):
    names = sorted(p.stem for p in CONVERTED.glob("*.npz"))
    if not use_skillcorner:
        names = [n for n in names if not n.startswith("skillcorner")]
    if "--pff" in sys.argv:
        names += sorted(p.stem for p in CONVERTED_PFF.glob("*.npz"))
    return names


def match_features(name, rng, swap=False, fix_gk=True, use_tags=True, tag_noise="normal", p_lose=None,
                   det_noise=None, test=False):
    z = np.load(_path(name))
    if det_noise is None:
        det_noise = "--det-noise" in sys.argv and not str(z["source"]) == "skillcorner"
    players, ball, team, trk, height, centre = degrade(z, rng, p_lose, det_noise, shift=SHIFT_TEST and test)
    label = z["label"].copy()
    tags = smart_tags.simulate_events(label, z["period"], int(z["fps"]), rng, tag_noise,
                                      sp_rng=rng.spawn(1)[0] if SP_CALIB else None) if use_tags else None
    if swap:
        team, label = 1 - team, features.swap_labels(label)
        if tags is not None:
            tags = ([(f, 1 - t) for f, t in tags[0]], tags[1])
    df = features.build(players, team, ball, z["period"], float(z["length"]), float(z["width"]), int(z["fps"]),
                        fix_gk=fix_gk, tags=tags, sticky="--sticky" in sys.argv, trk=trk, keeper=KEEPER, contacts=CONTACTS, height=height,
                        camera=centre)
    return df, label, int(z["fps"])


# -- decoding ----------------------------------------------------------------------------------

def viterbi(prob, switch_team, switch_dead):
    """Most likely H/A/D path given frame probabilities and a cost per change of state."""
    logp = np.log(np.clip(prob, 1e-6, 1.0))
    cost = np.array([[0, switch_team, switch_dead],
                     [switch_team, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=np.float64)
    n = len(prob)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].astype(np.float64)
    for i in range(1, n):
        cand = score[:, None] - cost          # from-state x to-state
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    path = np.zeros(n, np.int8)
    path[-1] = score.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def remove_flips(path, min_frames):
    """'H A H' where the A spell is shorter than min_frames -> 'H H H': a brief touch by the other team
    between two spells of the same team is a challenge, not a change of possession."""
    if min_frames <= 0:
        return path
    path = path.copy()
    starts = np.flatnonzero(np.diff(np.r_[-1, path]) != 0)
    stops = np.r_[starts[1:], len(path)]
    for i in range(1, len(starts) - 1):
        a, b = starts[i], stops[i]
        before, here, after = path[starts[i - 1]], path[a], path[b] if b < len(path) else -1
        if here in (0, 1) and before == after == 1 - here and b - a < min_frames:
            path[a:b] = before
    return path


def remove_short_dead(path, min_frames):
    """'A D A' where the D spell is shorter than min_frames -> 'A A A': a dead-ball blip inside one
    team's possession (typically slow play right after a restart) is not a real stoppage."""
    if min_frames <= 0:
        return path
    path = path.copy()
    starts = np.flatnonzero(np.diff(np.r_[-1, path]) != 0)
    stops = np.r_[starts[1:], len(path)]
    for i in range(1, len(starts) - 1):
        a, b = starts[i], stops[i]
        before, after = path[starts[i - 1]], path[b] if b < len(path) else -1
        if path[a] == 2 and before == after and before in (0, 1) and b - a < min_frames:
            path[a:b] = before
    return path


def decode(prob, period, switch_team, switch_dead, min_flip=0, min_dead=0):
    out = np.full(len(prob), X_CODE, np.int8)
    for p in (1, 2):
        sel = np.flatnonzero(period == p)
        if len(sel):
            out[sel] = remove_short_dead(remove_flips(viterbi(prob[sel], switch_team, switch_dead), min_flip), min_dead)
    return out


def score(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return {"spellF1": spells["mean_f1"], "macroF1": pm.macro_f1(cm3),
            "n_gt": spells["n_gt"], "n_pred": spells["n_pred"]}


def gt_codes(label, period):
    gt = label.astype(np.int8).copy()
    gt[period == 0] = X_CODE
    return gt


# -- main --------------------------------------------------------------------------------------

def main():
    use_sc = "--skillcorner" in sys.argv
    fix_gk = "--no-gkfix" not in sys.argv
    use_tags = "--no-tags" not in sys.argv
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "default")
    rng = np.random.default_rng(SEED)
    rng_sc = np.random.default_rng([SEED, 31])
    names = load_matches(use_sc)
    print("training matches:", names, flush=True)

    data = {}
    for name in names:
        # mixed tag noise: one copy with normal tags, the team-swapped copy with very noisy tags, so the
        # model learns to trust Smart Tagging only as far as it deserves
        # mixed ball noise: one copy with a detector-like ball, the other with the clean-tracker ball
        mixed = "--mixed-noise" in sys.argv and not name.startswith("skillcorner")
        r = rng_sc if name.startswith("skillcorner") else rng   # SkillCorner: its own draws, the rest unchanged
        df, lab, fps = match_features(name, r, fix_gk=fix_gk, use_tags=use_tags, det_noise=True if mixed else None)
        dfs, labs, _ = match_features(name, r, swap=True, fix_gk=fix_gk, use_tags=use_tags,
                                      tag_noise="heavy" if "--mixed-tags" in sys.argv else "normal",
                                      det_noise=False if mixed else None)
        z = np.load(_path(name))
        step = (PFF_STEP if name.startswith("pff") else SUBSAMPLE) if fps == 25 else 1
        if name.startswith(TRAIN_ONLY):  # training-only matches: keep just the rows used, to save memory
            df, lab, dfs, labs = df.iloc[::step].copy(), lab[::step], dfs.iloc[::step].copy(), labs[::step]   # copy: a strided slice keeps the whole match alive
            step = 1
        data[name] = dict(df=df, lab=lab, dfs=dfs, labs=labs, period=z["period"], fps=fps, step=step)
        print(f"  features {name}: {len(df)} frames", flush=True)
    cols = list(data[names[0]]["df"].columns)
    if "--no-stop-features" in sys.argv:
        cols = [c for c in cols if not c.startswith(features.STOP_PREFIX)]
    if "--no-tag-timing" in sys.argv:   # real Smart Tags come in bursts; simulated gaps between them mean dead ball
        cols = [c for c in cols if c not in ("st_since", "st_until")]
    if "--no-carry" in sys.argv:
        cols = [c for c in cols if not c.startswith(("pass_carry", "pass_duel"))]
    if "--no-own" in sys.argv:          # the ball-ownership columns (passes.py) came later: v35 / v36 were trained without
        cols = [c for c in cols if not c.startswith("own_")]
    hard = "--hard-test" in sys.argv

    def train_rows(keys):
        X, y = [], []
        for k in keys:
            d = data[k]
            step = d["step"]
            for df, lab in ((d["df"], d["lab"]), (d["dfs"], d["labs"])):
                keep = np.flatnonzero(lab[::step] < 3) * step
                X.append(df.iloc[keep][cols].to_numpy(np.float32))
                y.append(lab[keep])
        return np.concatenate(X), np.concatenate(y)

    if "--final-only" in sys.argv:   # skip CV: reuse the decoding found by an earlier CV run with this tag
        cv = json.loads((MODELS / f"cv_{tag}.json").read_text())
        save_final(names, train_rows, cols, fix_gk, use_tags, (cv["switch_team"], cv["switch_dead"], cv["min_flip"]),
                   cv["spellF1"], cv["macroF1"], tag)
        return

    # leave-one-match-out on the matches with real dead-ball labels at 25 fps
    held = [n for n in names if not n.startswith(TRAIN_ONLY)]
    probs, probs_hard, probs_hard_sw = {}, {}, {}
    for name in held:
        X, y = train_rows([k for k in names if k != name])
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), N_ROUNDS)
        test_df = data[name]["df"]
        if True:   # same fixed test copy for every version, so ensembles are compared fairly
            test_df, _, _ = match_features(name, np.random.default_rng(TEST_SEED + 3), fix_gk=fix_gk, use_tags=use_tags,
                                           det_noise=False, test=True)
        if "--heavy-test-tags" in sys.argv:   # robustness: test with much noisier tags than trained on
            test_df, _, _ = match_features(name, np.random.default_rng(TEST_SEED + 1), fix_gk=fix_gk, tag_noise="heavy",
                                           test=True)
        probs[name] = model.predict(test_df[cols].to_numpy(np.float32))
        if hard:   # the realistic test copy
            hdf, _, _ = match_features(name, np.random.default_rng(TEST_SEED + 2), fix_gk=fix_gk, use_tags=use_tags,
                                       p_lose=P_LOSE_TEST, det_noise=True, test=True)
            probs_hard[name] = model.predict(hdf[cols].to_numpy(np.float32))
            if SWAP_TTA:   # the same copy with the two teams' identities swapped, H and A swapped back afterwards
                sdf, _, _ = match_features(name, np.random.default_rng(TEST_SEED + 2), swap=True, fix_gk=fix_gk,
                                           use_tags=use_tags, p_lose=P_LOSE_TEST, det_noise=True, test=True)
                probs_hard_sw[name] = model.predict(sdf[cols].to_numpy(np.float32))[:, [1, 0, 2]]
        print(f"  fold {name}: trained on {len(y)} rows", flush=True)

    np.savez_compressed(MODELS / f"cv_probs_{tag}.npz", names=np.array(held),
                        **{f"prob_{n}": probs[n] for n in held},
                        **{f"lab_{n}": data[n]["lab"] for n in held},
                        **{f"period_{n}": data[n]["period"] for n in held},
                        **{f"exit_{n}": data[n]["df"]["since_exit"].to_numpy(np.float32) for n in held},
                        **{f"placed_{n}": data[n]["df"]["placed_since_exit"].to_numpy(np.float32) for n in held})
    grid = [(t, d, m) for t in (2, 5, 10, 20, 40) for d in (5, 20, 40, 80) for m in (0, 25, 50, 75)]
    results = []
    for t, d, m in grid:
        s = [score(gt_codes(data[n]["lab"], data[n]["period"]),
                   decode(probs[n], data[n]["period"], t, d, m)) for n in held]
        results.append((np.mean([x["spellF1"] for x in s]), np.mean([x["macroF1"] for x in s]), t, d, s, m))
    results.sort(key=lambda r: -r[0])
    best = results[0]
    print("\nbest switching penalties: team %s, dead %s, min flip %s frames -> mean spellF1 %.4f, macroF1 %.4f"
          % (best[2], best[3], best[5], best[0], best[1]))
    for n, s in zip(held, best[4]):
        print(f"  {n:22s} spellF1 {s['spellF1']:.4f}  macroF1 {s['macroF1']:.4f}  "
              f"possessions pred {s['n_pred']} / true {s['n_gt']}")
    print("\ndead-ball check (held-out, best decoding):")
    for n in held:
        gt = gt_codes(data[n]["lab"], data[n]["period"])
        pred = decode(probs[n], data[n]["period"], best[2], best[3], best[5])
        g = gt < 3
        tp = np.sum((pred == 2) & (gt == 2) & g)
        print(f"  {n:22s} D share true {np.mean(gt[g] == 2):.2f} pred {np.mean(pred[g] == 2):.2f}  "
              f"D precision {tp / max(1, np.sum((pred == 2) & g)):.2f} recall {tp / max(1, np.sum((gt == 2) & g)):.2f}  "
              f"ball missing & live -> called D: {np.mean(pred[g & (gt < 2) & (data[n]['df']['ball_seen'].to_numpy() == 0)] == 2):.2f}")
    raw = [score(gt_codes(data[n]["lab"], data[n]["period"]),
                 np.where(data[n]["period"] > 0, probs[n].argmax(1), X_CODE).astype(np.int8)) for n in held]
    print("  (no decoding: mean spellF1 %.4f, macroF1 %.4f)"
          % (np.mean([r["spellF1"] for r in raw]), np.mean([r["macroF1"] for r in raw])))

    (MODELS / f"cv_{tag}.json").parent.mkdir(exist_ok=True)
    (MODELS / f"cv_{tag}.json").write_text(json.dumps(
        {"fix_gk": fix_gk, "spellF1": best[0], "macroF1": best[1], "switch_team": best[2], "switch_dead": best[3],
         "min_flip": best[5],
         "per_match": {n: s for n, s in zip(held, best[4])}}, indent=2, default=float))
    if SWAP_TTA:
        np.savez_compressed(MODELS / f"cv_probs_{tag}_hardsw.npz", names=np.array(held),
                            **{f"prob_{n}": probs_hard_sw[n] for n in held},
                            **{f"lab_{n}": data[n]["lab"] for n in held},
                            **{f"period_{n}": data[n]["period"] for n in held})
    if hard:
        np.savez_compressed(MODELS / f"cv_probs_{tag}_hard.npz", names=np.array(held),
                            **{f"prob_{n}": probs_hard[n] for n in held},
                            **{f"lab_{n}": data[n]["lab"] for n in held},
                            **{f"period_{n}": data[n]["period"] for n in held})
        sh = [score(gt_codes(data[n]["lab"], data[n]["period"]),
                    decode(probs_hard[n], data[n]["period"], best[2], best[3], best[5])) for n in held]
        print("\nrealistic test (detector-like ball, kept on 3/4 of stoppages): mean spellF1 %.4f, macroF1 %.4f"
              % (np.mean([x["spellF1"] for x in sh]), np.mean([x["macroF1"] for x in sh])))
        (MODELS / f"cv_{tag}_hard.json").write_text(json.dumps(
            {"spellF1": np.mean([x["spellF1"] for x in sh]), "switch_team": best[2], "switch_dead": best[3],
             "min_flip": best[5], "per_match": {n: x for n, x in zip(held, sh)}}, indent=2, default=float))
    if "--cv-only" in sys.argv:
        return
    save_final(names, train_rows, cols, fix_gk, use_tags, (best[2], best[3], best[5]), best[0], best[1], tag)


def save_final(names, train_rows, cols, fix_gk, use_tags, decoding, cv_spell, cv_macro, tag):
    """Fit on every match; save as the active model and as possession_lgbm_<tag>.txt / config_<tag>.json."""
    X, y = train_rows(names)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), N_ROUNDS)
    MODELS.mkdir(exist_ok=True)
    cfg = json.dumps(
        {"columns": cols, "fix_gk": fix_gk, "switch_team": decoding[0], "switch_dead": decoding[1],
         "min_flip": decoding[2], "use_tags": use_tags, "sticky": "--sticky" in sys.argv, "trk": TRK, "keeper": KEEPER, "contacts": CONTACTS, "height": HEIGHT,
         "training_matches": names,
         "train_flags": [a for a in sys.argv[1:] if a.startswith("--")], "cv_spellF1": cv_spell, "cv_macroF1": cv_macro},
        indent=2)
    for suffix in ("", f"_{tag}"):
        model.save_model(str(MODELS / f"possession_lgbm{suffix}.txt"))
        (MODELS / f"config{suffix}.json").write_text(cfg)
    print(f"\nsaved final model trained on {len(y)} rows -> {MODELS}")


if __name__ == "__main__":
    main()
