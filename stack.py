"""Second stage: a small LightGBM that re-reads the first model's H/A/D probabilities over the
surrounding seconds (past and future, at several scales) and outputs corrected probabilities.

Trained leave-one-match-out on the held-out (out-of-fold) probabilities of a CV run, so every score
is still on a match neither stage saw. Uses both the normal and the realistic test copies.

Usage: python stack.py tagA[+tagB]      -> prints normal / realistic spellF1 with decode_search
       python stack.py tagA[+tagB] --save  also fits on all matches and saves models/stack_<tags>.txt
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import ctx_cache
import decode_search
import train

WINDOWS = (5, 13, 25, 51, 125, 251)
SHIFTS = (-50, -25, -12, 12, 25, 50)
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
ROUNDS = 300
STEP = 5
RUN_FEATURES = "--runs" in sys.argv
if "--more" in sys.argv:
    STEP, ROUNDS = 2, 600
if "--big2" in sys.argv:
    PARAMS.update(num_leaves=63, min_data_in_leaf=300)
    ROUNDS = 500
NO_TAG_CTX = "--no-tag-ctx" in sys.argv   # robustness: stage 2 without the Smart Tagging context columns


CTX = "--ctx" in sys.argv   # also give stage 2 some raw context (ctx_cache.py); trains on realistic copies only
ACTIONS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--actions=")), None)   # ball actions with team
#   (actions.py: models/actions_open_<variant>.npz from the open matches' event data, exact or video-like)


def stack_features(prob, ctx=None, ctx_cols=()):
    f = {}
    lp = np.log(np.clip(prob, 1e-4, 1))
    for k, c in enumerate("HAD"):
        s = pd.Series(prob[:, k])
        f[c] = prob[:, k]
        for w in WINDOWS:
            f[f"{c}_m{w}"] = s.rolling(w, center=True, min_periods=1).mean().to_numpy()
            f[f"{c}_past{w}"] = s.rolling(w, min_periods=1).mean().to_numpy()
            f[f"{c}_fut{w}"] = s[::-1].rolling(w, min_periods=1).mean().to_numpy()[::-1]
        for w in (25, 125):
            f[f"{c}_max{w}"] = s.rolling(w, center=True, min_periods=1).max().to_numpy()
            f[f"{c}_min{w}"] = s.rolling(w, center=True, min_periods=1).min().to_numpy()
        for sh in SHIFTS:
            f[f"{c}_sh{sh}"] = np.roll(prob[:, k], sh)
    f["hma"] = lp[:, 0] - lp[:, 1]
    f["live"] = 1 - prob[:, 2]
    if RUN_FEATURES:   # spell structure of the first stage's best guess
        am = prob.argmax(1)
        change = np.r_[False, am[1:] != am[:-1]]
        idx = np.arange(len(am))
        last = np.maximum.accumulate(np.where(change, idx, 0))
        nxt = np.minimum.accumulate(np.where(change, idx, len(am))[::-1])[::-1]
        f["run_since"] = np.minimum(idx - last, 500).astype(np.float32)
        f["run_until"] = np.minimum(nxt - idx, 500).astype(np.float32)
        f["run_len"] = np.minimum(nxt - last, 1000).astype(np.float32)
        for w in (51, 251):
            f[f"flips{w}"] = pd.Series(change.astype(np.float32)).rolling(w, center=True, min_periods=1).sum().to_numpy()
    if ctx is not None:
        for k, c in enumerate(ctx_cols):
            if NO_TAG_CTX and c.startswith("st_"):
                continue
            f[f"ctx_{c}"] = ctx[:, k]
    return pd.DataFrame(f).astype(np.float32)


def load(tags):
    out = {}
    ctx = np.load(train.MODELS / ctx_cache.FILE) if CTX else None
    for suffix in (("_hard",) if ("--hard-only" in sys.argv or CTX) else ("", "_hard")):
        d = decode_search.load(tags, suffix)
        for n, (p, gt, per) in d.items():
            c, cols = (ctx[n], list(ctx["columns"])) if ctx is not None else (None, [])
            if ACTIONS:
                acts = np.load(train.MODELS / f"actions_open_{ACTIONS}.npz")
                c, cols = np.hstack([c, acts[n]]), cols + list(acts["columns"])
            out[(n, suffix)] = (stack_features(p, c, cols), gt, per)
    return out


def main():
    tags = sys.argv[1].split("+")
    data = load(tags)
    names = sorted({n for n, _ in data})
    new = {}
    for held in names:
        train_keys = [k for k in data if k[0] != held and (k[1] == "_hard" or not ("--hard-only" in sys.argv or CTX))]
        X = np.concatenate([data[k][0].to_numpy()[::STEP] for k in train_keys])
        y = np.concatenate([data[k][1][::STEP] for k in train_keys])
        keep = y < 3
        m = lgb.train(PARAMS, lgb.Dataset(X[keep], y[keep]), ROUNDS)
        for suffix in (("_hard",) if CTX else ("", "_hard")):
            if (held, suffix) not in data:
                d = decode_search.load(tags, suffix)[held]
                data[(held, suffix)] = (stack_features(d[0]), d[1], d[2])
            f, gt, per = data[(held, suffix)]
            new[(held, suffix)] = (m.predict(f.to_numpy()), gt, per)
        print("  stacked", held, flush=True)
    normal = {n: new[(n, "")] for n in names} if not CTX else {n: new[(n, "_hard")] for n in names}
    out_tag = f"stacked_{'+'.join(tags)}{'_c' if CTX else ''}{'_p' if ctx_cache.PASSES else ''}{'c' if ctx_cache.CARRY else ''}{'_s' if train.SMOOTH else ''}{'_t' if ctx_cache.TRK else ''}{'o' if ctx_cache.OWN else ''}{'a' if ctx_cache.AIR else ''}{'k' if ctx_cache.TOUCH else ''}{'_b' if '--big2' in sys.argv else ''}{'_nt' if NO_TAG_CTX else ''}{ctx_cache.SHIFT}{'_ac' + ACTIONS if ACTIONS else ''}"
    np.savez_compressed(train.MODELS / f"cv_probs_{out_tag}_hard.npz", names=np.array(names),
                        **{f"prob_{n}": new[(n, "_hard")][0] for n in names},
                        **{f"lab_{n}": data[(n, "_hard")][1] for n in names},
                        **{f"period_{n}": data[(n, "_hard")][2] for n in names})
    hard = {n: new[(n, "_hard")] for n in names}
    best = None
    for params in [(t, td, fd, b, 0) for t in (2, 5, 10, 20) for td in (10, 20, 40, 80) for fd in (10, 20, 40, 80)
                   for b in (0.0,)]:
        s = (decode_search.evaluate(hard, params), decode_search.evaluate(normal, params))
        if best is None or sum(s) > sum(best[0]):
            best = (s, params)
            print(f"  realistic {s[0]:.4f} normal {s[1]:.4f} {params}", flush=True)
    print(f"BEST stack {'+'.join(tags)}: realistic {best[0][0]:.4f}, normal {best[0][1]:.4f}, params {best[1]}")
    print("  per match (realistic):", {n: round(train.score(gt, decode_search.decode(p, per, *best[1]))["spellF1"], 4)
                                       for n, (p, gt, per) in hard.items()})
    if "--save" in sys.argv:
        keys = [k for k in data if k[1] == "_hard" or not ("--hard-only" in sys.argv or CTX)]
        X = np.concatenate([data[k][0].to_numpy()[::STEP] for k in keys])
        y = np.concatenate([data[k][1][::STEP] for k in keys])
        m = lgb.train(PARAMS, lgb.Dataset(X[y < 3], y[y < 3]), ROUNDS)
        path = train.MODELS / f"stack_{'+'.join(tags)}{'_h' if '--hard-only' in sys.argv else ''}{'_r' if RUN_FEATURES else ''}{'_m' if '--more' in sys.argv else ''}{'_c' if CTX else ''}{'_p' if ctx_cache.PASSES else ''}{'c' if ctx_cache.CARRY else ''}{'_s' if train.SMOOTH else ''}{'_t' if ctx_cache.TRK else ''}{'o' if ctx_cache.OWN else ''}{'a' if ctx_cache.AIR else ''}{'k' if ctx_cache.TOUCH else ''}{'_b' if '--big2' in sys.argv else ''}{'_nt' if NO_TAG_CTX else ''}{ctx_cache.SHIFT}.txt"
        m.save_model(str(path))
        # the decoding chosen on the held-out matches travels with the model: predict_stack.py --decode=auto
        path.with_suffix(".json").write_text(json.dumps({"decode": list(best[1]), "realistic_spellF1": best[0][0]}))
        print("saved", best[1])


if __name__ == "__main__":
    main()
