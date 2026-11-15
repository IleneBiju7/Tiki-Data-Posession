"""Two-stage prediction for BAR-ATH: first-stage model(s) -> averaged H/A/D probabilities -> second-stage
stack model (stack.py) -> decoding. Writes predictions/<name>.csv.

Usage: python predict_stack.py <name> --stage1=v16[+v17] --stack=<stack tag> --decode=team,to_dead,from_dead,bias,min_flip
       [--ball=fused_m3]    our ball (models trained with --trk take the pre-sticky ball)
       [--seq=<dir>[+<dir>][,<dir>+...]]  also the sequence networks models/<dir>/seq_final.pt (seq_train.py),
                            averaged within each comma-separated group, each group one more first-stage member (the
                            stack must have been trained with the same groups: seq_cv.py)
       [--seq-weight=4[,2]]  each group's weight in that average (stack tag ...+seqC@4+seqG@2)
       [--ps=<dir>[+<dir>] --ps-weight=2]  also the ball-free player networks models/<dir>/ps_final.pt (ps_predict.py)
       [--swap-tta]         also predict with the two teams' identities swapped (slots and Smart Tag teams), H and A
                            swapped back, and average every first-stage member with it, as its CV files were
                            (train.py --swap-tta, seq_train.py --test=xh,xhs, tta_cv.py)
       [--swap-tta=seq]     the same for the sequence networks only (the trees gain nothing from it)
       [--swap-tta=seq:1,3] only for these --seq groups (counted from 0), the others as they are
       [--hold-hidden]      while our ball is hidden the team that last had it keeps it (lowers open-match CV; not the final)
  stage-1 models are models/possession_lgbm_<v>.txt with models/config_<v>.json
"""
import json
import sys

import lightgbm as lgb
import numpy as np

import atd_cache
import ball_height
import decode_search
import features
import predict
import smart_tags
import stack  # noqa: E402  (reads --runs / --more from argv to match the saved stack model)


def arg(key, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), default)


CONTEXT = {}


def stage1_probs(version, players, team, ball, period, tags_cache, trk=None, height=None, camera=None):
    cfg = json.loads((predict.HERE / "models" / f"config_{version}.json").read_text())
    model = lgb.Booster(model_file=str(predict.HERE / "models" / f"possession_lgbm_{version}.txt"))
    tags = tags_cache if cfg.get("use_tags") else None
    df = features.build(players, team, ball, period, *predict.PITCH, 25, fix_gk=cfg.get("fix_gk", False), tags=tags,
                        sticky=cfg.get("sticky", False), trk=trk if cfg.get("trk") else None,
                        keeper=cfg.get("keeper", False), contacts=cfg.get("contacts", False),
                        height=height if cfg.get("height") else None, camera=camera if cfg.get("height") else None)
    if tags is not None and "df" not in CONTEXT:
        CONTEXT["df"] = df
    return model.predict(df[cfg["columns"]].to_numpy(np.float32))


def seq_probs(players, team, ball, period, tags, trk, height, camera, groups, suffix=""):
    """The sequence networks (seq_train.py, models/<dir>/seq_final.pt), averaged per group: one more first-stage
    member per group, as in the CV files seq_cv.py made. Features as seq_export.py built them, every STEP-th frame,
    back to every frame."""
    import subprocess
    import seq_export
    cols = json.loads((predict.HERE / "models" / "seq" / "meta.json").read_text())["columns"]
    df = features.build(players, team, ball, period, *predict.PITCH, 25, fix_gk=True, tags=tags, sticky=True, trk=trk,
                        keeper=True, contacts=True, height=height, camera=camera)
    path = atd_cache.DATA_DIR / "cache" / f"BAR-ATH_seq_features{suffix}.npz"
    np.savez_compressed(path, x=df[cols].to_numpy(np.float16)[::seq_export.STEP])
    members = []
    for g, dirs in enumerate(groups):
        out = path.with_name(f"BAR-ATH_seq_probs_{g}{suffix}.npz")
        subprocess.run([sys.executable, str(predict.HERE / "seq_predict.py"), str(path), str(out)]
                       + [str(predict.HERE / "models" / d / "seq_final.pt") for d in dirs], check=True)
        p = np.load(out)["prob"]
        at = np.arange(len(p)) * seq_export.STEP
        members.append(np.stack([np.interp(np.arange(len(ball)), at, p[:, c]) for c in range(3)], 1))
    return members


def ps_probs(players, team, n, dirs):
    """The ball-free player networks, every frame, cleaned as in the training copies."""
    import subprocess
    import tracks
    path = atd_cache.DATA_DIR / "cache" / "BAR-ATH_ps_players.npz"
    clean = tracks.drop_drifting(tracks.drop_frozen(players), 25).astype(np.float32)
    np.savez_compressed(path, players=clean, team=team, length=predict.PITCH[0], width=predict.PITCH[1], fps=25)
    out = path.with_name("BAR-ATH_ps_probs.npz")
    subprocess.run([sys.executable, str(predict.HERE / "ps_predict.py"), str(path), str(out)]
                   + [str(predict.HERE / "models" / d / "ps_final.pt") for d in dirs], check=True)
    z = np.load(out)
    at = np.arange(len(z["prob"])) * int(z["step"])
    return np.stack([np.interp(np.arange(n), at, z["prob"][:, c]) for c in range(3)], 1)


def main():
    name = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    versions = arg("stage1").split("+")
    if arg("decode") == "auto":   # the decoding stack.py --save chose on the held-out matches
        decode = json.loads((predict.HERE / "models" / f"stack_{arg('stack')}.json").read_text())["decode"]
    else:
        decode = [float(x) for x in arg("decode").split(",")]
    decode[-1] = int(decode[-1])
    atd = atd_cache.load()
    players = atd["players"].astype(np.float32) * np.array(predict.PITCH, np.float32)
    team = atd["team"]
    ball_name = arg("ball", "fused_v6s")
    ball = np.load(atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_{ball_name}.npy").astype(np.float32)
    ball = ball * np.array(predict.PITCH, np.float32)
    n = len(ball)
    period = np.where(np.arange(n) < predict.HALF_SPLIT, 1, 2).astype(np.int8)
    tags = smart_tags.real_events(atd_cache.TACTICAL / "BAR-ATH_tactical_smart_tagging.xml", "Barcelona")
    trk = (atd["ball"] * np.array(predict.PITCH)).astype(np.float32)   # the provided tracker's own ball
    hpath = atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_height_{ball_name}.npy"   # ball_height.py
    height = np.load(hpath) if hpath.exists() else None
    if height is None:
        print(f"note: {hpath.name} is missing (ball_height.py): predicting without the ball's height", flush=True)
    camera = ball_height.competition_centres()
    members = [stage1_probs(v, players, team, ball, period, tags, trk, height, camera) for v in versions]
    groups = [g.split("+") for g in arg("seq").split(",")] if arg("seq") else []
    members += seq_probs(players, team, ball, period, tags, trk, height, camera, groups) if groups else []
    tta = next((a for a in sys.argv if a == "--swap-tta" or a.startswith("--swap-tta=")), None)
    if tta:   # the same match with the teams' identities swapped, H and A swapped back
        steam, stags = 1 - team, ([(f, 1 - t) for f, t in tags[0]], tags[1])
        seq_only = tta.startswith("--swap-tta=seq")   # the trees keep their own probabilities
        which = [int(i) for i in tta.split(":", 1)[1].split(",")] if ":" in tta else list(range(len(groups)))
        swapped = [] if seq_only else [stage1_probs(v, players, steam, ball, period, stags, trk, height, camera)
                                       for v in versions]
        chosen = [groups[i] for i in which]
        swapped += seq_probs(players, steam, ball, period, stags, trk, height, camera, chosen, "_sw") if chosen else []
        at = ([] if seq_only else list(range(len(versions)))) + [len(versions) + i for i in which]
        for i, b in zip(at, swapped, strict=True):
            members[i] = 0.5 * (members[i] + b[:, [1, 0, 2]])
    seq_weights = [float(w) for w in arg("seq-weight", ",".join(["1"] * len(groups))).split(",")] if groups else []
    assert len(seq_weights) == len(groups), "one --seq-weight per --seq group"
    ps_weight = []
    if arg("ps"):
        members.append(ps_probs(players, team, n, arg("ps").split("+")))
        ps_weight = [float(arg("ps-weight", "1"))]
    prob = np.average(members, 0, weights=[1.0] * len(versions) + seq_weights + ps_weight)

    graded = np.zeros(n, np.int8)
    for p, (a, b) in enumerate(predict.HALVES, start=1):
        graded[a:b + 1] = p
    stack_tag = arg("stack")
    if stack_tag:
        m = lgb.Booster(model_file=str(predict.HERE / "models" / f"stack_{stack_tag}.txt"))
        # stage-2 features are computed per graded half, like the per-match OOF probabilities it was trained on
        prob2 = prob.copy()
        for p in (1, 2):
            sel = np.flatnonzero(graded == p)
            ctx, cols = None, ()
            if stack.CTX:
                import ctx_cache
                ctx, cols = CONTEXT["df"][ctx_cache.COLUMNS].to_numpy(np.float32)[sel], list(ctx_cache.COLUMNS)
            prob2[sel] = m.predict(stack.stack_features(prob[sel], ctx, cols).to_numpy())
        prob = prob2
    codes = decode_search.decode(prob, graded, *decode)
    if "--hold-hidden" in sys.argv:   # while our ball is hidden the team that last had it keeps it (dead ball kept)
        idx = np.flatnonzero(CONTEXT["df"]["ball_seen"].to_numpy() < 0.5)
        cut = np.flatnonzero(np.diff(idx) > 1)
        for a, b in zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]) if len(idx) else ():
            if a > 0 and codes[a - 1] in (0, 1):
                seg = codes[a:b + 1]
                seg[(seg == 0) | (seg == 1)] = codes[a - 1]
    out = predict.HERE / "predictions" / f"{name}.csv"
    with open(out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{c}\n" for i, c in enumerate(predict.LETTERS[codes]))
    alive = codes[graded > 0]
    k = np.flatnonzero(np.diff(alive) != 0)
    print(f"wrote {out}: H {np.mean(alive == 0):.3f} A {np.mean(alive == 1):.3f} D {np.mean(alive == 2):.3f}, "
          f"{len(k) + 1} spells")


if __name__ == "__main__":
    main()
