"""Feature sequences for the sequence model (seq_train.py), open matches only: never the competition match.

Per match, the same two training copies train.py builds (normal and team-swapped) and, for the 10 matches that are
held out in CV, the fixed realistic test copy every CV run is scored on, all every STEP-th frame (5 per second at
25 fps) so the network sees continuous time. float16, one file per match in models/seq_x/.

Usage: python seq_export.py <train.py flags>   e.g. --mixed-tags --sticky --smooth --pff --trk --keeper --sp-calib
       --det-noise --contacts --height
  --skillcorner adds the 20 SkillCorner matches (training only, own random draws, folder models/seq_x_sc);
  --swap-tta adds xhs, the realistic test copy with the teams swapped (test-time averaging).
"""
import json
import sys

import numpy as np

import train

STEP = 5
OUT = train.MODELS / ("seq_x" + ("_sc" if "--skillcorner" in sys.argv else ""))   # own folders


def rows(df, cols, lab, fps):
    step = STEP if fps == 25 else 1
    return df[cols].to_numpy(np.float16)[::step], lab[::step].astype(np.int8)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(train.SEED)
    rng_sc = np.random.default_rng([train.SEED, 31])   # SkillCorner (--skillcorner): its own draws, the rest unchanged
    names = train.load_matches("--skillcorner" in sys.argv)
    cols = None
    for name in names:
        r = rng_sc if name.startswith("skillcorner") else rng
        df, lab, fps = train.match_features(name, r)
        dfs, labs, _ = train.match_features(name, r, swap=True,
                                            tag_noise="heavy" if "--mixed-tags" in sys.argv else "normal")
        if cols is None:
            cols = [c for c in df.columns if not ("--no-own" in sys.argv and c.startswith("own_"))]
        out = {}
        out["x"], y = rows(df, cols, lab, fps)
        out["xs"], ys = rows(dfs, cols, labs, fps)
        out["y"], out["ys"] = y, ys
        out["period"] = np.load(train._path(name))["period"][::STEP if fps == 25 else 1]
        if not name.startswith(train.TRAIN_ONLY):
            hdf, _, _ = train.match_features(name, np.random.default_rng(train.TEST_SEED + 2), p_lose=train.P_LOSE_TEST,
                                             det_noise=True, test=True)
            out["xh"] = hdf[cols].to_numpy(np.float16)[::STEP]
            if train.SWAP_TTA:   # the same copy with the teams swapped (seq_train.py --test=xh,xhs; swap H/A back)
                sdf, _, _ = train.match_features(name, np.random.default_rng(train.TEST_SEED + 2), swap=True,
                                                 p_lose=train.P_LOSE_TEST, det_noise=True, test=True)
                out["xhs"] = sdf[cols].to_numpy(np.float16)[::STEP]
        np.savez_compressed(OUT / f"{name}.npz", fps=fps, **out)
        print(f"  {name}: {len(out['x'])} steps", flush=True)
        del df, dfs
    (OUT / "meta.json").write_text(json.dumps({"columns": cols, "step": STEP, "flags": sys.argv[1:]}, indent=1))


if __name__ == "__main__":
    main()
