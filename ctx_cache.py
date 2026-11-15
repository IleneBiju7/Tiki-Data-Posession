"""Context columns for the second stage, computed on the fixed realistic test copy of every CV match
(identical to the copy each CV run predicted, so they line up with cv_probs_<tag>_hard.npz).

Usage: python ctx_cache.py --sticky   -> models/ctx_hard.npz
       python ctx_cache.py --sticky --passes   -> models/ctx_hard_p.npz (adds the pass-chain columns; --carry also carry / duel -> ctx_hard_pc.npz)
"""
import sys

import numpy as np

import train

CTX = ["ball_seen_m", "since_seen", "until_seen", "since_exit", "until_exit", "placed_since_exit", "since_still",
       "dd_s", "v_all_m", "v_all_l", "stop_frac_moving_m", "stop_since_calm", "stop_since_busy", "stop_corner_d_s",
       "stop_edge_still_s", "stop_box_max_m", "st_last_team", "st_since", "st_next_team", "st_until", "st_sp_since",
       "st_sp_until"]
PASS_CTX = ["pass_all_c", "pass_h_c", "pass_a_c", "pass_since", "pass_until", "pass_past", "pass_next"]
CARRY_CTX = ["pass_carry_s", "pass_carry_m", "pass_duel_s", "pass_duel_m"]
TRK_CTX = ["trk_seen_m", "trk_since", "trk_until", "trk_gap"]
TRK = "--trk" in sys.argv        # also the provided-tracker ball columns
OWN_CTX = ["own_team", "own_since", "own_until"]
OWN = "--own" in sys.argv        # also who owns the ball and when it last changed hands (passes.py)
AIR_CTX = ["h_air_m", "trk_h_s", "trk_h_m"]
AIR = "--air" in sys.argv        # also how high the ball is (needs --height: ball_height.py, features.build trk_h)
TOUCH_CTX = ["ctl_last", "ctl_next", "ctl_since", "ctl_until", "ctl_switch_since", "ctl_switch_until"]
TOUCH = "--touch" in sys.argv    # also the last / next controlled touch (needs --contacts: passes.contact_features)
PASSES = "--passes" in sys.argv
CARRY = "--carry" in sys.argv    # also the ball-carry / duel columns
COLUMNS = CTX + (PASS_CTX if PASSES else []) + (CARRY_CTX if CARRY else []) + (TRK_CTX if TRK else []) + (OWN_CTX if OWN else []) \
    + (AIR_CTX if AIR else []) + (TOUCH_CTX if TOUCH else [])
# test copies with the airborne ball moved as on the video: with ball height (--height) or without (--shift-test)
SHIFT = "_z" if train.HEIGHT else "_x" if train.SHIFT_TEST else ""
FILE = f"ctx_hard{'_p' if PASSES else ''}{'c' if CARRY else ''}{'_s' if '--smooth' in sys.argv else ''}{'_t' if TRK else ''}{'o' if OWN else ''}{'a' if AIR else ''}{'k' if TOUCH else ''}{SHIFT}{'e' if train.EXTERNAL else ''}.npz"


def main():
    if TOUCH and not train.CONTACTS:
        sys.exit("--touch needs --contacts")
    out = {}
    for name in train.load_matches(False):
        df, _, _ = train.match_features(name, np.random.default_rng(train.TEST_SEED + 2), p_lose=train.P_LOSE_TEST, det_noise=True,
                                        test=True)
        out[name] = df[COLUMNS].to_numpy(np.float32)
        print(" ", name, flush=True)
    np.savez_compressed(train.MODELS / FILE, columns=np.array(COLUMNS), **out)


if __name__ == "__main__":
    main()
