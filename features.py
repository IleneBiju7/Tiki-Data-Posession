"""Team-level, slot-agnostic features for frame-level possession.

Everything is computed in a frame where the home team attacks towards +x (decided per half from
where each team stands on average), so the same model works for any match and either side.

Input arrays (one match):
  players (n, slots, 2) metres, NaN where absent
  team    (slots,) 0 home / 1 away
  ball    (n, 2) metres, NaN where absent
  period  (n,) 1 / 2, 0 outside play
"""
import numpy as np
import pandas as pd

import ball_sticky
import passes
import smart_tags
import tracks

FPS = 25
AIR_OWN = 2.0   # m: an estimated height above this is out of reach of feet (passes.owners, touches)
NEAR = 3.0  # metres: "close to the ball"
STILL_SPEED = 1.0  # m/s: a placed ball
STILL_MIN = 0.4    # s
STOP_PREFIX = "stop_"  # player-only stoppage clues (work even when the ball is missing or wrong)
MOVING = 2.0           # m/s: a player faster than this is "in play"
BOX_X, BOX_Y = 16.5, 20.16


def orient(players, team, ball, period, length):
    """Flip x per half so that home attacks +x. Returns new arrays and the per-frame flip flag."""
    players, ball = players.copy(), ball.copy()
    flip = np.zeros(len(ball), bool)
    for p in np.unique(period[period > 0]):
        sel = period == p
        with np.errstate(invalid="ignore"):
            hx = np.nanmean(players[sel][:, team == 0, 0])
            ax = np.nanmean(players[sel][:, team == 1, 0])
        if hx > ax:  # home stands to the right on average -> defends right, attacks left
            flip[sel] = True
    players[flip, :, 0] = length - players[flip, :, 0]
    ball[flip, 0] = length - ball[flip, 0]
    return players, ball, flip


def _speed(xy, fps):
    """Speed (m/s) from a centred difference over +/-2 frames, NaN-safe."""
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    d = xy[4:] - xy[:-4]
    v[2:-2] = np.linalg.norm(d, axis=-1) * fps / 4
    return v


def _frames_since(mask):
    """Frames since mask was last False (0 where False). Used for 'time since ball seen'."""
    out = np.zeros(len(mask), np.float32)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run
    return out


def _roll(x, window):
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy(np.float32)


def build(players, team, ball, period, length, width, fps=FPS, fix_gk=True, tags=None, sticky=False, trk=None,
          keeper=False, contacts=False, height=None, camera=None):
    """DataFrame of features, one row per frame.
    trk: optional provided-tracker ball (metres), kept apart from our fused ball. It loses the ball in long
    blocks, mostly over stoppages, so its absence is a dead-ball clue; it also decides which of our own
    detections are trusted (ball_sticky.trusted). keeper: hold a hidden ball with the goalkeeper.
    contacts: also the touches of the ball and whether they were controlled (passes.contact_features).
    height: estimated ball height per frame (ball_height.py); an airborne ball cannot be at anyone's feet.
    camera: camera position (x, y, z in m; one, or one per frame) for comparing our ball with the tracker's."""
    air = {}
    if trk is not None and (camera is not None or height is not None):
        # the tracker places an airborne ball under it, our ball lands where the camera ray meets the grass: ours
        # is then farther from the camera by about height * distance / camera height
        if camera is None:   # a match without the true ball height (Metrica): no camera simulated, unknown
            air.update({k: np.full(len(ball), np.nan, np.float32) for k in ("trk_along", "trk_side", "trk_h")})
        else:
            cam = np.broadcast_to(np.asarray(camera, np.float64), (len(ball), 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                u = ball - cam[:, :2]
                r = np.hypot(u[:, 0], u[:, 1])
                d = ball - trk
                along = (d[:, 0] * u[:, 0] + d[:, 1] * u[:, 1]) / r
                air["trk_along"] = along.astype(np.float32)
                air["trk_side"] = ((d[:, 1] * u[:, 0] - d[:, 0] * u[:, 1]) / r).astype(np.float32)
                air["trk_h"] = (np.clip(along, 0, None) * np.abs(cam[:, 2]) / r).astype(np.float32)
        for w, tag in ((13, "s"), (51, "m")):
            air[f"trk_h_{tag}"] = _roll(np.nan_to_num(air["trk_h"]), w)
    players = tracks.drop_drifting(tracks.drop_frozen(players), fps)
    if sticky:  # hidden ball stays with the player who had it (ball_sticky.py)
        ball = ball_sticky.stick(ball, players, fps, trk=trk, team=team if keeper else None,
                                 length=length, width=width)[0].astype(np.float32)
    players, ball, _ = orient(players, team, ball, period, length)
    if fix_gk:
        team_f = tracks.fix_goalkeepers(players, team, length, width)
    else:
        team_f = np.broadcast_to(team, players.shape[:2])
    f = dict(air)
    present = np.isfinite(players[:, :, 0])
    pspeed = _speed(players, fps)
    for t, name in ((0, "h"), (1, "a")):
        mine = team_f == t
        P = np.where(mine[:, :, None], players, np.nan)
        with np.errstate(invalid="ignore"):
            f[f"n_{name}"] = (present & mine).sum(1).astype(np.float32)
            f[f"cx_{name}"] = np.nanmean(P[:, :, 0], 1)
            f[f"cy_{name}"] = np.nanmean(P[:, :, 1], 1)
            f[f"sx_{name}"] = np.nanstd(P[:, :, 0], 1)
            f[f"sy_{name}"] = np.nanstd(P[:, :, 1], 1)
            f[f"minx_{name}"] = np.nanmin(np.where(np.isfinite(P[:, :, 0]), P[:, :, 0], np.inf), 1)
            f[f"maxx_{name}"] = np.nanmax(np.where(np.isfinite(P[:, :, 0]), P[:, :, 0], -np.inf), 1)
            vs = np.where(mine, pspeed, np.nan)
            f[f"v_{name}"] = np.nanmean(vs, 1)
            f[f"vmax_{name}"] = np.nanmax(np.where(np.isfinite(vs), vs, -1), 1)
            # distance of every player of the team to the ball
            d = np.linalg.norm(P - ball[:, None, :], axis=-1)
            f[f"d_{name}"] = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
            f[f"near_{name}"] = (d < NEAR).sum(1).astype(np.float32)
    for k in list(f):
        f[k] = np.where(np.isfinite(f[k]), f[k], np.nan).astype(np.float32)

    seen = np.isfinite(ball[:, 0])
    bspeed = _speed(ball, fps)
    vel = np.gradient(np.where(seen[:, None], ball, np.nan), axis=0) * fps
    touch = np.full(len(ball), np.nan, np.float32)
    touch[2:-2] = np.linalg.norm(vel[4:] - vel[:-4], axis=1)
    f["ball_seen"] = seen.astype(np.float32)
    f["bx"], f["by"] = ball[:, 0], ball[:, 1]
    f["bspeed"] = bspeed
    f["btouch"] = touch
    f["ball_out"] = ((ball[:, 0] < -0.5) | (ball[:, 0] > length + 0.5) |
                     (ball[:, 1] < -0.5) | (ball[:, 1] > width + 0.5)).astype(np.float32)
    # ball went out: seen past a line, or lost within 1 m of a line for at least 1 s
    with np.errstate(invalid="ignore"):
        out_by = np.nanmax(np.stack([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width]), 0)
    with np.errstate(invalid="ignore"):
        moving = ~(_speed(ball, fps) < STILL_SPEED)               # a ball placed on the line is not an exit
    exit_ = seen & (out_by > 0) & moving
    last = np.flatnonzero(seen[:-1] & ~seen[1:])
    gap_end = np.r_[np.flatnonzero(seen), len(seen)]
    for a in last:
        nxt = gap_end[np.searchsorted(gap_end, a + 1)]
        if out_by[a] > -1.0 and nxt - a > fps:
            exit_[a] = True
    f["out_by"] = np.where(seen, out_by, np.nan).astype(np.float32)
    f["since_exit"] = np.minimum(_frames_since(~exit_), 30 * fps) / fps
    f["until_exit"] = np.minimum(_frames_since((~exit_)[::-1])[::-1], 30 * fps) / fps
    # a restart needs a placed (still) ball: has the ball been still for STILL_MIN since it last went out?
    with np.errstate(invalid="ignore"):
        still = _speed(ball, fps) < STILL_SPEED
    still_run = _frames_since(still) >= STILL_MIN * fps
    placed = np.zeros(len(ball), np.float32)
    since_still = np.full(len(ball), 30.0, np.float32)
    flag, last_still = 0.0, None
    for i in range(len(ball)):
        if exit_[i]:
            flag = 0.0
        if still_run[i]:
            flag, last_still = 1.0, i
        placed[i] = flag
        if last_still is not None:
            since_still[i] = min((i - last_still) / fps, 30.0)
    f["placed_since_exit"] = placed
    f["since_still"] = since_still
    f["since_seen"] = np.minimum(_frames_since(~seen), 50 * fps) / fps
    f["until_seen"] = np.minimum(_frames_since((~seen)[::-1])[::-1], 50 * fps) / fps

    # last / next known ball facts carried across gaps
    dd = np.where(seen, f["d_h"] - f["d_a"], np.nan)
    f["dd"] = dd
    f["dd_last"] = pd.Series(dd).ffill().to_numpy(np.float32)
    f["dd_next"] = pd.Series(dd).bfill().to_numpy(np.float32)
    f["bx_last"] = pd.Series(np.where(seen, ball[:, 0], np.nan)).ffill().to_numpy(np.float32)
    f["bx_next"] = pd.Series(np.where(seen, ball[:, 0], np.nan)).bfill().to_numpy(np.float32)
    f["by_last"] = pd.Series(np.where(seen, ball[:, 1], np.nan)).ffill().to_numpy(np.float32)
    f["by_last_edge"] = np.minimum(f["by_last"], width - f["by_last"])
    f["bx_last_edge"] = np.minimum(f["bx_last"], length - f["bx_last"])

    # derived team comparisons (invariant of slot identities)
    f["v_all"] = np.nanmean(np.stack([f["v_h"], f["v_a"]]), 0)
    f["cx_diff"] = f["cx_h"] - f["cx_a"]
    f["sx_diff"] = f["sx_h"] - f["sx_a"]
    f["sy_diff"] = f["sy_h"] - f["sy_a"]
    f["n_all"] = f["n_h"] + f["n_a"]
    f["near_diff"] = f["near_h"] - f["near_a"]

    # stoppage clues from the players alone: the whole field slows down, someone stands on a line or at a
    # corner flag (throw-in / corner taker), players crowd into a box (corner / free kick)
    present_all = np.isfinite(players[:, :, 0])
    with np.errstate(invalid="ignore"):
        moving_p = pspeed > MOVING
        f["stop_frac_moving"] = (moving_p.sum(1) / np.maximum(1, present_all.sum(1))).astype(np.float32)
        px, py = players[:, :, 0], players[:, :, 1]
        edge = np.minimum.reduce([px, length - px, py, width - py])          # negative = outside the pitch
        edge = np.where(np.isfinite(edge), edge, np.inf)
        f["stop_edge_min"] = np.minimum(edge.min(1), 30.0).astype(np.float32)
        slow = ~(pspeed > 1.0)
        f["stop_edge_still"] = np.minimum(np.where(slow, edge, np.inf).min(1), 30.0).astype(np.float32)
        cd = np.full(px.shape, np.inf)
        for cx in (0.0, length):
            for cy in (0.0, width):
                cd = np.minimum(cd, np.hypot(px - cx, py - cy))
        cd = np.where(np.isfinite(cd), cd, np.inf)
        f["stop_corner_d"] = np.minimum(cd.min(1), 40.0).astype(np.float32)
        inbox_y = np.abs(py - width / 2) < BOX_Y
        f["stop_box_l"] = ((px < BOX_X) & inbox_y).sum(1).astype(np.float32)
        f["stop_box_r"] = ((px > length - BOX_X) & inbox_y).sum(1).astype(np.float32)
        f["stop_box_max"] = np.maximum(f["stop_box_l"], f["stop_box_r"])
    f["stop_v_drop"] = (_roll(np.nan_to_num(f["v_all"]), 13) - _roll(np.nan_to_num(f["v_all"]), 251)).astype(np.float32)
    calm = f["stop_frac_moving"] < 0.15
    f["stop_since_calm"] = np.minimum(_frames_since(calm), 30 * fps) / fps
    f["stop_since_busy"] = np.minimum(_frames_since(~calm), 30 * fps) / fps

    # pass chains: a team stringing passes together is playing, however calm the rest of the field looks
    if height is not None:
        h = np.nan_to_num(np.asarray(height, np.float32))
        f["h_est"] = h
        f["h_air_s"] = _roll((h > 1.0).astype(np.float32), 13)
        f["h_air_m"] = _roll((h > 1.0).astype(np.float32), 51)
        feet = np.where((h > AIR_OWN)[:, None], np.nan, ball)       # nobody owns or touches a ball overhead
    else:
        feet = ball
    f.update(passes.features(players, team_f, feet, fps, contacts=contacts))

    for key in ("stop_frac_moving", "stop_edge_still", "stop_corner_d", "stop_box_max"):
        for w, tag in ((13, "s"), (51, "m"), (251, "l")):
            f[f"{key}_{tag}"] = _roll(f[key], w)

    for key in ("dd", "v_all", "v_h", "v_a", "ball_seen", "cx_diff", "sy_diff", "near_diff", "bspeed", "cx_h", "cx_a"):
        for w, tag in ((13, "s"), (51, "m"), (251, "l")):
            f[f"{key}_{tag}"] = _roll(np.nan_to_num(f[key], nan=0.0) if key == "dd" else f[key], w)
    if trk is not None:   # the provided tracker's ball on its own
        tseen = np.isfinite(trk[:, 0])
        f["trk_seen"] = tseen.astype(np.float32)
        since = np.minimum(_frames_since(~tseen), 60 * fps) / fps
        until = np.minimum(_frames_since((~tseen)[::-1])[::-1], 60 * fps) / fps
        f["trk_since"], f["trk_until"] = np.minimum(since, 30.0), np.minimum(until, 30.0)
        f["trk_gap"] = np.where(tseen, 0.0, np.minimum(since + until, 60.0)).astype(np.float32)
        for w, tag in ((13, "s"), (51, "m"), (251, "l")):
            f[f"trk_seen_{tag}"] = _roll(f["trk_seen"], w)
    f["period"] = period.astype(np.float32)
    if tags is not None:  # Smart Tagging (real or simulated): (phase_starts, set_piece_starts)
        f.update(smart_tags.features(tags[0], tags[1], len(ball), fps))
    return pd.DataFrame(f)


def swap_labels(labels):
    """Home <-> away labels, for the team-swap augmentation (build() with team = 1 - team)."""
    out = labels.copy()
    out[labels == 0], out[labels == 1] = 1, 0
    return out
