"""Algorithmic clean-up of automatic tracking: goalkeepers assigned to the wrong team.

Automatic team assignment works from kit colour, and a goalkeeper's kit matches neither side, so
the tracker sometimes puts a keeper (or a duplicate track of him) on the opposing team. The rule
below is general, applied to every frame, and uses positions only:

  an "opponent" track in a team's own penalty area, with no other opponent within ISOLATION
  metres, that is either deeper than every player of that team or shadowing that team's deepest
  player (within DUPLICATE metres -- a duplicate track of the keeper), for at least MIN_RUN
  consecutive frames, is that team's goalkeeper.

Works on oriented coordinates (home defends x = 0, away defends x = length; see features.orient).
Returns a per-frame team array (n, slots).
"""
import numpy as np

BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
ISOLATION = 12.0
DUPLICATE = 7.0
MIN_RUN = 50
KEEPER_ZONE = 6.0   # m from the goal line: deeper than every defender AND this close to goal -> keeper,
                    # even when a team-mate of its (wrong) label is nearby


FROZEN_FRAMES = 25


def drop_frozen(players, min_frames=FROZEN_FRAMES):
    """Remove tracks that do not move at all for min_frames or more.

    Real players always move a little, so an exactly constant position is a dead track the tracker
    left behind (e.g. a slot parked where it was before kickoff). Returns a copy with those NaN.
    """
    players = players.copy()
    same = np.all(players[1:] == players[:-1], axis=2) & np.isfinite(players[1:, :, 0])
    for s in range(players.shape[1]):
        f = np.r_[False, same[:, s], False].astype(np.int8)
        starts, stops = np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1)
        for a, b in zip(starts, stops):
            if b - a >= min_frames:
                players[a:b + 1, s] = np.nan
    return players


DRIFT_TOL = 0.02     # m/s: velocity unchanged ...
DRIFT_SPAN = 4.0     # ... across 4 s
DRIFT_MIN_SPEED = 0.05


def drop_drifting(players, fps=25, tol=DRIFT_TOL, span=DRIFT_SPAN, min_speed=DRIFT_MIN_SPEED):
    """Remove extrapolated tracks: after losing someone, the tracker keeps sliding the slot in a
    perfectly straight line at constant speed. Real players always change speed or direction within
    a few seconds, so a velocity that stays identical (within tol m/s) across `span` seconds is a
    ghost. Thresholds were set so that clean, corrected tracking (DFL open data) loses ~0.1% of
    player positions. Works in any units as long as tol/min_speed match (pass metres)."""
    players = players.copy()
    h, w = int(fps * 0.5), int(fps * span)
    v = np.full(players.shape, np.nan)
    v[h:-h] = (players[2 * h:] - players[:-2 * h]) / (2 * h / fps)
    dv = np.full(players.shape[:2], np.nan)
    dv[w // 2:w // 2 + len(v) - w] = np.linalg.norm(v[w:] - v[:-w], axis=2)
    with np.errstate(invalid="ignore"):
        ghost = (dv < tol) & (np.linalg.norm(v, axis=2) > min_speed)
    players[ghost] = np.nan
    return players


def fix_goalkeepers(players, team, length, width):
    n, slots, _ = players.shape
    team_f = np.broadcast_to(team, (n, slots)).copy()
    x, y = players[:, :, 0], players[:, :, 1]
    present = np.isfinite(x)
    in_box_y = np.abs(y - width / 2) < BOX_HALF_WIDTH

    flag = np.zeros((n, slots), bool)
    for defender, goal_x in ((0, 0.0), (1, length)):
        attacker = 1 - defender
        depth = np.abs(x - goal_x)                      # distance from that team's goal line
        own = present & (team == defender)
        own_depth = np.where(own, depth, np.inf)
        keeper = own_depth.argmin(1)                    # deepest player of the defending team
        has_keeper = np.isfinite(own_depth.min(1))
        kx = np.where(has_keeper, x[np.arange(n), keeper], np.nan)
        ky = np.where(has_keeper, y[np.arange(n), keeper], np.nan)
        deeper = depth < own_depth.min(1)[:, None]
        shadow = np.hypot(x - kx[:, None], y - ky[:, None]) < DUPLICATE
        cand = present & (team == attacker) & (depth < BOX_DEPTH) & in_box_y & (deeper | shadow)
        keeper_zone = present & (team == attacker) & (depth < KEEPER_ZONE) & in_box_y & deeper
        for i, s in zip(*np.nonzero(cand)):
            if keeper_zone[i, s]:
                flag[i, s] = True
                continue
            others = present[i] & (team == attacker)
            others[s] = False
            if others.any() and np.hypot(x[i, others] - x[i, s], y[i, others] - y[i, s]).min() < ISOLATION:
                continue
            flag[i, s] = True
    # keep only sustained runs: a duplicate keeper persists, a pressing striker passes through
    for s in range(slots):
        f = np.r_[False, flag[:, s], False].astype(np.int8)
        starts, stops = np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1)
        for a, b in zip(starts, stops):
            if b - a >= MIN_RUN:
                team_f[a:b, s] = 1 - team[s]
    return team_f


if __name__ == "__main__":
    import atd_cache
    import features

    d = atd_cache.load()
    pitch = np.array((105.3, 68.0), np.float32)
    P, B = d["players"] * pitch, d["ball"] * pitch
    period = np.where(np.arange(len(B)) < 100_730, 1, 2).astype(np.int8)
    P, B, _ = features.orient(P, d["team"], B, period, 105.3)
    team_f = fix_goalkeepers(P, d["team"], 105.3, 68.0)
    graded = np.zeros(len(B), bool)
    graded[11185:90652] = graded[110808:190198] = True
    changed = (team_f != d["team"]).any(1)
    print(f"frames with a goalkeeper re-assigned: {changed[graded].mean():.1%} of graded frames")
