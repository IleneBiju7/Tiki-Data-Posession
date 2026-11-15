"""Pass chains from tracking: a team stringing passes together is playing, however calm everyone else looks.

  owner        the player within ATTACH m of the ball, held for at least MIN_RUN frames
  pass         the ball leaves one owner and reaches a different player of the same team within MAX_FLIGHT
               seconds, having travelled at least MIN_TRAVEL m
Works in metres on any match (training or prediction); needs only player slots, teams and the ball.
"""
import numpy as np
import pandas as pd

ATTACH = 1.5       # m
MIN_RUN = 3        # frames
MIN_TRAVEL = 4.0   # m
MAX_FLIGHT = 4.0   # s
WINDOW = 8.0       # s: centred window for pass counts
CARRY_SPEED = 2.0  # m/s: an owned ball moving at least this fast is being carried
DUEL = 2.0         # m: players of both teams this close to the ball are contesting it
HALF = 4.0         # s: one-sided windows
# Contacts (--contacts): a sudden change of the ball's velocity next to a player is a touch by him (Link & Hoernig
# 2017, Vidal-Codina et al. 2022). The challenge counts a change of possession only on CONTROL, never on a bare
# touch or a deflection, so each touch is judged by what follows: the toucher keeps the ball (HOLD) or the next
# touch is a teammate's (a completed pass). The model learns from the training labels which touches flip possession.
KICK_WIN = 5       # frames before / after a touch over which the ball's velocity is measured
KICK_DV = 5.0      # m/s: change of velocity that counts as a touch
REACH = 2.0        # m: the toucher is the nearest player within this distance
HOLD = 0.4         # s the toucher must keep the ball (owners) within the next second
NEXT_TOUCH = 4.0   # s: a teammate's next touch within this is a completed pass


def owners(players, ball):
    """Per-frame owning slot (-1 = nobody close), with flicker shorter than MIN_RUN removed."""
    with np.errstate(invalid="ignore"):
        d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    own = np.where(d.min(1) < ATTACH, d.argmin(1), -1)
    change = np.flatnonzero(np.diff(np.r_[-2, own, -2]) != 0)
    for a, b in zip(change[:-1], change[1:]):
        if b - a < MIN_RUN:
            own[a:b] = -1
    return own


def pass_frames(players, team, ball, fps):
    """(frame of reception, receiving team) for every completed pass."""
    own = owners(players, ball)
    change = np.flatnonzero(np.diff(np.r_[-2, own, -2]) != 0)
    runs = [(a, b - 1, own[a]) for a, b in zip(change[:-1], change[1:]) if own[a] >= 0]
    team = np.asarray(team)
    out = []
    for (a1, e1, s1), (a2, e2, s2) in zip(runs[:-1], runs[1:]):
        t1 = team[e1, s1] if team.ndim == 2 else team[s1]
        t2 = team[a2, s2] if team.ndim == 2 else team[s2]
        if s1 == s2 or t1 != t2 or a2 - e1 > MAX_FLIGHT * fps:
            continue
        if np.hypot(*(ball[a2] - ball[e1])) >= MIN_TRAVEL:
            out.append((a2, int(t2)))
    return out


def touches(players, team, ball, fps):
    """(frame, slot, team, controlled) for every touch: a local peak of the ball's change of velocity above
    KICK_DV with a player within REACH."""
    n, k = len(ball), KICK_WIN
    dv = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        before = (ball[k:n - k] - ball[:n - 2 * k]) * fps / k
        after = (ball[2 * k:] - ball[k:n - k]) * fps / k
        dv[k:n - k] = np.hypot(*(after - before).T)
    ok = np.nan_to_num(dv) > KICK_DV
    peak = ok & (np.nan_to_num(dv) >= pd.Series(np.nan_to_num(dv)).rolling(2 * k + 1, center=True, min_periods=1).max().to_numpy())
    tm = np.asarray(team)
    own = owners(players, ball)
    out = []
    for f in np.flatnonzero(peak):
        with np.errstate(invalid="ignore"):
            d = np.hypot(players[f, :, 0] - ball[f, 0], players[f, :, 1] - ball[f, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        s = int(d.argmin())
        if d[s] < REACH:
            out.append([int(f), s, int(tm[f, s] if tm.ndim == 2 else tm[s]), False])
    for i, (f, s, t, _) in enumerate(out):
        held = (own[f:f + fps] == s).sum() >= HOLD * fps
        nxt = next((o for o in out[i + 1:] if o[1] != s or o[0] - f > fps // 3), None)
        passed = nxt is not None and nxt[0] - f <= NEXT_TOUCH * fps and nxt[2] == t
        out[i][3] = bool(held or passed)
    return out


def contact_features(players, team, ball, fps):
    """Last / next controlled touch (its team as +1 home / -1 away, seconds since / until), the last uncontrolled
    touch, and the last / next change of the controlling team."""
    n = len(ball)
    ev = touches(players, team, ball, fps)
    f = {}
    pos = np.arange(n)

    def since_until(idx):
        if not len(idx):
            return np.full(n, 10.0, np.float32), np.full(n, 10.0, np.float32), np.full(n, -1), np.full(n, -1)
        k = np.searchsorted(idx, pos, side="right") - 1
        k2 = np.searchsorted(idx, pos, side="left")
        since = np.where(k >= 0, pos - idx[np.maximum(k, 0)], 10 ** 9)
        until = np.where(k2 < len(idx), idx[np.minimum(k2, len(idx) - 1)] - pos, 10 ** 9)
        return (np.minimum(since / fps, 10.0).astype(np.float32), np.minimum(until / fps, 10.0).astype(np.float32),
                np.where(k >= 0, k, -1), np.where(k2 < len(idx), k2, -1))

    sign = lambda t: 1.0 if t == 0 else -1.0
    ctl = [(fr, t) for fr, _, t, c in ev if c]
    unc = [(fr, t) for fr, _, t, c in ev if not c]
    for name, lst in (("ctl", ctl), ("unc", unc)):
        idx = np.array([fr for fr, _ in lst], dtype=np.int64)
        tsign = np.array([sign(t) for _, t in lst], np.float32)
        since, until, kp, kn = since_until(idx)
        f[f"{name}_since"], f[f"{name}_until"] = since, until
        f[f"{name}_last"] = np.where(kp >= 0, tsign[np.maximum(kp, 0)] if len(lst) else 0, 0).astype(np.float32)
        f[f"{name}_next"] = np.where(kn >= 0, tsign[np.maximum(kn, 0)] if len(lst) else 0, 0).astype(np.float32)
    switch = np.array([ctl[i][0] for i in range(1, len(ctl)) if ctl[i][1] != ctl[i - 1][1]], dtype=np.int64)
    f["ctl_switch_since"], f["ctl_switch_until"], _, _ = since_until(switch)
    return f


def features(players, team, ball, fps, contacts=False):
    """Pass counts around each frame, per team and together, plus time since / until a pass."""
    n = len(ball)
    ev = pass_frames(players, team, ball, fps)
    f = {}
    marks = {}
    for t, name in ((0, "h"), (1, "a"), (None, "all")):
        m = np.zeros(n, np.float32)
        for fr, tt in ev:
            if t is None or tt == t:
                m[fr] = 1
        marks[name] = m
        f[f"pass_{name}_c"] = pd.Series(m).rolling(int(WINDOW * fps), center=True, min_periods=1).sum() \
            .to_numpy(np.float32)
    m = marks["all"]
    w = int(HALF * fps)
    f["pass_past"] = pd.Series(m).rolling(w, min_periods=1).sum().to_numpy(np.float32)
    f["pass_next"] = pd.Series(m[::-1]).rolling(w, min_periods=1).sum().to_numpy(np.float32)[::-1].copy()
    idx = np.flatnonzero(m)
    pos = np.arange(n)
    if len(idx):
        k = np.searchsorted(idx, pos, side="right") - 1
        since = np.where(k >= 0, pos - idx[np.maximum(k, 0)], 10 ** 9)
        k2 = np.searchsorted(idx, pos, side="left")
        until = np.where(k2 < len(idx), idx[np.minimum(k2, len(idx) - 1)] - pos, 10 ** 9)
    else:
        since = until = np.full(n, 10 ** 9)
    f["pass_since"] = np.minimum(since / fps, 30.0).astype(np.float32)
    f["pass_until"] = np.minimum(until / fps, 30.0).astype(np.float32)

    # a player running with the ball, and both teams fighting for it: both mean the ball is in play
    own = owners(players, ball)
    v = np.full(n, np.nan, np.float32)
    v[2:-2] = np.hypot(*(ball[4:] - ball[:-4]).T) * fps / 4
    with np.errstate(invalid="ignore"):
        carry = ((own >= 0) & (v > CARRY_SPEED)).astype(np.float32)
        d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    tm = np.asarray(team)
    tm = tm if tm.ndim == 2 else np.broadcast_to(tm, d.shape)
    duel = (((d < DUEL) & (tm == 0)).any(1) & ((d < DUEL) & (tm == 1)).any(1)).astype(np.float32)
    for key, x in (("carry", carry), ("duel", duel)):
        for w, tag in ((fps, "s"), (3 * fps, "m")):
            f[f"pass_{key}_{tag}"] = pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32)

    # who owns the ball: the team of the player on it, carried while nobody is (a loose ball stays with the last
    # team), how long that team has kept it, and how long ago it changed hands
    own_team = np.full(n, -1)
    has = own >= 0
    own_team[has] = (tm[np.flatnonzero(has), own[has]]).astype(int)
    last = pd.Series(np.where(own_team >= 0, own_team, np.nan)).ffill().to_numpy()
    sign = np.where(last == 0, 1.0, np.where(last == 1, -1.0, 0.0)).astype(np.float32)
    change = np.r_[False, (last[1:] != last[:-1]) & np.isfinite(last[1:]) & np.isfinite(last[:-1])]
    since = np.zeros(n, np.float32)
    run = 10 ** 6
    for i in range(n):
        run = 0 if change[i] else run + 1
        since[i] = run
    f["own_team"] = sign
    f["own_since"] = np.minimum(since / fps, 10.0).astype(np.float32)
    nxt = np.zeros(n, np.float32)
    run = 10 ** 6
    for i in range(n - 1, -1, -1):
        run = 0 if change[i] else run + 1
        nxt[i] = run
    f["own_until"] = np.minimum(nxt / fps, 10.0).astype(np.float32)
    if contacts:
        f.update(contact_features(players, team, ball, fps))
    return f
