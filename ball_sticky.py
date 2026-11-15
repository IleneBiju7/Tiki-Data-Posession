"""Keep the ball with its carrier while it is hidden.

A ball that vanishes next to a player is almost always at that player's feet, hidden by legs or a
body. Instead of leaving a gap (or letting it jump to a spurious detection), carry it with that
player until it reappears:

  carrier       the player within ATTACH m of the ball on the last frame it was seen, provided the
                ball was moving with them (< CONTROL_SPEED m/s relative to the player)
  gap fill      while the ball is missing, for at most MAX_GAP seconds, ball = carrier's position;
                stops early if the carrier's track disappears
  reachability  a sighting more than LEAVE m from the player who has the ball, or (with no carrier) one
                the ball could not physically reach (> MAX_SPEED from its last accepted position), is
                ignored unless consistent sightings there last CONFIRM_SPAN seconds, and (when raw
                detections are given) the detector is not more confident about something at the
                carrier's feet over the same moments
  glitch fix    a jump of more than JUMP m away from the carrier that lasts less than GLITCH seconds
                and comes back within ATTACH m of the carrier is replaced by the carrier's position

  trust         (when the provided tracker ball is given) a sighting is trusted if the tracker agrees with
                it, or if it is part of a consistent track lasting TRUST_RUN seconds. Untrusted sightings --
                short detector-only blips, typically a player's boot during a stoppage -- are dropped, so
                they can neither start a carry nor move the ball. A jump the tracker does not confirm needs
                CONFIRM_NO_TRK seconds of consistent sightings
  keeper hold   (when teams are given) a ball that disappears with a goalkeeper inside his own penalty area
                stays with him for up to KEEPER_HOLD seconds (Law 12: he must release it within 8 s)

Players are anonymous track slots; a carrier is followed by slot, so the fill stops if the slot ends.
Works in metres on any match (training or prediction).
"""
import os

import numpy as np

ATTACH = 1.5
CONTROL_SPEED = 6.0
MAX_GAP = float(os.environ.get("STICKY_GAP", 2.0))   # s; STICKY_GAP only for experiments
JUMP = 6.0
GLITCH = 0.4
MAX_SPEED = 35.0     # m/s: fastest a ball travels
MARGIN = 3.0         # m: position noise allowance
LEAVE = 4.0          # m: while a player has the ball, a sighting further than this from them needs confirming
CONFIRM_SPAN = 0.4   # s: an "impossible" jump is accepted only if consistent sightings there last this long
MATCH = 1.0          # m: a detection this close to a ball sighting is that sighting's source
RUN_GAP = 4          # frames: largest hole allowed inside a confirming run
TRUST_RUN = 0.6      # s: a detector-only track at least this long is trusted
TRACKER_AGREE = 2.0  # m: the tracker confirms a sighting this close to its own ball
CONFIRM_NO_TRK = 1.0 # s: an "impossible" jump the tracker does not confirm needs a run this long
KEEPER_HOLD = float(os.environ.get("STICKY_KEEPER", 8.0))    # s: Law 12, the goalkeeper must release the ball within 8 seconds
BOX_X, BOX_Y = 16.5, 20.16


def _carrier(ball_xy, players_f):
    d = np.hypot(players_f[:, 0] - ball_xy[0], players_f[:, 1] - ball_xy[1])
    if not np.isfinite(d).any():
        return -1
    s = int(np.nanargmin(d))
    return s if d[s] < ATTACH else -1


def trusted(ball, trk, fps=25):
    """Sightings confirmed by the tracker, or part of a consistent track lasting TRUST_RUN seconds."""
    seen = np.isfinite(ball[:, 0])
    with np.errstate(invalid="ignore"):
        agree = seen & np.isfinite(trk[:, 0]) & (np.hypot(*(ball - trk).T) < TRACKER_AGREE)
    out = np.zeros(len(ball), bool)
    idx = np.flatnonzero(seen)
    if not len(idx):
        return out, agree
    cut = np.flatnonzero(np.diff(idx) > 1 + 2) + 1          # holes of up to 2 frames stay inside a run
    for run in np.split(idx, cut):
        a, b = run[0], run[-1]
        step = np.hypot(*np.diff(ball[run], axis=0).T) * fps / np.maximum(np.diff(run), 1)
        consistent = not len(step) or np.all(step <= MAX_SPEED + MARGIN * fps / 4)
        if agree[a:b + 1].any() or (consistent and b - a + 1 >= TRUST_RUN * fps):
            out[run] = True
    return out, agree


def _keeper(players, team, f, s, length, width):
    """True when slot s is his team's deepest player and stands inside the penalty area at that end."""
    x, y = players[f, s]
    if not (np.isfinite(x) and np.isfinite(y)) or abs(y - width / 2) > BOX_Y:
        return False
    end = 0.0 if x < length / 2 else length
    if abs(x - end) > BOX_X:
        return False
    mates = np.flatnonzero(team == team[s])
    depth = np.abs(players[f, mates, 0] - end)
    depth = np.where(np.isfinite(depth), depth, np.inf)
    return mates[int(depth.argmin())] == s


def _conf_near(dets, f, xy, radius):
    """Highest detector confidence within radius of xy on frame f (the detector runs on every 2nd frame)."""
    best = 0.0
    for g in (f - 1, f, f + 1):
        for x, y, c in dets.get(g, ()):
            if np.hypot(x - xy[0], y - xy[1]) < radius:
                best = max(best, c)
    return best


def _carrier_wins(run, ball, players, carrier, dets):
    """True when the detector sees the ball at the carrier's feet more confidently than the run's own sightings."""
    if dets is None or carrier < 0:
        return False
    run_conf, carrier_conf = 0.0, 0.0
    for v in range(run[0], run[-1] + 1):
        if np.isfinite(ball[v, 0]) and v in run:
            c = _conf_near(dets, v, ball[v], MATCH)
            run_conf = max(run_conf, c if c > 0 else 1.0)   # no matching detection: came from the tracker, trust it
        if np.isfinite(players[v, carrier, 0]):
            carrier_conf = max(carrier_conf, _conf_near(dets, v, players[v, carrier], ATTACH))
    return carrier_conf > run_conf


def enforce_reachable(ball, players, fps=25, dets=None, agree=None):
    """Reject sightings the ball could not physically reach from its last accepted position, unless a run
    of mutually-consistent sightings lasting CONFIRM_SPAN says the ball really is there (our reference was
    wrong). While a sighting is rejected the ball stays with its carrier if there is one, else unknown."""
    ball = ball.copy()
    n = len(ball)
    obs = np.flatnonzero(np.isfinite(ball[:, 0]))
    if not len(obs):
        return ball, np.zeros(n, bool)
    fixed = np.zeros(n, bool)
    ref_t, ref = obs[0], ball[obs[0]].copy()
    carrier = _carrier(ref, players[ref_t])
    k = 1
    while k < len(obs):
        t = obs[k]
        held = carrier >= 0 and np.isfinite(players[t, carrier, 0])
        here = players[t, carrier] if held else ref
        limit = LEAVE if held else MARGIN + MAX_SPEED * (t - ref_t) / fps
        if np.hypot(*(ball[t] - here)) <= limit:
            ref_t, ref = t, ball[t].copy()
            carrier = _carrier(ref, players[t])
            k += 1
            continue
        # impossible from here: is it the start of a consistent run elsewhere?
        run = [t]
        j = k + 1
        span = int(CONFIRM_SPAN * fps)
        while j < len(obs):                   # follow the whole consistent run
            u, v = run[-1], obs[j]
            if np.hypot(*(ball[v] - ball[u])) <= MARGIN + MAX_SPEED * (v - u) / fps and v - u <= RUN_GAP:
                run.append(v)
                j += 1
            else:
                break
        need = span if agree is None or agree[run[0]:run[-1] + 1].any() else int(CONFIRM_NO_TRK * fps)
        if run[-1] - run[0] >= need and not _carrier_wins(run, ball, players, carrier, dets):
            ref_t, ref = run[-1], ball[run[-1]].copy()
            carrier = _carrier(ref, players[ref_t])
            k = j
            continue
        if carrier >= 0 and np.isfinite(players[t, carrier, 0]):
            ball[t] = players[t, carrier]
        else:
            ball[t] = np.nan
        fixed[t] = True
        k += 1
    return ball, fixed


def stick(ball, players, fps=25, dets=None, trk=None, team=None, length=105.0, width=68.0):
    """Returns (ball_filled, filled_mask). dets: optional {frame: [(x_m, y_m, conf), ...]} raw detections.
    trk: optional provided-tracker ball (metres) -> trust rules; team: optional slot teams -> keeper hold."""
    ball = ball.astype(np.float64).copy()
    n = len(ball)
    agree = None
    if trk is not None:
        ok, agree = trusted(ball, trk, fps)
        ball[~ok] = np.nan
    ball, filled = enforce_reachable(ball, players, fps, dets, agree)
    seen = np.isfinite(ball[:, 0])
    max_gap = int(MAX_GAP * fps)
    glitch = int(GLITCH * fps)

    # 1) glitch fix: short excursions away from a carrier that come back to it
    i = 1
    while i < n - 1:
        if seen[i - 1] and seen[i]:
            s = _carrier(ball[i - 1], players[i - 1])
            if s >= 0 and np.isfinite(players[i, s, 0]) and np.hypot(*(ball[i] - players[i, s])) > JUMP:
                j = i
                while j < min(n, i + glitch) and seen[j] and np.isfinite(players[j, s, 0]) \
                        and np.hypot(*(ball[j] - players[j, s])) > JUMP:
                    j += 1
                if j < n and j < i + glitch and seen[j] and np.isfinite(players[j, s, 0]) \
                        and np.hypot(*(ball[j] - players[j, s])) < ATTACH:
                    ball[i:j] = players[i:j, s]
                    filled[i:j] = True
                    i = j
                    continue
        i += 1

    # 2) gap fill: carry the ball with its carrier while it is hidden
    seen = np.isfinite(ball[:, 0])
    starts = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    for a in starts:
        last = a - 1
        s = _carrier(ball[last], players[last])
        if s < 0:
            continue
        rel = ball[last] - ball[last - 1] - (players[last, s] - players[last - 1, s])
        if not np.isfinite(rel).all():
            rel = ball[last] - ball[last - 1]
        if not np.isfinite(rel).all() or np.hypot(*rel) * fps > CONTROL_SPEED:
            continue                          # ball moving away from its player (pass/shot), not at their feet
        hold = max_gap
        if team is not None and _keeper(players, np.asarray(team), last, s, length, width):
            hold = int(KEEPER_HOLD * fps)
        b = a
        while b < n and b - a < hold and not seen[b] and np.isfinite(players[b, s, 0]):
            b += 1
        ball[a:b] = players[a:b, s]
        filled[a:b] = True
    return ball, filled


def main():
    """BAR-ATH: fused ball -> sticky ball (cache/BAR-ATH_ball_fused_v6s.npy), using the raw detections as tie-break.
    python ball_sticky.py [fused tag] [detections tag]  e.g.  v7 v3  ->  BAR-ATH_ball_fused_v7s.npy"""
    import sys
    import pandas as pd
    import atd_cache
    import tracks
    fused_tag = sys.argv[1] if len(sys.argv) > 1 else "v6"
    det_name = f"ball_detections_{sys.argv[2]}.csv.gz" if len(sys.argv) > 2 else "ball_detections_v2.csv.gz"
    pitch = np.array((105.3, 68.0))
    cache = atd_cache.DATA_DIR / "cache"
    players = tracks.drop_drifting(tracks.drop_frozen(atd_cache.load()["players"]) * pitch)
    fused = np.load(cache / f"BAR-ATH_ball_fused_{fused_tag}.npy") * pitch
    dets = {}
    det = pd.read_csv(cache / det_name)
    for f, x, y, c in det[["frame", "x", "y", "conf"]].itertuples(index=False):
        dets.setdefault(int(f), []).append((x * pitch[0], y * pitch[1], c))
    ball, changed = stick(fused, players, dets=dets)
    np.save(cache / f"BAR-ATH_ball_fused_{fused_tag}s.npy", (ball / pitch).astype(np.float32))
    print(f"sticky ball: {changed.mean():.1%} of frames changed, ball known on {np.isfinite(ball[:, 0]).mean():.1%}")


if __name__ == "__main__":
    main()
