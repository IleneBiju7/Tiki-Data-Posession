"""Smart Tagging as features: which team started a phase most recently, and when set pieces start.

Real tags (provided game): Metrica's automatic Smart Tagging XML. Phase codes are mapped to the team
that has the ball: attacking codes keep their team; DEFENDING ... and DEFENSIVE TRANSITION flip to
the other team. Mirrored pairs (e.g. BUILD UP / DEFENDING IN ATTACKING THIRD) collapse to one event.

Simulated tags (open training matches, which have none): drawn from the known labels with heavy,
deliberate noise -- missed tags, start jitter, spurious tags -- so the model learns to use them as a
weak hint, not an answer. No statistic of the provided game's ground truth is used.

Per-frame features:
  st_last_team   +1 home / -1 away for the most recent phase start within 30 s, else 0
  st_since       seconds since that phase start (capped)
  st_next_team   team of the next phase start within 30 s (offline: tags are known for the whole match)
  st_until       seconds until it (capped)
  st_sp_since    seconds since the last set-piece start (capped)
  st_sp_until    seconds until the next set-piece start (capped)
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

CAP = 30.0
SP_CAP = 40.0
ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
          "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
          "DEFENSIVE TRANSITION"}


def real_events(path, home_name, fps=25):
    """(phase_starts [(frame, team)], set_piece_starts [frame]) from the Smart Tagging XML."""
    phases, sps = set(), set()
    for inst in ET.parse(path).getroot().iter("instance"):
        code = inst.findtext("code")
        team = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}.get("Team", "N/A")
        start = round(float(inst.findtext("start")) * fps)
        if code == "SET PIECES":
            sps.add(start)
        elif team not in ("N/A", None) and (code in ATTACK or code in DEFEND):
            side = 0 if team == home_name else 1
            if code in DEFEND:
                side = 1 - side
            phases.add((start, side))
    return sorted(phases), sorted(sps)


NOISE = {"normal": dict(p_first=0.8, p_extra=0.7, jitter=1.5, fake_per_min=1 / 3),
         "heavy": dict(p_first=0.5, p_extra=0.4, jitter=4.0, fake_per_min=1.0)}



def simulate_events(label, period, fps, rng, noise="normal", sp_rng=None):
    """Noisy Smart-Tagging-like events from known H/A/D labels of an open training match.
    sp_rng: when given, set-piece tags and tag rates are drawn from it, from wide ranges per copy."""
    cfg = dict(NOISE[noise])
    if sp_rng is not None:
        if noise == "normal":
            cfg.update(p_first=sp_rng.uniform(0.5, 0.95), p_extra=sp_rng.uniform(0.4, 0.9),
                       jitter=sp_rng.uniform(1.0, 4.0), fake_per_min=sp_rng.uniform(0.1, 1.0))
        sp_share, sp_lead = sp_rng.uniform(0.3, 0.8), (1.0, 10.0)
    phases, sps = [], []
    n = len(label)
    live = (label < 2) & (period > 0)
    runs = np.flatnonzero(np.diff(np.r_[-1, label, -1]) != 0)
    for a, b in zip(runs[:-1], runs[1:]):
        lab, dur = label[a], (b - a) / fps
        if lab in (0, 1) and dur >= 3:
            if rng.random() < cfg["p_first"]:                         # most possessions get a phase tag
                phases.append((int(a + rng.normal(0, cfg["jitter"]) * fps), int(lab)))
            t = a + rng.uniform(6, 14) * fps
            while t < b:                                              # long possessions: more phases
                if rng.random() < cfg["p_extra"]:
                    phases.append((int(t + rng.normal(0, cfg["jitter"]) * fps), int(lab)))
                t += rng.uniform(6, 14) * fps
        if lab == 2 and dur >= 4 and b < n and rng.random() < 0.3:    # some restarts get a set piece
            lead = rng.uniform(2, 5)
            if sp_rng is None:
                sps.append(int(b - lead * fps))
        if sp_rng is not None and lab == 2 and dur >= 4 and b < n and sp_rng.random() < sp_share:
            sps.append(int(b - min(sp_rng.uniform(*sp_lead), dur - 0.5) * fps))
    alive = np.flatnonzero(live)
    if len(alive):
        n_fake = int(len(alive) / fps / 60 * cfg["fake_per_min"])     # spurious tags
        for f in rng.choice(alive, n_fake, replace=False):
            phases.append((int(f), int(rng.integers(0, 2))))
    phases = sorted((min(max(f, 0), n - 1), t) for f, t in phases)
    return phases, sorted(min(max(f, 0), n - 1) for f in sps)


def features(phases, sps, n, fps=25):
    f = {}
    team = np.zeros(n, np.float32)
    since = np.full(n, CAP, np.float32)
    nteam = np.zeros(n, np.float32)
    until = np.full(n, CAP, np.float32)
    for start, side in phases:
        sign = 1.0 if side == 0 else -1.0
        hi = min(n, start + int(CAP * fps))
        span = np.arange(start, hi)
        newer = since[span] >= (span - start) / fps
        team[span[newer]] = sign
        since[span[newer]] = (span[newer] - start) / fps
        lo = max(0, start - int(CAP * fps))
        span = np.arange(lo, start)
        nearer = until[span] >= (start - span) / fps
        nteam[span[nearer]] = sign
        until[span[nearer]] = (start - span[nearer]) / fps
    f["st_last_team"], f["st_since"], f["st_next_team"], f["st_until"] = team, since, nteam, until
    sp_since = np.full(n, SP_CAP, np.float32)
    sp_until = np.full(n, SP_CAP, np.float32)
    for start in sps:
        hi = min(n, start + int(SP_CAP * fps))
        span = np.arange(start, hi)
        sp_since[span] = np.minimum(sp_since[span], (span - start) / fps)
        lo = max(0, start - int(SP_CAP * fps))
        span = np.arange(lo, start)
        sp_until[span] = np.minimum(sp_until[span], (start - span) / fps)
    f["st_sp_since"], f["st_sp_until"] = sp_since, sp_until
    return f
