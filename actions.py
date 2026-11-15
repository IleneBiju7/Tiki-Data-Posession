"""Ball actions with the team that made them, as second-stage inputs (the idea behind T-DEED's team ball-action
spotting on the broadcast video).

Open matches: the human event data of other games (DFL / IDSSE open data, CC BY 4.0, through kloppy; Metrica sample
games) gives every on-ball action with its team and time: passes, shots, recoveries, duels and other ball actions.
simulate() then makes them look like a video model's output: some missed, some invented, some given to the wrong
team, timing jittered. A ceiling test (all actions, exact) tells whether building the video model is worth it.

Columns per frame (COLUMNS): seconds since the last / until the next action of each team (capped at CAP), and which
team made the last / next action (0 home, 1 away, 0.5 none within CAP).
Usage: python actions.py   -> models/actions_open_<variant>.npz for the 10 held-out open matches (exact / realistic)
"""
import numpy as np

import train

CAP = 10.0
COLUMNS = ["ac_since_h", "ac_until_h", "ac_since_a", "ac_until_a", "ac_last_team", "ac_next_team"]
ON_BALL = {"PASS", "SHOT", "RECOVERY", "GENERIC", "CLEARANCE", "INTERCEPTION", "TAKE_ON", "CARRY", "DUEL"}
METRICA_ON_BALL = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "CHALLENGE", "SET PIECE", "BALL OUT"}
REALISTIC = dict(recall=0.6, false_per_min=1.0, team_acc=0.9, jitter=0.3)   # rough video-model behaviour


def _dfl_events(match_id, period, label):
    from kloppy import sportec
    ev = sportec.load_open_event_data(match_id=match_id)
    starts = {p: int(np.flatnonzero(period == p)[0]) for p in (1, 2)}
    home = ev.metadata.teams[0].team_id
    frames, teams = [], []
    for e in ev.events:
        if e.team is None or e.event_type.name not in ON_BALL or e.period.id not in starts:
            continue
        frames.append(starts[e.period.id] + int(round(e.timestamp.total_seconds() * 25)))
        teams.append(0 if e.team.team_id == home else 1)
    return np.array(frames), np.array(teams)


def _metrica_events(game):
    import pandas as pd
    base = next(b for b in (train.ROOT / "data" / "metrica-sample-data", train.ROOT.parent / "data" / "metrica-sample-data")
                if b.exists())   # the main folder, or the rebuild folder next to it
    root = base / "data" / f"Sample_Game_{game}"
    if game in (1, 2):
        d = pd.read_csv(root / f"Sample_Game_{game}_RawEventsData.csv")
        d = d[d["Type"].str.upper().isin(METRICA_ON_BALL)]
        return d["Start Frame"].to_numpy() - 1, (d["Team"] != "Home").astype(int).to_numpy()
    from kloppy import metrica
    ev = metrica.load_event(event_data=str(root / f"Sample_Game_{game}_events.json"),
                            meta_data=str(root / f"Sample_Game_{game}_metadata.xml"))
    home = ev.metadata.teams[0].team_id
    frames, teams = [], []
    for e in ev.events:
        raw = getattr(e, "raw_event", None) or {}
        f = raw.get("start", {}).get("frame") if isinstance(raw, dict) else None
        if e.team is None or f is None or e.event_type.name not in ON_BALL | {"BALL_OUT"}:
            continue
        frames.append(int(f) - 1)
        teams.append(0 if e.team.team_id == home else 1)
    return np.array(frames), np.array(teams)


def open_events(name, period, label):
    """(frames, teams) of the on-ball actions of an open match, in its converted frame numbering."""
    if name.startswith("dfl_"):
        return _dfl_events(name[4:], period, label)
    return _metrica_events(int(name.split("_")[1]))


def simulate(frames, teams, n, fps, rng, recall=1.0, false_per_min=0.0, team_acc=1.0, jitter=0.0):
    keep = rng.random(len(frames)) < recall
    f = frames[keep] + np.round(rng.normal(0, jitter, keep.sum()) * fps).astype(int)
    t = np.where(rng.random(keep.sum()) < team_acc, teams[keep], 1 - teams[keep])
    n_false = rng.poisson(false_per_min * n / fps / 60)
    f = np.r_[f, rng.integers(0, n, n_false)]
    t = np.r_[t, rng.integers(0, 2, n_false)]
    order = np.argsort(f)
    return np.clip(f[order], 0, n - 1), t[order]


def columns(frames, teams, n, fps):
    idx = np.arange(n)
    out = np.zeros((n, len(COLUMNS)), np.float32)
    for k, team in enumerate((0, 1)):
        w = np.sort(frames[teams == team])
        i = np.searchsorted(w, idx, side="right")
        out[:, 2 * k] = np.where(i > 0, (idx - w[np.maximum(i - 1, 0)]) / fps, CAP) if len(w) else CAP
        j = np.searchsorted(w, idx, side="left")
        out[:, 2 * k + 1] = np.where(j < len(w), (w[np.minimum(j, len(w) - 1)] - idx) / fps, CAP) if len(w) else CAP
    out[:, :4] = np.minimum(out[:, :4], CAP)
    last = np.where(out[:, 0] < out[:, 2], 0.0, 1.0)
    last[(out[:, 0] >= CAP) & (out[:, 2] >= CAP)] = 0.5
    nxt = np.where(out[:, 1] < out[:, 3], 0.0, 1.0)
    nxt[(out[:, 1] >= CAP) & (out[:, 3] >= CAP)] = 0.5
    out[:, 4], out[:, 5] = last, nxt
    return out


def main():
    names = [n for n in train.load_matches(False) if not n.startswith(train.TRAIN_ONLY)]
    variants = {"exact": dict(), "realistic": REALISTIC}
    res = {v: {} for v in variants}
    for name in names:
        z = np.load(train._path(name))
        lab, period, fps = z["label"], z["period"], int(z["fps"])
        frames, teams = open_events(name, period, lab)
        ok = (frames >= 0) & (frames < len(lab))
        frames, teams = frames[ok], teams[ok]
        live = lab[frames] < 2
        agree = np.mean(lab[frames][live] == teams[live]) if live.any() else float("nan")
        print(f"{name}: {len(frames)} actions, {np.mean(live):.0%} in live play, team = labelled team on {agree:.0%}", flush=True)
        for v, kw in variants.items():
            f, t = simulate(frames, teams, len(lab), fps, np.random.default_rng(7), **kw)
            res[v][name] = columns(f, t, len(lab), fps)
    for v in variants:
        np.savez_compressed(train.MODELS / f"actions_open_{v}.npz", columns=np.array(COLUMNS), **res[v])
        print("->", train.MODELS / f"actions_open_{v}.npz")


if __name__ == "__main__":
    main()
