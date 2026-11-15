"""Convert open training matches (other games only) into one common .npz format.

Common format, 25 fps where the source allows it:
  players: (n, slots, 2) metres, x along the pitch length, NaN where absent
  team:    (slots,) 0 = home, 1 = away
  ball:    (n, 2) metres, NaN where absent
  label:   (n,) 0 = H (home in possession, ball alive), 1 = A, 2 = D (dead), 3 = unknown
  period:  (n,) 1 or 2, 0 outside play
  length, width, fps, source

Sources:
  dfl          DFL / IDSSE open data (Bassek et al. 2025, CC BY 4.0), labels from the
               corrected tracking's ball status + ball possession flags
  skillcorner  SkillCorner open data (MIT), broadcast tracking resampled 10 -> 25 fps; labels from its
               phases of play (team in possession), and dead ball between phases (phases only exist
               while the ball is in play)
  metrica      Metrica sample data games 1-3 (game 3 in EPTS, like the competition files), labels
               derived from its event data
  pff          PFF FC FIFA World Cup 2022, 64 matches (broadcast tracking by Sportlogiq + PFF events),
               resampled 29.97 -> 25 fps; raw (not event-smoothed) players and ball; labels from the
               events: team of the player on the ball, dead from "ball out" or a foul until the tagged
               restart (kick-off, free kick, goal kick, throw-in, corner, penalty). Written to
               converted_pff/ (training-only matches, used with train.py --pff)

Usage: python convert_open.py dfl|skillcorner|metrica|pff [game ids...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "open" / "converted"
DFL_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def save(name: str, **arrays) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    lab = arrays["label"]
    print(f"{name}: {len(lab)} frames, H {np.mean(lab == 0):.2f} A {np.mean(lab == 1):.2f} "
          f"D {np.mean(lab == 2):.2f} unknown {np.mean(lab == 3):.2f}", flush=True)


def convert_dfl() -> None:
    from kloppy import sportec
    from kloppy.domain import BallState

    for match_id in DFL_MATCHES:
        if (OUT / f"dfl_{match_id}.npz").exists():
            continue
        ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
        length = ds.metadata.pitch_dimensions.pitch_length
        width = ds.metadata.pitch_dimensions.pitch_width
        home = ds.metadata.teams[0]
        slot_of, team = {}, []
        for t_idx, t in enumerate(ds.metadata.teams):
            for p in t.players:
                slot_of[p.player_id] = len(team)
                team.append(t_idx)
        n = len(ds.records)
        players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        ball_z = np.full(n, np.nan, dtype=np.float32)   # height in metres (the corrected tracking has it)
        label = np.full(n, 3, dtype=np.int8)
        period = np.zeros(n, dtype=np.int8)
        for i, f in enumerate(ds.records):
            period[i] = f.period.id
            if f.ball_coordinates is not None and f.ball_coordinates.x is not None:
                ball[i] = (f.ball_coordinates.x * length, f.ball_coordinates.y * width)
                if getattr(f.ball_coordinates, "z", None) is not None:
                    ball_z[i] = f.ball_coordinates.z
            for p, d in f.players_data.items():
                if d.coordinates is not None and p.player_id in slot_of:
                    players[i, slot_of[p.player_id]] = (d.coordinates.x * length, d.coordinates.y * width)
            if f.ball_state == BallState.DEAD:
                label[i] = 2
            elif f.ball_owning_team is not None:
                label[i] = 0 if f.ball_owning_team == home else 1
        save(f"dfl_{match_id}", players=players, team=np.array(team, np.int8), ball=ball, ball_z=ball_z, label=label,
             period=period, length=length, width=width, fps=ds.metadata.frame_rate, source="dfl")


def convert_skillcorner() -> None:
    base = ROOT / "data" / "open" / "skillcorner" / "data" / "matches"
    for folder in sorted(base.iterdir()):
        mid = folder.name
        meta = json.loads((folder / f"{mid}_match.json").read_text())
        length, width = meta["pitch_length"], meta["pitch_width"]
        home_id = meta["home_team"]["id"]
        team_of = {p["id"]: 0 if p["team_id"] == home_id else 1 for p in meta["players"]}
        slot_of = {pid: i for i, pid in enumerate(team_of)}
        team = np.array(list(team_of.values()), np.int8)
        rows = [json.loads(line) for line in open(folder / f"{mid}_tracking_extrapolated.jsonl")]
        n = len(rows)
        players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        label = np.full(n, 3, dtype=np.int8)
        period = np.zeros(n, dtype=np.int8)
        for i, r in enumerate(rows):
            if not r.get("period"):
                label[i] = 3
                continue
            period[i] = r["period"]
            b = r["ball_data"]
            # SkillCorner centres the pitch on (0, 0); shift to a corner origin
            if b.get("x") is not None and b.get("is_detected"):
                ball[i] = (b["x"] + length / 2, b["y"] + width / 2)
            for p in r["player_data"]:
                if p["player_id"] in slot_of:  # extrapolated players kept, like a tracker's carry-over
                    players[i, slot_of[p["player_id"]]] = (p["x"] + length / 2, p["y"] + width / 2)
        # phases of play exist only while the ball is in play: inside a phase -> its team, between -> dead
        frame_no = np.array([r["frame"] for r in rows])
        label[period > 0] = 2
        phases = pd.read_csv(folder / f"{mid}_phases_of_play.csv")
        for a, b, tid in phases[["frame_start", "frame_end", "team_in_possession_id"]].itertuples(index=False):
            inside = (frame_no >= a) & (frame_no < b) & (period > 0)
            label[inside] = 0 if tid == home_id else 1
        players, ball, label, period = upsample(players, ball, label, period, 10, 25)
        save(f"skillcorner_{mid}", players=players, team=team, ball=ball, label=label, period=period,
             length=length, width=width, fps=25, source="skillcorner")


def upsample(players, ball, label, period, src_fps, dst_fps):
    """Linear interpolation of positions (only between two known samples), nearest earlier sample for labels."""
    n = len(ball)
    t = np.arange(int(n * dst_fps / src_fps)) * src_fps / dst_fps
    i0 = np.minimum(t.astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (t - i0).astype(np.float32)

    def interp(a):
        w_ = w.reshape((-1,) + (1,) * (a.ndim - 1))
        return a[i0] * (1 - w_) + a[i1] * w_          # NaN if either neighbour is unknown

    return interp(players), interp(ball), label[i0], period[i0]


def convert_metrica() -> None:
    import metrica_labels as pb

    base = ROOT / "data" / "metrica-sample-data" / "data"
    for game in (1, 2):
        meta, ball, (home, away) = pb.load_tracking(base / f"Sample_Game_{game}", game)
        events = pd.read_csv(base / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv")
        ref = pb.reference_labels(events, len(meta))
        label = np.array([{"Home": 0, "Away": 1, "None": 2}[x] for x in ref], np.int8)
        players = np.concatenate([home, away], axis=1).astype(np.float32)
        team = np.array([0] * home.shape[1] + [1] * away.shape[1], np.int8)
        save(f"metrica_{game}", players=players, team=team, ball=ball.astype(np.float32), label=label,
             period=meta["Period"].to_numpy().astype(np.int8), length=pb.PITCH_X, width=pb.PITCH_Y,
             fps=25, source="metrica")

    # Game 3: EPTS tracking (same format as the competition data) + JSON events
    from kloppy import metrica
    g3 = base / "Sample_Game_3"
    ds = metrica.load_tracking_epts(meta_data=str(g3 / "Sample_Game_3_metadata.xml"),
                                    raw_data=str(g3 / "Sample_Game_3_tracking.txt"), coordinates="metrica")
    home_team = ds.metadata.teams[0]
    plist = [p for t in ds.metadata.teams for p in t.players]
    slot = {p.player_id: i for i, p in enumerate(plist)}
    team = np.array([0 if p.team == home_team else 1 for p in plist], np.int8)
    n = len(ds.frames)
    players = np.full((n, len(plist), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    period = np.zeros(n, np.int8)
    for i, fr in enumerate(ds.frames):
        period[i] = fr.period.id
        if fr.ball_coordinates is not None:
            ball[i] = (fr.ball_coordinates.x, fr.ball_coordinates.y)
        for pl, c in fr.players_coordinates.items():
            if c is None:
                continue
            players[i, slot[pl.player_id]] = (c.x, c.y)
    scale = np.array([pb.PITCH_X, pb.PITCH_Y], np.float32)
    players, ball = players * scale, ball * scale
    rows = []
    for e in json.loads((g3 / "Sample_Game_3_events.json").read_text())["data"]:
        sub = e["subtypes"]
        sub = "-".join(x["name"] for x in sub) if isinstance(sub, list) else (sub or {}).get("name", "")
        rows.append({"Team": "Home" if e["team"]["id"] == home_team.team_id else "Away", "Type": e["type"]["name"],
                     "Subtype": sub, "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or 0})
    ref = pb.reference_labels(pd.DataFrame(rows), n)
    label = np.array([{"Home": 0, "Away": 1, "None": 2}[x] for x in ref], np.int8)
    save("metrica_3", players=players, team=team, ball=ball, label=label, period=period,
         length=pb.PITCH_X, width=pb.PITCH_Y, fps=25, source="metrica")


PFF_DIR = ROOT / "data" / "open" / "pff"
PFF_OUT = ROOT / "data" / "open" / "converted_pff"
RESTARTS = {"K", "F", "G", "T", "C", "P"}


def _pff_file(kind, game):
    """The game's file of one kind; the download came in several parts, event data in several versions
    (the top-level copy is the current release, dated subfolders are older ones)."""
    hits = sorted(PFF_DIR.glob(f"*/{kind}/{game}.*")) or sorted(PFF_DIR.glob(f"*/{kind}/*/{game}.*"), reverse=True)
    return hits[0]


def pff_labels(events, t):
    """H/A/D code at each time t (video seconds) from PFF event data."""
    marks, last_end, pending_out = [], None, None
    for e in sorted(events, key=lambda e: (e["startTime"], e["endTime"] or e["startTime"])):
        g = e.get("gameEvents") or {}
        kind, sp = g.get("gameEventType"), g.get("setpieceType")
        start = float(e["startTime"])
        end = float(e["endTime"] if e["endTime"] is not None else start)
        team = 0 if g.get("homeTeam") else 1
        if kind in ("FIRSTKICKOFF", "SECONDKICKOFF"):
            marks.append((start, team))
            last_end, pending_out = max(start, end), None
        elif kind == "OUT":
            pending_out = start if pending_out is None else pending_out
        elif kind == "VID":
            marks.append((start, 3))
        elif kind == "OTB":
            if sp in RESTARTS and last_end is not None:
                if pending_out is not None:
                    dead_from = pending_out
                elif sp in ("F", "P"):                 # foul: dead from the last touch before the whistle
                    dead_from = last_end
                else:                                  # untagged exit: the ball leaves soon after the last touch
                    dead_from = last_end + min(1.0, 0.5 * (start - last_end))
                if start > dead_from:
                    marks.append((dead_from, 2))
            elif pending_out is not None:              # out with no tagged restart
                marks.append((pending_out, 2))
            marks.append((start, team))
            last_end, pending_out = max(start, end), None
    marks.sort(key=lambda m: m[0])
    times = np.array([m[0] for m in marks])
    codes = np.array([m[1] for m in marks] + [3], np.int8)
    k = np.searchsorted(times, t, side="right") - 1
    return np.where(k >= 0, codes[np.maximum(k, 0)], 3).astype(np.int8)


def convert_pff_game(game: str) -> None:
    import bz2
    meta = json.loads(_pff_file("Metadata", game).read_text())
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = (meta.get("stadium") or {}).get("pitches") or [{}]
    length, width = float(pitch[0].get("length") or 105.0), float(pitch[0].get("width") or 68.0)
    events = json.loads(_pff_file("Event Data", game).read_text())
    if meta.get("startPeriod1") is None:   # extra-time games: half boundaries only in the events
        marks = sorted((float(e["startTime"]), (e.get("gameEvents") or {}).get("gameEventType")) for e in events)
        k1 = next(t for t, k in marks if k == "FIRSTKICKOFF")
        k2 = next(t for t, k in marks if k == "SECONDKICKOFF")
        meta["startPeriod1"], meta["startPeriod2"] = k1, k2
        meta["endPeriod1"] = max(t for t, k in marks if k == "END" and t < k2)
        meta["endPeriod2"] = min(t for t, k in marks if k == "END" and t > k2)
    vt, rows = [], []
    with bz2.open(_pff_file("Tracking Data", game), "rt") as handle:
        for line in handle:
            fr = json.loads(line)
            vt.append(fr["videoTimeMs"] / 1000.0)
            rows.append((fr.get("homePlayers") or [], fr.get("awayPlayers") or [], fr.get("balls") or []))
    vt = np.array(vt)
    order = np.argsort(vt, kind="stable")
    slots = {}
    for side in (0, 1):
        for i in order:
            for p in rows[i][side]:
                slots.setdefault((side, str(p.get("jerseyNum"))), len(slots))
    t = np.arange(float(meta["startPeriod1"]), float(meta["endPeriod2"]), 1 / 25)
    k = np.clip(np.searchsorted(vt[order], t), 1, len(order) - 1)
    k = np.where(np.abs(vt[order][k - 1] - t) < np.abs(vt[order][k] - t), k - 1, k)
    src = order[k]
    ok = np.abs(vt[src] - t) < 0.06
    n = len(t)
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    ball_z = np.full(n, np.nan, np.float32)
    for j in np.flatnonzero(ok):
        home, away, balls = rows[src[j]]
        for side, plist in ((0, home), (1, away)):
            for p in plist:
                s = slots[(side, str(p.get("jerseyNum")))]
                if np.isnan(players[j, s, 0]) and p.get("x") is not None:
                    players[j, s] = (p["x"] + length / 2, p["y"] + width / 2)
        if len(balls) == 1 and balls[0].get("x") is not None:   # several balls only during stoppages
            ball[j] = (balls[0]["x"] + length / 2, balls[0]["y"] + width / 2)
            if balls[0].get("z") is not None:
                ball_z[j] = balls[0]["z"]
    team = np.array([side for side, _ in sorted(slots, key=slots.get)], np.int8)
    period = np.zeros(n, np.int8)
    period[(t >= meta["startPeriod1"]) & (t <= meta["endPeriod1"])] = 1
    period[(t >= meta["startPeriod2"]) & (t <= meta["endPeriod2"])] = 2
    label = pff_labels(events, t)
    label[(period == 0) | ~ok] = 3
    global OUT
    OUT = PFF_OUT
    save(f"pff_{game}", players=players, team=team, ball=ball, ball_z=ball_z, label=label, period=period,
         length=length, width=width, fps=25, source="pff")


def convert_pff() -> None:
    games = sys.argv[2:] or sorted({p.name.split(".")[0] for p in PFF_DIR.glob("*/Metadata/*.json")})
    for game in games:
        convert_pff_game(game)


if __name__ == "__main__":
    {"dfl": convert_dfl, "skillcorner": convert_skillcorner, "metrica": convert_metrica, "pff": convert_pff}[sys.argv[1]]()
