"""Frame labels for the Metrica sample games, from Metrica's event data: the team of the latest on-ball event has the
ball; the ball is dead from a stop (ball out, foul, a shot's goal or miss, the end of a half) until the next on-ball
event. Used by convert_open.py. Labels: 'Home', 'Away', 'None' (dead ball)."""
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 105.0, 68.0
ON_BALL = {"PASS", "BALL LOST", "RECOVERY", "SET PIECE", "SHOT"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def load_tracking(game_dir: Path, game: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = {}
    for side in ("Home", "Away"):
        df = pd.read_csv(game_dir / f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv", skiprows=2)
        cols = df.columns
        players = [c for c in cols if c.startswith("Player")]
        xy = np.stack(
            [np.stack([df[p].to_numpy(), df.iloc[:, cols.get_loc(p) + 1].to_numpy()], -1) for p in players], 1
        )
        frames[side] = xy * [PITCH_X, PITCH_Y]
        base = df
    ball = base.iloc[:, -2:].to_numpy() * [PITCH_X, PITCH_Y]
    meta = base[["Period", "Frame", "Time [s]"]].copy()
    return meta, ball, (frames["Home"], frames["Away"])


def reference_labels(events: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Possession = team of the latest on-ball event; dead ball from a stop until the restart."""
    label = np.full(n_frames + 2, "None", dtype=object)
    current = "None"
    marks = []  # (frame, label)
    for _, e in events.sort_values("Start Frame", kind="stable").iterrows():
        if e["Type"] in ON_BALL:
            marks.append((int(e["Start Frame"]), e["Team"]))
        if e["Type"] in STOPS or (e["Type"] == "SHOT" and str(e["Subtype"]).endswith(("GOAL", "OUT"))):
            end = int(e["End Frame"]) if e["End Frame"] > 0 else int(e["Start Frame"])
            marks.append((end, "None"))
        if e["Type"] == "BALL LOST" and e["Subtype"] == "END HALF":
            marks.append((int(e["Start Frame"]), "None"))
    marks.sort(key=lambda m: m[0])
    j = 0
    for f in range(1, n_frames + 1):
        while j < len(marks) and marks[j][0] <= f:
            current = marks[j][1]
            j += 1
        label[f] = current
    return label[1 : n_frames + 1]
