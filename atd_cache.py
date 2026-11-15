"""Parse the tactical ATD once into a compact numpy cache.

players: (n_frames, n_slots, 2) normalised 0-1 x/y, NaN where a slot is empty
ball:    (n_frames, 2)          normalised 0-1 x/y, NaN where the ball is missing
team:    (n_slots,)             0 = home (Barcelona), 1 = away (Athletic)

Row i is video frame i (ATD frame_id - 1), i.e. submission row i.

Usage: python3 atd_cache.py            (reads $MTAH_DATA_DIR/tactical, writes cache next to it)
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "mtah2027"))
TACTICAL = DATA_DIR / "tactical"
CACHE = DATA_DIR / "cache" / "BAR-ATH_tactical_atd.npz"
N_FRAMES = 199_319


def team_of_each_slot(meta_path: Path) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")], dtype=np.int8)


def parse(raw_path: Path, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    players = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, slots, ball_txt = line.rstrip("\n").split(":")
            i = int(frame_id) - 1
            values = np.array(slots.replace(";", ",").split(","), dtype=np.float32).reshape(n_slots, 3)
            players[i] = values[:, :2]
            ball[i] = np.array(ball_txt.split(",")[:2], dtype=np.float32)
    return players, ball


def load() -> dict:
    """Load the cache, building it first if needed."""
    if not CACHE.exists():
        build()
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def build() -> None:
    team = team_of_each_slot(TACTICAL / "BAR-ATH_tactical_atd.xml")
    players, ball = parse(TACTICAL / "BAR-ATH_tactical_atd.txt", len(team))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, players=players, ball=ball, team=team)
    print(f"cached {len(ball)} frames, {len(team)} slots -> {CACHE}")
    print(f"ball present on {np.isfinite(ball[:, 0]).mean():.1%} of all frames")


if __name__ == "__main__":
    sys.exit(build())
