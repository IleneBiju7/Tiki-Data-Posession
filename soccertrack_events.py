"""Where the ball is in SoccerTrack v2's panoramic videos, from its annotations only (other matches, CC BY 4.0).

SoccerTrack v2's ball track is rebuilt from the ball-action events and rounded to a 1.05 m x 0.68 m grid, too coarse
for a 5-8 px ball. But at a kick (pass, high pass, cross, shot, free kick) or a dribble ("DRIVE") the ball is at the
acting player's feet, and the game-state labels give that player's box in the video's pixels at that frame. This
writes one row per such event: half, frame (1-based video frame), label, team, player_id and the actor's box
(x, y, w, h, pixels, top-left), plus the video size. ball_pseudo_st.py then looks for the ball at those feet.

Reads data/open/soccertrack/bas/<match>/<match>_12_class_events.json and gsr/<match>/<match>_<half>.json (the
game-state files are 2.7 GB each: read line by line, only the actors' records at event frames are kept).
Usage: python soccertrack_events.py [match ...]   -> data/open/soccertrack/events_feet.csv (all matches by default)
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "open" / "soccertrack"
AT_FEET = {"PASS", "DRIVE", "HIGH PASS", "CROSS", "SHOT", "FREE KICK"}
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
NUM = re.compile(r'"(\w+)": (-?[\d.]+)')


def events(match):
    acts = json.loads((ROOT / "bas" / match / f"{match}_12_class_events.json").read_text())["actions"]
    out = {"1st": {}, "2nd": {}}
    for a in acts:
        if a["label"] in AT_FEET and a.get("player_id"):
            half = "1st" if a["gameTime"].startswith("1") else "2nd"
            out[half].setdefault(int(a["frame"]), []).append(a)
    return out


def actor_boxes(path, wanted):
    """{(frame, player_id): (x, y, w, h)} for the wanted frames' players, and the video size, in one pass."""
    found, size = {}, {}
    in_ann, frame, pid, box, in_box = False, None, None, {}, False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not in_ann:   # the images list: the first entry's width and height
                for key in ("width", "height"):
                    if key not in size and s.startswith(f'"{key}"'):
                        size[key] = int(s.split(":")[1].strip(" ,"))
                in_ann = s.startswith('"annotations"')
                continue
            if s.startswith('"image_id"'):
                frame = int(s.split('"')[3][-6:])
                pid, box, in_box = None, {}, False
            elif frame not in wanted:
                continue
            elif s.startswith('"player_id"'):
                pid = s.split(":")[1].strip(" ,")
            elif s.startswith('"bbox_image"'):
                in_box = True
            elif in_box:
                m = NUM.search(s)
                if m:
                    box[m.group(1)] = float(m.group(2))
                if s.startswith("}"):
                    in_box = False
                    if pid in wanted[frame] and all(k in box for k in "xywh"):
                        found[(frame, pid)] = (box["x"], box["y"], box["w"], box["h"])
    return found, (size.get("width"), size.get("height"))


def main():
    matches = [m for m in sys.argv[1:] if m in MATCHES] or MATCHES
    out = ROOT / "events_feet.csv"
    rows = []
    for match in matches:
        ev = events(match)
        for half in ("1st", "2nd"):
            wanted = {f: {str(a["player_id"]) for a in acts} for f, acts in ev[half].items()}
            found, size = actor_boxes(ROOT / "gsr" / match / f"{match}_{half}.json", wanted)
            n = 0
            for f, acts in sorted(ev[half].items()):
                for a in acts:
                    b = found.get((f, str(a["player_id"])))
                    if b is not None:
                        rows.append([match, half, f, a["label"], a["team"], a["player_id"], *b, *size])
                        n += 1
            print(f"{match} {half}: {sum(len(v) for v in ev[half].values())} events at the feet, {n} with the actor's box, "
                  f"video {size[0]}x{size[1]}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match", "half", "frame", "label", "team", "player_id", "x", "y", "w", "h", "width", "height"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
