"""Where do the ball detector's sightings land on SoccerNet test (other matches, human boxes, never trained on)?

Uses the detections eval_chains.py cached (ball_combined_s.pt, crops as in our pipeline) and the human person
boxes in gt.txt. Each sighting is put on the person box that contains it (a 10% margin above and below), split by
height in the box: head (top 20%), body (20-60%), legs (60-85%), feet (85-110%). "Trusted" = the two rules our
ball merge uses: confident chains (conf >= 0.5, >= 3 sightings) and weak chains (fuse_ball.py --chains: conf >=
0.3 over 11 frames). Run from mtah-tools.
"""
import configparser

import numpy as np

import eval_chains as ec
import eval_soccernet as es

ZONES = ("head", "body", "legs", "feet", "off player")


def person_boxes(seq):
    info = configparser.ConfigParser()
    info.read(seq / "gameinfo.ini")
    kind = {int(k.split("_")[1]): v.split(";")[0].strip().lower()
            for k, v in info["Sequence"].items() if k.startswith("trackletid_")}
    by = {}
    for line in open(seq / "gt" / "gt.txt"):
        f, tid, x, y, w, h = map(float, line.split(",")[:6])
        if "ball" not in kind.get(int(tid), "ball"):
            by.setdefault(int(f), []).append((x, y, w, h, kind[int(tid)]))
    return by


def zone_of(px, py, boxes):
    best = None
    for x, y, w, h, kind in boxes:
        rel = (py - y) / h
        if x <= px <= x + w and -0.1 <= rel <= 1.1:
            dc = abs(px - (x + w / 2)) / w
            if best is None or dc < best[0]:
                best = (dc, rel, kind)
    if best is None:
        return 4, ""
    rel = best[1]
    return (0 if rel < 0.2 else 1 if rel < 0.6 else 2 if rel < 0.85 else 3), best[2]


def trusted(d):
    sel = np.zeros(len(d), bool)
    for min_conf, min_len in ((0.5, 3), (ec.fb_conf, ec.fb_len)):
        cid = ec.chains(d, min_conf)
        ok = cid >= 0
        ids, inv = np.unique(cid[ok], return_inverse=True)
        fr = d[ok, 2]
        hi = np.zeros(len(ids)); np.maximum.at(hi, inv, fr)
        lo = np.full(len(ids), np.inf); np.minimum.at(lo, inv, fr)
        span, cnt = hi - lo + 1, np.bincount(inv)
        keep = (cnt >= 3) if min_len == 3 else (span >= min_len) & (cnt >= 0.5 * span / ec.EVERY)
        s = np.zeros(len(d), bool)
        s[np.flatnonzero(ok)] = keep[inv]
        sel |= s
    return sel


def main():
    ec.fb_conf, ec.fb_len = 0.3, 11
    z = np.load(ec.CACHE)
    d = z["rows"]
    seqs = sorted(es.ROOT.iterdir())
    right = np.hypot(d[:, 4] - d[:, 6], d[:, 5] - d[:, 7]) <= np.maximum(8, d[:, 8])
    zone = np.full(len(d), 4)
    keeper = np.zeros(len(d), bool)
    for k in np.unique(d[:, 0]).astype(int):
        boxes = person_boxes(seqs[k])
        for i in np.flatnonzero(d[:, 0] == k):
            zone[i], kind = zone_of(d[i, 4], d[i, 5], boxes.get(int(d[i, 2]), []))
            keeper[i] = "goalkeeper" in kind
    tr = trusted(d)
    print(f"{len(d)} sightings; trusted by our merge: {tr.sum()} ({right[tr].mean():.1%} right)")
    for name, sel in (("trusted WRONG", tr & ~right), ("trusted right", tr & right), ("all wrong", ~right)):
        n = sel.sum()
        parts = ", ".join(f"{z_} {100 * np.mean(zone[sel] == j):.0f}%" for j, z_ in enumerate(ZONES))
        print(f"  {name:14s} {n:6d}: {parts}")
    for cut, label in ((1, "head"), (2, "head or body")):
        drop = tr & (zone < cut)
        kept = tr & ~drop
        print(f"  drop trusted sightings on a player's {label}: removes {np.sum(drop & ~right)} of "
              f"{np.sum(tr & ~right)} wrong, {np.sum(drop & right)} of {np.sum(tr & right)} right -> "
              f"{right[kept].mean():.1%} right")
    print(f"  trusted wrong sightings on a goalkeeper: {np.sum(tr & ~right & keeper)}")


if __name__ == "__main__":
    main()
