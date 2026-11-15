"""The chain rules for trusting ball sightings: a run of sightings that moves like a ball. fuse_simple.py merges with them."""
import numpy as np

PITCH = np.array((105.3, 68.0))
HALVES = [(11185, 90651), (110808, 190197)]
MAX_SPEED = 35.0       # m/s, a hard shot
AGREE = 2.5            # m
ANCHOR_CONF = 0.5
CHAIN_CONF = 0.3
CHAIN_SECONDS = 0.44   # 11 frames


def anchored_set(by_frame, step=2, dt=2 / 25):
    """(frame, rounded x, rounded y) of confident detections that sit in a chain of >= 3 consecutive
    confident sightings (one missed sighting allowed) along a physically possible path."""
    strong = {f: (g[g.conf >= ANCHOR_CONF][["x", "y"]].to_numpy() * PITCH) for f, g in by_frame.items()}

    def linked(f, xy, g):
        for k in (1, 2):
            other = strong.get(f + g * k * step)
            if other is not None and len(other) and \
                    (np.hypot(*(other - xy).T) / (k * dt) <= MAX_SPEED).any():
                return f + g * k * step, other[np.argmin(np.hypot(*(other - xy).T))]
        return None

    out = set()
    for f, pts in strong.items():
        for xy in pts:
            back, fwd = linked(f, xy, -1), linked(f, xy, +1)
            chain = (back is not None) + (fwd is not None)
            if chain == 2 or (fwd and linked(fwd[0], fwd[1], +1)) or (back and linked(back[0], back[1], -1)):
                out.add((f, round(xy[0], 2), round(xy[1], 2)))
    return out


def weak_chain_set(by_frame, min_conf, min_frames, every=5):
    """(frame, rounded x, rounded y) of detections >= min_conf in a chain spanning >= min_frames. Built exactly as
    validated in eval_chains.py, whose SoccerNet images are every 5th frame: on each of the 5 offsets, a sighting
    links to the nearest one up to 3 images back along a physically possible path, with sightings on >= half the
    images. Sightings on the frames in between that lie on a chain (within AGREE m of it) are trusted too.
    Also returns the chains' paths (frame -> interpolated positions), which carry the ball between sightings."""
    pts = {f: g[g.conf >= min_conf][["x", "y"]].to_numpy() * PITCH for f, g in by_frame.items()}
    out, path = set(), {}
    for phase in range(every):
        members, open_ = {}, []          # chain id -> [(frame, xy)]; open chains: (id, last frame, xy)
        for f in sorted(k for k in pts if k % every == phase):
            open_ = [o for o in open_ if f - o[1] <= 3 * every]
            taken = set()
            for xy in pts[f]:
                best, bd = None, None
                for j, (c, lf, lxy) in enumerate(open_):
                    if j in taken or f == lf:
                        continue
                    dist = np.hypot(*(xy - lxy))
                    if dist <= MAX_SPEED * (f - lf) / 25 and (bd is None or dist < bd):
                        best, bd = j, dist
                if best is None:
                    c = len(members)
                    members[c] = []
                    open_.append((c, f, xy))
                    taken.add(len(open_) - 1)
                else:
                    c = open_[best][0]
                    open_[best] = (c, f, xy)
                    taken.add(best)
                members[c].append((f, xy))
        for m in members.values():
            span = m[-1][0] - m[0][0] + 1
            if span >= min_frames and len({f for f, _ in m}) >= 0.5 * span / every:
                fr = np.array([f for f, _ in m]); xy = np.array([p for _, p in m])
                for f in range(fr[0], fr[-1] + 1):
                    path.setdefault(f, []).append((np.interp(f, fr, xy[:, 0]), np.interp(f, fr, xy[:, 1])))
    for f, line in path.items():
        for xy in pts.get(f, ()):
            if min(np.hypot(xy[0] - x, xy[1] - y) for x, y in line) < AGREE:
                out.add((f, round(xy[0], 2), round(xy[1], 2)))
    return out, path
