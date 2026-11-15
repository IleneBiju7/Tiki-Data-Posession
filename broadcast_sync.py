"""Line the broadcast recording up with the tactical one, from the two trackings alone (no labels, no video).

Both trackings give positions on the same pitch, so the same moment shows the same ball path and the same players:
  1. every 2-s ball path of the broadcast tracking that moves more than 3 m is matched to the most similar 2-s path
     of the tactical tracking (nearest neighbour); in each half almost all close matches share one offset
     (tactical frame - broadcast frame): the broadcast is the tactical timeline minus its cuts, with replays and
     close-ups inserted
  2. every broadcast frame then takes its half's offset, refined by up to FINE frames, where its tracked players sit
     where the tactical players are at that moment (median distance to the nearest tactical player < LIVE m);
     the rest (replays, close-ups, crowd shots) are not live and are not used
Measured: the two offsets are 4,014 and 26,920 frames; 56% of broadcast frames are live, covering 52% of the graded
tactical frames.

Usage: python broadcast_sync.py   -> cache/BAR-ATH_broadcast_sync.npz (for each tactical frame its live broadcast
                                     frame or -1, and the broadcast ball there in normalised pitch coordinates)
"""
import numpy as np
from scipy.spatial import cKDTree

import atd_cache
import predict

BROADCAST = atd_cache.DATA_DIR / "broadcast"
OUT = atd_cache.DATA_DIR / "cache" / "BAR-ATH_broadcast_sync.npz"
N_BROADCAST = 164488   # broadcast tracking records (its video has one frame less)
W, S = 50, 5           # 2-s ball paths, sampled every 5th frame
MOVE = 3.0             # m: a path must move this much to be distinctive
MATCH = 1.5            # m rms: a close path match
FINE = 2               # frames
LIVE = 1.0             # m


def load_broadcast():
    team = atd_cache.team_of_each_slot(BROADCAST / "BAR-ATH_broadcast_atd.xml")
    players, ball = atd_cache.parse(BROADCAST / "BAR-ATH_broadcast_atd.txt", len(team))
    return players[:N_BROADCAST, :, :2].astype(np.float64), ball[:N_BROADCAST].astype(np.float64)


def paths(ball, starts):
    v = ball[starts[:, None] + np.arange(0, W, S)[None, :]]
    ok = np.isfinite(v).all((1, 2)) & (np.ptp(v[..., 0], 1) + np.ptp(v[..., 1], 1) > MOVE)
    return v.reshape(len(starts), -1), ok


def half_offsets(tb, bb):
    """(broadcast frame where the second half's offset starts, first offset, second offset)."""
    st_t, st_b = np.arange(len(tb) - W), np.arange(0, len(bb) - W, S)
    vt, okt = paths(tb, st_t)
    vb, okb = paths(bb, st_b)
    d, j = cKDTree(vt[okt]).query(vb[okb], k=1)
    good = d / np.sqrt(W // S) < MATCH
    at, off = st_b[okb][good], (st_t[okt][j] - st_b[okb])[good]
    # the two halves: the two offsets most matches share, and the broadcast frame that best separates them
    values, counts = np.unique(off, return_counts=True)
    a = values[np.argmax(counts)]
    b = values[np.argmax(np.where(np.abs(values - a) > 1000, counts, 0))]
    first, second = sorted((a, b), key=lambda o: np.median(at[np.abs(off - o) <= FINE]))
    sel = (np.abs(off - first) <= FINE) | (np.abs(off - second) <= FINE)
    order = np.argsort(at[sel])
    x, is_second = at[sel][order], (np.abs(off[sel] - second) <= FINE)[order]
    i = int(np.argmax(np.cumsum(~is_second) - np.cumsum(is_second)))
    return int(x[i]) + 1, int(first), int(second), good.mean()


def main():
    P = np.array(predict.PITCH, np.float64)
    atd = atd_cache.load()
    pt, tb = atd["players"].astype(np.float64) * P, atd["ball"].astype(np.float64) * P
    pb, bb = load_broadcast()
    pb, bb = pb * P, bb * P
    split, first, second, share = half_offsets(tb, bb)
    print(f"offsets {first} (first half) and {second} (from broadcast frame {split}); {share:.0%} of moving ball "
          f"paths matched within {MATCH} m")
    off = np.where(np.arange(len(bb)) < split, first, second)
    best, fine = np.full(len(bb), np.inf), np.zeros(len(bb), int)
    for b in range(len(bb)):
        a = pb[b][np.isfinite(pb[b, :, 0])]
        if len(a) < 4:
            continue
        for d in range(-FINE, FINE + 1):
            t = b + off[b] + d
            if 0 <= t < len(tb):
                c = pt[t][np.isfinite(pt[t, :, 0])]
                if len(c) >= 10:
                    m = np.median(np.min(np.hypot(a[:, None, 0] - c[None, :, 0], a[:, None, 1] - c[None, :, 1]), 1))
                    if m < best[b]:
                        best[b], fine[b] = m, d
    live = best < LIVE
    t_of_b = np.where(live, np.arange(len(bb)) + off + fine, -1)
    b_of_t = np.full(len(tb), -1)
    ok = (t_of_b >= 0) & (t_of_b < len(tb))
    b_of_t[t_of_b[ok]] = np.flatnonzero(ok)
    ball = np.full((len(tb), 2), np.nan, np.float32)
    has = b_of_t >= 0
    ball[has] = (bb[b_of_t[has]] / P).astype(np.float32)
    graded = np.zeros(len(tb), bool)
    for a, b in predict.HALVES:
        graded[a:b + 1] = True
    print(f"live broadcast frames {live.mean():.1%}; graded tactical frames with a live broadcast picture "
          f"{(graded & has).sum() / graded.sum():.1%}, with its ball {(graded & np.isfinite(ball[:, 0])).sum() / graded.sum():.1%}")
    np.savez_compressed(OUT, broadcast_frame=b_of_t, ball=ball, offsets=np.array([first, second, split]))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
