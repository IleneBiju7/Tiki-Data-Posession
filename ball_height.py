"""Ball height from one camera: a ball in the air falls with gravity, which only a vertical motion can explain.

The homography maps the pitch (z = 0) to pixels. Assuming square pixels and the principal point at the image
centre, it gives the full camera (focal length, rotation, position), so a 3-D point can be projected too.
Where our ball is airborne, its pitch position is where the camera ray meets the grass, not under the ball.

Estimate: around every frame, fit the ball's pixels over +-WIN frames twice: rolling on the grass (straight line,
z = 0) and in free flight (straight line plus gravity, any start height and vertical speed). The height is the
flight fit's height at that frame when it explains the pixels clearly better (by FLIGHT_GAIN in mean squared
pixel error) and is at least MIN_AIR high; otherwise 0. (Rojas et al.; Metzler & Pagel 2013: a parabolic flight
is recoverable from one calibrated camera.)

Training matches have the true height (DFL, PFF). Their ball goes through the same camera model (simulate()), then
through the rest of the degradation (gaps, false sightings, jitter), and only the finished ball gets the height
estimate (from_pitch()), as our ball does on the competition video. So the model learns from heights exactly as
noisy, and as often missing, as on the video.
"""
import numpy as np

G = 9.81
WIN = 12           # frames each side (0.48 s at 25 fps). WIN and FLIGHT_GAIN set on open matches through generic
#                    cameras (height_tune.py, run from mtah-tools): finds 35% of flights > 2 m, 0.21% false on the ground
#                    (WIN 8, gain 0.5, the first setting: 4.9%, 0.12%)
FLIGHT_GAIN = 0.7  # flight fit must cut the squared pixel error to this share of the grass fit
MIN_AIR = 0.8      # m
MAX_H = 30.0
CENTRE = (960.0, 540.0)
H_HI = 3.0         # m: above this our pipeline may lose a flight, and our ball then keeps the tracker's, under it
Q_HI = 0.5         # share of flights where it does, when simulate() is given no other share


def camera(h_norm, pitch=(105.3, 68.0)):
    """3x4 projection matrix (metres -> pixels, z up) from a homography in normalised pitch coordinates."""
    H = h_norm @ np.diag([1 / pitch[0], 1 / pitch[1], 1.0])
    cx, cy = CENTRE
    a = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H       # principal point to the origin
    num = a[0, 0] * a[0, 1] + a[1, 0] * a[1, 1]
    if abs(a[2, 0] * a[2, 1]) < 1e-12 or -num / (a[2, 0] * a[2, 1]) <= 0:
        f2 = (np.hypot(a[0, 0], a[1, 0]) / max(np.hypot(a[2, 0], 1e-9), 1e-9)) ** 2   # fallback, rarely used
    else:
        f2 = -num / (a[2, 0] * a[2, 1])
    f = np.sqrt(f2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    m = np.linalg.inv(K) @ H
    s = 2 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    r1, r2, t = m[:, 0] * s, m[:, 1] * s, m[:, 2] * s
    r3 = np.cross(r1, r2)
    P = K @ np.column_stack([r1, r2, r3, t])
    # z up: a point above the pitch centre must appear higher in the image (smaller v)
    c = np.array([pitch[0] / 2, pitch[1] / 2])
    v0 = P @ np.r_[c, 0, 1]
    v1 = P @ np.r_[c, 2, 1]
    if v1[1] / v1[2] > v0[1] / v0[2]:
        P[:, 2] *= -1
    return P


def project(P, xyz):
    q = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return q[:, :2] / q[:, 2:3]


def estimate(P, uv, fps=25, chunk=20000):
    """Height per frame (m). P: one camera (3, 4) or one per frame (n, 3, 4); uv (n, 2) ball pixels, NaN where
    unseen. Around each seen frame, the pixels within +-WIN frames are fitted by linear least squares on the
    projection equations u (P3.X) = P1.X, v (P3.X) = P2.X, with X = (x0 + vx t, y0 + vy t, z): z = 0 (grass) or
    z = z0 + vz t - g t^2 / 2 (flight). Residuals are divided by the point's depth, so they are about pixels."""
    n = len(uv)
    out = np.zeros(n, np.float32)
    seen = np.isfinite(uv[:, 0])
    off = np.arange(-WIN, WIN + 1)
    t = off / fps
    for lo in range(0, int(seen.sum()), chunk):
        c = np.flatnonzero(seen)[lo:lo + chunk]
        j = c[:, None] + off[None, :]
        valid = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        valid &= seen[jj]
        Pc = P[c][:, None] if P.ndim == 3 else P[None, None]          # (m or 1, 1, 3, 4)
        w = np.where(valid[..., None], uv[jj], 0.0)                       # (m, W, 2)
        cu = w[..., 0:1] * Pc[..., 2, :] - Pc[..., 0, :]                 # (m, W, 4): cu . (x, y, z, 1) = 0
        cv = w[..., 1:2] * Pc[..., 2, :] - Pc[..., 1, :]
        coef = np.concatenate([cu, cv], 1)                                # (m, 2W, 4)
        tt = np.r_[t, t][None, :]
        m2 = np.concatenate([valid, valid], 1).astype(float)
        base = [coef[..., 0], coef[..., 1], coef[..., 0] * tt, coef[..., 1] * tt]
        fits = {}
        for flight in (False, True):
            cols = base + ([coef[..., 2], coef[..., 2] * np.broadcast_to(tt, coef[..., 2].shape)] if flight else [])
            A = np.stack(cols, -1) * m2[..., None]
            y = (-coef[..., 3] + (coef[..., 2] * 0.5 * G * tt ** 2 if flight else 0)) * m2
            p = np.einsum("mij,mj->mi", np.linalg.pinv(A), y)
            z = (p[:, 4:5] + p[:, 5:6] * t - 0.5 * G * t ** 2) if flight else np.zeros((len(c), len(t)))
            X = np.stack([p[:, 0:1] + p[:, 2:3] * t, p[:, 1:2] + p[:, 3:4] * t, z, np.ones_like(z)], -1)
            depth = np.abs(np.einsum("mwk,mwk->mw", X, np.broadcast_to(Pc[..., 2, :], X.shape)))
            res = (np.einsum("mij,mj->mi", A, p) - y) / np.maximum(np.concatenate([depth, depth], 1), 1e-6)
            fits[flight] = (p, (res ** 2 * m2).sum(1) / np.maximum(m2.sum(1), 1), z)
        (_, e0, _), (p1, e1, z1) = fits[False], fits[True]
        ok = (valid.sum(1) >= WIN) & (e1 < FLIGHT_GAIN * e0) & (p1[:, 4] >= MIN_AIR) & (p1[:, 4] <= MAX_H) \
            & np.all((z1 > -0.5) | ~valid, 1)
        out[c[ok]] = p1[ok, 4]
    return out


def generic_camera(rng, pitch=(105.3, 68.0)):
    """A wide main-stand camera from generic ranges, no competition data (train.py): 15-60 m behind one
    touchline, 8-35 m up, aimed near the middle of the pitch, focal length from the whole pitch fitting the
    1920x1080 frame to 1.6x zoomed in (from the whole pitch in view to a closer view). Returns the 3x4 projection matrix (metres -> pixels, z up), as camera() does."""
    L, W = pitch
    C = np.array([L * rng.uniform(0.35, 0.65), -rng.uniform(15.0, 60.0), rng.uniform(8.0, 35.0)])
    T = np.array([L * rng.uniform(0.4, 0.6), W * rng.uniform(0.35, 0.65), 0.0])
    fwd = (T - C) / np.linalg.norm(T - C)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])                  # image x right, y down, z forward
    corners = (np.array([[0, 0, 0], [L, 0, 0], [0, W, 0], [L, W, 0]], float) - C) @ R.T
    tan = np.abs(corners[:, :2] / corners[:, 2:3])
    f = min(0.95 * CENTRE[0] / tan[:, 0].max(), 0.95 * CENTRE[1] / tan[:, 1].max()) * rng.uniform(0.85, 1.6)
    K = np.array([[f, 0, CENTRE[0]], [0, f, CENTRE[1]], [0, 0, 1.0]])
    return K @ np.column_stack([R, -R @ C])


def competition(ball_tag="fused_v9"):
    """Height of our ball on the competition video: its pitch position back to pixels (exact, through the
    homography), then the same estimate as for the training copies. -> cache/BAR-ATH_ball_height_<tag>.npy"""
    import atd_cache
    homs = np.loadtxt(atd_cache.TACTICAL / "BAR-ATH_tactical_homography.csv", delimiter=",", skiprows=1,
                      usecols=range(1, 10))
    ball = np.load(atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_{ball_tag}.npy").astype(np.float64)
    n = len(ball)
    uv = np.full((n, 2), np.nan)
    ok = np.isfinite(ball[:, 0])
    for i in np.flatnonzero(ok):
        q = homs[i].reshape(3, 3) @ np.r_[ball[i], 1.0]
        uv[i] = q[:2] / q[2]
    P = np.zeros((n, 3, 4))
    for i in np.flatnonzero(ok):
        P[i] = camera(homs[i].reshape(3, 3))
    h = estimate(P, uv)
    out = atd_cache.DATA_DIR / "cache" / f"BAR-ATH_ball_height_{ball_tag}.npy"
    np.save(out, h)
    print(f"airborne (> 1 m) on {np.mean(h[ok] > 1.0):.1%} of frames with a ball -> {out}")


def simulate(ball, ball_z, P, rng, noise_px=1.5, pitch=(105.3, 68.0), lost=5.0, q_hi=None):
    """Training copy: true 3-D ball -> pixels through camera P (+ detector noise) -> our pitch position (where the
    ray meets the grass), as it would be on the competition video. A ball over the pitch whose ray lands more than
    `lost` m outside it is lost, as our ball pipeline only keeps balls on the pitch. The height is estimated later,
    on the finished copy (from_pitch())."""
    ok = np.isfinite(ball[:, 0]) & np.isfinite(ball_z)
    z = np.where(ok, np.maximum(ball_z, 0), 0)
    uv = np.full((len(ball), 2), np.nan)
    uv[ok] = project(P, np.c_[ball[ok], z[ok]]) + rng.normal(0, noise_px, (ok.sum(), 2))
    Hg = P[:, [0, 1, 3]]
    g = np.c_[uv[ok], np.ones(ok.sum())] @ np.linalg.inv(Hg).T
    ground = ball.copy()
    ground[ok] = g[:, :2] / g[:, 2:3]
    L, W = pitch
    with np.errstate(invalid="ignore"):
        inside = (ball[:, 0] > -0.5) & (ball[:, 0] < L + 0.5) & (ball[:, 1] > -0.5) & (ball[:, 1] < W + 0.5)
        far = (ground[:, 0] < -lost) | (ground[:, 0] > L + lost) | (ground[:, 1] < -lost) | (ground[:, 1] > W + lost)
    ground[inside & far] = np.nan
    q_hi = Q_HI if q_hi is None else q_hi
    if q_hi > 0:   # in some flights the detector loses the high ball and our ball keeps the tracker's (under the ball)
        air = np.isfinite(ball_z) & (ball_z > 0.5)
        d = np.diff(np.r_[0, air.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if rng.random() < q_hi:
                high = a + np.flatnonzero(ball_z[a:b] > H_HI)
                ground[high] = ball[high]
    return ground.astype(np.float32)


def centre(P):
    """Camera position (x, y, z in m) of projection matrix P."""
    c = np.linalg.svd(P)[2][-1]
    return c[:3] / c[3]


def competition_centres():
    """Camera position on every frame of the competition video, from its homography (cached)."""
    import atd_cache
    path = atd_cache.DATA_DIR / "cache" / "BAR-ATH_camera_centre.npy"
    if path.exists():
        return np.load(path)
    homs = np.loadtxt(atd_cache.TACTICAL / "BAR-ATH_tactical_homography.csv", delimiter=",", skiprows=1,
                      usecols=range(1, 10))
    out = np.full((len(homs), 3), np.nan)
    for i in np.flatnonzero(np.isfinite(homs).all(1)):
        out[i] = centre(camera(homs[i].reshape(3, 3)))
    np.save(path, out)
    return out


def from_pitch(P, ball):
    """Height estimate for a ball given on the pitch (m, NaN where unseen) as seen by camera P: back to pixels
    through the camera's view of the grass, then estimate(), as competition() does with our ball on the video."""
    uv = np.full((len(ball), 2), np.nan)
    ok = np.isfinite(ball[:, 0])
    q = np.c_[ball[ok].astype(np.float64), np.ones(ok.sum())] @ P[:, [0, 1, 3]].T
    uv[ok] = q[:, :2] / q[:, 2:3]
    return estimate(P, uv)


if __name__ == "__main__":
    import sys
    competition(sys.argv[1] if len(sys.argv) > 1 else "fused_v9")
