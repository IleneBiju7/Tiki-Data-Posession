"""Tune ball_height.estimate (WIN, FLIGHT_GAIN) on open matches seen through generic cameras (external data only):
recall of true flights (ball_z > 2 m) and false 'airborne' rate on the ground (ball_z < 0.3 m)."""
import sys

import numpy as np
import ball_height
import train
names = [n for n in sorted(p.stem for p in train.CONVERTED.glob("dfl_*.npz"))][:4]
data = []
rng = np.random.default_rng(11)
for name in names:
    z = np.load(train._path(name))
    pitch = (float(z["length"]), float(z["width"]))
    for k in range(3):
        P = ball_height.generic_camera(rng, pitch)
        g = ball_height.simulate(z["ball"].astype(np.float64), z["ball_z"], P, np.random.default_rng(k), pitch=pitch, q_hi=0.0)
        data.append((P, g, z["ball_z"]))
for win in (8, 12, 16):
    for gain in (0.5, 0.7, 0.85):
        ball_height.WIN, ball_height.FLIGHT_GAIN = win, gain
        hit = air = fa = ground = 0
        for P, g, bz in data:
            h = ball_height.from_pitch(P, g)
            seen = np.isfinite(g[:, 0])
            a, gr = seen & np.isfinite(bz) & (bz > 2), seen & np.isfinite(bz) & (bz < 0.3)
            hit += (h[a] > 1).sum(); air += a.sum(); fa += (h[gr] > 1).sum(); ground += gr.sum()
        print(f"WIN {win:2d} FLIGHT_GAIN {gain}: flights found {hit / air:.1%}, false on ground {fa / ground:.2%}", flush=True)
