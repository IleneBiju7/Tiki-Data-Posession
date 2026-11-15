"""The ball-free player networks (kaggle/player_set, ps_final.pt) on one match's player tracking, averaged. Its own
process: PyTorch and LightGBM do not share one on the Mac. The network idea follows PathCRF (Kim et al., KDD 2026).

Usage: python ps_predict.py <players.npz with players (metres), team, length, width, fps> <out.npz> <ps_final.pt>...
Output: prob, H/A/D per step (every STEP-th frame).
"""
import os
import sys

import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Linear(6, d)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, 0.1, batch_first=True)
        self.social = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.score = nn.Linear(d, 1)
        self.gru = nn.GRU(d * 4 + 2, 128, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(256, 3)
        self.back = nn.Linear(256, d)
        self.own = nn.Linear(d, 1)

    def forward(self, x):
        b, t, k, _ = x.shape
        x = x.reshape(b * t, k, -1)
        m = x[..., 5] < 0.5
        m[m.all(1), 0] = False
        h = self.social(self.emb(x), src_key_padding_mask=m)
        s = self.score(h).squeeze(-1).masked_fill(m, -1e4)
        w = s.softmax(1).unsqueeze(-1)
        keep = (~m).unsqueeze(-1).float()
        half = k // 2
        pool = [(h[:, a:c] * w[:, a:c]).sum(1) for a, c in ((0, half), (half, k))]
        mean = [(h[:, a:c] * keep[:, a:c]).sum(1) / keep[:, a:c].sum(1).clamp(min=1) for a, c in ((0, half), (half, k))]
        share = torch.stack([w[:, :half].sum((1, 2)), w[:, half:].sum((1, 2))], 1)
        g, _ = self.gru(torch.cat(pool + mean + [share], 1).reshape(b, t, -1))
        return self.out(g)


def tokens(players, team, length, width, fps, step, tok):
    v = np.zeros_like(players)
    v[step:] = (players[step:] - players[:-step]) * fps / step
    p, v = players[::step], v[::step]
    out = np.zeros((len(p), tok, 6), np.float32)
    rows = np.arange(len(p))[:, None]
    for t, base in ((0, 0), (1, tok // 2)):
        slots = np.flatnonzero(team == t)
        ok = np.isfinite(p[:, slots, 0])
        pick = np.argsort(~ok, axis=1, kind="stable")[:, :tok // 2]
        k = pick.shape[1]
        xy, vel, m = p[:, slots][rows, pick], v[:, slots][rows, pick], ok[rows, pick]
        out[:, base:base + k, 0] = np.nan_to_num(xy[..., 0]) / length
        out[:, base:base + k, 1] = np.nan_to_num(xy[..., 1]) / width
        out[:, base:base + k, 2:4] = np.clip(np.nan_to_num(vel), -12, 12) / 12
        out[:, base:base + k, 4] = t
        out[:, base:base + k, 5] = m
    return out


@torch.no_grad()
def predict(net, x, window):
    n, tot, cnt = len(x), np.zeros((len(x), 3)), np.zeros(len(x))
    for a in range(0, max(1, n - window // 2), window // 2):
        xb = torch.from_numpy(x[a:a + window][None]).to(DEV)
        tot[a:a + xb.shape[1]] += net(xb)[0].softmax(-1).float().cpu().numpy()
        cnt[a:a + xb.shape[1]] += 1
    return tot / cnt[:, None]


def main():
    z = np.load(sys.argv[1])
    total = None
    for path in sys.argv[3:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        x = tokens(z["players"].astype(np.float64), z["team"], float(z["length"]), float(z["width"]), int(z["fps"]), ck["step"], ck["tok"])
        net = Net()
        net.load_state_dict({k: v for k, v in ck["state"].items()})
        net = net.to(DEV).eval()
        p = predict(net, x, ck["window"])
        total = p if total is None else total + p
    prob = total / len(sys.argv[3:])
    np.savez_compressed(sys.argv[2], prob=prob.astype(np.float32), step=ck["step"])
    print(f"{len(sys.argv) - 3} player networks on {len(prob)} steps", flush=True)


if __name__ == "__main__":
    main()
