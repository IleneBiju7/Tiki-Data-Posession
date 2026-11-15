"""A ball-free possession network: every player is a token, nothing is read but the players' tracking. The idea follows
PathCRF (Kim et al., KDD 2026, arXiv 2602.12080): model who holds the ball from player movement alone. Written from
scratch for our folds and degraded copies; their CRF is left out, our Viterbi decoding comes later.

Per step (5 a second): set attention over up to 13 players a team -> team summaries and a score per player -> a
bidirectional GRU over about 51 s -> H/A/D per step. A second output learns who has the ball, the owning team's player
nearest the true ball, used only as a training target. Same leave-one-match-out split and test copies as train.py.
Open data and code only. Output: cv_seq.npz (prob_<match>), as seq_train.py, for seq_cv.py and stack.py.
Env PS_SMOKE=1: two matches, a few steps, to check that it runs.
"""
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # LightGBM and PyTorch in one process
SMOKE = os.environ.get("PS_SMOKE") == "1"
if os.path.isdir("/kaggle/input"):
    new_code = [p for p in glob.glob("/kaggle/input/**/mtah-tools/train.py", recursive=True) if "open-sc" in p]
    work = "/tmp/run"
    shutil.copytree(os.path.dirname(sorted(new_code)[0]), f"{work}/mtah-tools")
    os.makedirs(f"{work}/data/open/converted")
    for p in glob.glob("/kaggle/input/**/open/converted/*.npz", recursive=True):
        d = f"{work}/data/open/converted/{os.path.basename(p)}"
        if not os.path.exists(d):
            os.symlink(p, d)
    os.symlink(sorted(glob.glob("/kaggle/input/**/open/converted_pff", recursive=True))[0], f"{work}/data/open/converted_pff")
    os.symlink(os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/evaluation/possession_metrics.py", recursive=True))[0])), f"{work}/official-repo")
    code, OUT = f"{work}/mtah-tools", Path("/kaggle/working")
else:
    code, OUT = str(Path(__file__).resolve().parents[2]), Path(os.environ.get("PS_OUT", "."))
os.chdir(code)
sys.path.insert(0, code)
sys.argv = ["x"] + ("--mixed-tags --sticky --smooth --pff --trk --keeper --sp-calib --det-noise --contacts --height --external").split()
import features  # noqa: E402
import train  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

SEED0, FOLDS, FINAL = 7, True, True   # set per notebook
STEP, TOK, WINDOW, BATCH = 5, 26, 256, 32
EPOCHS, STEPS = (1, 3) if SMOKE else (8, 300)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tokens(players, team, length, width, fps, clean, label):
    """(steps, TOK, 6): x, y, vx, vy, team, present; and who has the ball per step (token index, -1 unknown)."""
    v = np.zeros_like(players)
    v[STEP:] = (players[STEP:] - players[:-STEP]) * fps / STEP
    p, v, y, b = players[::STEP], v[::STEP], label[::STEP], clean[::STEP]
    n = len(p)
    out = np.zeros((n, TOK, 6), np.float32)
    owner = np.full(n, -1, np.int16)
    for t, base in ((0, 0), (1, TOK // 2)):
        slots = np.flatnonzero(team == t)
        ok = np.isfinite(p[:, slots, 0])
        pick = np.argsort(~ok, axis=1, kind="stable")[:, :TOK // 2]
        rows = np.arange(n)[:, None]
        k = pick.shape[1]
        xy, vel, m = p[:, slots][rows, pick], v[:, slots][rows, pick], ok[rows, pick]
        out[:, base:base + k, 0] = np.nan_to_num(xy[..., 0]) / length
        out[:, base:base + k, 1] = np.nan_to_num(xy[..., 1]) / width
        out[:, base:base + k, 2:4] = np.clip(np.nan_to_num(vel), -12, 12) / 12
        out[:, base:base + k, 4] = t
        out[:, base:base + k, 5] = m
        d = np.hypot(*(np.moveaxis(xy - b[:, None], -1, 0)))
        d = np.where(np.isfinite(d), d, 1e9)
        near, dist = d.argmin(1), d.min(1)
        take = (y == t) & (dist <= 5)
        owner[take] = base + near[take]
    return out.astype(np.float16), owner, y.astype(np.int8)


def copy(name, rng, swap=False, test=False):
    z = np.load(train._path(name))
    pl, _, team, _, _, _ = train.degrade(z, rng, train.P_LOSE_TEST if test else None, True, train.SHIFT_TEST and test)
    lab = z["label"]
    if swap:
        team, lab = 1 - team, features.swap_labels(lab)
    return tokens(pl.astype(np.float64), team, float(z["length"]), float(z["width"]), int(z["fps"]), z["ball"].astype(np.float64), lab)


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

    def forward(self, x):   # (B, T, TOK, 6)
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
        ctx = self.back(g).reshape(b * t, 1, -1)
        own = self.own(h * ctx).squeeze(-1).masked_fill(m, -1e4)
        return self.out(g), own.reshape(b, t, k)


def fit(data, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    total = EPOCHS * STEPS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=total)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    net.train()
    for it in range(total):
        xs, ys, os_ = [], [], []
        for _ in range(BATCH):
            x, o, y = data[rng.integers(len(data))]
            a = rng.integers(max(1, len(y) - WINDOW))
            xs.append(x[a:a + WINDOW]); ys.append(y[a:a + WINDOW]); os_.append(o[a:a + WINDOW])
        xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(DEV)
        yb = torch.from_numpy(np.stack(ys).astype(np.int64)).to(DEV)
        ob = torch.from_numpy(np.stack(os_).astype(np.int64)).to(DEV)
        with torch.amp.autocast(DEV, enabled=DEV == "cuda"):
            p, q = net(xb)
            loss = nn.functional.cross_entropy(p.transpose(1, 2), yb, ignore_index=3)
            if (ob >= 0).any():
                loss = loss + 0.5 * nn.functional.cross_entropy(q.reshape(-1, TOK), ob.reshape(-1), ignore_index=-1)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 300 == 0:
            print(f"    step {it}/{total} loss {loss.item():.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def predict(net, x):
    n, tot, cnt = len(x), np.zeros((len(x), 3)), np.zeros(len(x))
    for a in range(0, max(1, n - WINDOW // 2), WINDOW // 2):
        xb = torch.from_numpy(x[a:a + WINDOW].astype(np.float32)[None]).to(DEV)
        tot[a:a + xb.shape[1]] += net(xb)[0].softmax(-1)[0].float().cpu().numpy()
        cnt[a:a + xb.shape[1]] += 1
    return (tot / cnt[:, None]).astype(np.float32)


def main():
    names = train.load_matches(False)
    if SMOKE:
        names = [n for n in names if not n.startswith("pff")][:2]
    rng = np.random.default_rng(train.SEED)
    t0 = time.time()
    data, tests = {}, {}
    for n in names:
        data[n] = [copy(n, rng)[:3], copy(n, rng, swap=True)[:3]]
        if not n.startswith(train.TRAIN_ONLY):
            tests[n] = copy(n, np.random.default_rng(train.TEST_SEED + 2), test=True)[0]
    print(f"data ready: {len(names)} matches, {len(tests)} held out, {time.time() - t0:.0f} s, device {DEV}", flush=True)
    probs = {}
    for i, held in enumerate(tests if FOLDS else []):
        t = time.time()
        pool = [c for n in names if n != held for c in data[n]]
        net = fit(pool, SEED0 + i)
        probs[held] = predict(net, tests[held])
        np.savez_compressed(OUT / "cv_seq.npz", **{f"prob_{k}": v for k, v in probs.items()})
        print(f"  {held}: done in {(time.time() - t) / 60:.1f} min", flush=True)
    print("all folds done", flush=True)
    if FINAL:
        net = fit([c for n in names for c in data[n]], SEED0 + 1000)
        torch.save({"state": net.state_dict(), "step": STEP, "tok": TOK, "window": WINDOW}, OUT / "ps_final.pt")
        print("final fit saved", flush=True)


main()
