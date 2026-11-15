import sys
sys.argv = ["seq_train.py", "--data=/kaggle/input", "--out=/kaggle/working", "--epochs=12", "--seed=21", "--arch=gru"]   # --external rebuild
"""Sequence model: a dilated 1-D convolutional network over the feature sequences of seq_export.py, looking at about
25 s either side of every step, trained with the same leave-one-match-out split as train.py (PFF training-only).

Writes, to --out:
  cv_seq.npz      held-out probabilities on each match's realistic test copy (5 per second), for stack.py
  seq_final.pt    the network trained on every match, plus the column scaling, for the competition match

Usage: python seq_train.py --data=<dir with seq_export.py files> --out=<dir> [--epochs=12] [--window=256] [--seed=7]
       [--folds=all|none|<name>,...] [--final=0] [--arch=cnn|gru]
  Borrowed from temporal action segmentation, whose segmental F1 at an IoU threshold is what spellF1 measures
  (all off by default, so earlier runs are unchanged):
       [--boundary=1]     a second output, "the label changes here" (Ishikawa et al., ASRF, WACV 2021), learnt next to
                          H/A/D with this loss weight; target spread over --bd-sigma=1 steps; cv_seq.npz then also holds
                          its held-out probability (bd_<match>)
       [--stages=3]       refinement stages after the network, each re-reading the previous stage's probabilities over
                          about 50 s, every stage trained on the labels (Farha & Gall, MS-TCN, CVPR 2019)
       [--smooth=0.15]    MS-TCN's truncated smoothing loss on frame-to-frame log-probability changes
       [--test=xh,xhs]    held-out test copies to score: the first goes to cv_seq.npz, each other one to
                          cv_seq_<key>.npz
Needs torch (GPU if there is one). Pure training: no data is downloaded or uploaded.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
DATA, OUT = Path(ARGS.get("data", ".")), Path(ARGS.get("out", "."))
EPOCHS = int(ARGS.get("epochs", 12))
WINDOW = int(ARGS.get("window", 256))   # steps per training window (51 s at 5 per second)
BATCH = 64
STEPS_PER_EPOCH = int(ARGS.get("steps", 400))
TRAIN_ONLY = ("skillcorner", "pff")
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = int(ARGS.get("seed", 7))
BOUNDARY = float(ARGS.get("boundary", 0))
BD_SIGMA = float(ARGS.get("bd-sigma", 1.0))
STAGES = int(ARGS.get("stages", 1))
SMOOTH = float(ARGS.get("smooth", 0))
COPIES = ARGS.get("copies", "")
TESTS = ARGS.get("test", "xh").split(",")
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class Block(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation), nn.GELU(),
                                  nn.Conv1d(ch, ch, 1), nn.Dropout(0.1))
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        return x + self.conv(self.norm(x))


class Net(nn.Module):
    """Non-causal: possession is labelled after the fact, so the future of a frame is as useful as its past."""

    def __init__(self, n_in, ch=128, dilations=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64), n_out=3):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dilations])
        self.out = nn.Conv1d(ch, n_out, 1)

    def forward(self, x):          # x: batch, features, time
        return self.out(self.blocks(self.inp(x)))


class GRUNet(nn.Module):
    """A bidirectional GRU over the same inputs: a different kind of model, for a more varied ensemble."""

    def __init__(self, n_in, ch=128, n_out=3):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.gru = nn.GRU(ch, ch, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Conv1d(2 * ch, n_out, 1)

    def forward(self, x):
        h, _ = self.gru(self.inp(x).transpose(1, 2))
        return self.out(h.transpose(1, 2))


NETS = {"cnn": Net, "gru": GRUNet}
ARCH = ARGS.get("arch", "cnn")


def stage_input(out):
    """What a refinement stage reads: the H/A/D probabilities and, if there is one, the boundary probability."""
    return torch.cat([torch.softmax(out[:, :3], 1), torch.sigmoid(out[:, 3:])], 1)


class MultiStage(nn.Module):
    """The network, then stages that each re-read the previous stage's output over about 50 s (MS-TCN)."""

    def __init__(self, base, n_out, stages, ch=64):
        super().__init__()
        self.base = base
        self.refine = nn.ModuleList(Net(n_out, ch, (1, 2, 4, 8, 16, 32, 64, 128), n_out) for _ in range(stages - 1))

    def forward(self, x):          # every stage's output, the last one is the prediction
        outs = [self.base(x)]
        for stage in self.refine:
            outs.append(stage(stage_input(outs[-1])))
        return outs


def build(n_in, arch=ARCH, boundary=BOUNDARY > 0, stages=STAGES):
    """The plain network when no option is on (so earlier runs and saved networks are unchanged)."""
    n_out = 4 if boundary else 3
    base = NETS[arch](n_in, n_out=n_out)
    return base if stages == 1 else MultiStage(base, n_out, stages)


def outputs(net, x):
    out = net(x)
    return out if isinstance(out, list) else [out]


def boundary_target(y):
    """1 where the label changes between two graded steps, spread as a Gaussian over BD_SIGMA steps; -1 where a step
    is not graded (no loss there)."""
    t = np.zeros(len(y), np.float32)
    at = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < 3) & (y[:-1] < 3)) + 0.5
    if len(at):
        steps = np.arange(len(y))
        i = np.searchsorted(at, steps)
        near = np.minimum(np.abs(steps - at[np.clip(i - 1, 0, len(at) - 1)]), np.abs(steps - at[np.clip(i, 0, len(at) - 1)]))
        t = np.exp(-0.5 * (near / BD_SIGMA) ** 2).astype(np.float32)
    t[y >= 3] = -1
    return t


def segment_loss(outs, yb, bb, loss_fn):
    """H/A/D cross-entropy on every stage, plus the boundary and smoothing terms when they are on."""
    total = 0
    for out in outs:
        total = total + loss_fn(out[:, :3], yb)
        if BOUNDARY > 0:
            m = bb >= 0
            total = total + BOUNDARY * nn.functional.binary_cross_entropy_with_logits(out[:, 3][m], bb[m])
        if SMOOTH > 0:
            lp = torch.log_softmax(out[:, :3], 1)
            d = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
            both = ((yb[:, 1:] < 3) & (yb[:, :-1] < 3)).unsqueeze(1)
            total = total + SMOOTH * torch.clamp(d, max=16)[both.expand_as(d)].mean()
    return total


def load():
    matches = {}
    for p in sorted(DATA.rglob("*.npz")):   # Kaggle mounts datasets one or two folders down
        if p.stem == "cv_seq":               # earlier runs' outputs, not a match
            continue
        z = np.load(p)
        matches[p.stem] = {k: z[k] for k in z.files}
    return matches


def scaling(matches, names):
    x = np.concatenate([matches[n][k + COPIES][::10].astype(np.float32) for n in names for k in ("x", "xs")])
    mean, std = np.nanmean(x, 0), np.nanstd(x, 0)
    return np.nan_to_num(mean), np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)


def prep(x, mean, std):
    x = (x.astype(np.float32) - mean) / std
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)


def train_net(matches, names, mean, std):
    seqs, weight = [], []
    for n in names:
        for kx, ky in (("x" + COPIES, "y"), ("xs" + COPIES, "ys")):
            if len(matches[n][ky]) > WINDOW:
                y = matches[n][ky].astype(np.int64)
                seqs.append((torch.from_numpy(prep(matches[n][kx], mean, std)), torch.from_numpy(y),
                             torch.from_numpy(boundary_target(y))))
                weight.append(1.0)
    sizes = np.array([len(y) for _, y, _ in seqs], float) * weight
    net = build(len(mean)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    total = EPOCHS * STEPS_PER_EPOCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total)
    loss_fn = nn.CrossEntropyLoss(ignore_index=3)
    net.train()
    for it in range(total):
        pick = rng.choice(len(seqs), BATCH, p=sizes / sizes.sum())
        xb, yb, bb = [], [], []
        for i in pick:
            x, y, b = seqs[i]
            s = rng.integers(0, len(y) - WINDOW)
            xb.append(x[s:s + WINDOW])
            yb.append(y[s:s + WINDOW])
            bb.append(b[s:s + WINDOW])
        xb = torch.stack(xb).transpose(1, 2).to(DEV)
        yb = torch.stack(yb).to(DEV)
        bb = torch.stack(bb).to(DEV)
        loss = segment_loss(outputs(net, xb), yb, bb, loss_fn)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % STEPS_PER_EPOCH == 0:
            print(f"    epoch {(it + 1) // STEPS_PER_EPOCH}: loss {loss.item():.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, x, mean, std, with_boundary=False):
    """H/A/D probabilities per step from the last stage (and the boundary probability, if asked and learnt)."""
    net.eval()
    xt = torch.from_numpy(prep(x, mean, std)).T[None].to(DEV)
    out = outputs(net, xt)[-1][0]
    prob = torch.softmax(out[:3], 0).T.cpu().numpy()
    if with_boundary:
        return prob, torch.sigmoid(out[3]).cpu().numpy() if len(out) > 3 else None
    return prob


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = load()
    names = list(matches)
    held = [n for n in names if not n.startswith(TRAIN_ONLY)]
    folds = held if ARGS.get("folds", "all") == "all" else [] if ARGS["folds"] == "none" else ARGS["folds"].split(",")
    print(f"{len(names)} matches, {len(held)} held out in turn, device {DEV}", flush=True)
    probs, bds = {t: {} for t in TESTS}, {t: {} for t in TESTS}
    for name in folds:
        t0 = time.time()
        rest = [n for n in names if n != name]
        mean, std = scaling(matches, rest)
        net = train_net(matches, rest, mean, std)
        y = matches[name]["y"]
        g = y < 3
        for k, t in enumerate(TESTS):
            probs[t][name], bd = predict(net, matches[name][t], mean, std, with_boundary=True)
            if bd is not None:
                bds[t][name] = bd
            acc = np.mean(probs[t][name].argmax(1)[g] == y[g]) if len(y) == len(probs[t][name]) else float("nan")
            print(f"  fold {name}: {time.time() - t0:.0f} s, step accuracy {acc:.3f} on {t}", flush=True)
            np.savez_compressed(OUT / ("cv_seq.npz" if k == 0 else f"cv_seq_{t}.npz"), names=np.array(list(probs[t])),
                                **{f"prob_{n}": p for n, p in probs[t].items()},
                                **{f"bd_{n}": b for n, b in bds[t].items()})
    if ARGS.get("final", "1") == "1":
        mean, std = scaling(matches, names)
        net = train_net(matches, names, mean, std)
        torch.save({"state": net.state_dict(), "mean": mean, "std": std, "n_in": len(mean), "arch": ARCH,
                    "boundary": BOUNDARY > 0, "stages": STAGES}, OUT / "seq_final.pt")
        print("saved final network", flush=True)
    run = {"epochs": EPOCHS, "window": WINDOW, "seed": SEED, "arch": ARCH, "folds": folds}
    if BOUNDARY > 0 or STAGES > 1 or SMOOTH > 0:
        run.update(boundary=BOUNDARY, bd_sigma=BD_SIGMA, stages=STAGES, smooth=SMOOTH)
    if COPIES or TESTS != ["xh"]:
        run.update(copies=COPIES, test=TESTS)
    (OUT / "seq_run.json").write_text(json.dumps(run))


if __name__ == "__main__":
    main()
