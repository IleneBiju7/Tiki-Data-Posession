"""Run the sequence networks (seq_train.py seq_final.pt) on one feature sequence and average them. Its own process:
torch and lightgbm do not share one (predict_stack.py --seq calls this).

Usage: python seq_predict.py <features.npz with x> <out.npz> <seq_final.pt>...
"""
import sys

import numpy as np
import torch

import seq_train


def main():
    x = np.load(sys.argv[1])["x"]
    probs = []
    for path in sys.argv[3:]:
        saved = torch.load(path, map_location="cpu", weights_only=False)   # our own file: weights plus the scaling
        net = seq_train.build(saved["n_in"], saved.get("arch", "cnn"), saved.get("boundary", False),
                              saved.get("stages", 1)).to(seq_train.DEV)
        net.load_state_dict(saved["state"])
        probs.append(seq_train.predict(net, x, saved["mean"], saved["std"]))
    np.savez_compressed(sys.argv[2], prob=np.mean(probs, 0))
    print(f"{len(probs)} networks on {len(x)} steps", flush=True)


if __name__ == "__main__":
    main()
