"""eval_chains.py's TV-scale detections for other weights, into their own cache (same crops, same seed).
Usage: python tv_eval.py <weights name>   (run from mtah-tools)
"""
import sys

import eval_chains as ec

name = sys.argv[1]
ec.CACHE = ec.CACHE.with_name(f"soccernet_chain_dets_{name}.npz")
ec.detect(str(ec.HERE / "models" / "weights" / f"{name}.pt"))
print("->", ec.CACHE)
