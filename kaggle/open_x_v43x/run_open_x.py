"""One CPU step of the --external rebuild on Kaggle (organisers, 3 Nov: train and tune on external data only).
Input: dataset possession-open-x (our code, the organisers' scoring code possession_metrics.py and the open
matches: DFL, Metrica, PFF). No competition data.
STEP below is set per notebook: train.py for one first-stage model, or seq_export.py for the sequence networks.
Everything train.py / seq_export.py write to models/ ends up in the notebook's output."""
import glob
import os
import shutil
import subprocess
import sys

STEP = ['train.py', '--tag=v43x', '--det-noise']
FLAGS = "--mixed-tags --sticky --hard-test --big --smooth --pff --trk --keeper --sp-calib --contacts --height --external"
EXPORT_FLAGS = "--mixed-tags --sticky --smooth --pff --trk --keeper --sp-calib --det-noise --contacts --height --external"

src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/mtah-tools/train.py", recursive=True))[0])
data = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/open/converted_pff", recursive=True))[0]))
work = "/tmp/run"
shutil.copytree(src, f"{work}/mtah-tools")
os.symlink(data, f"{work}/data")
official = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/evaluation/possession_metrics.py", recursive=True))[0]))
os.symlink(official, f"{work}/official-repo")   # the organisers' scoring code (MIT licence), nothing else
os.makedirs(f"{work}/mtah-tools/models", exist_ok=True)
flags = (EXPORT_FLAGS if STEP[0] == "seq_export.py" else FLAGS).split()
print("running", STEP + flags, flush=True)
subprocess.run([sys.executable, *STEP, *flags], cwd=f"{work}/mtah-tools", check=True)
out = f"{work}/mtah-tools/models"
if STEP[0] == "seq_export.py":
    out += "/seq_x"
for p in glob.glob(f"{out}/*"):
    if os.path.isfile(p):
        shutil.copy(p, "/kaggle/working/")
print("outputs:", sorted(os.listdir("/kaggle/working")), flush=True)
