"""One CPU step on Kaggle for the SkillCorner / team-swap tests (open data only, no competition data).
Inputs: possession-open-x (DFL, Metrica, PFF) and possession-open-sc (newer code, SkillCorner, the organisers' scoring
code possession_metrics.py, MIT). The code comes from possession-open-sc; both datasets' matches are linked into one
data/open/converted folder. STEP is set per notebook; everything written to models/ ends up in the output."""
import glob
import os
import shutil
import subprocess
import sys

STEP = ['seq_export.py', '--skillcorner', '--swap-tta']
FLAGS = "--mixed-tags --sticky --hard-test --big --smooth --pff --trk --keeper --sp-calib --contacts --height --external"
EXPORT_FLAGS = "--mixed-tags --sticky --smooth --pff --trk --keeper --sp-calib --det-noise --contacts --height --external"

new_code = [p for p in glob.glob("/kaggle/input/**/mtah-tools/train.py", recursive=True) if "open-sc" in p]
src = os.path.dirname(sorted(new_code)[0])
work = "/tmp/run"
shutil.copytree(src, f"{work}/mtah-tools")
os.makedirs(f"{work}/data/open/converted")
for p in glob.glob("/kaggle/input/**/open/converted/*.npz", recursive=True):
    dst = f"{work}/data/open/converted/{os.path.basename(p)}"
    if not os.path.exists(dst):
        os.symlink(p, dst)
os.symlink(sorted(glob.glob("/kaggle/input/**/open/converted_pff", recursive=True))[0], f"{work}/data/open/converted_pff")
official = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/evaluation/possession_metrics.py", recursive=True))[0]))
os.symlink(official, f"{work}/official-repo")
os.makedirs(f"{work}/mtah-tools/models", exist_ok=True)
print("matches:", len(os.listdir(f"{work}/data/open/converted")), "+ PFF", flush=True)
flags = (EXPORT_FLAGS if STEP[0] == "seq_export.py" else FLAGS).split()
print("running", STEP + flags, flush=True)
subprocess.run([sys.executable, *STEP, *flags], cwd=f"{work}/mtah-tools", check=True)
outs = glob.glob(f"{work}/mtah-tools/models/seq_x*/*") if STEP[0] == "seq_export.py" else glob.glob(f"{work}/mtah-tools/models/*")
for p in outs:
    if os.path.isfile(p):
        shutil.copy(p, "/kaggle/working/")
print("outputs:", sorted(os.listdir("/kaggle/working")), flush=True)
