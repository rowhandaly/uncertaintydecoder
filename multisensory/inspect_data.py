"""
Inventory of the zarya visual-vestibular sessions, so the new notebook can be
written against the real data format.

For each session it lists the files, the shape and range of every .npy, the
columns of every .tsv, and then looks for the trial/behaviour files (heading,
modality, coherence, choice, wager, stimulus times), which the kilosort folder
doesn't contain. Everything is printed and saved to data_inventory.txt.

Run on Rockfish:   python inspect_data.py      (or  %run inspect_data.py  in Jupyter)
Then send back data_inventory.txt.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cfetsch1/vast-cfetsch1/data/zarya/wavemap")
SESSIONS = ["060225", "030625", "052325", "071025", "041725", "070225"]   # mmddyy
SEARCH = ROOT.parent          # where to look for behaviour files
MAX_DEPTH = 4

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def describe_npy(path):
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    info = f"shape {a.shape}, {a.dtype}"
    if a.size and np.issubdtype(a.dtype, np.number):
        flat = np.asarray(a).ravel()
        info += f", min {flat.min()}, max {flat.max()}"
        if flat.size <= 2_000_000 or path.name in ("spike_clusters.npy", "spike_templates.npy"):
            info += f", {len(np.unique(flat))} unique"
    return info


def describe_tsv(path):
    t = pd.read_csv(path, sep="\t")
    say(f"      columns: {list(t.columns)}, {len(t)} rows")
    for col in t.columns[1:]:
        if t[col].dtype == object:
            say(f"      {col}: {t[col].value_counts().to_dict()}")
    say("      " + t.head(3).to_string().replace("\n", "\n      "))


def describe_mat(path):
    try:
        from scipy.io import whosmat
        for name, shape, kind in whosmat(path):
            say(f"      {name}: {shape} {kind}")
    except (NotImplementedError, ValueError):       # MATLAB v7.3 files are HDF5
        import h5py
        with h5py.File(path, "r") as f:
            f.visititems(lambda name, obj: say(f"      {name}: {getattr(obj, 'shape', '')} {getattr(obj, 'dtype', '')}"))


say(f"top level of {ROOT}:")
for p in sorted(ROOT.iterdir())[:200]:
    say(f"  {p.name}{'/' if p.is_dir() else ''}")

for s in SESSIONS:
    folder = ROOT / s
    say(f"\n===== {s} =====")
    if not folder.exists():
        say("  MISSING")
        continue
    for p in sorted(folder.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(folder)
        say(f"  {rel}  ({p.stat().st_size / 1e6:.1f} MB)")
        try:
            if p.suffix == ".npy":
                say(f"      {describe_npy(p)}")
            elif p.suffix == ".tsv":
                describe_tsv(p)
            elif p.suffix == ".mat":
                describe_mat(p)
            elif p.suffix in (".py", ".txt", ".json") and p.stat().st_size < 20_000:
                say("      " + p.read_text(errors="replace").replace("\n", "\n      "))
        except Exception as problem:
            say(f"      could not read: {problem}")

# behaviour lives somewhere else: find anything named after a session date
say(f"\n===== files under {SEARCH} named after a session (depth {MAX_DEPTH}) =====")
found = {s: [] for s in SESSIONS}
start_depth = len(SEARCH.parts)
for dirpath, dirnames, filenames in os.walk(SEARCH):
    if len(Path(dirpath).parts) - start_depth >= MAX_DEPTH:
        dirnames[:] = []
    if Path(dirpath) == ROOT:
        dirnames[:] = []                      # already listed above
    for name in dirnames + filenames:
        for s in SESSIONS:
            dates = [s, f"20{s[4:]}{s[:4]}", f"20{s[4:]}-{s[:2]}-{s[2:4]}", f"{s[4:]}{s[:4]}"]
            if any(d in name for d in dates):
                found[s].append(Path(dirpath) / name)
for s, paths in found.items():
    say(f"  {s}: {len(paths)} matches")
    for p in paths[:20]:
        say(f"    {p}")

# peek inside the first behaviour-looking file of one session
for p in found[SESSIONS[3]]:
    if p.suffix in (".mat", ".csv", ".h5", ".hdf5", ".pkl"):
        say(f"\n===== contents of {p} =====")
        try:
            if p.suffix == ".mat":
                describe_mat(p)
            elif p.suffix == ".csv":
                t = pd.read_csv(p)
                say(f"  columns: {list(t.columns)}, {len(t)} rows")
                say("  " + t.head(5).to_string().replace("\n", "\n  "))
            else:
                say(f"  ({p.suffix} file, not opened)")
        except Exception as problem:
            say(f"  could not read: {problem}")
        break

Path("data_inventory.txt").write_text("\n".join(lines))
print("\nsaved data_inventory.txt")
