"""
Run every session through load_session, decode, axes and behaviour, for
each analysis, then pool. A step is skipped if its output already exists.

Usage:  python run_all.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
"""

import subprocess
import sys
from pathlib import Path
from load_session import analysis_name

ANALYSES = [
    dict(start=0.08, length=0.25, max_coherence=None),     # 80-330 ms after onset
    dict(start=0.08, length=0.25, max_coherence=0.128),    # same, only |coh| <= 12.8%
]

data_root = Path(sys.argv[1])
session_folders = sorted(folder for folder in data_root.iterdir()
                         if folder.is_dir() and not folder.name.startswith("@"))
for folder_name in ["data", "figures", "logs"]:
    Path(folder_name).mkdir(exist_ok=True)


def run(command, log_file):
    with open(log_file, "w") as log:
        finished = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
    return finished.returncode == 0


problems = []
for folder in session_folders:
    for analysis in ANALYSES:
        name = analysis_name(analysis["start"], analysis["length"], analysis["max_coherence"])
        session_file = f"data/{folder.name}__{name}.npz"
        load_command = ["load_session.py", str(folder),
                        "--start", str(analysis["start"]), "--length", str(analysis["length"])]
        if analysis["max_coherence"] is not None:
            load_command += ["--max-coherence", str(analysis["max_coherence"])]

        steps = [("load", load_command, session_file),
                 ("decode", ["decode.py", session_file], f"data/{folder.name}__{name}_decoded.npz"),
                 ("axes", ["axes.py", session_file], f"data/{folder.name}__{name}_axes.npz"),
                 ("behaviour", ["behaviour.py", session_file], f"data/{folder.name}__{name}_behaviour.npz")]

        print(f"{folder.name}__{name} ...", end=" ", flush=True)
        steps_run = []
        for step, command, output in steps:
            if Path(output).exists():
                continue
            log_file = f"logs/{folder.name}__{name}_{step}.txt"
            if not run(command, log_file):
                last_line = Path(log_file).read_text().strip().splitlines()[-1]
                print(f"stopped at {step}: {last_line}")
                problems.append(f"{folder.name}__{name}: {step}: {last_line}")
                break
            steps_run.append(step)
        else:
            print(f"done ({', '.join(steps_run)})" if steps_run else "already done")

print()
for analysis in ANALYSES:
    name = analysis_name(analysis["start"], analysis["length"], analysis["max_coherence"])
    worked = run(["pool.py", name], f"logs/pooled__{name}.txt")
    print(f"pooled {name}: {'logs/pooled__' + name + '.txt' if worked else 'FAILED'}")

if problems:
    print("\nsessions left out:")
    for problem in problems:
        print(f"  {problem}")
