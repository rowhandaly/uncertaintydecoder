"""
Run the whole pipeline on every session, for each spike window, then pool.

For each session folder and each window:
    step 0  load the session, count spikes in the window
    step 2  decode likelihoods (all decoders, cross-validated)
    step 3  describe the shared axes
    step 4  predict choice and wager
Then for each window:
    step 5  pool all sessions

Each step's printed output goes to logs/<session>__<window>_stepN.txt.
A session that fails a check in step 0 (e.g. no labelled MT units) is
skipped and listed at the end; the others carry on.

Usage (on Rockfish, from the repository folder):
    python run_all_sessions.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
or as a batch job:  sbatch run_all_sessions.sbatch
"""

import subprocess
import sys
from pathlib import Path
from step0_load_session import window_tag

# Windows to run: (aligned to, start in s, length in s). All 250 ms long.
WINDOWS = [
    ("onset", 0.08, 0.25),     # early stimulus response (80-330 ms)
    ("onset", 0.18, 0.25),     # 180-430 ms
    ("onset", 0.28, 0.25),     # 280-530 ms
    ("offset", -0.30, 0.25),   # last part of the stimulus, before the saccade
]

data_root = Path(sys.argv[1])
session_folders = sorted(p for p in data_root.iterdir()
                         if p.is_dir() and not p.name.startswith("@"))
Path("logs").mkdir(exist_ok=True)
print(f"{len(session_folders)} sessions, {len(WINDOWS)} windows\n")


def run(command, log_file):
    """Run one step, save its output to a log. Returns True if it worked."""
    with open(log_file, "w") as log:
        result = subprocess.run([sys.executable, *command], stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0


problems = []
for folder in session_folders:
    for align, start, length in WINDOWS:
        tag = window_tag(align, start, length)
        name = f"{folder.name}__{tag}"
        print(f"{name} ...", end=" ", flush=True)

        ok = run(["step0_load_session.py", str(folder), "--align", align,
                  "--start", str(start), "--length", str(length)],
                 f"logs/{name}_step0.txt")
        if not ok:
            # The last line of the log says why (e.g. "SKIP: ...").
            reason = Path(f"logs/{name}_step0.txt").read_text().strip().splitlines()[-1]
            print(f"skipped: {reason}")
            problems.append(f"{name}: {reason}")
            continue

        session_file = f"data/{name}.npz"
        for step in ["step2_decode_likelihoods.py", "step3_shared_axes.py",
                     "step4_predict_behaviour.py"]:
            if not run([step, session_file], f"logs/{name}_{step[:5]}.txt"):
                print(f"FAILED at {step} (see logs/{name}_{step[:5]}.txt)")
                problems.append(f"{name}: failed at {step}")
                break
        else:
            print("done")

print()
for align, start, length in WINDOWS:
    tag = window_tag(align, start, length)
    ok = run(["step5_pool_sessions.py", tag], f"logs/pooled__{tag}.txt")
    print(f"pooled {tag}: {'logs/pooled__' + tag + '.txt' if ok else 'FAILED'}")

if problems:
    print("\nproblems:")
    for p in problems:
        print(f"  {p}")
