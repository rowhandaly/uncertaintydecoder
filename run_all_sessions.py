"""
Run the whole pipeline on every session, for each analysis, then pool.

For each session folder and each analysis (a brain area + a spike window):
    step 0  load the session, count spikes in the window
    step 2  decode likelihoods (all decoders, cross-validated)
    step 3  describe the shared axes
    step 4  predict choice and wager from the likelihood
    step 6  decode the wager (and choice) directly; population directions
and, once a session has both MT and LIP counted in the same window:
    step 7  does MT predict LIP's position along its wager direction?
Then for each analysis:
    step 5  pool all sessions

The early windows (before and just after motion onset) only run steps 0,
6 and 7: they ask WHEN the wager signal appears, not about likelihoods
(there is little or no stimulus response to decode that early).

Each step's printed output goes to logs/<session>__<analysis>_stepN.txt.
A session that fails a check in step 0 (e.g. too few units) is skipped and
listed at the end; the others carry on.

Each step only runs if its output file doesn't exist yet, so new steps or
analyses can be added without redoing old ones. To redo something, delete
its files (data/*__<analysis>*, figures/*__<analysis>*).

Usage (on Rockfish, from the repository folder):
    python run_all_sessions.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
or as a batch job:  sbatch run_all_sessions.sbatch
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
from step0_load_session import window_tag

MT = ["MT", "MST"]
LIP = ["LIP"]

# Analyses to run. All windows are 250 ms long.
#   align:          "onset" or "offset" (motion onset, or motion offset = saccade)
#   start:          window start in s relative to that event
#   rt_band:        keep only trials with RT in this range (or None)
#   max_coherence:  keep only trials with |coherence| up to this (or None)
#   allow_early:    allow a window before MT's response latency
#   steps:          which per-session steps to run (default: all)
ALL_STEPS = ("step0", "step2", "step3", "step4", "step6")
EARLY_STEPS = ("step0", "step6")
ANALYSES = []
for regions in [MT, LIP]:
    ANALYSES += [
        dict(regions=regions, align="onset", start=0.08),     # 80-330 ms
        dict(regions=regions, align="onset", start=0.18),     # 180-430 ms
        dict(regions=regions, align="onset", start=0.28),     # 280-530 ms
        dict(regions=regions, align="offset", start=-0.30),   # last 250 ms before the saccade
        # Same pre-saccade window, only RTs of 0.5-0.8 s, so the window sits
        # at nearly the same time after motion onset on every trial.
        dict(regions=regions, align="offset", start=-0.30, rt_band=(0.5, 0.8)),
        # When does the wager signal start? The 250 ms before the dots
        # appear (no evidence yet), then two short windows after onset.
        dict(regions=regions, align="onset", start=-0.25, allow_early=True,
             steps=EARLY_STEPS),
        dict(regions=regions, align="onset", start=0.0, length=0.15,
             allow_early=True, steps=EARLY_STEPS),
        dict(regions=regions, align="onset", start=0.15, length=0.15,
             allow_early=True, steps=EARLY_STEPS),
    ]
# MT's early window with only the difficult coherences (|coh| <= 12.8%):
# is the second shared axis needed there too, or only for strong motion?
ANALYSES.append(dict(regions=MT, align="onset", start=0.08, max_coherence=0.128))

for analysis in ANALYSES:
    analysis.setdefault("length", 0.25)
    analysis.setdefault("rt_band", None)
    analysis.setdefault("max_coherence", None)
    analysis.setdefault("allow_early", False)
    analysis.setdefault("steps", ALL_STEPS)
    analysis["tag"] = window_tag(analysis["align"], analysis["start"],
                                 analysis["length"], analysis["rt_band"],
                                 analysis["regions"], analysis["max_coherence"])

data_root = Path(sys.argv[1])
session_folders = sorted(p for p in data_root.iterdir()
                         if p.is_dir() and not p.name.startswith("@"))
Path("logs").mkdir(exist_ok=True)
print(f"{len(session_folders)} sessions, {len(ANALYSES)} analyses\n")


def run(command, log_file):
    """Run one step, save its output to a log. Returns True if it worked."""
    with open(log_file, "w") as log:
        result = subprocess.run([sys.executable, *command], stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0


problems = []
for folder in session_folders:
    for analysis in ANALYSES:
        name = f"{folder.name}__{analysis['tag']}"
        session_file = f"data/{name}.npz"
        print(f"{name} ...", end=" ", flush=True)

        # Each step and the file it makes; a step is skipped if its file exists.
        step0 = ["step0_load_session.py", str(folder),
                 "--align", analysis["align"],
                 "--start", str(analysis["start"]),
                 "--length", str(analysis["length"]),
                 "--regions", *analysis["regions"]]
        if analysis["rt_band"] is not None:
            step0 += ["--rt-band", *map(str, analysis["rt_band"])]
        if analysis["max_coherence"] is not None:
            step0 += ["--max-coherence", str(analysis["max_coherence"])]
        if analysis["allow_early"]:
            step0 += ["--allow-early"]

        # Session files made before trial history was added: redo step 0
        # (same counts, plus history) and step 6 (which now uses it).
        if Path(session_file).exists() and "prev_wager" not in np.load(session_file).files:
            Path(session_file).unlink()
            Path(f"data/{name}_wager.npz").unlink(missing_ok=True)

        steps = [
            ("step0", step0, session_file),
            ("step2", ["step2_decode_likelihoods.py", session_file], f"data/{name}_decoded.npz"),
            ("step3", ["step3_shared_axes.py", session_file], f"figures/{name}_step3.png"),
            ("step4", ["step4_predict_behaviour.py", session_file], f"data/{name}_behaviour.npz"),
            ("step6", ["step6_wager_decoder.py", session_file], f"data/{name}_wager.npz"),
        ]
        steps = [step for step in steps if step[0] in analysis["steps"]]

        ran = []
        for label, command, output in steps:
            if Path(output).exists():
                continue
            log = f"logs/{name}_{label}.txt"
            if not run(command, log):
                # The last line of the log says why (e.g. "SKIP: ...").
                reason = Path(log).read_text().strip().splitlines()[-1]
                print(f"{'skipped' if label == 'step0' else 'FAILED at ' + label}: {reason}")
                problems.append(f"{name}: {label}: {reason}")
                break
            ran.append(label)
        else:
            print(f"done ({', '.join(ran)})" if ran else "already done")

    # Step 7, for every window where this session has both MT and LIP.
    for analysis in ANALYSES:
        if analysis["regions"] != MT:
            continue
        name = f"{folder.name}__{analysis['tag']}"
        mt_file = Path(f"data/{name}.npz")
        lip_file = Path(f"data/{folder.name}__LIP_{analysis['tag']}.npz")
        output = Path(f"data/{name}_mt_lip.npz")
        if mt_file.exists() and lip_file.exists() and not output.exists():
            print(f"{name} step7 ...", end=" ", flush=True)
            log = f"logs/{name}_step7.txt"
            if run(["step7_mt_to_lip.py", str(mt_file)], log):
                print("done")
            else:
                reason = Path(log).read_text().strip().splitlines()[-1]
                print(f"FAILED: {reason}")
                problems.append(f"{name}: step7: {reason}")

print()
for analysis in ANALYSES:
    tag = analysis["tag"]
    ok = run(["step5_pool_sessions.py", tag], f"logs/pooled__{tag}.txt")
    print(f"pooled {tag}: {'logs/pooled__' + tag + '.txt' if ok else 'FAILED'}")

if problems:
    print("\nproblems:")
    for p in problems:
        print(f"  {p}")
