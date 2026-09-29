"""
Pool every session for one analysis. Each comparison is reported two ways:
all trials together, and one mean per session tested against zero.

Usage:  python pool.py onset+80_250ms
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

analysis = sys.argv[1]
sessions = sorted(f.name.split("__")[0] for f in Path("data").glob(f"*__{analysis}_behaviour.npz"))
if not sessions:
    raise SystemExit(f"no results for {analysis}")
print(f"{analysis}: {len(sessions)} sessions\n")


def load(session, suffix):
    return np.load(Path("data") / f"{session}__{analysis}{suffix}.npz")


def report(label, differences_per_session):
    all_trials = np.concatenate(differences_per_session)
    standard_error = all_trials.std() / np.sqrt(len(all_trials))
    session_means = np.array([differences.mean() for differences in differences_per_session])
    across_sessions = ttest_1samp(session_means, 0)
    print(f"  {label:48s} {all_trials.mean():+.4f} ± {standard_error:.4f}   "
          f"sessions positive {np.sum(session_means > 0)}/{len(sessions)}, "
          f"p = {across_sessions.pvalue:.2g}")


print("decoding coherence (A - B, held-out log-likelihood per trial):")
for a, b in [("rank 2", "rank 1"), ("rank 3", "rank 2"), ("rank 2", "full"), ("rank 2", "fixed width")]:
    differences = []
    for session in sessions:
        saved = load(session, "_decoded")
        scores = dict(zip(saved["names"], saved["scores"]))
        differences.append(scores[a] - scores[b])
    report(f"{a} - {b}", differences)

comparisons = [("likelihood beyond stimulus", "fixed width", "baseline"),
               ("Walker's test", "rank 2", "fixed width"),
               ("extra axis", "rank 2", "rank 1"),
               ("shuffle control", "shuffled", "fixed width"),
               ("neurons directly", "direct", "baseline")]
for behaviour in ["choice", "wager"]:
    print(f"\npredicting the {behaviour} (A - B, held-out log-likelihood per trial):")
    for question, a, b in comparisons:
        differences = []
        for session in sessions:
            saved = load(session, "_behaviour")
            scores = dict(zip(saved["names"], saved[f"{behaviour}_scores"]))
            differences.append(scores[a] - scores[b])
        report(f"{question} ({a} - {b})", differences)

print("\nsession                      trials  units")
for session in sessions:
    counts = load(session, "")["counts"]
    print(f"{session:28s} {counts.shape[0]:6d} {counts.shape[1]:6d}")
