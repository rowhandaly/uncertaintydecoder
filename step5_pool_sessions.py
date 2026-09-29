"""
Step 5: pool the results of all sessions analysed with one window.

Each session has its own decoders (its units are different), so sessions
are combined at the level of results, not data. For every comparison we
report two things:

  pooled trials   all trials from all sessions together: the most
                  sensitive test, like Walker et al.'s paired t-tests
  across sessions one number per session (its mean difference), then a
                  t-test of those numbers against 0: asks whether the
                  effect is consistent, so a single session can't drive it

Reads, for each session, the files steps 2 and 4 saved:
    data/<session>__<window>_decoded.npz     (decoding scores)
    data/<session>__<window>_behaviour.npz   (behaviour scores)

Usage:
    python step5_pool_sessions.py onset+80_250ms
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

window = sys.argv[1]
behaviour_files = sorted(Path("data").glob(f"*__{window}_behaviour.npz"))
if not behaviour_files:
    raise SystemExit(f"no files matching data/*__{window}_behaviour.npz")
sessions = [f.name.split("__")[0] for f in behaviour_files]
print(f"window {window}: {len(sessions)} sessions\n")


def summarise(label, differences_per_session):
    """Print the pooled-trials and across-sessions tests for one comparison.

    differences_per_session: one array of per-trial differences (A - B)
    per session. Positive = A predicts better.
    """
    pooled = np.concatenate(differences_per_session)
    pooled_sem = pooled.std() / np.sqrt(len(pooled))
    pooled_t = pooled.mean() / pooled_sem
    session_means = np.array([d.mean() for d in differences_per_session])
    across = ttest_1samp(session_means, 0)
    n_positive = np.sum(session_means > 0)
    print(f"  {label:38s} pooled {pooled.mean():+.4f} ± {pooled_sem:.4f}"
          f" (t = {pooled_t:+.2f})   sessions: {n_positive}/{len(session_means)}"
          f" positive, t = {across.statistic:+.2f}, p = {across.pvalue:.2g}")
    return session_means


# ---------------------------------------------------------------------------
# 1. Decoding: does the second shared axis help? (from step 2)
# ---------------------------------------------------------------------------
print("decoding coherence (held-out log-likelihood per trial, A - B):")
decoding = {"rank 2 - rank 1": [], "rank 3 - rank 2": [], "rank 2 - full": []}
n_trials, n_units = [], []
for session in sessions:
    decoded = np.load(Path("data") / f"{session}__{window}_decoded.npz")
    names = list(decoded["names"])
    scores = dict(zip(names, decoded["scores"]))
    for label in decoding:
        a, b = label.split(" - ")
        decoding[label].append(scores[a] - scores[b])
    data = np.load(Path("data") / f"{session}__{window}.npz")
    n_trials.append(data["counts"].shape[0])
    n_units.append(data["counts"].shape[1])
for label, differences in decoding.items():
    summarise(label, differences)
print(f"  ({sum(n_trials)} trials; {min(n_units)}-{max(n_units)} units per session)\n")

# ---------------------------------------------------------------------------
# 2. Behaviour (from step 4)
# ---------------------------------------------------------------------------
comparisons = [
    ("neurons add anything?", "fixed width", "baseline"),
    ("Walker's test", "rank 2", "fixed width"),
    ("extra axis", "rank 2", "rank 1"),
    ("full vs fixed", "full", "fixed width"),
    ("shuffle control", "shuffled", "fixed width"),
]
per_session = {}
for behaviour in ["choice", "wager"]:
    print(f"predicting {behaviour} (held-out log-likelihood per trial, A - B):")
    for question, a, b in comparisons:
        differences = []
        for f in behaviour_files:
            saved = np.load(f)
            scores = dict(zip(saved["names"], saved[f"{behaviour}_scores"]))
            differences.append(scores[a] - scores[b])
        per_session[(behaviour, question)] = summarise(
            f"{question} ({a} - {b})", differences)
    print()

print("per session:")
print(f"  {'session':28s} {'trials':>6s} {'units':>5s}   wager: neurons   Walker's test")
wager_neurons = per_session[("wager", "neurons add anything?")]
wager_walker = per_session[("wager", "Walker's test")]
for i, session in enumerate(sessions):
    print(f"  {session:28s} {n_trials[i]:6d} {n_units[i]:5d}"
          f"   {wager_neurons[i]:+14.4f}   {wager_walker[i]:+13.4f}")

# ---------------------------------------------------------------------------
# 2b. Direct wager decoder and population directions (from step 6)
# ---------------------------------------------------------------------------
wager_files = [Path("data") / f"{s}__{window}_wager.npz" for s in sessions]
if all(f.exists() for f in wager_files):
    print("\ndirect decoders (held-out log-likelihood per trial, A - B):")
    direct = {"wager": [], "choice": [], "wager direct vs likelihood": []}
    for f_wager, f_behaviour in zip(wager_files, behaviour_files):
        saved = np.load(f_wager)
        scores = dict(zip(saved["names"], saved["scores"]))
        for label in ["wager", "choice"]:
            direct[label].append(scores[f"{label} direct"] - scores[f"{label} baseline"])
        # Does decoding the wager directly beat going through the
        # likelihood (step 4's rank-2 features)? Each is measured against
        # its own baseline, so the two gains can be compared trial by trial.
        step4 = np.load(f_behaviour)
        step4_scores = dict(zip(step4["names"], step4["wager_scores"]))
        likelihood_gain = step4_scores["rank 2"] - step4_scores["baseline"]
        direct["wager direct vs likelihood"].append(
            (scores["wager direct"] - scores["wager baseline"]) - likelihood_gain)
    summarise("wager: direct decoder - baseline", direct["wager"])
    summarise("choice: direct decoder - baseline", direct["choice"])
    summarise("wager: direct gain - likelihood gain", direct["wager direct vs likelihood"])

    print("\nangles between population directions (degrees; 90 = orthogonal)")
    print("null = wager decoder refitted with wagers shuffled within coherence")
    print(f"  {'session':28s} {'units':>5s}   wager-coh (null)   p(aligned)"
          f"   wager-choice (null)   p(aligned)   choice-coh")
    observed_minus_null = {"wager vs coherence axis": [], "wager vs choice": []}
    for i, f in enumerate(wager_files):
        saved = np.load(f)
        angles = dict(zip(saved["angle_names"], saved["angles"]))
        null_coh = saved["null_wager_coherence"]
        null_choice = saved["null_wager_choice"]
        observed_minus_null["wager vs coherence axis"].append(
            angles["wager vs coherence axis"] - null_coh.mean())
        observed_minus_null["wager vs choice"].append(
            angles["wager vs choice"] - null_choice.mean())
        print(f"  {sessions[i]:28s} {n_units[i]:5d}"
              f"   {angles['wager vs coherence axis']:5.1f} ({null_coh.mean():4.1f})"
              f"   {np.mean(null_coh <= angles['wager vs coherence axis']):10.2f}"
              f"   {angles['wager vs choice']:8.1f} ({null_choice.mean():4.1f})"
              f"   {np.mean(null_choice <= angles['wager vs choice']):12.2f}"
              f"   {angles['choice vs coherence axis']:10.1f}")
    # Across sessions: is the wager direction consistently closer to (or
    # further from) the other directions than a noise-fitted decoder?
    for name, values in observed_minus_null.items():
        values = np.array(values)
        test = ttest_1samp(values, 0)
        print(f"  {name}: observed - null = {values.mean():+.1f} deg on average "
              f"(negative = more aligned than chance), t = {test.statistic:+.2f}, "
              f"p = {test.pvalue:.2g}")

# ---------------------------------------------------------------------------
# 3. Figure: one dot per session, for the key comparisons.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, behaviour in zip(axes, ["choice", "wager"]):
    for x, (question, a, b) in enumerate(comparisons):
        values = per_session[(behaviour, question)]
        ax.plot(np.full(len(values), x) + np.random.uniform(-0.1, 0.1, len(values)),
                values, "o", color="gray", alpha=0.6)
        ax.plot(x, values.mean(), "ks", markersize=8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(len(comparisons)),
                  [q for q, _, _ in comparisons], rotation=30, ha="right")
    ax.set_title(f"predicting the {behaviour}")
axes[0].set_ylabel("held-out log-likelihood per trial\n(A - B), one dot per session")
fig.suptitle(f"all sessions, window {window}")
fig.tight_layout()
fig.savefig(f"figures/pooled__{window}.png", dpi=120)
print(f"\nsaved figures/pooled__{window}.png")
