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

Reads, for each session, whichever of these files exist (each section
uses the sessions that have its file, and is skipped if none do):
    data/<session>__<window>_decoded.npz     (step 2: decoding scores)
    data/<session>__<window>_behaviour.npz   (step 4: behaviour scores)
    data/<session>__<window>_wager.npz       (step 6: direct decoders, angles)
    data/<session>__<window>_mt_lip.npz      (step 7: MT predicting LIP)

Usage:
    python step5_pool_sessions.py onset+80_250ms
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

window = sys.argv[1]
session_files = sorted(Path("data").glob(f"*__{window}.npz"))
if not session_files:
    raise SystemExit(f"no files matching data/*__{window}.npz")
all_sessions = [f.name.split("__")[0] for f in session_files]
print(f"window {window}: {len(all_sessions)} sessions\n")


def with_file(suffix):
    """The sessions that have data/<session>__<window><suffix>."""
    return [s for s in all_sessions
            if (Path("data") / f"{s}__{window}{suffix}").exists()]


def load(session, suffix):
    return np.load(Path("data") / f"{session}__{window}{suffix}")


n_trials, n_units = {}, {}
for session in all_sessions:
    counts = load(session, ".npz")["counts"]
    n_trials[session], n_units[session] = counts.shape


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
    print(f"  {label:50s} pooled {pooled.mean():+.4f} ± {pooled_sem:.4f}"
          f" (t = {pooled_t:+.2f})   sessions: {n_positive}/{len(session_means)}"
          f" positive, t = {across.statistic:+.2f}, p = {across.pvalue:.2g}")
    return session_means


# ---------------------------------------------------------------------------
# 1. Decoding: does the second shared axis help? (from step 2)
# ---------------------------------------------------------------------------
sessions = with_file("_decoded.npz")
if sessions:
    print(f"decoding coherence ({len(sessions)} sessions; "
          f"held-out log-likelihood per trial, A - B):")
    decoding = {"rank 2 - rank 1": [], "rank 3 - rank 2": [], "rank 2 - full": []}
    for session in sessions:
        decoded = load(session, "_decoded.npz")
        scores = dict(zip(decoded["names"], decoded["scores"]))
        for label in decoding:
            a, b = label.split(" - ")
            decoding[label].append(scores[a] - scores[b])
    for label, differences in decoding.items():
        summarise(label, differences)
    units = [n_units[s] for s in sessions]
    print(f"  ({sum(n_trials[s] for s in sessions)} trials; "
          f"{min(units)}-{max(units)} units per session)\n")

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
behaviour_sessions = with_file("_behaviour.npz")
per_session = {}
if behaviour_sessions:
    for behaviour in ["choice", "wager"]:
        print(f"predicting {behaviour} ({len(behaviour_sessions)} sessions; "
              f"held-out log-likelihood per trial, A - B):")
        for question, a, b in comparisons:
            differences = []
            for session in behaviour_sessions:
                saved = load(session, "_behaviour.npz")
                scores = dict(zip(saved["names"], saved[f"{behaviour}_scores"]))
                differences.append(scores[a] - scores[b])
            per_session[(behaviour, question)] = summarise(
                f"{question} ({a} - {b})", differences)
        print()

    print("per session:")
    print(f"  {'session':28s} {'trials':>6s} {'units':>5s}   wager: neurons   Walker's test")
    wager_neurons = per_session[("wager", "neurons add anything?")]
    wager_walker = per_session[("wager", "Walker's test")]
    for i, session in enumerate(behaviour_sessions):
        print(f"  {session:28s} {n_trials[session]:6d} {n_units[session]:5d}"
              f"   {wager_neurons[i]:+14.4f}   {wager_walker[i]:+13.4f}")

# ---------------------------------------------------------------------------
# 2b. Direct wager decoder and population directions (from step 6)
# ---------------------------------------------------------------------------
sessions = with_file("_wager.npz")
if sessions:
    print(f"\ndirect decoders ({len(sessions)} sessions; "
          f"held-out log-likelihood per trial, A - B):")
    direct = {}
    for session in sessions:
        saved = load(session, "_wager.npz")
        scores = dict(zip(saved["names"], saved["scores"]))
        pairs = {f"{b}: direct decoder - baseline": (f"{b} direct", f"{b} baseline")
                 for b in ["wager", "choice"]}
        # With trial history in the baseline (saved by newer runs of
        # step 6): does the neural signal survive it? And how much does
        # history predict on its own?
        for b in ["wager", "choice"]:
            pairs[f"{b}: direct - baseline, both +history"] = (
                f"{b} direct+history", f"{b} baseline+history")
            pairs[f"{b}: history alone (baseline+history - baseline)"] = (
                f"{b} baseline+history", f"{b} baseline")
        for label, (a, b) in pairs.items():
            if a in scores and b in scores:
                direct.setdefault(label, []).append(scores[a] - scores[b])
        # Does decoding the wager directly beat going through the
        # likelihood (step 4's rank-2 features)? Each is measured against
        # its own baseline, so the two gains can be compared trial by trial.
        if (Path("data") / f"{session}__{window}_behaviour.npz").exists():
            step4 = load(session, "_behaviour.npz")
            step4_scores = dict(zip(step4["names"], step4["wager_scores"]))
            likelihood_gain = step4_scores["rank 2"] - step4_scores["baseline"]
            direct.setdefault("wager: direct gain - likelihood gain", []).append(
                (scores["wager direct"] - scores["wager baseline"]) - likelihood_gain)
    for label, differences in direct.items():
        note = "" if len(differences) == len(sessions) else f"  [{len(differences)} sessions]"
        summarise(label + note, differences)

    print("\nangles between population directions (degrees; 90 = orthogonal)")
    print("null = wager decoder refitted with wagers shuffled within coherence")
    print(f"  {'session':28s} {'units':>5s}   wager-coh (null)   p(aligned)"
          f"   wager-choice (null)   p(aligned)   choice-coh")
    observed_minus_null = {"wager vs coherence axis": [], "wager vs choice": []}
    for session in sessions:
        saved = load(session, "_wager.npz")
        angles = dict(zip(saved["angle_names"], saved["angles"]))
        null_coh = saved["null_wager_coherence"]
        null_choice = saved["null_wager_choice"]
        observed_minus_null["wager vs coherence axis"].append(
            angles["wager vs coherence axis"] - null_coh.mean())
        observed_minus_null["wager vs choice"].append(
            angles["wager vs choice"] - null_choice.mean())
        print(f"  {session:28s} {n_units[session]:5d}"
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
# 2c. MT predicting LIP's wager and choice positions (from step 7)
# ---------------------------------------------------------------------------
sessions = with_file("_mt_lip.npz")
if sessions:
    print(f"\nMT predicting LIP's position along its directions ({len(sessions)} "
          f"sessions; held-out R^2, within coherence)")
    print(f"  {'session':28s} {'trials':>6s}   wager (shuffled MT)   p"
          f"      choice (shuffled MT)   p      random directions")
    beyond = {"wager - shuffled MT": [], "choice - shuffled MT": [],
              "wager - random directions": []}
    for session in sessions:
        saved = load(session, "_mt_lip.npz")
        r2_wager, r2_choice = float(saved["r2_wager"]), float(saved["r2_choice"])
        null_wager, null_choice = saved["null_wager"], saved["null_choice"]
        print(f"  {session:28s} {int(saved['n_trials']):6d}"
              f"   {r2_wager:+.4f} ({null_wager.mean():+.4f})"
              f"   {np.mean(null_wager >= r2_wager):4.2f}"
              f"   {r2_choice:+.4f} ({null_choice.mean():+.4f})"
              f"   {np.mean(null_choice >= r2_choice):4.2f}"
              f"   {saved['r2_random'].mean():+.4f}")
        beyond["wager - shuffled MT"].append(r2_wager - null_wager.mean())
        beyond["choice - shuffled MT"].append(r2_choice - null_choice.mean())
        beyond["wager - random directions"].append(r2_wager - saved["r2_random"].mean())
    # Across sessions: one number per session, tested against 0.
    for label, values in beyond.items():
        values = np.array(values)
        test = ttest_1samp(values, 0)
        print(f"  R^2 {label:26s} {values.mean():+.4f} on average, "
              f"{np.sum(values > 0)}/{len(values)} sessions positive, "
              f"t = {test.statistic:+.2f}, p = {test.pvalue:.2g}")

# ---------------------------------------------------------------------------
# 3. Figure: one dot per session, for the key comparisons (step 4).
# ---------------------------------------------------------------------------
if not per_session:
    raise SystemExit("\n(no step 4 results for this window, so no figure)")
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
