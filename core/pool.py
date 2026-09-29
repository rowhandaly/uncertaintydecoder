"""
Pool every session for one analysis. Each comparison is reported two ways:
all trials together, and one mean per session tested against zero.
Also plots the rank-2 loadings and the decoded uncertainty (the width of
each trial's likelihood) against |coherence|, averaged over sessions.

Usage:  python pool.py onset+80_250ms
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
for a, b in [("rank 2", "rank 1"), ("rank 3", "rank 2"), ("rank 2", "full"),
             ("rank 2", "fixed width"), ("full", "fixed width")]:
    differences = []
    for session in sessions:
        saved = load(session, "_decoded")
        scores = dict(zip(saved["names"], saved["scores"]))
        differences.append(scores[a] - scores[b])
    report(f"{a} - {b}", differences)

comparisons = [("likelihood beyond stimulus", "fixed width", "baseline"),
               ("rank 2 beyond stimulus", "rank 2", "baseline"),
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


def uncertainty(log_likelihood, coherence_values):
    """Two measures of each trial's decoded uncertainty:
    spread: entropy of the likelihood over the 11 coherences, in bits
            (0 = all on one coherence, 3.46 = spread evenly over all)
    direction: probability the likelihood gives the less likely direction
            (0 = sure which way, 0.5 = no idea)"""
    likelihood = np.exp(log_likelihood - np.logaddexp.reduce(log_likelihood, axis=1, keepdims=True))
    spread = -np.sum(likelihood * np.log2(likelihood + 1e-12), axis=1)
    zero = likelihood[:, coherence_values == 0].sum(axis=1)
    rightward = likelihood[:, coherence_values > 0].sum(axis=1) + zero / 2
    direction = np.minimum(rightward, 1 - rightward)
    return spread, direction


decoder_names = ["fixed width", "rank 1", "rank 2", "rank 3", "full"]
strengths = None
spread_per_session = {name: [] for name in decoder_names}
direction_per_session = {name: [] for name in decoder_names}
loadings_per_session = {name: [] for name in decoder_names}
for session in sessions:
    data = load(session, "")
    decoded = load(session, "_decoded")
    log_likelihoods = dict(zip(decoded["names"], decoded["log_likelihoods"]))
    strength = np.abs(data["coherence"])
    strengths = np.unique(strength)
    for name in decoder_names:
        spread, direction = uncertainty(log_likelihoods[name], data["coherence_values"])
        spread_per_session[name].append([spread[strength == value].mean() for value in strengths])
        direction_per_session[name].append([direction[strength == value].mean() for value in strengths])

    saved_axes = load(session, "_axes")
    for name in decoder_names:
        key = f"loadings_{name.replace(' ', '_')}"
        if key in saved_axes.files:              # older axes files only have rank 2
            loadings = saved_axes[key]
            loadings_per_session[name].append(loadings / np.abs(loadings).max(axis=0))

for measure, per_session in [("spread (entropy, bits)", spread_per_session),
                             ("P(other direction)", direction_per_session)]:
    print(f"\ndecoded uncertainty, {measure}, average over sessions:")
    print("  |coh|   " + "".join(f"{name:>13s}" for name in decoder_names))
    for row, value in enumerate(strengths):
        print(f"  {value * 100:5.1f}%  " + "".join(
            f"{np.mean(per_session[name], axis=0)[row]:13.3f}" for name in decoder_names))

Path("figures").mkdir(exist_ok=True)
coherence_values = load(sessions[0], "")["coherence_values"]

figure, (spread_ax, direction_ax) = plt.subplots(1, 2, figsize=(10, 4))
for ax, per_session, label in [(spread_ax, spread_per_session, "spread of the likelihood\n(entropy, bits)"),
                               (direction_ax, direction_per_session, "P(other direction)")]:
    for name in decoder_names:
        ax.plot(strengths * 100, np.mean(per_session[name], axis=0), "o-", label=name)
    ax.set_xlabel("|coherence| (%)")
    ax.set_ylabel(label)
    ax.set_title("decoded uncertainty vs difficulty")
spread_ax.legend()
figure.suptitle(f"{analysis}: {len(sessions)} sessions")
figure.tight_layout()
figure.savefig(f"figures/pooled__{analysis}_uncertainty.png", dpi=120)

figure, panels = plt.subplots(1, len(decoder_names), figsize=(4 * len(decoder_names), 3.5), sharey=True)
for panel, name in zip(panels, decoder_names):
    if not loadings_per_session[name]:
        continue
    mean_loadings = np.mean(loadings_per_session[name], axis=0)
    standard_error = np.std(loadings_per_session[name], axis=0) / np.sqrt(len(loadings_per_session[name]))
    for axis in range(mean_loadings.shape[1]):
        panel.errorbar(coherence_values * 100, mean_loadings[:, axis], yerr=standard_error[:, axis],
                       fmt="o-", label=f"axis {axis + 1}")
    panel.axhline(0, color="gray", lw=0.5)
    panel.set_title(f"{name} ({len(loadings_per_session[name])} sessions)")
    panel.set_xlabel("coherence (%)")
    panel.legend(fontsize=8)
panels[0].set_ylabel("loading (scaled to max 1 per session)")
figure.suptitle(f"{analysis}: shared axes of each decoder, average over sessions")
figure.tight_layout()
figure.savefig(f"figures/pooled__{analysis}_axes.png", dpi=120)
print(f"\nsaved figures/pooled__{analysis}_uncertainty.png and figures/pooled__{analysis}_axes.png")
