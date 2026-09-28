"""
Step 2: decode a likelihood function over coherence on every trial,
with decoders of increasing flexibility, and compare them.

Decoders (see likelihood_decoders.py):
  fixed width   one bump of fixed width that only moves (Walker's
                "fixed-uncertainty" decoder)
  rank 1, 2, 3  every coherence's log-likelihood is read out from the same
                1, 2 or 3 population axes, shared across all difficulties
  full          every coherence gets its own readout weights

The question for the shared-structure hypothesis: how many shared axes
does it take before extra flexibility stops helping on held-out trials?
If rank 2 does as well as full, then the 11 per-coherence readouts are
really just 2 population axes mixed in different amounts.

The decoders never see the monkey's choice or wager - only coherence.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from likelihood_decoders import (FullDecoder, LowRankDecoder,
                                 FixedWidthDecoder,
                                 cross_validated_log_likelihoods)

# Which session to analyse: a file made by step 0 (real data) or step 1
# (the simulation, the default). Run as:  python step2_decode_likelihoods.py data/<session>.npz
session_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/simulated_session.npz")
session_name = session_file.stem
decoded_file = Path("data") / f"{session_name}_decoded.npz"
data = np.load(session_file)
spike_counts = data["counts"].astype(float)       # (trials, neurons)
coherence = data["coherence"]                     # (trials,)
coherence_values = data["coherence_values"]       # the 11 coherences, sorted
n_neurons = spike_counts.shape[1]

# Which of the 11 coherences was shown, as a number 0-10.
coherence_index = np.searchsorted(coherence_values, coherence)

# ---------------------------------------------------------------------------
# 1. Decode every trial with every decoder (5-fold cross-validated).
# ---------------------------------------------------------------------------
decoders = {
    "fixed width": lambda: FixedWidthDecoder(n_neurons, coherence_values),
    "rank 1": lambda: LowRankDecoder(n_neurons, coherence_values, rank=1),
    "rank 2": lambda: LowRankDecoder(n_neurons, coherence_values, rank=2),
    "rank 3": lambda: LowRankDecoder(n_neurons, coherence_values, rank=3),
    "full": lambda: FullDecoder(n_neurons, coherence_values),
}

torch.manual_seed(0)      # same random starting weights every run
log_likelihoods = {}
scores = {}
for name, make_decoder in decoders.items():
    print(f"decoding with: {name}")
    log_likelihoods[name], scores[name] = cross_validated_log_likelihoods(
        make_decoder, spike_counts, coherence_index)

# ---------------------------------------------------------------------------
# 2. Compare them on held-out trials.
# ---------------------------------------------------------------------------
# Score = log probability the decoder gave to the true coherence on a trial
# it never saw. We report each decoder relative to the full one, trial by
# trial, so the error bar is for the paired difference.
# A positive number means the decoder did BETTER than full on held-out
# trials. The 95% confidence interval tells us whether "as good as full"
# is a fair statement: if it is narrow and includes (or sits above) zero,
# the extra flexibility of the full decoder buys nothing.
print("\ndecoder        held-out score relative to full   95% CI")
for name in decoders:
    difference = scores[name] - scores["full"]
    sem = difference.std() / np.sqrt(len(difference))
    low, high = difference.mean() - 1.96 * sem, difference.mean() + 1.96 * sem
    print(f"{name:12s}   {difference.mean():+.4f}"
          f"                      [{low:+.4f}, {high:+.4f}]")

# ---------------------------------------------------------------------------
# 2b. How different are the curves themselves?
# ---------------------------------------------------------------------------
# KL divergence between the full decoder's likelihood and each other
# decoder's, trial by trial: 0 means identical curves. This says how much
# two decoders DISAGREE, not which one is right (the score above does that).
print("\ndecoder        mean KL(full || decoder), in nats")
full_likelihood = np.exp(log_likelihoods["full"])
for name in decoders:
    kl_per_trial = np.sum(full_likelihood
                          * (log_likelihoods["full"] - log_likelihoods[name]),
                          axis=1)
    print(f"{name:12s}   {kl_per_trial.mean():.4f}")

# ---------------------------------------------------------------------------
# 3. Figures.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: held-out score for each decoder.
names = list(decoders)
means = [np.mean(scores[n] - scores["full"]) for n in names]
sems = [np.std(scores[n] - scores["full"]) / np.sqrt(len(coherence))
        for n in names]
axes[0].errorbar(range(len(names)), means, yerr=sems, fmt="ko")
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].set_xticks(range(len(names)), names)
axes[0].set_ylabel("held-out log-likelihood\nrelative to full decoder")
axes[0].set_title("how much flexibility does decoding need?")

# Right: the average decoded likelihood for each true coherence
# (the LIP version of Walker et al. Fig. 4), from the full decoder.
likelihood = np.exp(log_likelihoods["full"])
colors = plt.cm.coolwarm(np.linspace(0, 1, len(coherence_values)))
for c, color in zip(coherence_values, colors):
    trials = coherence == c
    axes[1].plot(coherence_values * 100, likelihood[trials].mean(axis=0),
                 "o-", color=color, label=f"{c * 100:g}%")
axes[1].set_xlabel("hypothesised coherence (%)")
axes[1].set_ylabel("average likelihood (normalised)")
axes[1].set_title("decoded likelihood, by true coherence")
axes[1].legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(f"figures/{session_name}_step2.png", dpi=120)
print(f"\nsaved figures/{session_name}_step2.png")

np.savez(decoded_file,
         names=names,
         log_likelihoods=np.array([log_likelihoods[n] for n in names]),
         scores=np.array([scores[n] for n in names]),
         coherence_values=coherence_values)
print(f"saved {decoded_file}")
