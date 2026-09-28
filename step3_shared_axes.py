"""
Step 3: what are the shared axes, and do they recover the per-coherence
readouts?

A decoder's readout can be written as one weight matrix

    readout_weights: (11 coherences, neurons)
    log L(c) = readout_weights[c] @ spike_counts + bias[c]

Row c is "the weights for coherence c" - the thing Walker et al. let vary
freely. The low-rank decoder forces all 11 rows to be mixtures of the
same few population axes. Here we:

  1. fit the best low-rank decoder from step 2 on all trials
  2. find its axes in a unique, ordered way (SVD of the readout weights)
  3. plot how much each coherence uses each axis (the "loadings") -
     this shows how the readout changes with difficulty
  4. check how well its rows match the rows of the unconstrained (full)
     decoder, coherence by coherence
  5. (simulation only) check what the axes track, using the known
     attention level

Note: fitting on all trials is fine here because we only look at weights.
Anything that is compared with behaviour (step 4) uses the
cross-validated likelihoods from step 2.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from likelihood_decoders import FullDecoder, LowRankDecoder, train

data = np.load("data/simulated_session.npz")
spike_counts = data["counts"].astype(float)
coherence = data["coherence"]
coherence_values = data["coherence_values"]
n_neurons = spike_counts.shape[1]
coherence_index = np.searchsorted(coherence_values, coherence)

# Pick the rank that did best on held-out trials in step 2.
decoded = np.load("data/decoded_likelihoods.npz")
names = list(decoded["names"])
mean_scores = decoded["scores"].mean(axis=1)
low_rank_names = [n for n in names if n.startswith("rank")]
best_name = max(low_rank_names, key=lambda n: mean_scores[names.index(n)])
best_rank = int(best_name.split()[1])
print(f"best low-rank decoder in step 2: {best_name}")

# ---------------------------------------------------------------------------
# 1. Fit the low-rank and the full decoder on all trials.
# ---------------------------------------------------------------------------
z_scored = (spike_counts - spike_counts.mean(axis=0)) / spike_counts.std(axis=0)
counts_tensor = torch.tensor(z_scored, dtype=torch.float32)
index_tensor = torch.tensor(coherence_index)

torch.manual_seed(0)
low_rank = train(LowRankDecoder(n_neurons, coherence_values, best_rank),
                 counts_tensor, index_tensor)
full = train(FullDecoder(n_neurons, coherence_values),
             counts_tensor, index_tensor)

with torch.no_grad():
    # Low-rank readout = (axes -> coherences) times (neurons -> axes).
    low_rank_weights = (low_rank.axes_to_log_likelihood.weight
                        @ low_rank.neurons_to_axes.weight).numpy()
    full_weights = full.neurons_to_log_likelihood.weight.numpy()

# Adding the same vector to every row changes nothing after the softmax,
# so remove it: only differences between coherences are meaningful.
low_rank_weights = low_rank_weights - low_rank_weights.mean(axis=0)
full_weights = full_weights - full_weights.mean(axis=0)

# ---------------------------------------------------------------------------
# 2. Unique, ordered axes: SVD of the readout weights.
# ---------------------------------------------------------------------------
# readout_weights = loadings @ axes
#   axes:     (rank, neurons)  population directions, most important first
#   loadings: (11, rank)       how much each coherence uses each axis
U, singular_values, axes = np.linalg.svd(low_rank_weights,
                                         full_matrices=False)
axes = axes[:best_rank]
loadings = U[:, :best_rank] * singular_values[:best_rank]

# The sign of each axis is arbitrary (flipping both the axis and its
# loadings changes nothing). Fix it so the most rightward coherence loads
# positively on every axis, so the output is the same every run.
for k in range(best_rank):
    if loadings[-1, k] < 0:
        loadings[:, k] *= -1
        axes[k] *= -1

# ---------------------------------------------------------------------------
# 3. Does each coherence's readout from the low-rank decoder match the
#    one the full decoder found on its own?
# ---------------------------------------------------------------------------
print("\ncoherence   correlation of low-rank and full readout weights")
for c, low_rank_row, full_row in zip(coherence_values,
                                     low_rank_weights, full_weights):
    r = np.corrcoef(low_rank_row, full_row)[0, 1]
    print(f"{c * 100:+6.1f}%    {r:.2f}")

# ---------------------------------------------------------------------------
# 4. What does each axis track on single trials?
# ---------------------------------------------------------------------------
position_on_axes = z_scored @ axes.T          # (trials, rank)
print("\naxis   corr. with signed coherence   corr. with attention*")
for k in range(best_rank):
    r_coh = np.corrcoef(position_on_axes[:, k], coherence)[0, 1]
    r_att = np.corrcoef(position_on_axes[:, k], data["attention"])[0, 1]
    print(f"{k + 1:4d}   {r_coh:+27.2f}   {r_att:+20.2f}")
print("* attention is only known because this is a simulation")

# ---------------------------------------------------------------------------
# 5. Figure: loadings of each coherence on each shared axis.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
for k in range(best_rank):
    ax.plot(coherence_values * 100, loadings[:, k], "o-",
            label=f"axis {k + 1}")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("coherence (%)")
ax.set_ylabel("loading (how much this coherence\nuses the axis)")
ax.set_title(f"shared axes of the {best_name} decoder")
ax.legend()
fig.tight_layout()
fig.savefig("figures/step3_shared_axes.png", dpi=120)
print("\nsaved figures/step3_shared_axes.png")
