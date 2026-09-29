"""
The two shared axes of the rank-2 decoder: how much each coherence uses
each axis (the loadings), and whether the rank-2 readout for each
coherence matches what the full decoder finds on its own.

Usage:  python axes.py data/<session>__<analysis>.npz
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from decoders import FullDecoder, LowRankDecoder, train

RANK = 2

session_file = Path(sys.argv[1])
session = np.load(session_file)
spike_counts = session["counts"].astype(float)
coherence_values = session["coherence_values"]
coherence_index = torch.tensor(np.searchsorted(coherence_values, session["coherence"]))
n_neurons = spike_counts.shape[1]
n_coherences = len(coherence_values)

z_scored = torch.tensor((spike_counts - spike_counts.mean(axis=0)) / spike_counts.std(axis=0),
                        dtype=torch.float32)


def centred(readout_weights):
    """Adding the same weights to every coherence changes nothing after the
    softmax, so only each coherence's difference from the average matters."""
    return readout_weights - readout_weights.mean(axis=0)


# Axes and loadings: the SVD splits the rank-2 readout into
# (coherences x axes) loadings times (axes x neurons) axes.
torch.manual_seed(0)
rank_2 = train(LowRankDecoder(n_neurons, n_coherences, RANK), z_scored, coherence_index)
left_vectors, strengths, axes = np.linalg.svd(centred(rank_2.readout_weights()), full_matrices=False)
axes = axes[:RANK]
loadings = left_vectors[:, :RANK] * strengths[:RANK]
for axis in range(RANK):
    if loadings[-1, axis] < 0:            # the sign is arbitrary: make +51.2% positive
        loadings[:, axis] *= -1
        axes[axis] *= -1

# Does rank 2 recover each coherence's own readout? Fit on two independent
# halves: full vs full is how reproducible the full readout is at all (the
# ceiling); rank 2 vs full is how much of it rank 2 recovers.
half_a = np.random.default_rng(0).permutation(len(coherence_index)) < len(coherence_index) // 2
half_b = ~half_a
full_a = centred(train(FullDecoder(n_neurons, n_coherences), z_scored[half_a], coherence_index[half_a]).readout_weights())
full_b = centred(train(FullDecoder(n_neurons, n_coherences), z_scored[half_b], coherence_index[half_b]).readout_weights())
rank_2_a = centred(train(LowRankDecoder(n_neurons, n_coherences, RANK), z_scored[half_a], coherence_index[half_a]).readout_weights())

print("coherence   ceiling (full A vs full B)   rank 2 A vs full B")
for coherence, full_a_row, full_b_row, rank_2_a_row in zip(coherence_values, full_a, full_b, rank_2_a):
    ceiling = np.corrcoef(full_a_row, full_b_row)[0, 1]
    match = np.corrcoef(rank_2_a_row, full_b_row)[0, 1]
    print(f"{coherence * 100:+6.1f}%   {ceiling:26.2f}   {match:18.2f}")

figure, ax = plt.subplots(figsize=(5, 4))
for axis in range(RANK):
    ax.plot(coherence_values * 100, loadings[:, axis], "o-", label=f"axis {axis + 1}")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("coherence (%)")
ax.set_ylabel("loading")
ax.set_title("how much each coherence uses each shared axis")
ax.legend()
figure.tight_layout()
Path("figures").mkdir(exist_ok=True)
figure.savefig(f"figures/{session_file.stem}_axes.png", dpi=120)

np.savez(Path("data") / f"{session_file.stem}_axes.npz",
         loadings=loadings, axes=axes, coherence_values=coherence_values)
print(f"saved figures/{session_file.stem}_axes.png")
