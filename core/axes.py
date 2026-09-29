"""
The shared population axes of each decoder, and how much each coherence
uses each axis (the loadings). Every decoder's readout can be written as
(coherences x axes) loadings times (axes x neurons) axes:

    fixed width   1 axis, loadings forced to be a straight line in coherence
    rank 1/2/3    1/2/3 axes, loadings free
    full          up to 11 axes; the top 3 are kept

Also checks whether the rank-2 readout for each coherence matches what the
full decoder finds on its own, against a split-half ceiling.

Usage:  python axes.py data/<session>__<analysis>.npz
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from decoders import FullDecoder, LowRankDecoder, FixedWidthDecoder, train

session_file = Path(sys.argv[1])
session = np.load(session_file)
spike_counts = session["counts"].astype(float)
coherence_values = session["coherence_values"]
coherence_index = torch.tensor(np.searchsorted(coherence_values, session["coherence"]))
n_neurons = spike_counts.shape[1]
n_coherences = len(coherence_values)

z_scored = torch.tensor((spike_counts - spike_counts.mean(axis=0)) / spike_counts.std(axis=0),
                        dtype=torch.float32)

decoders = {
    "fixed width": (lambda: FixedWidthDecoder(n_neurons, coherence_values), 1),
    "rank 1": (lambda: LowRankDecoder(n_neurons, n_coherences, rank=1), 1),
    "rank 2": (lambda: LowRankDecoder(n_neurons, n_coherences, rank=2), 2),
    "rank 3": (lambda: LowRankDecoder(n_neurons, n_coherences, rank=3), 3),
    "full": (lambda: FullDecoder(n_neurons, n_coherences), 3),
}


def centred(readout_weights):
    """Adding the same weights to every coherence changes nothing after the
    softmax, so only each coherence's difference from the average matters."""
    return readout_weights - readout_weights.mean(axis=0)


def axes_and_loadings(readout_weights, n_axes):
    left_vectors, strengths, axes = np.linalg.svd(centred(readout_weights), full_matrices=False)
    axes = axes[:n_axes]
    loadings = left_vectors[:, :n_axes] * strengths[:n_axes]
    for axis in range(n_axes):
        if loadings[-1, axis] < 0:            # the sign is arbitrary: make +51.2% positive
            loadings[:, axis] *= -1
            axes[axis] *= -1
    return axes, loadings


torch.manual_seed(0)
loadings = {}
for name, (make_decoder, n_axes) in decoders.items():
    decoder = train(make_decoder(), z_scored, coherence_index)
    _, loadings[name] = axes_and_loadings(decoder.readout_weights(), n_axes)

# Does rank 2 recover each coherence's own readout? Fit on two independent
# halves: full vs full is how reproducible the full readout is at all (the
# ceiling); rank 2 vs full is how much of it rank 2 recovers.
half_a = np.random.default_rng(0).permutation(len(coherence_index)) < len(coherence_index) // 2
half_b = ~half_a
full_a = centred(train(FullDecoder(n_neurons, n_coherences), z_scored[half_a], coherence_index[half_a]).readout_weights())
full_b = centred(train(FullDecoder(n_neurons, n_coherences), z_scored[half_b], coherence_index[half_b]).readout_weights())
rank_2_a = centred(train(LowRankDecoder(n_neurons, n_coherences, 2), z_scored[half_a], coherence_index[half_a]).readout_weights())

print("coherence   ceiling (full A vs full B)   rank 2 A vs full B")
for coherence, full_a_row, full_b_row, rank_2_a_row in zip(coherence_values, full_a, full_b, rank_2_a):
    ceiling = np.corrcoef(full_a_row, full_b_row)[0, 1]
    match = np.corrcoef(rank_2_a_row, full_b_row)[0, 1]
    print(f"{coherence * 100:+6.1f}%   {ceiling:26.2f}   {match:18.2f}")

figure, panels = plt.subplots(1, len(decoders), figsize=(4 * len(decoders), 3.5), sharex=True)
for panel, name in zip(panels, decoders):
    for axis in range(loadings[name].shape[1]):
        panel.plot(coherence_values * 100, loadings[name][:, axis], "o-", label=f"axis {axis + 1}")
    panel.axhline(0, color="gray", lw=0.5)
    panel.set_title(name)
    panel.set_xlabel("coherence (%)")
    panel.legend(fontsize=8)
panels[0].set_ylabel("loading")
figure.tight_layout()
Path("figures").mkdir(exist_ok=True)
figure.savefig(f"figures/{session_file.stem}_axes.png", dpi=120)

np.savez(Path("data") / f"{session_file.stem}_axes.npz",
         loadings=loadings["rank 2"],
         **{f"loadings_{name.replace(' ', '_')}": values for name, values in loadings.items()})
print(f"saved figures/{session_file.stem}_axes.png")
