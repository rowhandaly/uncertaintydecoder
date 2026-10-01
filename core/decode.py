"""
Decode coherence on held-out trials with each decoder, and ask how many
shared axes decoding needs.

Usage:  python decode.py data/<session>__<analysis>.npz
"""

import sys
from pathlib import Path
import numpy as np
import torch
from decoders import FullDecoder, LowRankDecoder, FixedWidthDecoder, decode_held_out

session_file = Path(sys.argv[1])
session = np.load(session_file)
spike_counts = session["counts"].astype(float)
coherence_values = session["coherence_values"]
coherence_index = np.searchsorted(coherence_values, session["coherence"])
n_neurons = spike_counts.shape[1]
n_coherences = len(coherence_values)

decoders = {
    "fixed width": lambda: FixedWidthDecoder(n_neurons, coherence_values),
    "rank 1": lambda: LowRankDecoder(n_neurons, n_coherences, rank=1),
    "rank 2": lambda: LowRankDecoder(n_neurons, n_coherences, rank=2),
    "rank 3": lambda: LowRankDecoder(n_neurons, n_coherences, rank=3),
    "full": lambda: FullDecoder(n_neurons, n_coherences),
}

torch.manual_seed(0)
log_likelihoods, scores = {}, {}
for name, make_decoder in decoders.items():
    log_likelihoods[name], scores[name] = decode_held_out(make_decoder, spike_counts, coherence_index)

print("held-out log-likelihood of the true coherence, per trial (A - B), with 95% CI:")
for first, second in [("rank 2", "rank 1"), ("rank 3", "rank 2"),
                      ("rank 2", "full"), ("rank 2", "fixed width")]:
    difference = scores[first] - scores[second]
    margin = 1.96 * difference.std() / np.sqrt(len(difference))
    print(f"  {first} - {second:12s} {difference.mean():+.4f}  "
          f"[{difference.mean() - margin:+.4f}, {difference.mean() + margin:+.4f}]")

output = Path("data") / f"{session_file.stem}_decoded.npz"
np.savez(output,
         names=list(decoders),
         log_likelihoods=np.array([log_likelihoods[name] for name in decoders]),
         scores=np.array([scores[name] for name in decoders]))
print(f"saved {output}")
