"""
Companion to choice_wager_figure.py: how well each decoder's likelihood
decodes the stimulus itself.

    a) all coherences             b) |coh| <= 12.8% only

Decoders: fixed width, rank 1, rank 2, full. The score is decode.py's:
the held-out log-likelihood of the coherence actually shown, per trial,
minus chance (a uniform guess over the coherences in that analysis,
log 1/11 or log 1/7). One dot per session, black = mean ± SEM over sessions.

Reads the files decode.py already saved, so it runs in seconds.

Usage:  python decoding_figure.py onset+80_250ms onset+80_250ms_coh128
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

DECODERS = ["fixed width", "rank 1", "rank 2", "full"]

analyses = sys.argv[1:3]
titles = ["a) all coherences", "b) |coherence| ≤ 12.8%"]

figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, analysis, title in zip(axes, analyses, titles):
    decoded_files = sorted(Path("data").glob(f"*__{analysis}_decoded.npz"))
    if not decoded_files:
        raise SystemExit(f"no decoded results for {analysis}")
    gains = {name: [] for name in DECODERS}
    for decoded_file in decoded_files:
        decoded = np.load(decoded_file)
        n_coherences = decoded["log_likelihoods"].shape[2]
        scores = dict(zip(decoded["names"], decoded["scores"]))
        for name in DECODERS:
            gains[name].append(np.mean(scores[name]) - np.log(1 / n_coherences))

    print(f"\n{analysis}: {len(decoded_files)} sessions, log likelihood over chance, mean ± SEM")
    for position, name in enumerate(DECODERS):
        values = np.array(gains[name])
        print(f"  {name:12s} {values.mean():+.4f} ± {values.std(ddof=1) / np.sqrt(len(values)):.4f}   "
              f"p = {ttest_1samp(values, 0).pvalue:.2g}")
        jitter = np.random.default_rng(position).uniform(-0.12, 0.12, len(values))
        ax.plot(position + jitter, values, "o", color="gray", alpha=0.6)
        ax.errorbar(position, values.mean(), yerr=values.std(ddof=1) / np.sqrt(len(values)),
                    fmt="s", color="black", markersize=8, capsize=4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(len(DECODERS)), DECODERS)
    ax.set_title(title)
    ax.set_ylabel("log likelihood over chance")
figure.suptitle("Decoding performance of likelihood")
figure.tight_layout()
Path("figures").mkdir(exist_ok=True)
output = f"figures/pooled__{analyses[0]}_decoding.png"
figure.savefig(output, dpi=120)
print(f"\nsaved {output}")
