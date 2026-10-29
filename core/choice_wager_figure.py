"""
Four panels: how well each decoder's likelihood predicts the monkey's
choice and wager, with and without the stimulus already known.

    a) choice                     b) wager
    c) choice beyond stimulus     d) wager beyond stimulus

Decoders: fixed width, rank 1, rank 2, full. Features are the same as in
behaviour.py (choice: log odds of rightward motion; wager: |log odds| and
choice * log odds). The score is the held-out log-likelihood of what the
monkey did, per trial, minus that of a reference model. No drift terms:
    a, b: chance, i.e. an intercept only (the overall rate of rightward
          choices or high bets)
    c, d: the stimulus alone, one intercept per coherence
One dot per session, black = mean ± SEM over sessions.

Reads the files decode.py already saved, so it runs in a minute or two.

Usage:  python choice_wager_figure.py onset+80_250ms
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DECODERS = ["fixed width", "rank 1", "rank 2", "full"]

analysis = sys.argv[1]
sessions = sorted(f.name.split("__")[0] for f in Path("data").glob(f"*__{analysis}_decoded.npz"))
if not sessions:
    raise SystemExit(f"no decoded results for {analysis}")


def load(session, suffix):
    return np.load(Path("data") / f"{session}__{analysis}{suffix}.npz")


def log_odds_rightward(log_likelihood, coherence_values):
    likelihood = np.exp(log_likelihood - np.logaddexp.reduce(log_likelihood, axis=1, keepdims=True))
    zero = likelihood[:, coherence_values == 0].sum(axis=1)
    rightward = likelihood[:, coherence_values > 0].sum(axis=1) + zero / 2
    leftward = likelihood[:, coherence_values < 0].sum(axis=1) + zero / 2
    return np.log(rightward) - np.log(leftward)


def held_out_log_likelihood(features, behaviour):
    """Log probability of what the monkey did, from a model that never saw that trial."""
    model = LogisticRegression(C=100, max_iter=2000)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    probabilities = cross_val_predict(model, features, behaviour, cv=folds, method="predict_proba")
    column = np.searchsorted(np.unique(behaviour), behaviour)
    return np.log(probabilities[np.arange(len(behaviour)), column])


# gains[panel][decoder] = one mean gain per session
panels = ["choice", "wager", "choice beyond stimulus", "wager beyond stimulus"]
gains = {panel: {name: [] for name in DECODERS} for panel in panels}
for session in sessions:
    data = load(session, "")
    decoded = load(session, "_decoded")
    log_likelihoods = dict(zip(decoded["names"], decoded["log_likelihoods"]))
    coherence, coherence_values = data["coherence"], data["coherence_values"]
    choice, wager = data["choice"], data["wager"]

    chance = np.zeros((len(coherence), 1))          # intercept only
    coherence_columns = (coherence[:, None] == coherence_values).astype(float)
    references = {"": chance, " beyond stimulus": coherence_columns}

    for suffix, reference in references.items():
        reference_choice = held_out_log_likelihood(reference, choice)
        reference_wager = held_out_log_likelihood(reference, wager)
        for name in DECODERS:
            odds = log_odds_rightward(log_likelihoods[name], coherence_values)
            choice_score = held_out_log_likelihood(np.column_stack([reference, odds]), choice)
            wager_score = held_out_log_likelihood(
                np.column_stack([reference, np.abs(odds), choice * odds]), wager)
            gains["choice" + suffix][name].append(np.mean(choice_score - reference_choice))
            gains["wager" + suffix][name].append(np.mean(wager_score - reference_wager))
    print(f"{session} done")

print(f"\n{analysis}: {len(sessions)} sessions")
print("gain in held-out log-likelihood per trial over the reference, mean ± SEM over sessions")
for panel in panels:
    print(f"\n{panel}:")
    for name in DECODERS:
        values = np.array(gains[panel][name])
        print(f"  {name:12s} {values.mean():+.4f} ± {values.std(ddof=1) / np.sqrt(len(values)):.4f}   "
              f"sessions positive {np.sum(values > 0)}/{len(values)}, "
              f"p = {ttest_1samp(values, 0).pvalue:.2g}")

Path("figures").mkdir(exist_ok=True)
figure, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = {"choice": "a) choice",
          "wager": "b) wager",
          "choice beyond stimulus": "c) choice, beyond the stimulus",
          "wager beyond stimulus": "d) wager, beyond the stimulus"}
ylabels = {"choice": "log likelihood over chance", "wager": "log likelihood over chance",
           "choice beyond stimulus": "log likelihood over stimulus alone",
           "wager beyond stimulus": "log likelihood over stimulus alone"}
for ax, panel in zip(axes.flat, panels):
    for position, name in enumerate(DECODERS):
        values = np.array(gains[panel][name])
        jitter = np.random.default_rng(position).uniform(-0.12, 0.12, len(values))
        ax.plot(position + jitter, values, "o", color="gray", alpha=0.6)
        ax.errorbar(position, values.mean(), yerr=values.std(ddof=1) / np.sqrt(len(values)),
                    fmt="s", color="black", markersize=8, capsize=4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(len(DECODERS)), DECODERS)
    ax.set_title(titles[panel])
    ax.set_ylabel(ylabels[panel])
figure.suptitle("Predictive power of likelihood")
figure.tight_layout()
output = f"figures/pooled__{analysis}_choice_wager.png"
figure.savefig(output, dpi=120)
print(f"\nsaved {output}")
