"""
Step 4: does the decoded uncertainty predict the monkey's wager, beyond
what the stimulus and the point estimate already explain?

This is Walker et al.'s main test, applied to the wager.

For each decoder from step 2 we turn every trial's likelihood function
into the population's belief about direction:

    log_odds_right = log P(right | spikes) - log P(left | spikes)

and use it in a small decision model:

    choice:  P(right choice) depends on log_odds_right
    wager:   P(high bet)     depends on |log_odds_right|  (= confidence)

Both models also get a separate intercept for every coherence. That is
how we "condition on the stimulus": anything the stimulus explains is
absorbed by the intercepts, so the decoder can only help through
trial-to-trial fluctuations at a fixed coherence.

Comparisons (all on held-out trials):
  fixed width vs best low-rank decoder
      the fixed-width likelihood only moves, it never gets narrower or
      wider, so it carries a point estimate but no extra uncertainty
      (the full decoder is shown too, for reference)
  shuffle control
      keep each trial's fixed-width likelihood (its point estimate) but
      give it the extra shape from a different trial with the same
      coherence. If the benefit comes from real trial-by-trial
      uncertainty, it should disappear.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(seed=1)

data = np.load("data/simulated_session.npz")
coherence = data["coherence"]
choice = data["choice"]            # +1 right, -1 left
wager = data["wager"]              # 1 high, 0 low

decoded = np.load("data/decoded_likelihoods.npz")
names = list(decoded["names"])
coherence_values = decoded["coherence_values"]
log_likelihoods = dict(zip(names, decoded["log_likelihoods"]))

# The best low-rank decoder from step 2.
mean_scores = decoded["scores"].mean(axis=1)
best_name = max([n for n in names if n.startswith("rank")],
                key=lambda n: mean_scores[names.index(n)])
print(f"comparing 'fixed width' with '{best_name}'\n")


def normalise(log_likelihood):
    """Rescale each row so that exp(row) sums to 1."""
    return log_likelihood - np.logaddexp.reduce(log_likelihood, axis=1,
                                                keepdims=True)


def log_odds_right(log_likelihood):
    """Population's log odds that the motion was rightward.

    Adds up the likelihood of all rightward coherences vs all leftward
    ones; 0% counts half for each.
    """
    likelihood = np.exp(normalise(log_likelihood))
    zero = likelihood[:, coherence_values == 0].sum(axis=1)
    p_right = likelihood[:, coherence_values > 0].sum(axis=1) + zero / 2
    p_left = likelihood[:, coherence_values < 0].sum(axis=1) + zero / 2
    return np.log(p_right) - np.log(p_left)


# ---------------------------------------------------------------------------
# 1. The shuffle control.
# ---------------------------------------------------------------------------
# Extra shape = what the flexible decoder adds on top of the fixed-width one.
fixed = log_likelihoods["fixed width"]
extra_shape = log_likelihoods[best_name] - fixed

shuffled_extra_shape = np.zeros_like(extra_shape)
for c in coherence_values:
    trials = np.flatnonzero(coherence == c)
    shuffled_extra_shape[trials] = extra_shape[rng.permutation(trials)]

log_likelihoods["shuffled"] = normalise(fixed + shuffled_extra_shape)

# ---------------------------------------------------------------------------
# 2. Fit the decision models and score them on held-out trials.
# ---------------------------------------------------------------------------
# One column per coherence, 1 if that coherence was shown: the intercepts.
coherence_columns = (coherence[:, None] == coherence_values).astype(float)


def held_out_log_likelihood(neural_feature, behaviour):
    """Log probability of the behaviour on each trial, from a model that
    never saw that trial. Higher = better prediction."""
    features = np.column_stack([coherence_columns, neural_feature])
    model = LogisticRegression(C=100, max_iter=1000)
    p = cross_val_predict(model, features, behaviour, cv=10,
                          method="predict_proba")
    classes = np.unique(behaviour)
    return np.log(p[np.arange(len(behaviour)),
                    np.searchsorted(classes, behaviour)])


models = ["fixed width", best_name, "full", "shuffled"]
choice_scores, wager_scores = {}, {}
for name in models:
    odds = log_odds_right(log_likelihoods[name])
    choice_scores[name] = held_out_log_likelihood(odds, choice)
    wager_scores[name] = held_out_log_likelihood(np.abs(odds), wager)

# ---------------------------------------------------------------------------
# 3. Report, relative to the fixed-width model (like Walker et al. Fig. 5).
# ---------------------------------------------------------------------------
for label, scores in [("choice", choice_scores), ("wager", wager_scores)]:
    print(f"predicting {label}: held-out log-likelihood per trial, "
          f"relative to fixed width")
    for name in models[1:]:
        difference = scores[name] - scores["fixed width"]
        t, p = ttest_rel(scores[name], scores["fixed width"])
        print(f"  {name:10s} {difference.mean():+.4f}   "
              f"(total {difference.sum():+.1f}, t = {t:.2f}, p = {p:.1g})")
    # Accuracy, for intuition: the model "gets a trial right" when it gave
    # the monkey's actual behaviour a probability above 0.5.
    accuracies = "   ".join(f"{name} {np.mean(np.exp(scores[name]) > 0.5):.3f}"
                            for name in models)
    print(f"  accuracy: {accuracies}")
    print()

# ---------------------------------------------------------------------------
# 4. Figure: the same comparison, split by difficulty.
# ---------------------------------------------------------------------------
difficulties = np.unique(np.abs(coherence))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, label, scores in [(axes[0], "choice", choice_scores),
                          (axes[1], "wager", wager_scores)]:
    for name, style in [(best_name, "o-"), ("shuffled", "o--")]:
        difference = scores[name] - scores["fixed width"]
        means = [difference[np.abs(coherence) == d].mean()
                 for d in difficulties]
        sems = [difference[np.abs(coherence) == d].std()
                / np.sqrt(np.sum(np.abs(coherence) == d))
                for d in difficulties]
        ax.errorbar(difficulties * 100, means, yerr=sems, fmt=style,
                    label=name)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("|coherence| (%)")
    ax.set_title(f"predicting the {label}")
axes[0].set_ylabel("held-out log-likelihood per trial\n"
                   "relative to fixed-width model")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/step4_behaviour.png", dpi=120)
print("saved figures/step4_behaviour.png")
