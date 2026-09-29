"""
Step 4: does the decoded uncertainty predict the monkey's wager, beyond
what the stimulus and the point estimate already explain?

This is Walker et al.'s main test, applied to the wager.

For each decoder from step 2 we turn every trial's likelihood function
into the population's belief about direction:

    log_odds_right = log P(right | spikes) - log P(left | spikes)

and use it in a small decision model:

    choice:  P(right choice) depends on log_odds_right
    wager:   P(high bet)     depends on |log_odds_right|
                             and on choice * log_odds_right

The wager gets two features, because confidence can mean two things:
  |log_odds_right|          how sure the population is about direction,
                            whichever way it points
  choice * log_odds_right   the population's evidence FOR THE DIRECTION THE
                            MONKEY CHOSE (choice is +1 right, -1 left):
                            positive when MT agrees with the choice,
                            negative when it favours the other direction -
                            "was my choice right?", like the posterior of the
                            chosen option in Walker et al.'s decision model
Every model gets both, so the comparisons stay like for like.

Both models also get a separate intercept for every coherence. That is
how we "condition on the stimulus": anything the stimulus explains is
absorbed by the intercepts, so the decoder can only help through
trial-to-trial fluctuations at a fixed coherence. They also get a smooth
function of trial order, so a slow drift over the session (in the
neurons AND in how often the monkey bets high) can't masquerade as a
trial-by-trial link.

The flexible decoder is always rank 2 - the hypothesis - rather than
whichever rank happened to decode best, so every session tests the same
thing and the sessions can be pooled (step 5).

Comparisons (all on held-out trials):
  fixed width vs rank 2
      the fixed-width likelihood only moves, it never gets narrower or
      wider, so it carries a point estimate but no extra uncertainty
      (the full decoder is shown too, for reference)
  shuffle control
      keep each trial's fixed-width likelihood (its point estimate) but
      give it the extra shape from a different trial with the same
      coherence. If the benefit comes from real trial-by-trial
      uncertainty, it should disappear.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(seed=1)

# Which session to analyse: a file made by step 0 (real data) or step 1
# (the simulation, the default). Run as:  python step4_predict_behaviour.py data/<session>.npz
session_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/simulated_session.npz")
session_name = session_file.stem
decoded_file = Path("data") / f"{session_name}_decoded.npz"
data = np.load(session_file)
coherence = data["coherence"]
choice = data["choice"]            # +1 right, -1 left
wager = data["wager"]              # 1 high, 0 low

decoded = np.load(decoded_file)
names = list(decoded["names"])
coherence_values = decoded["coherence_values"]
log_likelihoods = dict(zip(names, decoded["log_likelihoods"]))

flexible_name = "rank 2"
print(f"comparing 'fixed width' with '{flexible_name}'\n")


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
extra_shape = log_likelihoods[flexible_name] - fixed

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

# Slow drift: trial order scaled to -1..1, and its square and cube, so
# the model can follow a smooth trend over the session.
trial_order = data["trial_number"] if "trial_number" in data.files else np.arange(len(coherence))
position = np.interp(trial_order, (trial_order.min(), trial_order.max()), (-1, 1))
drift_columns = np.column_stack([position, position ** 2, position ** 3])
baseline_columns = np.column_stack([coherence_columns, drift_columns])


def held_out_log_likelihood(neural_feature, behaviour):
    """Log probability of the behaviour on each trial, from a model that
    never saw that trial. Higher = better prediction.

    neural_feature=None fits the baseline alone (coherence intercepts and
    slow drift): it knows the stimulus but nothing about the neurons."""
    if neural_feature is None:
        features = baseline_columns
    else:
        features = np.column_stack([baseline_columns, neural_feature])
    model = LogisticRegression(C=100, max_iter=1000)
    p = cross_val_predict(model, features, behaviour, cv=10,
                          method="predict_proba")
    classes = np.unique(behaviour)
    return np.log(p[np.arange(len(behaviour)),
                    np.searchsorted(classes, behaviour)])


models = ["fixed width", "rank 1", flexible_name, "rank 3", "full", "shuffled"]
choice_scores = {"baseline": held_out_log_likelihood(None, choice)}
wager_scores = {"baseline": held_out_log_likelihood(None, wager)}
for name in models:
    odds = log_odds_right(log_likelihoods[name])
    choice_scores[name] = held_out_log_likelihood(odds, choice)
    how_sure = np.abs(odds)
    evidence_for_choice = choice * odds
    wager_scores[name] = held_out_log_likelihood(
        np.column_stack([how_sure, evidence_for_choice]), wager)

# ---------------------------------------------------------------------------
# 3. Report the comparisons that matter, in order.
# ---------------------------------------------------------------------------
# Each line: model A minus model B, held-out log-likelihood per trial.
#   1. Do the neurons add ANYTHING beyond the baseline (stimulus + slow
#      drift)? If not, there is
#      nothing for the likelihood's shape to explain, and 2-4 will be ~0.
#   2. Walker's test: flexible likelihood vs fixed width.
#   3. Stricter: best low-rank vs rank 1 (both decode well; only the
#      extra axis differs).
#   4. Shuffle control: should be <= 0 if 2 is real.
comparisons = [
    ("neurons add anything?", "fixed width", "baseline"),
    ("Walker's test", flexible_name, "fixed width"),
    ("extra axis", flexible_name, "rank 1"),
    ("full vs fixed", "full", "fixed width"),
    ("shuffle control", "shuffled", "fixed width"),
]
for label, scores in [("choice", choice_scores), ("wager", wager_scores)]:
    print(f"predicting {label}: held-out log-likelihood per trial (A - B)")
    for question, a, b in comparisons:
        if a == b:
            continue
        difference = scores[a] - scores[b]
        t, p = ttest_rel(scores[a], scores[b])
        print(f"  {question:22s} {a:>11s} - {b:<14s} {difference.mean():+.4f}"
              f"   (total {difference.sum():+.1f}, t = {t:.2f}, p = {p:.1g})")
    # Accuracy, for intuition: the model "gets a trial right" when it gave
    # the monkey's actual behaviour a probability above 0.5.
    accuracies = "   ".join(f"{name} {np.mean(np.exp(scores[name]) > 0.5):.3f}"
                            for name in scores)
    print(f"  accuracy: {accuracies}")
    print()

# ---------------------------------------------------------------------------
# 4. Figure: the same comparison, split by difficulty.
# ---------------------------------------------------------------------------
difficulties = np.unique(np.abs(coherence))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, label, scores in [(axes[0], "choice", choice_scores),
                          (axes[1], "wager", wager_scores)]:
    for name, style in [(flexible_name, "o-"), ("shuffled", "o--")]:
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
fig.savefig(f"figures/{session_name}_step4.png", dpi=120)
print(f"saved figures/{session_name}_step4.png")

# ---------------------------------------------------------------------------
# 5. Save every trial's scores, so step 5 can pool sessions.
# ---------------------------------------------------------------------------
behaviour_file = Path("data") / f"{session_name}_behaviour.npz"
np.savez(behaviour_file,
         names=list(choice_scores),
         choice_scores=np.array(list(choice_scores.values())),
         wager_scores=np.array(list(wager_scores.values())),
         coherence=coherence)
print(f"saved {behaviour_file}")
