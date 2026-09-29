"""
Does MT predict the monkey's choice and wager beyond the stimulus?

Every model knows the coherence (one intercept per coherence) and slow
drift over the session (trial position, squared, cubed): the baseline.
On top of that:

    fixed width, rank 1, rank 2   features from that decoder's likelihood
                                  (choice: log odds of rightward motion;
                                   wager: |log odds| and choice * log odds)
    shuffled                      fixed-width likelihood plus rank 2's extra
                                  shape taken from another trial of the same
                                  coherence (should lose any rank-2 benefit)
    direct                        the spike counts themselves, regularised

Walker's test is rank 2 vs fixed width. Scores are held-out
log-likelihoods of what the monkey did, per trial.

Usage:  python behaviour.py data/<session>__<analysis>.npz
"""

import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
random = np.random.default_rng(1)

session_file = Path(sys.argv[1])
session = np.load(session_file)
decoded = np.load(Path("data") / f"{session_file.stem}_decoded.npz")
coherence = session["coherence"]
coherence_values = session["coherence_values"]
choice = session["choice"]
wager = session["wager"]
spike_counts = session["counts"].astype(float)
log_likelihoods = dict(zip(decoded["names"], decoded["log_likelihoods"]))


def normalise(log_likelihood):
    return log_likelihood - np.logaddexp.reduce(log_likelihood, axis=1, keepdims=True)


def log_odds_rightward(log_likelihood):
    likelihood = np.exp(normalise(log_likelihood))
    zero = likelihood[:, coherence_values == 0].sum(axis=1)
    rightward = likelihood[:, coherence_values > 0].sum(axis=1) + zero / 2
    leftward = likelihood[:, coherence_values < 0].sum(axis=1) + zero / 2
    return np.log(rightward) - np.log(leftward)


extra_shape = log_likelihoods["rank 2"] - log_likelihoods["fixed width"]
shuffled_extra_shape = np.zeros_like(extra_shape)
for value in coherence_values:
    trials = np.flatnonzero(coherence == value)
    shuffled_extra_shape[trials] = extra_shape[random.permutation(trials)]
log_likelihoods["shuffled"] = normalise(log_likelihoods["fixed width"] + shuffled_extra_shape)

coherence_columns = (coherence[:, None] == coherence_values).astype(float)
trial_number = session["trial_number"]
position = np.interp(trial_number, (trial_number.min(), trial_number.max()), (-1, 1))
baseline = np.column_stack([coherence_columns, position, position ** 2, position ** 3])

z_scored = (spike_counts - spike_counts.mean(axis=0)) / spike_counts.std(axis=0)
# The direct model's penalty should shrink the neurons' weights, not the
# baseline's: scaling the baseline up makes its weights (and penalty) tiny.
BASELINE_SCALE = 1000


def held_out_log_likelihood(features, behaviour, choose_penalty=False):
    """Log probability of what the monkey did, from a model that never saw that trial."""
    if choose_penalty:
        model = LogisticRegressionCV(Cs=5, cv=3, max_iter=2000, scoring="neg_log_loss")
    else:
        model = LogisticRegression(C=100, max_iter=2000)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    probabilities = cross_val_predict(model, features, behaviour, cv=folds, method="predict_proba")
    column = np.searchsorted(np.unique(behaviour), behaviour)
    return np.log(probabilities[np.arange(len(behaviour)), column])


choice_scores = {"baseline": held_out_log_likelihood(baseline, choice)}
wager_scores = {"baseline": held_out_log_likelihood(baseline, wager)}
for name in ["fixed width", "rank 1", "rank 2", "shuffled"]:
    odds = log_odds_rightward(log_likelihoods[name])
    choice_scores[name] = held_out_log_likelihood(np.column_stack([baseline, odds]), choice)
    wager_features = np.column_stack([baseline, np.abs(odds), choice * odds])
    wager_scores[name] = held_out_log_likelihood(wager_features, wager)

direct_features = np.column_stack([baseline * BASELINE_SCALE, z_scored])
choice_scores["direct"] = held_out_log_likelihood(direct_features, choice, choose_penalty=True)
wager_scores["direct"] = held_out_log_likelihood(direct_features, wager, choose_penalty=True)

comparisons = [("likelihood beyond stimulus", "fixed width", "baseline"),
               ("Walker's test", "rank 2", "fixed width"),
               ("extra axis", "rank 2", "rank 1"),
               ("shuffle control", "shuffled", "fixed width"),
               ("neurons directly", "direct", "baseline")]
for behaviour_name, scores in [("choice", choice_scores), ("wager", wager_scores)]:
    print(f"{behaviour_name}: held-out log-likelihood per trial (A - B)")
    for question, a, b in comparisons:
        print(f"  {question:28s} {a:>11s} - {b:<11s} {np.mean(scores[a] - scores[b]):+.4f}")

output = Path("data") / f"{session_file.stem}_behaviour.npz"
np.savez(output,
         names=list(choice_scores),
         choice_scores=np.array(list(choice_scores.values())),
         wager_scores=np.array(list(wager_scores.values())))
print(f"saved {output}")
