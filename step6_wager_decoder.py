"""
Step 6: is the wager readable from these neurons directly, and along
which population direction?

Steps 2-4 only look at the wager THROUGH the likelihood over coherence.
Here we skip the likelihood and decode the wager straight from the spike
counts (as Vivar-Lazo & Fetsch did in LIP), then ask how the direction that
predicts the wager relates to the direction that encodes coherence.

  1. Direct wager decoder: logistic regression of the wager on the spike
     counts, plus the same baseline as step 4 (one intercept per
     coherence + slow drift), so it too must predict BEYOND the stimulus.
     Cross-validated; the amount of regularisation is chosen
     automatically (inside each training set).
  2. The same for the choice, for comparison.
     Both are repeated with TRIAL HISTORY added to the baseline (the
     previous trial's choice, wager, correctness and whether it was a lost
     high bet). If the neurons only predict the wager because both carry
     over from the last trial, the "+history" decoder loses that benefit.
  3. Angles between population directions (0 = the same direction,
     90 = orthogonal):
        wager  vs coherence axis   (rank-1 axis of the likelihood decoder)
        wager  vs choice
        choice vs coherence axis
     compared with a null: the wager decoder refitted with the wagers
     shuffled among trials of the same coherence. The null shows what
     angle a decoder fitted to noise gives with these same neurons.

     Important: with tens of units, almost any two directions are close to
     90 degrees apart, so the null itself sits near 90. "Orthogonal" alone
     is therefore not a finding. What the test can detect is ALIGNMENT: a
     wager direction closer to the coherence axis (or the choice) than a
     noise-fitted decoder would be. The printout reports how often the
     null is at least that close.

Usage:  python step6_wager_decoder.py data/<session>__<window>.npz
        python step6_wager_decoder.py data/<session>__<window>.npz --history-only
The second form only adds the "+history" decoders to an existing result
(for sessions analysed before trial history was saved).
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import ttest_rel
from likelihood_decoders import LowRankDecoder, train

# scikit-learn prints long notices about future changes to
# LogisticRegressionCV's defaults; they don't affect these results.
warnings.filterwarnings("ignore", category=FutureWarning)

N_SHUFFLES = 100
rng = np.random.default_rng(seed=2)

session_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/simulated_session.npz")
history_only = "--history-only" in sys.argv
session_name = session_file.stem
data = np.load(session_file)
spike_counts = data["counts"].astype(float)
coherence = data["coherence"]
coherence_values = data["coherence_values"]
wager = data["wager"]
choice = data["choice"]
n_neurons = spike_counts.shape[1]

# z-score each neuron, so decoder weights are comparable across neurons.
# (Unsupervised, so doing it on all trials leaks nothing about behaviour.)
z_scored = (spike_counts - spike_counts.mean(axis=0)) / spike_counts.std(axis=0)

# The same baseline as step 4: an intercept per coherence + slow drift.
coherence_columns = (coherence[:, None] == coherence_values).astype(float)
trial_order = data["trial_number"] if "trial_number" in data.files else np.arange(len(coherence))
position = np.interp(trial_order, (trial_order.min(), trial_order.max()), (-1, 1))
baseline_columns = np.column_stack([coherence_columns,
                                    position, position ** 2, position ** 3])

# The direct decoders are regularised (a penalty on large weights) so the
# neurons can't overfit - but the penalty should not also shrink the
# baseline, or the model would lose the stimulus information and do worse
# than the baseline on its own. Scaling the baseline columns up by 1000 lets
# their weights be 1000 times smaller for the same effect, which makes their
# penalty negligible. (The neurons are z-scored, so they're unaffected.)
UNPENALISED = 1000

# How the regularisation strength is chosen: this many strengths, each
# scored by this many folds of cross-validation within the training data.
# (A coarse search is enough - it only sets how much the neurons' weights
# are shrunk - and it is the slowest part of this step.)
N_STRENGTHS = 5
INNER_FOLDS = 3

# Trial history (saved by step 0; the simulation has none).
HISTORY = ["prev_choice", "prev_wager", "prev_correct", "prev_timeout"]
has_history = all(name in data.files for name in HISTORY)
if has_history:
    history_columns = np.column_stack([data[name] for name in HISTORY])


# ---------------------------------------------------------------------------
# 1-2. Direct decoders: do the neurons predict wager / choice beyond the
#      stimulus? (held-out log-likelihood, like step 4)
# ---------------------------------------------------------------------------
def held_out_log_likelihood(features, behaviour, choose_regularisation):
    """Log probability of the actual behaviour on each held-out trial."""
    if choose_regularisation:
        # Tries N_STRENGTHS strengths, picks the best by cross-validation
        # inside each training set - so the held-out trials never
        # influence the choice.
        model = LogisticRegressionCV(Cs=N_STRENGTHS, cv=INNER_FOLDS,
                                     max_iter=2000, scoring="neg_log_loss")
    else:
        model = LogisticRegression(C=100, max_iter=2000)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    p = cross_val_predict(model, features, behaviour, cv=folds,
                          method="predict_proba")
    classes = np.unique(behaviour)
    return np.log(p[np.arange(len(behaviour)), np.searchsorted(classes, behaviour)])


def report(a, b):
    difference = scores[a] - scores[b]
    t, p = ttest_rel(scores[a], scores[b])
    print(f"  {a:>21s} - {b:<23s} {difference.mean():+.4f} per trial "
          f"(total {difference.sum():+.1f}, t = {t:.2f}, p = {p:.1g})")


baselines = {"": baseline_columns}
output = Path("data") / f"{session_name}_wager.npz"
if history_only:
    # Keep everything already saved; only the "+history" decoders are new.
    if not has_history:
        raise SystemExit(f"{session_file} has no trial history (rerun step 0)")
    saved = dict(np.load(output))
    scores = dict(zip(saved["names"], saved["scores"]))
    baselines = {}
else:
    scores = {}
if has_history:
    baselines["+history"] = np.column_stack([baseline_columns, history_columns])
for label, behaviour in [("wager", wager), ("choice", choice)]:
    print(f"predicting {label} (held-out log-likelihood per trial, A - B):")
    for extra, columns in baselines.items():
        scores[f"{label} baseline{extra}"] = held_out_log_likelihood(
            columns, behaviour, False)
        scores[f"{label} direct{extra}"] = held_out_log_likelihood(
            np.column_stack([columns * UNPENALISED, z_scored]), behaviour, True)
        report(f"{label} direct{extra}", f"{label} baseline{extra}")
    if has_history:
        # How much does history alone predict, beyond stimulus and drift?
        report(f"{label} baseline+history", f"{label} baseline")
print()

if history_only:
    saved["names"], saved["scores"] = list(scores), np.array(list(scores.values()))
    np.savez(output, **saved)
    print(f"added the +history decoders to {output}")
    sys.exit()

# ---------------------------------------------------------------------------
# 3. Population directions, fitted on all trials.
# ---------------------------------------------------------------------------
def neural_direction(behaviour, C):
    """The weights a behaviour decoder puts on the neurons (baseline
    columns included in the fit, but their weights dropped)."""
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(np.column_stack([baseline_columns * UNPENALISED, z_scored]), behaviour)
    return model.coef_[0, baseline_columns.shape[1]:]


def angle(a, b):
    """Angle between two directions, 0-90 degrees (the sign of a
    direction is arbitrary, so 100 degrees counts as 80)."""
    cosine = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine, 0, 1)))


# The regularisation for each decoder, chosen once on all trials.
features = np.column_stack([baseline_columns * UNPENALISED, z_scored])
C_wager = LogisticRegressionCV(Cs=N_STRENGTHS, cv=INNER_FOLDS, max_iter=2000,
                               scoring="neg_log_loss").fit(features, wager).C_[0]
C_choice = LogisticRegressionCV(Cs=N_STRENGTHS, cv=INNER_FOLDS, max_iter=2000,
                                scoring="neg_log_loss").fit(features, choice).C_[0]
wager_direction = neural_direction(wager, C_wager)
choice_direction = neural_direction(choice, C_choice)

# Coherence axis: the rank-1 likelihood decoder's population direction.
torch.manual_seed(0)
coherence_index = np.searchsorted(coherence_values, coherence)
rank1 = train(LowRankDecoder(n_neurons, coherence_values, rank=1),
              torch.tensor(z_scored, dtype=torch.float32),
              torch.tensor(coherence_index))
with torch.no_grad():
    coherence_axis = rank1.neurons_to_axes.weight.numpy()[0]

observed = {
    "wager vs coherence axis": angle(wager_direction, coherence_axis),
    "wager vs choice": angle(wager_direction, choice_direction),
    "choice vs coherence axis": angle(choice_direction, coherence_axis),
}

# Null: shuffle the wagers among trials of the same coherence, refit.
null = {"wager vs coherence axis": [], "wager vs choice": []}
for _ in range(N_SHUFFLES):
    shuffled = wager.copy()
    for c in coherence_values:
        trials = np.flatnonzero(coherence == c)
        shuffled[trials] = wager[rng.permutation(trials)]
    if len(np.unique(shuffled)) < 2:
        continue
    direction = neural_direction(shuffled, C_wager)
    null["wager vs coherence axis"].append(angle(direction, coherence_axis))
    null["wager vs choice"].append(angle(direction, choice_direction))

print(f"\nangles between population directions ({n_neurons} units; "
      f"null = wager shuffled within coherence, {N_SHUFFLES} times)")
for name, value in observed.items():
    if name in null:
        values = np.array(null[name])
        # How often a noise-fitted decoder is at least this ALIGNED (small
        # angle). Small = the wager direction is closer than chance.
        p_aligned = np.mean(values <= value)
        print(f"  {name:26s} {value:5.1f} deg   null {values.mean():5.1f} deg "
              f"[{np.percentile(values, 2.5):.1f}, {np.percentile(values, 97.5):.1f}]"
              f"   p(aligned) = {p_aligned:.2f}")
    else:
        print(f"  {name:26s} {value:5.1f} deg")

np.savez(output,
         names=list(scores), scores=np.array(list(scores.values())),
         angle_names=list(observed), angles=np.array(list(observed.values())),
         null_wager_coherence=np.array(null["wager vs coherence axis"]),
         null_wager_choice=np.array(null["wager vs choice"]))
print(f"\nsaved {output}")
