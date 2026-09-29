"""
Step 7: does MT's activity predict where LIP sits along its WAGER
direction, on the same trial, beyond the stimulus?

Steps 4 and 6 found no wager information in MT on its own, but a strong
wager signal in LIP from early on. One possibility: LIP computes something
like confidence from MT's activity, through a readout that step 6's single
wager decoder in MT couldn't find. If so, MT's trial-to-trial fluctuations
should predict LIP's wager signal.

  1. LIP's wager and choice directions (as in step 6: logistic regression
     beyond the stimulus and drift). Each is CROSS-FITTED: fitted on one
     half of the trials and used on the other half, so a trial's position
     along the direction never comes from a decoder that saw its wager.
  2. On every trial, LIP's position along each direction, and MT's counts,
     minus their average for that coherence: only the fluctuations at a
     fixed stimulus are left.
  3. Ridge regression from MT's fluctuations to LIP's position,
     cross-validated. The score is the held-out R^2: the fraction of LIP's
     trial-to-trial variation along that direction that MT predicts.
  4. Comparisons:
       choice direction     MT is known to carry some choice-related
                            activity, so it should predict LIP's choice
                            position somewhat: a positive reference
       random directions    how much MT predicts ANY direction in LIP.
                            MT and LIP share slow, brain-wide fluctuations
                            (arousal, attention), which would make MT
                            predict every LIP direction a little. The
                            wager direction is only special if it beats these.
       shuffled MT          MT's trials shuffled within coherence: what
                            R^2 is from fitting noise (slightly below 0)

Both areas must have been counted in the same window (step 0 with and
without --regions LIP). Trials are matched by trial number.

Usage:  python step7_mt_to_lip.py data/<session>__<window>.npz
        (the MT file; the LIP file data/<session>__LIP_<window>.npz is
        found from it)
"""

import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)

N_SHUFFLES = 100
N_RANDOM_DIRECTIONS = 20
UNPENALISED = 1000       # as in step 6: don't penalise the baseline
rng = np.random.default_rng(seed=3)

mt_file = Path(sys.argv[1])
session, window = mt_file.stem.split("__")
lip_file = mt_file.parent / f"{session}__LIP_{window}.npz"
mt, lip = np.load(mt_file), np.load(lip_file)

# Keep the trials both files have, in the same order.
common = np.intersect1d(mt["trial_number"], lip["trial_number"])
mt_row_of = {t: row for row, t in enumerate(mt["trial_number"])}
lip_row_of = {t: row for row, t in enumerate(lip["trial_number"])}
mt_rows = [mt_row_of[t] for t in common]
lip_rows = [lip_row_of[t] for t in common]
coherence = lip["coherence"][lip_rows]
coherence_values = np.unique(coherence)
wager = lip["wager"][lip_rows]
choice = lip["choice"][lip_rows]
print(f"{session}, window {window}: {len(common)} trials in both files, "
      f"{mt['counts'].shape[1]} MT units, {lip['counts'].shape[1]} LIP units\n")


def z_score(counts):
    return (counts - counts.mean(axis=0)) / counts.std(axis=0)


mt_counts = z_score(mt["counts"][mt_rows].astype(float))
lip_counts = z_score(lip["counts"][lip_rows].astype(float))

# The same baseline as steps 4 and 6: an intercept per coherence + drift.
coherence_columns = (coherence[:, None] == coherence_values).astype(float)
position = np.interp(common, (common.min(), common.max()), (-1, 1))
baseline_columns = np.column_stack([coherence_columns,
                                    position, position ** 2, position ** 3])


def minus_coherence_average(values):
    """Each trial minus the average of the trials with its coherence."""
    residual = values.astype(float).copy()
    for c in coherence_values:
        residual[coherence == c] -= values[coherence == c].mean(axis=0)
    return residual


# ---------------------------------------------------------------------------
# 1-2. LIP's cross-fitted positions along its wager and choice directions.
# ---------------------------------------------------------------------------
def cross_fitted_position(behaviour):
    """Each trial's position along LIP's direction for this behaviour,
    from a decoder fitted on the OTHER half of the trials."""
    halves = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    position_along = np.zeros(len(behaviour))
    features = np.column_stack([baseline_columns * UNPENALISED, lip_counts])
    for fit_half, use_half in halves.split(lip_counts, behaviour):
        model = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000,
                                     scoring="neg_log_loss")
        model.fit(features[fit_half], behaviour[fit_half])
        direction = model.coef_[0, baseline_columns.shape[1]:]
        position_along[use_half] = lip_counts[use_half] @ direction
    return position_along


lip_positions = {"wager": cross_fitted_position(wager),
                 "choice": cross_fitted_position(choice)}
for i in range(N_RANDOM_DIRECTIONS):
    lip_positions[f"random {i}"] = lip_counts @ rng.standard_normal(lip_counts.shape[1])
lip_positions = {name: minus_coherence_average(p) for name, p in lip_positions.items()}
mt_fluctuations = minus_coherence_average(mt_counts)

# Sanity check: does the cross-fitted wager position predict the wager
# (beyond the stimulus)? If not, there is no LIP wager signal to explain.
for label, behaviour in [("wager", wager), ("choice", choice)]:
    high = lip_positions[label][behaviour == behaviour.max()].mean()
    low = lip_positions[label][behaviour == behaviour.min()].mean()
    print(f"LIP {label} position (cross-fitted, within coherence): "
          f"{high:+.2f} when {label} = {behaviour.max()}, "
          f"{low:+.2f} when {label} = {behaviour.min()}")
print()


# ---------------------------------------------------------------------------
# 3. How well does MT predict each LIP position? (held-out R^2)
# ---------------------------------------------------------------------------
def held_out_r2(mt_activity, target):
    folds = KFold(n_splits=10, shuffle=True, random_state=0)
    ridge = RidgeCV(alphas=np.logspace(-1, 5, 13))   # strength chosen inside each fold
    predicted = cross_val_predict(ridge, mt_activity, target, cv=folds)
    return 1 - np.sum((target - predicted) ** 2) / np.sum((target - target.mean()) ** 2)


r2 = {name: held_out_r2(mt_fluctuations, target) for name, target in lip_positions.items()}
r2_random = np.array([r2.pop(f"random {i}") for i in range(N_RANDOM_DIRECTIONS)])

# Null: MT's trials shuffled among trials of the same coherence.
null = {"wager": [], "choice": []}
for _ in range(N_SHUFFLES):
    shuffled = mt_fluctuations.copy()
    for c in coherence_values:
        trials = np.flatnonzero(coherence == c)
        shuffled[trials] = mt_fluctuations[rng.permutation(trials)]
    for name in null:
        null[name].append(held_out_r2(shuffled, lip_positions[name]))
null = {name: np.array(values) for name, values in null.items()}

print("MT predicting LIP's position along each direction (held-out R^2):")
for name in ["wager", "choice"]:
    print(f"  {name:6s} direction   R^2 = {r2[name]:+.4f}   "
          f"shuffled MT {null[name].mean():+.4f} "
          f"[{np.percentile(null[name], 2.5):+.4f}, {np.percentile(null[name], 97.5):+.4f}]"
          f"   p = {np.mean(null[name] >= r2[name]):.2f}")
print(f"  random directions  R^2 = {r2_random.mean():+.4f} on average "
      f"[{r2_random.min():+.4f} to {r2_random.max():+.4f}]")
print(f"  wager beats {np.mean(r2['wager'] > r2_random) * 100:.0f}% of random directions")

output = Path("data") / f"{session}__{window}_mt_lip.npz"
np.savez(output,
         r2_wager=r2["wager"], r2_choice=r2["choice"], r2_random=r2_random,
         null_wager=null["wager"], null_choice=null["choice"],
         n_trials=len(common))
print(f"\nsaved {output}")
