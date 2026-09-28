"""
Step 2: decode a likelihood function over coherence on every trial.

This is the core idea of Walker, Cotton, Ma & Tolias (2020), moved from
V1/orientation to LIP/motion coherence.

What we want on each trial i is the likelihood function

    L_i(c) = P(spike counts on trial i | coherence = c)

for each of the 11 possible signed coherences c. Its centre tells us which
coherence the population "thinks" it saw; its width tells us how uncertain
the population is.

How Walker et al. get it (their Methods, eqs 1-11):
  - a network f(r) outputs one number per stimulus value, meant to be
    log L(c) up to a constant
  - add the log prior and take a softmax -> posterior over stimulus
  - train with ordinary cross-entropy against the true stimulus
Because the experimenter picks coherence uniformly at random, the log
prior is the same for every c and drops out: the classifier's posterior
IS the normalised likelihood. If f(r) is linear, this is just
multinomial logistic regression (the "Poisson-like" special case in the
paper). We start there because every line of it is easy to follow;
a small neural network can be swapped in later with the same interface.

The decoder never sees the monkey's choice or wager - only coherence.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

data = np.load("data/simulated_session.npz")
counts = data["counts"]                   # (trials, neurons)
coherence = data["coherence"]             # (trials,)
coh_values = data["coherence_values"]     # the 11 possible coherences

# ---------------------------------------------------------------------------
# 1. Fit the decoder and get a likelihood function for every trial.
# ---------------------------------------------------------------------------
# z-score each neuron so all inputs are on the same scale. This does not
# change what can be decoded, it just makes the fit converge much faster.
X = (counts - counts.mean(axis=0)) / counts.std(axis=0)

# cross_val_predict splits the trials into 5 folds, trains on 4 and
# predicts the held-out one, so every trial's likelihood comes from a
# decoder that never saw that trial.
decoder = LogisticRegression(C=1.0, max_iter=2000)
likelihood = cross_val_predict(decoder, X, coherence,
                               cv=5, method="predict_proba")
# likelihood has shape (trials, 11). Each row sums to 1.
# Its columns follow the sorted class labels, which are coh_values.

# ---------------------------------------------------------------------------
# 2. Summarise each likelihood function with a few numbers.
# ---------------------------------------------------------------------------
# Centre (mean) and width (standard deviation) over coherence.
mean = likelihood @ coh_values
sd = np.sqrt(likelihood @ coh_values**2 - mean**2)

# Probability that the motion was rightward, according to the population.
# The 0% column is split evenly between the two directions.
p_right = likelihood[:, coh_values > 0].sum(axis=1) \
    + 0.5 * likelihood[:, coh_values == 0].sum(axis=1)

# Confidence: the population's own probability of being right about the
# direction, whichever direction it favours. Ranges from 0.5 to 1.
confidence = np.maximum(p_right, 1 - p_right)

# ---------------------------------------------------------------------------
# 3. Sanity checks.
# ---------------------------------------------------------------------------
best_guess = coh_values[likelihood.argmax(axis=1)]
print(f"exact coherence decoded on {np.mean(best_guess == coherence):.2f}"
      f" of trials (chance = {1 / len(coh_values):.2f})")

direction_ok = np.sign(mean) == np.sign(coherence)
print(f"direction decoded correctly on "
      f"{direction_ok[coherence != 0].mean():.2f} of non-zero trials")

print("\n|coh|   width (sd)   confidence")
for c in np.unique(np.abs(coherence)):
    trials = np.abs(coherence) == c
    print(f"{c:5.3f}   {sd[trials].mean():10.3f}   {confidence[trials].mean():10.3f}")

# ---------------------------------------------------------------------------
# 4. Figure: average decoded likelihood for each true coherence
#    (the LIP version of Walker et al. Fig. 4).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = plt.cm.coolwarm(np.linspace(0, 1, len(coh_values)))
for c, color in zip(coh_values, colors):
    trials = coherence == c
    axes[0].plot(coh_values * 100, likelihood[trials].mean(axis=0),
                 "o-", color=color, label=f"{c * 100:g}%")
axes[0].set_xlabel("hypothesised coherence (%)")
axes[0].set_ylabel("average likelihood (normalised)")
axes[0].set_title("decoded likelihood, by true coherence")
axes[0].legend(fontsize=7, ncol=2)

for c in np.unique(np.abs(coherence)):
    trials = np.abs(coherence) == c
    axes[1].scatter(np.full(trials.sum(), c * 100)
                    + np.random.uniform(-1, 1, trials.sum()),
                    confidence[trials], s=2, alpha=0.2, color="gray")
    axes[1].plot(c * 100, confidence[trials].mean(), "ko")
axes[1].set_xlabel("|coherence| (%)")
axes[1].set_ylabel("decoded confidence  max(P(right), P(left))")
axes[1].set_title("trial-by-trial uncertainty")

fig.tight_layout()
fig.savefig("figures/step2_likelihoods.png", dpi=120)
print("\nsaved figures/step2_likelihoods.png")

np.savez("data/decoded_likelihoods.npz",
         likelihood=likelihood, mean=mean, sd=sd, p_right=p_right,
         confidence=confidence,
         coherence_values=coh_values)
print("saved data/decoded_likelihoods.npz")
