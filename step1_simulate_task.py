"""
Step 1: simulate one session of the peri-decision wagering task in LIP.

The task (Vivar-Lazo & Fetsch, 2025):
  - a random-dot motion stimulus with a signed coherence (negative = leftward)
  - the monkey makes ONE saccade to one of four targets:
      left-low, left-high, right-low, right-high
    so each trial gives us a choice (left/right) AND a wager (low/high).

The model of the brain we simulate is the "parallel" race model the paper
favours:
  - two accumulators, one for rightward evidence and one for leftward,
    with partially anti-correlated noise
  - the first one to hit the bound decides the choice
  - the LOSING accumulator decides the wager: if it is far from its own
    bound, the evidence was one-sided, so bet high

LIP neurons are modelled very simply: each neuron's firing rate is a
weighted sum of the two accumulators, and we count Poisson spikes in one
fixed window at the start of the trial.

One extra ingredient gives trials a genuine uncertainty signal: an
"attention" level that changes from trial to trial. On high-attention
trials the evidence is stronger (bigger drift) AND the neurons fire more
spikes, so the population's likelihood is sharper. Set ATTENTION_SD = 0
to switch it off and get a session where no such signal exists.

The real data are not public yet, so everything downstream only needs the
arrays saved at the end of this file. Swap them for real data later.
"""

import numpy as np

rng = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------------
# Task settings
# ---------------------------------------------------------------------------
COHERENCES = np.array([-0.512, -0.256, -0.128, -0.064, -0.032, 0.0,
                       0.032, 0.064, 0.128, 0.256, 0.512])
N_TRIALS = 6000

# ---------------------------------------------------------------------------
# Accumulator (race model) settings
# ---------------------------------------------------------------------------
DT = 0.01               # time step, in seconds
MAX_TIME = 3.0          # give up after this long
DRIFT_GAIN = 6.0        # how strongly coherence pushes the accumulators
NOISE_CORR = -0.7       # correlation between the two accumulators' noise
BOUND = 1.0             # decision bound
WAGER_CRITERION = -0.6  # bet high if the loser ended below this value
ATTENTION_SD = 0.3      # trial-to-trial spread of attention (0 = none)

# ---------------------------------------------------------------------------
# LIP population settings
# ---------------------------------------------------------------------------
N_NEURONS = 40
BASELINE_RATE = 20.0    # spikes per second
WINDOW = 0.4            # count spikes in the first 400 ms of the decision


def simulate_trial(coherence, attention):
    """Run the race between the two accumulators for one trial.

    Returns the full time course of both accumulators (frozen once the
    decision is made), the choice, the wager and the decision time.
    """
    n_steps = int(MAX_TIME / DT)

    # Noise for the two accumulators, drawn together so they are correlated.
    noise_cov = [[1.0, NOISE_CORR], [NOISE_CORR, 1.0]]
    noise = rng.multivariate_normal([0, 0], noise_cov, size=n_steps)
    noise = noise * np.sqrt(DT)

    # Mean change per step: rightward motion pushes "right" up and "left" down.
    # Higher attention means stronger evidence per unit time.
    drift = DRIFT_GAIN * attention * coherence * DT
    steps_right = drift + noise[:, 0]
    steps_left = -drift + noise[:, 1]

    # Accumulate.
    right = np.cumsum(steps_right)
    left = np.cumsum(steps_left)

    # Find the first time step where either one reaches the bound.
    crossed = (right >= BOUND) | (left >= BOUND)
    if crossed.any():
        t_end = np.argmax(crossed)          # index of first True
    else:
        t_end = n_steps - 1                 # no decision: take the last step

    # After the decision, LIP activity stays where it was (freeze).
    right[t_end:] = right[t_end]
    left[t_end:] = left[t_end]

    # Choice: whichever accumulator is higher at the end (+1 right, -1 left).
    choice = 1 if right[t_end] > left[t_end] else -1

    # Wager: look at the accumulator that lost.
    loser = min(right[t_end], left[t_end])
    wager = 1 if loser < WAGER_CRITERION else 0     # 1 = high, 0 = low

    decision_time = (t_end + 1) * DT
    return right, left, choice, wager, decision_time


def make_neurons():
    """Give each neuron a baseline and a weight on each accumulator.

    Weights are random, so neurons show mixed selectivity: some like
    right, some like left, some care about both (e.g. total evidence).
    """
    baseline = BASELINE_RATE * rng.uniform(0.5, 1.5, size=N_NEURONS)
    weight_right = rng.normal(0, 10, size=N_NEURONS)   # spikes/s per unit
    weight_left = rng.normal(0, 10, size=N_NEURONS)
    return baseline, weight_right, weight_left


def spike_counts(right, left, attention,
                 baseline, weight_right, weight_left):
    """Poisson spike counts in the first WINDOW seconds of one trial."""
    n_window = int(WINDOW / DT)
    r = right[:n_window]                     # shape (time,)
    l = left[:n_window]

    # Firing rate of every neuron at every time step: shape (time, neurons).
    rate = baseline + np.outer(r, weight_right) + np.outer(l, weight_left)
    rate = np.clip(rate, 0, None)            # rates cannot be negative
    rate = attention * rate                  # attention scales all firing

    # Expected number of spikes = rate integrated over the window.
    expected_count = rate.sum(axis=0) * DT
    return rng.poisson(expected_count)


if __name__ == "__main__":
    baseline, weight_right, weight_left = make_neurons()

    coherence = rng.choice(COHERENCES, size=N_TRIALS)
    # Attention varies around 1 from trial to trial (log-normal, always > 0).
    attention = np.exp(ATTENTION_SD * rng.normal(size=N_TRIALS))
    choice = np.zeros(N_TRIALS, dtype=int)
    wager = np.zeros(N_TRIALS, dtype=int)
    decision_time = np.zeros(N_TRIALS)
    counts = np.zeros((N_TRIALS, N_NEURONS), dtype=int)

    for i in range(N_TRIALS):
        right, left, choice[i], wager[i], decision_time[i] = \
            simulate_trial(coherence[i], attention[i])
        counts[i] = spike_counts(right, left, attention[i],
                                 baseline, weight_right, weight_left)

    # Quick sanity check of the behaviour, one line per unsigned coherence.
    correct = choice == np.sign(coherence)
    print("|coh|   P(correct)  P(high bet)  mean DT (s)")
    for c in np.unique(np.abs(coherence)):
        trials = np.abs(coherence) == c
        # At 0% there is no correct answer, so accuracy is not meaningful.
        acc = correct[trials].mean() if c > 0 else np.nan
        print(f"{c:5.3f}   {acc:9.2f}  {wager[trials].mean():11.2f}"
              f"  {decision_time[trials].mean():11.2f}")

    np.savez("data/simulated_session.npz",
             coherence=coherence, choice=choice, wager=wager,
             decision_time=decision_time, counts=counts,
             attention=attention,
             coherence_values=COHERENCES)
    print("saved data/simulated_session.npz")
