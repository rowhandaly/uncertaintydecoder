"""
A fake session where uncertainty really does vary from trial to trial, to
check the pipeline can find it. An "attention" level scales both the
evidence and the firing, so on some trials the population is sharper
than on others at the same coherence. ATTENTION_SPREAD = 0 switches it off.

Usage:  python simulate.py        then run decode.py, axes.py, behaviour.py
        on data/simulated__onset+80_250ms.npz
"""

from pathlib import Path
import numpy as np

random = np.random.default_rng(0)

COHERENCES = np.array([-0.512, -0.256, -0.128, -0.064, -0.032, 0.0, 0.032, 0.064, 0.128, 0.256, 0.512])
N_TRIALS = 6000
N_NEURONS = 40
ATTENTION_SPREAD = 0.3
TIME_STEP = 0.01
WINDOW = 0.4
BOUND = 1.0
HIGH_BET_IF_LOSER_BELOW = -0.6

coherence = random.choice(COHERENCES, size=N_TRIALS)
attention = np.exp(ATTENTION_SPREAD * random.normal(size=N_TRIALS))

baseline_rate = 20 * random.uniform(0.5, 1.5, N_NEURONS)
weight_on_right = random.normal(0, 10, N_NEURONS)
weight_on_left = random.normal(0, 10, N_NEURONS)

choice = np.zeros(N_TRIALS, dtype=int)
wager = np.zeros(N_TRIALS, dtype=int)
counts = np.zeros((N_TRIALS, N_NEURONS), dtype=int)
n_steps = int(3.0 / TIME_STEP)
for trial in range(N_TRIALS):
    # Two racing accumulators with anti-correlated noise.
    noise = random.multivariate_normal([0, 0], [[1, -0.7], [-0.7, 1]], size=n_steps) * np.sqrt(TIME_STEP)
    drift = 6.0 * attention[trial] * coherence[trial] * TIME_STEP
    right = np.cumsum(drift + noise[:, 0])
    left = np.cumsum(-drift + noise[:, 1])
    reached_bound = (right >= BOUND) | (left >= BOUND)
    end = np.argmax(reached_bound) if reached_bound.any() else n_steps - 1
    right[end:], left[end:] = right[end], left[end]

    choice[trial] = 1 if right[end] > left[end] else -1
    wager[trial] = int(min(right[end], left[end]) < HIGH_BET_IF_LOSER_BELOW)

    in_window = int(WINDOW / TIME_STEP)
    rates = baseline_rate + np.outer(right[:in_window], weight_on_right) + np.outer(left[:in_window], weight_on_left)
    rates = attention[trial] * np.clip(rates, 0, None)
    counts[trial] = random.poisson(rates.sum(axis=0) * TIME_STEP)

Path("data").mkdir(exist_ok=True)
np.savez("data/simulated__onset+80_250ms.npz",
         counts=counts, coherence=coherence, coherence_values=COHERENCES,
         choice=choice, wager=wager, trial_number=np.arange(N_TRIALS), attention=attention)
print("saved data/simulated__onset+80_250ms.npz")
