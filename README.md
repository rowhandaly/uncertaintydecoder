# Uncertainty decoding in LIP

This project applies the likelihood-decoding approach of
**Walker, Cotton, Ma & Tolias (2020)**, *A neural basis of probabilistic
computation in visual cortex*, to the LIP peri-decision wagering task of
**Vivar-Lazo & Fetsch (2025)**, *Neural basis of concurrent deliberation
toward a choice and confidence judgment*.

It asks two questions:

1. Do trial-to-trial fluctuations in the *uncertainty* decoded from LIP
   predict the monkey's wager, even when the stimulus is held fixed?
2. Do those trial-to-trial fluctuations lie on a low-dimensional manifold
   that stays the same across task difficulty (|coherence|)?

The code is kept small and built one step at a time. Each step is a plain
script that reads the previous step's output from `data/`.

## Translating one paper onto the other

| Walker et al. (V1)                       | Here (LIP)                                   |
|------------------------------------------|----------------------------------------------|
| stimulus: orientation                    | stimulus: signed motion coherence (11 values) |
| difficulty: contrast                     | difficulty: \|coherence\|                     |
| spikes in 500 ms of stimulus             | spikes in the first 400 ms of deliberation    |
| likelihood over orientation              | likelihood over signed coherence              |
| decision: class 1 vs class 2             | decision: choice (L/R) **and wager (high/low)** |
| DNN, trained with cross-entropy          | same objective; linear version first          |

The wager is what makes this task a good fit: it is an explicit,
trial-by-trial report of confidence, so we can ask directly whether the
decoded uncertainty tracks it.

## Steps

- [x] **Step 1 – `step1_simulate_task.py`.** The real data are not public
  yet, so we simulate a session: a two-accumulator race (the paper's
  "parallel" model) sets choice, wager and decision time; 40 LIP-like
  neurons read out the accumulators with random weights; we count Poisson
  spikes in a fixed window. Swap in real data by saving the same arrays.
- [x] **Step 2 – `step2_likelihood_decoder.py`.** For every trial, decode a
  likelihood function over the 11 coherences (cross-validated, never sees
  behaviour). With a uniform prior over coherence, Walker et al.'s
  training objective is exactly multinomial logistic regression when the
  network is linear, so we start there. Summaries per trial: mean, sd,
  P(right) and confidence = max(P(right), P(left)).
- [ ] **Step 3 – does decoded uncertainty predict the wager?** Walker's
  test: compare a *full-likelihood* model with a *fixed-uncertainty* model
  (same centre, fixed shape), fit separately at each coherence, and check
  it with their shuffle control (swap likelihood shapes between trials of
  the same coherence).
- [ ] **Step 4 – a difficulty-invariant manifold?** Take each trial's
  log-likelihood vector, subtract the average for its coherence (leaving
  only trial-to-trial fluctuations), and run PCA separately for each
  |coherence|. Then ask: (a) do a few components explain most of the
  fluctuation? (b) do the components found at one difficulty explain the
  fluctuations at another difficulty as well as that difficulty's own
  components do? (the "alignment index"; compared against random
  subspaces). (c) is the wager readable from the shared components?

## Running

```
pip install -r requirements.txt
mkdir -p data figures
python step1_simulate_task.py
python step2_likelihood_decoder.py
```
