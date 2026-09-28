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

| file | what it does |
|---|---|
| `step1_simulate_task.py` | Simulates one session (real data aren't public yet): a two-accumulator race sets choice, wager and decision time; 40 LIP-like neurons read out the accumulators; Poisson spike counts in the first 400 ms. A trial-to-trial **attention** level scales both the evidence and the firing, which gives trials a real uncertainty signal. `ATTENTION_SD = 0` switches it off (the null case). |
| `likelihood_decoders.py` | The decoders and their training (Walker et al.'s objective: `log_softmax(log L + log prior)` trained with cross-entropy against the true coherence). **Full**: free weights for every coherence. **Low rank**: every coherence is read out from the same few population axes. **Fixed width**: a bump of fixed width that only moves (Walker's "fixed-uncertainty" decoder). |
| `step2_decode_likelihoods.py` | Decodes a cross-validated likelihood over the 11 coherences on every trial with each decoder, and compares them on held-out trials. *How many shared axes does decoding need?* |
| `step3_shared_axes.py` | Takes the best low-rank decoder, finds its axes (SVD of the readout weights) and plots how much each coherence uses each axis. Checks the per-coherence readouts against the unconstrained decoder. |
| `step4_predict_behaviour.py` | Walker's behavioural test on choice **and wager**: at fixed coherence, does the flexible likelihood predict behaviour better than the fixed-width one? Includes the shuffle control (swap likelihood shapes between trials of the same coherence). |

## What the simulation gives (as a check that the pipeline works)

With attention on (`ATTENTION_SD = 0.3`):
- Rank 2 decodes held-out trials best, better than rank 1 *and* the full decoder.
- Axis 1 is loaded in proportion to signed coherence (the point estimate);
  axis 2 is U-shaped in |coherence| and tracks the hidden attention level
  on single trials (r = 0.95), even though the decoder was only ever
  trained on coherence.
- The rank-2 likelihood predicts choice and wager better than the
  fixed-width one; shuffling shapes within coherence removes the benefit.

With attention off (`ATTENTION_SD = 0`): rank 1 wins and the benefit for
behaviour disappears.

## Later ideas

- Walker's nonlinear decoder (add hidden layers + ReLU in `likelihood_decoders.py`).
- Timing: decode in sliding windows / aligned to saccade, and relate to RT.
- A pre-saccade window, where the wager signal in LIP is strongest.

## Running

```
pip install -r requirements.txt
python step1_simulate_task.py
python step2_decode_likelihoods.py     # ~45 s
python step3_shared_axes.py
python step4_predict_behaviour.py
```
Figures go to `figures/`, intermediate arrays to `data/`.
