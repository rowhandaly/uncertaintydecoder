# Uncertainty decoding in LIP

**Current target: MT/MST** (recorded alongside LIP in the same sessions).
MT is a sensory area, so it is the closest match to Walker et al.'s V1
setting: the question is whether the trial-by-trial width of MT's
likelihood over motion coherence predicts the monkey's wager.

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
| `step0_load_session.py` | **Real data.** Runs on Rockfish next to a session folder (Open Ephys binary + Kilosort/Phy). Reads the task's network text messages (coherence, direction, choice, PDW, RT), motion onset/offset from TTL channel 4, and spike times of units labelled MT/MST (not noise). Counts spikes in a fixed window after motion onset and saves `data/<session>.npz` in the same format as step 1. Prints a table to check the choice/wager conventions. |
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

Simulation (steps 2-4 default to the simulated session):
```
pip install -r requirements.txt
python step1_simulate_task.py
python step2_decode_likelihoods.py     # ~45 s
python step3_shared_axes.py
python step4_predict_behaviour.py
```

Real data (on Rockfish), one session:
```
python step0_load_session.py "/home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary/hanzo_2021-09-14_13-27-18"
python step2_decode_likelihoods.py data/hanzo_2021-09-14_13-27-18.npz
python step3_shared_axes.py        data/hanzo_2021-09-14_13-27-18.npz
python step4_predict_behaviour.py  data/hanzo_2021-09-14_13-27-18.npz
```
Figures go to `figures/<session>_stepN.png`, intermediate arrays to `data/`.

## Before trusting real-data results

- [ ] Check the conventions table printed by step 0 (choice coding, which
      PDW value is the high bet, which Direction is rightward).
- [ ] Choose the spike window (step 0 settings). MT responds ~50-80 ms
      after motion onset; longer windows keep fewer short-RT trials.
- [ ] Loop over sessions and pool the step 4 comparisons.
- [ ] Check what axis 2 tracks (step 3 prints its correlation with RT and
      trial number) before calling it uncertainty.
- [ ] Compare with a decoder trained directly on the wager
      (Vivar-Lazo & Fetsch's approach).
- [ ] Optional: Walker's nonlinear decoder; a sensory (MT-like) version of
      the simulation.
