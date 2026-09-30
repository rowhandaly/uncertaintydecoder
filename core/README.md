# Shared coherence axes and uncertainty in MT

Applies the likelihood decoders of Walker et al. (2020) to MT/MST recordings
from the peri-decision wagering task of Vivar-Lazo & Fetsch (2025).

Two questions:

1. **Shared structure across difficulty.** If every coherence's readout has
   to be built from the same few population axes, how many axes does
   decoding need?
2. **Walker's test.** At a fixed coherence, does a flexible likelihood
   (one whose width can vary from trial to trial) predict the choice or
   wager better than a fixed-width one?

## Files, in the order they run

| file | what it does |
|---|---|
| `load_session.py` | Reads one session (Open Ephys + Kilosort + task messages). Counts spikes of MT/MST units 80–330 ms after motion onset, removes slow drift, and saves `data/<session>__<analysis>.npz`. |
| `decoders.py` | Fixed-width, low-rank and full decoders; training; held-out decoding. |
| `decode.py` | Decodes coherence on held-out trials with fixed width, rank 1–3 and full, and compares them. |
| `axes.py` | Splits each decoder's readout into population axes and each coherence's loadings on them (fixed width: 1 axis, loadings a straight line; ranks 1–3; full: top 3). Checks the rank-2 readouts against the full decoder's with a split-half ceiling. |
| `behaviour.py` | Predicts choice and wager beyond the stimulus from each likelihood (Walker's test), with a shuffle control and a direct decoder. |
| `pool.py` | Pools sessions for one analysis: all comparisons, plus two figures: each decoder's axes averaged over sessions, and decoded uncertainty (likelihood entropy; probability of the other direction) vs \|coherence\| for each decoder. |
| `run_all.py`, `run_all.sbatch` | Runs everything for every session. |
| `firing_over_time.py` | Firing over time for each coherence; shows why the window is aligned to motion onset. |
| `simulate.py` | A fake session with a real trial-to-trial uncertainty signal, to check the pipeline finds it. |

## Running

On Rockfish, from this folder:
```
sbatch run_all.sbatch                  # then: tail -f logs/run_all.txt
python firing_over_time.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
```
Results: `logs/pooled__onset+80_250ms.txt` and `logs/pooled__onset+80_250ms_coh128.txt`.

Check on fake data first:
```
python simulate.py
python decode.py data/simulated__onset+80_250ms.npz
python axes.py data/simulated__onset+80_250ms.npz
python behaviour.py data/simulated__onset+80_250ms.npz
```

## Results (12 sessions, 80–330 ms after onset)

Decoding coherence, held-out log-likelihood per trial:

| | all coherences | only \|coh\| ≤ 12.8% |
|---|---|---|
| rank 2 − rank 1 | +0.025 (10/12 sessions) | −0.009 (1/12) |
| rank 3 − rank 2 | −0.016 (0/12) | −0.012 (2/12) |
| rank 2 − full | +0.047 (12/12) | +0.030 (10/12) |
| rank 2 − fixed width | +0.038 (10/12) | −0.011 (2/12) |

Predicting behaviour beyond the stimulus:

| | choice | wager |
|---|---|---|
| rank 2 − baseline | +0.0017 (8/12, p = 0.2) | −0.0006 (4/12, p = 0.2) |
| Walker's test (rank 2 − fixed width) | +0.0001 (6/12, p = 0.6) | +0.0004 (8/12, p = 0.2) |
| spike counts directly − baseline | −0.0010 (6/12, p = 0.4) | +0.0014 (9/12, p = 0.3) |

- Across all coherences, two shared axes decode best. Averaged over sessions,
  axis 1 is a straight line in signed coherence and axis 2 is U-shaped (it
  separates the strongest motion). The full decoder's top two axes are the
  same line and U; a third axis is noise.
- Across the difficult range, the best decoder is a single axis whose
  loadings are a straight line: the fixed-width decoder does as well as or
  better than every flexible one.
- Decoded uncertainty (likelihood entropy, or the probability of the other
  direction) falls with |coherence| in the same way for every decoder.
- No decoder predicts the choice or the wager beyond the stimulus, including
  a direct decoder on the spike counts. With 5–26 units per session that
  is limited power, not proof of absence.
- Windows aligned to the saccade confound coherence with time since onset
  (fast trials sample the onset response). `firing_over_time.py` shows this.
