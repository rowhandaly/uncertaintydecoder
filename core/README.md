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
| `axes.py` | Splits the rank-2 readout into two population axes and each coherence's loadings on them. Checks them against the full decoder's readouts with a split-half ceiling. |
| `behaviour.py` | Predicts choice and wager beyond the stimulus from each likelihood (Walker's test), with a shuffle control and a direct decoder. |
| `pool.py` | Pools sessions for one analysis. |
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

## Results so far (12 sessions, 80–330 ms after onset)

| | per trial | sessions |
|---|---|---|
| rank 2 − rank 1 | +0.025 | 10/12 |
| rank 3 − rank 2 | −0.016 | 0/12 |
| rank 2 − full | +0.047 | 12/12 |
| rank 2 − rank 1, only \|coh\| ≤ 12.8% | −0.009 | 1/12 |
| Walker's test, wager | +0.0006 | 8/12, n.s. |
| Walker's test, choice | +0.0005 | 9/12, p = 0.03 |

- Two shared axes decode coherence best. Axis 1 is signed coherence; axis 2
  separates the strongest motion. Across the difficult range one axis is enough.
- With the readout that low-dimensional, the likelihood's shape follows its
  position, so there is no room for a separate uncertainty signal: the
  flexible likelihood does not predict the wager beyond a fixed-width one.
- Power is limited (5–26 units per session): MT does not predict the choice
  beyond the stimulus either.
- Windows aligned to the saccade confound coherence with time since onset
  (fast trials sample the onset response). `firing_over_time.py` shows this.
