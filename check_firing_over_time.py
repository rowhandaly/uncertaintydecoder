"""
Check: how does firing change over the trial, and does it depend on
where the window is aligned?

Average firing rate over time for each |coherence|, pooled over units and
sessions, aligned two ways:
    to motion onset   (every trial measured from when the dots appeared)
    to the saccade    (motion offset: what a pre-saccade window sees)
split into each unit's preferred and null direction.

Typical MT: a burst ~100-200 ms after onset, then lower, steady firing
until the dots go off - no ramp towards the saccade. If so, a window just
before the saccade lands near the burst on fast (strong-motion) trials and
in the steady part on slow (weak-motion) trials: the timing confound that
the RT-band control removes.

Each unit's preferred direction is the one it fires more for at the
strongest coherence, 80-330 ms after onset.

Only times inside the motion period are used (the onset-aligned curves
stop at each trial's saccade; the saccade-aligned ones start at its motion
onset), plus a short stretch before onset / after the saccade.

Needs the trial lists that step 0 cached (data/<session>_trials.csv).

Usage (on Rockfish):
    python check_firing_over_time.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
    python check_firing_over_time.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary --regions LIP
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from step0_load_session import (SAMPLE_RATE, REGIONS, RIGHTWARD_DIRECTION,
                                LEFTWARD_DIRECTION, MIN_UNITS, one_match, load_units)

BIN = 0.02                                          # 20 ms bins
ONSET_BINS = np.arange(-0.2, 0.8 + 1e-9, BIN)       # relative to motion onset
SACCADE_BINS = np.arange(-0.8, 0.2 + 1e-9, BIN)     # relative to motion offset

parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=Path)
parser.add_argument("--regions", nargs="+", default=REGIONS)
args = parser.parse_args()


def rates(spikes, events, bin_edges):
    """(trials, bins) firing rate, in bins relative to each event."""
    edges = events[:, None] + np.round(bin_edges * SAMPLE_RATE).astype(np.int64)
    counts = np.diff(np.searchsorted(spikes, edges), axis=1)
    return counts / BIN


# Every unit's average rate over time, for each |coherence| and for its
# preferred / null direction: (units, 2 directions, coherences, bins).
all_onset, all_saccade = [], []
coherence_levels = None
for folder in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
    trials_file = Path("data") / f"{folder.name}_trials.csv"
    if not trials_file.exists():
        continue
    trials = pd.read_csv(trials_file, dtype={"GoodOrBadTrial": str})
    if "PDW" not in trials.columns:
        continue
    task = trials[trials["block"].eq("Dots_PDW")
                  & trials["choice"].notna() & trials["PDW"].notna()
                  & trials["GoodOrBadTrial"].astype(str).eq("1")
                  & trials["motion_on"].ge(0) & trials["motion_off"].ge(0)
                  & trials["Direction"].astype(float).isin([RIGHTWARD_DIRECTION,
                                                            LEFTWARD_DIRECTION])]
    try:
        recording = one_match(folder, "Record Node */experiment*/recording*")
        units, spikes_per_unit = load_units(recording, args.regions)
    except SystemExit:
        continue
    if len(units) < MIN_UNITS:
        continue

    on = task["motion_on"].to_numpy(np.int64)
    off = task["motion_off"].to_numpy(np.int64)
    rightward = task["Direction"].astype(float).to_numpy() == RIGHTWARD_DIRECTION
    strength = np.round(task["Coherence"].astype(float).to_numpy(), 4)
    levels = np.unique(strength)
    coherence_levels = levels if coherence_levels is None else coherence_levels
    if not np.array_equal(levels, coherence_levels):
        print(f"{folder.name}: different coherences, skipped")
        continue

    # Blank out times outside the motion period (keeping the stretch
    # before onset / after the saccade).
    onset_times = on[:, None] + np.round(ONSET_BINS[:-1] * SAMPLE_RATE)
    after_saccade = onset_times >= off[:, None]
    saccade_times = off[:, None] + np.round(SACCADE_BINS[:-1] * SAMPLE_RATE)
    before_onset = saccade_times < on[:, None]

    for spikes in spikes_per_unit:
        onset_rate = rates(spikes, on, ONSET_BINS)
        onset_rate[after_saccade] = np.nan
        saccade_rate = rates(spikes, off, SACCADE_BINS)
        saccade_rate[before_onset] = np.nan

        # Preferred direction: more spikes 80-330 ms after onset at the
        # strongest coherence.
        early = (ONSET_BINS[:-1] >= 0.08) & (ONSET_BINS[:-1] < 0.33)
        strongest = strength == levels.max()
        right = np.nanmean(onset_rate[strongest & rightward][:, early])
        left = np.nanmean(onset_rate[strongest & ~rightward][:, early])
        preferred = rightward if right >= left else ~rightward

        unit_onset = np.full((2, len(levels), len(ONSET_BINS) - 1), np.nan)
        unit_saccade = np.full((2, len(levels), len(SACCADE_BINS) - 1), np.nan)
        for d, direction in enumerate([preferred, ~preferred]):
            for c, level in enumerate(levels):
                chosen = direction & (strength == level)
                if level == 0:            # 0% has no direction: use all
                    chosen = strength == 0
                with np.errstate(all="ignore"):
                    unit_onset[d, c] = np.nanmean(onset_rate[chosen], axis=0)
                    unit_saccade[d, c] = np.nanmean(saccade_rate[chosen], axis=0)
        all_onset.append(unit_onset)
        all_saccade.append(unit_saccade)
    print(f"{folder.name}: {len(units)} units, {len(task)} trials")

if not all_onset:
    raise SystemExit("no sessions with enough units")
with np.errstate(all="ignore"):
    onset_curves = np.nanmean(all_onset, axis=0)       # (2, coherences, bins)
    saccade_curves = np.nanmean(all_saccade, axis=0)
area = "+".join(args.regions)
print(f"\n{len(all_onset)} {area} units")

# Printed summary: preferred-direction rate (spikes/s) at a few times.
onset_times = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
saccade_times = [-0.5, -0.3, -0.2, -0.1]
print("\npreferred direction, spikes/s (average over units)")
header = "".join(f"{t * 1000:+5.0f}" for t in onset_times)
header += "  |" + "".join(f"{t * 1000:+5.0f}" for t in saccade_times)
print(f"  |coh|    after onset (ms):{' ' * 18}| before saccade (ms):")
print(f"  {'':6s}{header}")
for c, level in enumerate(coherence_levels):
    row = "".join(f"{onset_curves[0, c, np.searchsorted(ONSET_BINS, t - 1e-9)]:5.1f}"
                  for t in onset_times)
    row += "  |" + "".join(f"{saccade_curves[0, c, np.searchsorted(SACCADE_BINS, t - 1e-9)]:5.1f}"
                           for t in saccade_times)
    print(f"  {level * 100:5.1f}%{row}")

# Figure: preferred (top) and null (bottom), onset- and saccade-aligned.
colours = plt.cm.viridis(np.linspace(0, 0.9, len(coherence_levels)))
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row",
                         gridspec_kw={"width_ratios": [1, 1]})
for d, name in enumerate(["preferred direction", "null direction"]):
    for ax, curves, bins, label in [
            (axes[d, 0], onset_curves, ONSET_BINS, "time from motion onset (s)"),
            (axes[d, 1], saccade_curves, SACCADE_BINS, "time from saccade (s)")]:
        for c, level in enumerate(coherence_levels):
            ax.plot(bins[:-1] + BIN / 2, curves[d, c], color=colours[c],
                    label=f"{level * 100:.1f}%")
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(label)
    axes[d, 0].set_ylabel(f"{name}\nspikes/s")
axes[0, 1].legend(title="|coherence|", fontsize=8)
fig.suptitle(f"{area}: {len(all_onset)} units")
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/firing_over_time_{area}.png", dpi=120)
print(f"\nsaved figures/firing_over_time_{area}.png")
