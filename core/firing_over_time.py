"""
Average MT firing over time for each |coherence|, aligned to motion onset
and to the saccade, for each unit's preferred and null direction. Shows why
the analysis window is aligned to motion onset.

Usage:  python firing_over_time.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from load_session import SAMPLE_RATE, MIN_UNITS, RIGHTWARD, find_one, load_task_trials, load_units

warnings.filterwarnings("ignore", category=RuntimeWarning)

BIN = 0.02
BINS_FROM_ONSET = np.arange(-0.2, 0.8 + 1e-9, BIN)
BINS_FROM_SACCADE = np.arange(-0.8, 0.2 + 1e-9, BIN)


def firing_rates(spikes, event_times, bin_edges):
    """(trials, bins) spikes per second, in bins relative to each event."""
    edges = event_times[:, None] + np.round(bin_edges * SAMPLE_RATE).astype(np.int64)
    return np.diff(np.searchsorted(spikes, edges), axis=1) / BIN


from_onset_per_unit, from_saccade_per_unit = [], []
strengths = None
for folder in sorted(folder for folder in Path(sys.argv[1]).iterdir() if folder.is_dir()):
    try:
        recording = find_one(folder, "Record Node */experiment*/recording*")
        task = load_task_trials(folder, recording)
        units, spikes_per_unit = load_units(recording)
    except SystemExit:
        continue
    if len(units) < MIN_UNITS:
        continue
    print(f"{folder.name}: {len(units)} units")

    motion_on = task["motion_on"].to_numpy(np.int64)
    motion_off = task["motion_off"].to_numpy(np.int64)
    rightward = task["Direction"].astype(float).to_numpy() == RIGHTWARD
    strength = task["coherence"].abs().to_numpy()
    strengths = np.unique(strength)

    # Leave out time after the dots went off (onset-aligned) or before they came on (saccade-aligned).
    after_motion = motion_on[:, None] + np.round(BINS_FROM_ONSET[:-1] * SAMPLE_RATE) >= motion_off[:, None]
    before_motion = motion_off[:, None] + np.round(BINS_FROM_SACCADE[:-1] * SAMPLE_RATE) < motion_on[:, None]
    early = (BINS_FROM_ONSET[:-1] >= 0.08) & (BINS_FROM_ONSET[:-1] < 0.33)

    for spikes in spikes_per_unit:
        from_onset = firing_rates(spikes, motion_on, BINS_FROM_ONSET)
        from_onset[after_motion] = np.nan
        from_saccade = firing_rates(spikes, motion_off, BINS_FROM_SACCADE)
        from_saccade[before_motion] = np.nan

        strongest = strength == strengths.max()
        prefers_right = (np.nanmean(from_onset[strongest & rightward][:, early])
                         >= np.nanmean(from_onset[strongest & ~rightward][:, early]))
        preferred = rightward if prefers_right else ~rightward

        unit_from_onset = np.full((2, len(strengths), len(BINS_FROM_ONSET) - 1), np.nan)
        unit_from_saccade = np.full((2, len(strengths), len(BINS_FROM_SACCADE) - 1), np.nan)
        for row, direction in enumerate([preferred, ~preferred]):
            for column, value in enumerate(strengths):
                trials = (strength == 0) if value == 0 else (direction & (strength == value))
                unit_from_onset[row, column] = np.nanmean(from_onset[trials], axis=0)
                unit_from_saccade[row, column] = np.nanmean(from_saccade[trials], axis=0)
        from_onset_per_unit.append(unit_from_onset)
        from_saccade_per_unit.append(unit_from_saccade)

from_onset = np.nanmean(from_onset_per_unit, axis=0)
from_saccade = np.nanmean(from_saccade_per_unit, axis=0)

colours = plt.cm.viridis(np.linspace(0, 0.9, len(strengths)))
figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
for row, direction_name in enumerate(["preferred direction", "null direction"]):
    for ax, curves, bins, label in [(axes[row, 0], from_onset, BINS_FROM_ONSET, "time from motion onset (s)"),
                                    (axes[row, 1], from_saccade, BINS_FROM_SACCADE, "time from saccade (s)")]:
        for column, value in enumerate(strengths):
            ax.plot(bins[:-1] + BIN / 2, curves[row, column], color=colours[column], label=f"{value * 100:.1f}%")
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(label)
    axes[row, 0].set_ylabel(f"{direction_name}\nspikes/s")
axes[0, 1].legend(title="|coherence|", fontsize=8)
figure.suptitle(f"MT/MST: {len(from_onset_per_unit)} units")
figure.tight_layout()
Path("figures").mkdir(exist_ok=True)
figure.savefig("figures/firing_over_time.png", dpi=120)
print("saved figures/firing_over_time.png")
