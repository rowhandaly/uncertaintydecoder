"""
Turn one recorded session into spike counts for MT/MST units, in one
window after motion onset, plus the task variables.

Usage:
    python load_session.py /path/to/hanzo_2021-09-14_13-27-18
    python load_session.py /path/to/hanzo_2021-09-14_13-27-18 --max-coherence 0.128
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_RATE = 30000
AREAS = ["MT", "MST"]
MIN_UNITS = 5
MIN_TRIALS = 200
DRIFT_POLYNOMIAL_DEGREE = 3

RIGHTWARD = 0
LEFTWARD = 180
RIGHT_CHOICE = 2
HIGH_BET = 1
MOTION_TTL_CHANNEL = 4


def analysis_name(start, length, max_coherence=None):
    name = f"onset{round(start * 1000):+d}_{round(length * 1000)}ms"
    if max_coherence is not None:
        name += f"_coh{round(max_coherence * 1000)}"
    return name


def find_one(folder, pattern):
    matches = sorted(Path(folder).glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"SKIP: expected one {pattern!r} in {folder}, found {len(matches)}")
    return matches[0]


def read_trials(recording):
    """One row per trial, from the task's "name value" text messages."""
    folder = find_one(recording, "events/Network_Events-*/TEXT_group_1")
    message_times = np.load(folder / "timestamps.npy")
    messages = np.load(folder / "text.npy", mmap_mode="r")

    trials = []
    current_block = None
    current_trial = None
    for time, raw_message in zip(message_times, messages):
        message = raw_message.decode(errors="replace").strip()
        name, _, value = message.partition(" ")
        if name in ["Dots_PDW", "mappingMT", "mappingLIP"]:
            current_block = name
        elif name == "TrialStart":
            current_trial = {"trial_number": int(value), "start_time": time,
                             "block": current_block}
            trials.append(current_trial)
        elif current_trial is None or name == "IsRecording":
            continue
        elif value:
            current_trial[name] = value
        else:
            current_trial[name + "_time"] = time
    return pd.DataFrame(trials)


def find_motion_times(recording, trial_start_times):
    """Motion onset and offset on each trial, from the motion TTL."""
    folder = find_one(recording, "events/Rhythm_FPGA-*/TTL_1")
    ttl_times = np.load(folder / "timestamps.npy")
    ttl_states = np.load(folder / "channel_states.npy")
    all_onsets = ttl_times[ttl_states == MOTION_TTL_CHANNEL]
    all_offsets = ttl_times[ttl_states == -MOTION_TTL_CHANNEL]

    next_trial_start = np.append(trial_start_times[1:], np.iinfo(np.int64).max)
    onsets = np.full(len(trial_start_times), -1, dtype=np.int64)
    offsets = np.full(len(trial_start_times), -1, dtype=np.int64)
    for trial, (start, end) in enumerate(zip(trial_start_times, next_trial_start)):
        first_onset = np.searchsorted(all_onsets, start)
        if first_onset < len(all_onsets) and all_onsets[first_onset] < end:
            onsets[trial] = all_onsets[first_onset]
            first_offset = np.searchsorted(all_offsets, onsets[trial])
            if first_offset < len(all_offsets):
                offsets[trial] = all_offsets[first_offset]
    return onsets, offsets


def load_units(recording):
    """Spike times (in clock samples) of every MT/MST unit not labelled noise."""
    continuous = find_one(recording, "continuous/Rhythm_FPGA-*")
    kilosort = continuous / "kilosort"
    unit_info = pd.read_csv(kilosort / "cluster_info.tsv", sep="\t")
    if "Region" not in unit_info.columns:
        raise SystemExit("SKIP: units have no Region label")
    if "group" not in unit_info.columns:
        unit_info["group"] = np.nan
    units = unit_info[unit_info["Region"].isin(AREAS) & (unit_info["group"] != "noise")]

    # Kilosort counts from the start of the file; the clock starts at clock[0].
    clock = np.load(continuous / "timestamps.npy", mmap_mode="r")
    spike_times = np.load(kilosort / "spike_times.npy").ravel().astype(np.int64) + int(clock[0])
    spike_units = np.load(kilosort / "spike_clusters.npy").ravel()

    spikes_per_unit = [np.sort(spike_times[spike_units == unit_id]) for unit_id in units["id"]]
    return units.reset_index(drop=True), spikes_per_unit


def load_task_trials(session_folder, recording):
    """Completed wagering trials with left/right motion (parsing is cached)."""
    Path("data").mkdir(exist_ok=True)
    cache = Path("data") / f"{session_folder.name}_trials.csv"
    if cache.exists():
        trials = pd.read_csv(cache, dtype={"GoodOrBadTrial": str})
    else:
        trials = read_trials(recording)
        trials["motion_on"], trials["motion_off"] = find_motion_times(
            recording, trials["start_time"].to_numpy())
        trials.to_csv(cache, index=False)

    for column in ["choice", "PDW", "RT", "Coherence", "Direction"]:
        if column not in trials.columns:
            raise SystemExit(f"SKIP: no {column!r} messages")
    completed = (trials["block"].eq("Dots_PDW")
                 & trials["choice"].notna() & trials["PDW"].notna()
                 & trials["GoodOrBadTrial"].astype(str).eq("1")
                 & trials["motion_on"].ge(0) & trials["motion_off"].ge(0)
                 & trials["Direction"].astype(float).isin([RIGHTWARD, LEFTWARD]))
    task = trials[completed].copy()

    direction_sign = np.where(task["Direction"].astype(float) == RIGHTWARD, 1, -1)
    task["coherence"] = np.round(task["Coherence"].astype(float) * direction_sign, 4) + 0.0
    task["choice"] = np.where(task["choice"].astype(int) == RIGHT_CHOICE, 1, -1)
    task["wager"] = (task["PDW"].astype(int) == HIGH_BET).astype(int)
    task["rt"] = task["RT"].astype(float)
    return task


def check_codings(task):
    motion_duration = (task["motion_off"] - task["motion_on"]) / SAMPLE_RATE
    if (motion_duration - task["rt"]).abs().median() > 0.005:
        raise SystemExit("SKIP: motion TTL does not match RT")

    by_coherence = task.groupby("coherence").agg(
        trials=("wager", "size"),
        p_right=("choice", lambda choice: np.mean(choice == 1)),
        p_high_bet=("wager", "mean"),
        mean_rt=("rt", "mean"))
    print(by_coherence.round(3).to_string(), "\n")

    strongest = by_coherence.index.max()
    if not (by_coherence.loc[strongest, "p_right"] > 0.8 and by_coherence.loc[-strongest, "p_right"] < 0.2):
        raise SystemExit("SKIP: choice coding looks wrong")
    weak = by_coherence.index.to_series().abs() <= 0.032
    if not by_coherence.loc[~weak, "p_high_bet"].mean() > by_coherence.loc[weak, "p_high_bet"].mean():
        raise SystemExit("SKIP: wager coding looks wrong")


def count_spikes(spikes_per_unit, window_starts, window_ends):
    counts = np.zeros((len(window_starts), len(spikes_per_unit)), dtype=int)
    for unit, spikes in enumerate(spikes_per_unit):
        counts[:, unit] = np.searchsorted(spikes, window_ends) - np.searchsorted(spikes, window_starts)
    return counts


def remove_slow_drift(counts, coherence):
    """Subtract each unit's smooth trend over the session, fitted after
    removing each coherence's average so the stimulus can't leak in."""
    stimulus_part = np.zeros(counts.shape)
    for value in np.unique(coherence):
        stimulus_part[coherence == value] = counts[coherence == value].mean(axis=0)
    left_over = counts - stimulus_part

    position_in_session = np.linspace(-1, 1, len(counts))
    without_drift = np.zeros(counts.shape)
    for unit in range(counts.shape[1]):
        fit = np.polyfit(position_in_session, left_over[:, unit], DRIFT_POLYNOMIAL_DEGREE)
        trend = np.polyval(fit, position_in_session)
        without_drift[:, unit] = counts[:, unit] - (trend - trend.mean())
    return without_drift


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_folder", type=Path)
    parser.add_argument("--start", type=float, default=0.08, help="seconds after motion onset")
    parser.add_argument("--length", type=float, default=0.25, help="seconds")
    parser.add_argument("--max-coherence", type=float, default=None)
    args = parser.parse_args()

    name = analysis_name(args.start, args.length, args.max_coherence)
    recording = find_one(args.session_folder, "Record Node */experiment*/recording*")
    print(f"{args.session_folder.name}   {name}\n")

    task = load_task_trials(args.session_folder, recording)
    check_codings(task)

    window_starts = task["motion_on"].to_numpy() + round(args.start * SAMPLE_RATE)
    window_ends = window_starts + round(args.length * SAMPLE_RATE)
    keep = window_ends <= task["motion_off"].to_numpy()
    if args.max_coherence is not None:
        keep = keep & (task["coherence"].abs() <= args.max_coherence + 1e-9).to_numpy()
    task = task[keep]
    window_starts, window_ends = window_starts[keep], window_ends[keep]

    units, spikes_per_unit = load_units(recording)
    print(f"{len(units)} units, {len(task)} trials")
    if len(units) < MIN_UNITS or len(task) < MIN_TRIALS:
        raise SystemExit(f"SKIP: {len(units)} units and {len(task)} trials "
                         f"(need at least {MIN_UNITS} and {MIN_TRIALS})")

    raw_counts = count_spikes(spikes_per_unit, window_starts, window_ends)
    counts = remove_slow_drift(raw_counts, task["coherence"].to_numpy())

    output = Path("data") / f"{args.session_folder.name}__{name}.npz"
    np.savez(output,
             counts=counts,
             coherence=task["coherence"].to_numpy(dtype=float),
             coherence_values=np.sort(task["coherence"].unique()).astype(float),
             choice=task["choice"].to_numpy(dtype=int),
             wager=task["wager"].to_numpy(dtype=int),
             rt=task["rt"].to_numpy(dtype=float),
             trial_number=task["trial_number"].to_numpy(dtype=int),
             unit_id=units["id"].to_numpy(dtype=int))
    print(f"saved {output}")
