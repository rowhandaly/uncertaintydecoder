"""
Step 0: turn one recorded session into the arrays the pipeline uses.

Runs on Rockfish, next to the data. It reads only small files (spike times,
unit labels, TTL events, network messages), never the raw voltage, and
saves one small .npz per session with the same arrays that
step1_simulate_task.py makes:

    counts            (trials, units)  spike counts in a fixed window
    coherence         (trials,)        signed: + rightward, - leftward
    choice            (trials,)        +1 right, -1 left
    wager             (trials,)        1 high, 0 low
    coherence_values  the sorted list of signed coherences

plus rt, unit labels and trial numbers, for later analyses.

Where everything comes from (Open Ephys binary format + Kilosort/Phy):
  - behaviour: the task computer sends "name value" text messages to Open
    Ephys on every trial (TrialStart, Coherence, Direction, choice, PDW, RT)
  - motion onset/offset: TTL channel 4 goes on when the dots appear and
    off when they disappear (its duration equals the RT message)
  - spikes: Kilosort's spike_times (sample index from the start of the
    recording) and spike_clusters, with the lab's unit labels in
    cluster_info.tsv
All times are in samples of the 30 kHz acquisition clock.

Usage (on Rockfish):
    python step0_load_session.py "/path/to/hanzo_2021-09-14_13-27-18"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 30000

# Spike-counting window, relative to motion onset (seconds). MT responds
# about 50-80 ms after the dots appear, so the window starts after that.
# Trials whose motion ended before the window did are left out.
WINDOW_START = 0.08
WINDOW_LENGTH = 0.25

# Which units to use: the lab's Region label, and not curated as noise.
REGIONS = ["MT", "MST"]

# Task conventions. CHECK THESE against the table this script prints:
#   - at high coherence, rightward trials should almost always get the
#     "right" choice
#   - the high bet should become more common as coherence increases
RIGHTWARD_DIRECTION = 0      # Direction message value for rightward motion
LEFTWARD_DIRECTION = 180
RIGHT_CHOICE = 2             # choice message value for a rightward choice
HIGH_BET = 1                 # PDW message value for a high bet

MOTION_TTL_CHANNEL = 4       # TTL channel that is on while the dots are shown


def find_recording_folder(session_folder):
    """The folder that holds continuous/, events/ and structure.oebin."""
    matches = list(Path(session_folder).glob("Record Node */experiment*/recording*"))
    if len(matches) != 1:
        raise SystemExit(f"expected one recording folder, found: {matches}")
    return matches[0]


def read_messages(recording):
    """All network text messages, as (times, list of strings)."""
    folder = recording / "events/Network_Events-110.0/TEXT_group_1"
    times = np.load(folder / "timestamps.npy")
    # Each message is stored in a fixed 64 KB slot; mmap avoids loading
    # all 1.9 GB at once, and numpy drops the empty padding for us.
    text = np.load(folder / "text.npy", mmap_mode="r")
    messages = [text[i].decode(errors="replace").strip() for i in range(len(text))]
    return times, messages


def messages_to_trials(times, messages):
    """Group the messages into one dictionary per trial.

    A trial starts at "TrialStart N". Every "name value" message after it
    is stored under that name (a later message overwrites an earlier one,
    e.g. GoodOrBadTrial 0 -> 1). Messages with no value (e.g.
    "MotionStart") are events: we store their time instead. Block names
    (e.g. "Dots_PDW") come between trials and apply to the trials after.
    """
    block_names = {"Dots_PDW", "mappingMT", "mappingLIP"}
    trials = []
    block = None
    current = None
    for time, message in zip(times, messages):
        name, _, value = message.partition(" ")
        if name in block_names:
            block = name
        elif name == "TrialStart":
            current = {"trial_number": int(value), "start_time": time,
                       "block": block}
            trials.append(current)
        elif current is None or name == "IsRecording":
            continue
        elif value:
            current[name] = value
        else:
            current[name + "_time"] = time
    return pd.DataFrame(trials)


def motion_times(recording, trial_start_times):
    """Motion onset and offset for each trial, from the motion TTL.

    For each trial we take the first motion onset after the trial started
    (and before the next trial started), and the first offset after that.
    """
    folder = recording / "events/Rhythm_FPGA-100.0/TTL_1"
    ttl_times = np.load(folder / "timestamps.npy")
    ttl_states = np.load(folder / "channel_states.npy")
    onsets = ttl_times[ttl_states == MOTION_TTL_CHANNEL]
    offsets = ttl_times[ttl_states == -MOTION_TTL_CHANNEL]

    next_start = np.append(trial_start_times[1:], np.iinfo(np.int64).max)
    onset = np.full(len(trial_start_times), -1, dtype=np.int64)
    offset = np.full(len(trial_start_times), -1, dtype=np.int64)
    for i, (start, end) in enumerate(zip(trial_start_times, next_start)):
        k = np.searchsorted(onsets, start)
        if k < len(onsets) and onsets[k] < end:
            onset[i] = onsets[k]
            j = np.searchsorted(offsets, onsets[k])
            if j < len(offsets):
                offset[i] = offsets[j]
    return onset, offset


def load_units(recording):
    """Spike times (in clock samples) for each selected unit."""
    kilosort = recording / "continuous/Rhythm_FPGA-100.0/kilosort"
    info = pd.read_csv(kilosort / "cluster_info.tsv", sep="\t")
    selected = info[info["Region"].isin(REGIONS) & (info["group"] != "noise")]

    # Kilosort counts samples from the start of the .dat file; the clock
    # started at the first entry of timestamps.npy (see sync_messages.txt).
    clock = np.load(recording / "continuous/Rhythm_FPGA-100.0/timestamps.npy",
                    mmap_mode="r")
    first_sample = int(clock[0])
    spike_times = np.load(kilosort / "spike_times.npy").ravel().astype(np.int64)
    spike_times = spike_times + first_sample
    spike_clusters = np.load(kilosort / "spike_clusters.npy").ravel()

    spikes_per_unit = [np.sort(spike_times[spike_clusters == unit_id])
                       for unit_id in selected["id"]]
    return selected.reset_index(drop=True), spikes_per_unit


def count_spikes(spikes_per_unit, window_starts, window_ends):
    """(trials, units) spike counts between each start and end time."""
    counts = np.zeros((len(window_starts), len(spikes_per_unit)), dtype=int)
    for u, spikes in enumerate(spikes_per_unit):
        counts[:, u] = (np.searchsorted(spikes, window_ends)
                        - np.searchsorted(spikes, window_starts))
    return counts


if __name__ == "__main__":
    session_folder = Path(sys.argv[1])
    session_name = session_folder.name
    recording = find_recording_folder(session_folder)
    print(f"session: {session_name}\nrecording folder: {recording}\n")

    # 1. Behaviour from the network messages.
    times, messages = read_messages(recording)
    trials = messages_to_trials(times, messages)
    print("trials per block:")
    print(trials["block"].value_counts(dropna=False).to_string(), "\n")

    # 2. Motion onset/offset from the TTL, checked against the RT message.
    trials["motion_on"], trials["motion_off"] = motion_times(
        recording, trials["start_time"].to_numpy())

    # 3. Keep completed wagering trials with rightward or leftward motion.
    keep = (trials["block"].eq("Dots_PDW")
            & trials["choice"].notna() & trials["PDW"].notna()
            & trials["GoodOrBadTrial"].eq("1")
            & trials["motion_on"].ge(0) & trials["motion_off"].ge(0))
    task = trials[keep].copy()
    task["rt"] = task["RT"].astype(float)
    task["direction"] = task["Direction"].astype(float)
    task = task[task["direction"].isin([RIGHTWARD_DIRECTION, LEFTWARD_DIRECTION])]
    print(f"completed wagering trials: {len(task)}")

    ttl_duration = (task["motion_off"] - task["motion_on"]) / SAMPLE_RATE
    mismatch = (ttl_duration - task["rt"]).abs()
    print(f"motion TTL duration vs RT message: median difference "
          f"{mismatch.median() * 1000:.1f} ms, worst {mismatch.max() * 1000:.1f} ms\n")

    # 4. Signed coherence, choice (+1/-1), wager (1/0).
    unsigned = task["Coherence"].astype(float)
    sign = np.where(task["direction"] == RIGHTWARD_DIRECTION, 1, -1)
    # (+ 0.0 turns -0.0, from 0% leftward trials, into plain 0.0)
    task["coherence"] = np.round(unsigned * sign, 4) + 0.0
    task["choice_lr"] = np.where(task["choice"].astype(int) == RIGHT_CHOICE, 1, -1)
    task["wager"] = (task["PDW"].astype(int) == HIGH_BET).astype(int)

    # The convention check: read this table before trusting anything else.
    print("CHECK THE CONVENTIONS: P(right choice) should rise from ~0 to ~1")
    print("across signed coherence, and P(high bet) should rise with |coh|.")
    check = task.groupby("coherence").agg(
        n=("wager", "size"),
        p_right_choice=("choice_lr", lambda c: np.mean(c == 1)),
        p_high_bet=("wager", "mean"),
        mean_rt=("rt", "mean"))
    print(check.round(3).to_string(), "\n")

    # 5. Leave out trials whose motion ended before the counting window did.
    long_enough = task["rt"] >= WINDOW_START + WINDOW_LENGTH
    print(f"window {WINDOW_START * 1000:.0f}-{(WINDOW_START + WINDOW_LENGTH) * 1000:.0f} ms"
          f" after motion onset: keeping {long_enough.sum()} of {len(task)} trials")
    print("trials kept per |coherence|:")
    print(task[long_enough].groupby(task["coherence"].abs()).size().to_string(), "\n")
    task = task[long_enough]

    # 6. Spike counts for the selected units.
    units, spikes_per_unit = load_units(recording)
    print(f"units in {REGIONS}, not noise: {len(units)}")
    print(units[["id", "Region", "group", "KSLabel", "TargSelect", "Unit", "fr"]]
          .to_string(index=False), "\n")
    window_starts = task["motion_on"].to_numpy() + int(WINDOW_START * SAMPLE_RATE)
    window_ends = window_starts + int(WINDOW_LENGTH * SAMPLE_RATE)
    counts = count_spikes(spikes_per_unit, window_starts, window_ends)
    print(f"mean spike count per unit in the window: "
          f"{np.round(counts.mean(axis=0), 1)}")

    output = Path("data") / f"{session_name}.npz"
    output.parent.mkdir(exist_ok=True)
    # Explicit types, so the file opens with a plain np.load.
    np.savez(output,
             counts=counts,
             coherence=task["coherence"].to_numpy(dtype=float),
             choice=task["choice_lr"].to_numpy(dtype=int),
             wager=task["wager"].to_numpy(dtype=int),
             rt=task["rt"].to_numpy(dtype=float),
             coherence_values=np.sort(task["coherence"].unique()).astype(float),
             trial_number=task["trial_number"].to_numpy(dtype=int),
             unit_id=units["id"].to_numpy(dtype=int),
             unit_region=units["Region"].astype(str).to_numpy(dtype=str),
             unit_target=units["TargSelect"].astype(str).to_numpy(dtype=str),
             unit_type=units["Unit"].astype(str).to_numpy(dtype=str),
             window=np.array([WINDOW_START, WINDOW_LENGTH]))
    print(f"saved {output}")
