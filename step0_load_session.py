"""
Step 0: turn one recorded session into the arrays the pipeline uses.

Runs on Rockfish, next to the data. It reads only small files (spike times,
unit labels, TTL events, network messages), never the raw voltage, and
saves one small .npz per session and window with the same arrays that
step1_simulate_task.py makes:

    counts            (trials, units)  spike counts in a fixed window,
                                       with each unit's slow drift removed
    coherence         (trials,)        signed: + rightward, - leftward
    choice            (trials,)        +1 right, -1 left
    wager             (trials,)        1 high, 0 low
    coherence_values  the sorted list of signed coherences

plus the raw counts, rt, unit labels and trial numbers.

Where everything comes from (Open Ephys binary format + Kilosort/Phy):
  - behaviour: the task computer sends "name value" text messages to Open
    Ephys on every trial (TrialStart, Coherence, Direction, choice, PDW, RT)
  - motion onset/offset: TTL channel 4 goes on when the dots appear and
    off when they disappear (its duration equals the RT message)
  - spikes: Kilosort's spike_times (sample index from the start of the
    recording) and spike_clusters, with the lab's unit labels in
    cluster_info.tsv
All times are in samples of the 30 kHz acquisition clock.

The counting window always has the SAME LENGTH on every trial. (A window
that grew with RT would give long trials more spikes, hence sharper
likelihoods, and RT is related to confidence.) It is placed either:
  --align onset   relative to motion onset, e.g. --start 0.08 = 80 ms after
  --align offset  relative to motion offset (the saccade), e.g.
                  --start -0.30 = starting 300 ms before the dots went off
Trials whose window would fall outside the motion period are left out.

Usage (on Rockfish):
    python step0_load_session.py "/path/to/hanzo_2021-09-14_13-27-18"
    python step0_load_session.py "/path/to/..." --align offset --start -0.30
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 30000

# MT responds about 50-80 ms after the dots appear. A window must start at
# least this long after motion onset to contain only stimulus responses.
RESPONSE_LATENCY = 0.08

# Which units to use: the lab's Region label, and not curated as noise.
REGIONS = ["MT", "MST"]

# Sessions with fewer units or trials than this are skipped.
MIN_UNITS = 5
MIN_TRIALS = 200

# Slow drift: for each unit, a smooth curve over the session (a polynomial
# in trial order) is fitted and subtracted. The curve is fitted to each
# count MINUS the average count for that trial's coherence, so the
# stimulus can't leak into it (even if, by chance, the random coherence
# sequence leans one way for part of the session).
DRIFT_POLYNOMIAL_DEGREE = 3

# Task conventions (checked automatically below, and printed as a table).
RIGHTWARD_DIRECTION = 0      # Direction message value for rightward motion
LEFTWARD_DIRECTION = 180
RIGHT_CHOICE = 2             # choice message value for a rightward choice
HIGH_BET = 1                 # PDW message value for a high bet

MOTION_TTL_CHANNEL = 4       # TTL channel that is on while the dots are shown


def window_tag(align, start, length):
    """Short name for a window, used in file names: e.g. onset+80_250ms."""
    return f"{align}{round(start * 1000):+d}_{round(length * 1000)}ms"


def one_match(folder, pattern):
    """The single path matching a glob pattern (folder names differ a
    little between sessions, e.g. 'Record Node 109' vs 'Record Node 104')."""
    matches = sorted(Path(folder).glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"SKIP: expected one match for {pattern!r} in "
                         f"{folder}, found {len(matches)}")
    return matches[0]


def read_trials(recording):
    """One row per trial, parsed from the network text messages.

    A trial starts at "TrialStart N". Every "name value" message after it
    is stored under that name (a later message overwrites an earlier one,
    e.g. GoodOrBadTrial 0 -> 1). Messages with no value (e.g.
    "MotionStart") are events: we store their time instead. Block names
    (e.g. "Dots_PDW") come between trials and apply to the trials after.
    """
    folder = one_match(recording, "events/Network_Events-*/TEXT_group_1")
    times = np.load(folder / "timestamps.npy")
    # Each message is stored in a fixed 64 KB slot; mmap avoids loading
    # all of it at once, and numpy drops the empty padding for us.
    text = np.load(folder / "text.npy", mmap_mode="r")

    block_names = {"Dots_PDW", "mappingMT", "mappingLIP"}
    trials = []
    block = None
    current = None
    for i, time in enumerate(times):
        message = text[i].decode(errors="replace").strip()
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
    folder = one_match(recording, "events/Rhythm_FPGA-*/TTL_1")
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
    continuous = one_match(recording, "continuous/Rhythm_FPGA-*")
    kilosort = continuous / "kilosort"
    info = pd.read_csv(kilosort / "cluster_info.tsv", sep="\t")
    if "Region" not in info.columns:
        raise SystemExit("SKIP: cluster_info.tsv has no Region column "
                         "(units not labelled in Phy)")
    # The other lab labels are only printed and saved for reference, and
    # not every session has all of them: add any that are missing as blank.
    for column in ["group", "KSLabel", "TargSelect", "Unit", "fr"]:
        if column not in info.columns:
            info[column] = np.nan
    selected = info[info["Region"].isin(REGIONS) & (info["group"] != "noise")]

    # Kilosort counts samples from the start of the .dat file; the clock
    # started at the first entry of timestamps.npy (see sync_messages.txt).
    clock = np.load(continuous / "timestamps.npy", mmap_mode="r")
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


def remove_slow_drift(counts, coherence):
    """Subtract each unit's smooth trend over the session.

    The trend is fitted to the counts after removing each coherence's
    average, and then centred on zero, so each unit keeps its mean and
    its stimulus response.
    """
    stimulus_part = np.zeros(counts.shape)
    for c in np.unique(coherence):
        stimulus_part[coherence == c] = counts[coherence == c].mean(axis=0)
    residual = counts - stimulus_part

    trial_position = np.linspace(-1, 1, len(counts))
    detrended = np.zeros(counts.shape)
    for u in range(counts.shape[1]):
        trend = np.polyval(np.polyfit(trial_position, residual[:, u],
                                      DRIFT_POLYNOMIAL_DEGREE), trial_position)
        detrended[:, u] = counts[:, u] - (trend - trend.mean())
    return detrended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("session_folder", type=Path)
    parser.add_argument("--align", choices=["onset", "offset"], default="onset")
    parser.add_argument("--start", type=float, default=0.08,
                        help="window start (s) relative to the alignment event")
    parser.add_argument("--length", type=float, default=0.25,
                        help="window length (s)")
    args = parser.parse_args()

    session_name = args.session_folder.name
    tag = window_tag(args.align, args.start, args.length)
    recording = one_match(args.session_folder,
                          "Record Node */experiment*/recording*")
    print(f"session: {session_name}   window: {tag}\n"
          f"recording folder: {recording}\n")

    # 1. Behaviour from the network messages. Parsing reads the whole
    #    (1-2 GB) message file, so the result is cached per session.
    trials_file = Path("data") / f"{session_name}_trials.csv"
    trials_file.parent.mkdir(exist_ok=True)
    if trials_file.exists():
        trials = pd.read_csv(trials_file, dtype={"GoodOrBadTrial": str})
        print(f"read cached trials from {trials_file}")
    else:
        trials = read_trials(recording)
        trials["motion_on"], trials["motion_off"] = motion_times(
            recording, trials["start_time"].to_numpy())
        trials.to_csv(trials_file, index=False)
        print(f"parsed messages, cached trials in {trials_file}")
    print("trials per block:")
    print(trials["block"].value_counts(dropna=False).to_string(), "\n")

    # 2. Keep completed wagering trials with rightward or leftward motion.
    for column in ["choice", "PDW", "RT", "Coherence", "Direction"]:
        if column not in trials.columns:
            raise SystemExit(f"SKIP: no {column!r} messages in this session")
    keep = (trials["block"].eq("Dots_PDW")
            & trials["choice"].notna() & trials["PDW"].notna()
            & trials["GoodOrBadTrial"].astype(str).eq("1")
            & trials["motion_on"].ge(0) & trials["motion_off"].ge(0))
    task = trials[keep].copy()
    task["rt"] = task["RT"].astype(float)
    task["direction"] = task["Direction"].astype(float)
    task = task[task["direction"].isin([RIGHTWARD_DIRECTION, LEFTWARD_DIRECTION])]
    print(f"completed wagering trials: {len(task)}")

    # 3. Check the motion TTL against the RT message.
    ttl_duration = (task["motion_off"] - task["motion_on"]) / SAMPLE_RATE
    mismatch = (ttl_duration - task["rt"]).abs()
    print(f"motion TTL duration vs RT message: median difference "
          f"{mismatch.median() * 1000:.1f} ms, worst {mismatch.max() * 1000:.1f} ms")
    if mismatch.median() > 0.005:
        raise SystemExit("SKIP: motion TTL does not match RT - check which "
                         "TTL channel marks the motion in this session")

    # 4. Signed coherence, choice (+1/-1), wager (1/0).
    unsigned = task["Coherence"].astype(float)
    sign = np.where(task["direction"] == RIGHTWARD_DIRECTION, 1, -1)
    # (+ 0.0 turns -0.0, from 0% leftward trials, into plain 0.0)
    task["coherence"] = np.round(unsigned * sign, 4) + 0.0
    task["choice_lr"] = np.where(task["choice"].astype(int) == RIGHT_CHOICE, 1, -1)
    task["wager"] = (task["PDW"].astype(int) == HIGH_BET).astype(int)

    print("\nconventions: P(right choice) should rise from ~0 to ~1 across")
    print("signed coherence, and P(high bet) should rise with |coherence|.")
    check = task.groupby("coherence").agg(
        n=("wager", "size"),
        p_right_choice=("choice_lr", lambda c: np.mean(c == 1)),
        p_high_bet=("wager", "mean"),
        mean_rt=("rt", "mean"))
    print(check.round(3).to_string(), "\n")
    strongest = check.index.max()
    if not (check.loc[strongest, "p_right_choice"] > 0.8
            and check.loc[-strongest, "p_right_choice"] < 0.2):
        raise SystemExit("SKIP: choice coding looks wrong (strong motion is "
                         "not followed by the matching choice)")
    weak = check.index.to_series().abs() <= 0.032
    if not (check.loc[~weak, "p_high_bet"].mean()
            > check.loc[weak, "p_high_bet"].mean()):
        raise SystemExit("SKIP: wager coding looks wrong (high bets are not "
                         "more common at high coherence)")

    # 5. Place the window, and keep trials where it lies within the motion
    #    period (and after MT's response latency).
    align_time = task["motion_on"] if args.align == "onset" else task["motion_off"]
    window_starts = (align_time + round(args.start * SAMPLE_RATE)).to_numpy()
    window_ends = window_starts + round(args.length * SAMPLE_RATE)
    inside = ((window_starts >= task["motion_on"] + RESPONSE_LATENCY * SAMPLE_RATE)
              & (window_ends <= task["motion_off"])).to_numpy()
    print(f"window {tag}: keeping {inside.sum()} of {len(task)} trials")
    print("trials kept per |coherence|:")
    print(task[inside].groupby(task["coherence"].abs()).size().to_string(), "\n")
    task = task[inside]
    window_starts, window_ends = window_starts[inside], window_ends[inside]

    # 6. Spike counts for the selected units, then remove slow drift.
    units, spikes_per_unit = load_units(recording)
    print(f"units in {REGIONS}, not noise: {len(units)}")
    print(units[["id", "Region", "group", "KSLabel", "TargSelect", "Unit", "fr"]]
          .to_string(index=False), "\n")
    if len(units) < MIN_UNITS or len(task) < MIN_TRIALS:
        raise SystemExit(f"SKIP: {len(units)} units and {len(task)} trials "
                         f"(need at least {MIN_UNITS} and {MIN_TRIALS})")
    raw_counts = count_spikes(spikes_per_unit, window_starts, window_ends)
    counts = remove_slow_drift(raw_counts, task["coherence"].to_numpy())
    print(f"mean spike count per unit in the window: "
          f"{np.round(raw_counts.mean(axis=0), 1)}")

    output = Path("data") / f"{session_name}__{tag}.npz"
    # Explicit types, so the file opens with a plain np.load.
    np.savez(output,
             counts=counts,
             raw_counts=raw_counts,
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
             window=np.array([args.start, args.length]),
             window_align=args.align)
    print(f"saved {output}")
