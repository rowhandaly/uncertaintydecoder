"""
Likelihood decoding of motion coherence from MT/MST (Walker et al. 2020 style).

For every session: count spikes 80-330 ms after motion onset, decode coherence
with four decoders (fixed width, rank 1, rank 2, full), pull out each decoder's
axes, and ask how well its likelihood predicts choice and wager. Then plot:

    firing_over_time.png    why the window starts at motion onset
    loadings.png            loading vs coherence for each decoder
    uncertainty.png         likelihood spread vs |coherence|
    decoding.png            how well each decoder decodes the stimulus
    predictive_power.png    how well each likelihood predicts choice and wager

Per-session results are saved in results/, so a rerun only redoes new sessions.

Usage:  python analysis.py /home/cfetsch1/vast-cfetsch1/data/hanzo_neuro_binary
"""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from torch import nn

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

FS = 30000                      # samples per second
WINDOW = (0.08, 0.33)           # seconds after motion onset
HARD = 0.128                    # max |coherence| for the hard-trials run
MIN_UNITS, MIN_TRIALS = 5, 200
MODELS = ["fixed width", "rank 1", "rank 2", "full"]


# ---------- loading a session ----------

def find(folder, pattern):
    hits = sorted(Path(folder).glob(pattern))
    if len(hits) != 1:
        raise ValueError(f"expected one {pattern} in {folder}")
    return hits[0]


def read_trials(rec):
    """One row per trial from the task's text messages, plus motion on/off times."""
    folder = find(rec, "events/Network_Events-*/TEXT_group_1")
    times = np.load(folder / "timestamps.npy")
    texts = np.load(folder / "text.npy", mmap_mode="r")

    rows, block, trial = [], None, None
    for t, text in zip(times, texts):
        key, _, value = text.decode(errors="replace").strip().partition(" ")
        if key in ["Dots_PDW", "mappingMT", "mappingLIP"]:
            block = key
        elif key == "TrialStart":
            trial = {"trial_number": int(value), "start_time": t, "block": block}
            rows.append(trial)
        elif trial is None or key == "IsRecording":
            continue
        elif value:
            trial[key] = value
        else:
            trial[key + "_time"] = t
    trials = pd.DataFrame(rows)

    # motion on/off come from TTL channel 4
    ttl = find(rec, "events/Rhythm_FPGA-*/TTL_1")
    ttl_times = np.load(ttl / "timestamps.npy")
    states = np.load(ttl / "channel_states.npy")
    ons, offs = ttl_times[states == 4], ttl_times[states == -4]

    starts = trials["start_time"].to_numpy()
    next_starts = np.append(starts[1:], np.iinfo(np.int64).max)
    motion_on = np.full(len(starts), -1, dtype=np.int64)
    motion_off = np.full(len(starts), -1, dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, next_starts)):
        j = np.searchsorted(ons, start)
        if j < len(ons) and ons[j] < end:
            motion_on[i] = ons[j]
            k = np.searchsorted(offs, ons[j])
            if k < len(offs):
                motion_off[i] = offs[k]
    trials["motion_on"], trials["motion_off"] = motion_on, motion_off
    return trials


def load_task(session, rec):
    """Completed left/right wagering trials. Parsing is slow, so it's cached."""
    cache = Path("results") / f"{session.name}_trials.csv"
    if cache.exists():
        trials = pd.read_csv(cache, dtype={"GoodOrBadTrial": str})
    else:
        trials = read_trials(rec)
        trials.to_csv(cache, index=False)

    direction = trials["Direction"].astype(float)
    done = (trials["block"].eq("Dots_PDW") & trials["choice"].notna() & trials["PDW"].notna()
            & trials["GoodOrBadTrial"].astype(str).eq("1")
            & trials["motion_on"].ge(0) & trials["motion_off"].ge(0)
            & direction.isin([0, 180]))
    task = trials[done].copy()

    sign = np.where(task["Direction"].astype(float) == 0, 1, -1)    # rightward = +
    task["coherence"] = np.round(task["Coherence"].astype(float) * sign, 4) + 0.0
    task["choice"] = np.where(task["choice"].astype(int) == 2, 1, -1)
    task["wager"] = (task["PDW"].astype(int) == 1).astype(int)
    return task


def load_units(rec):
    """Spike times of every MT/MST unit that isn't labelled noise."""
    cont = find(rec, "continuous/Rhythm_FPGA-*")
    info = pd.read_csv(cont / "kilosort" / "cluster_info.tsv", sep="\t")
    if "group" not in info.columns:
        info["group"] = np.nan
    units = info[info["Region"].isin(["MT", "MST"]) & (info["group"] != "noise")]

    clock_start = int(np.load(cont / "timestamps.npy", mmap_mode="r")[0])
    spike_times = np.load(cont / "kilosort" / "spike_times.npy").ravel().astype(np.int64) + clock_start
    spike_units = np.load(cont / "kilosort" / "spike_clusters.npy").ravel()
    return [np.sort(spike_times[spike_units == u]) for u in units["id"]]


def spike_counts(task, spikes, max_coh=None):
    """Counts in the window, with each unit's slow drift over the session removed."""
    start = task["motion_on"].to_numpy() + round(WINDOW[0] * FS)
    end = start + round((WINDOW[1] - WINDOW[0]) * FS)
    keep = end <= task["motion_off"].to_numpy()
    if max_coh is not None:
        keep &= (task["coherence"].abs() <= max_coh + 1e-9).to_numpy()
    task, start, end = task[keep], start[keep], end[keep]

    counts = np.column_stack([np.searchsorted(s, end) - np.searchsorted(s, start) for s in spikes])
    coh = task["coherence"].to_numpy(dtype=float)

    # fit drift on what's left after removing each coherence's mean
    residual = counts.astype(float)
    for c in np.unique(coh):
        residual[coh == c] -= counts[coh == c].mean(axis=0)
    x = np.linspace(-1, 1, len(counts))
    corrected = np.zeros(counts.shape)
    for u in range(counts.shape[1]):
        trend = np.polyval(np.polyfit(x, residual[:, u], 3), x)
        corrected[:, u] = counts[:, u] - (trend - trend.mean())
    return task, corrected


# ---------- decoders ----------

class LowRank(nn.Module):
    """Every coherence is read out from the same few population axes."""
    def __init__(self, n_units, n_coh, rank):
        super().__init__()
        self.axes = nn.Linear(n_units, rank, bias=False)
        self.loadings = nn.Linear(rank, n_coh)

    def forward(self, x):
        return self.loadings(self.axes(x))

    def weights(self):
        return self.loadings.weight.detach().numpy() @ self.axes.weight.detach().numpy()


class Full(nn.Module):
    """Separate weights for every coherence."""
    def __init__(self, n_units, n_coh):
        super().__init__()
        self.readout = nn.Linear(n_units, n_coh)

    def forward(self, x):
        return self.readout(x)

    def weights(self):
        return self.readout.weight.detach().numpy()


class FixedWidth(nn.Module):
    """A Gaussian bump of one fixed width; only its centre moves."""
    def __init__(self, n_units, coh_values):
        super().__init__()
        self.coh = torch.tensor(coh_values, dtype=torch.float32)
        self.centre = nn.Linear(n_units, 1)
        self.log_width = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))

    def forward(self, x):
        return -0.5 * ((self.coh - self.centre(x)) / torch.exp(self.log_width)) ** 2

    def weights(self):
        # only c * centre / width^2 mixes coherence and neurons: one axis, loadings ∝ c
        w = self.centre.weight.detach().numpy()[0]
        return np.outer(self.coh.numpy(), w) / torch.exp(self.log_width).item() ** 2


def make_model(name, n_units, coh_values):
    n_coh = len(coh_values)
    if name == "fixed width":
        return FixedWidth(n_units, coh_values)
    if name == "full":
        return Full(n_units, n_coh)
    return LowRank(n_units, n_coh, rank=int(name[-1]))


def train(model, x, y, steps=1000):
    """Walker's objective: add the log prior, softmax, cross-entropy."""
    prior = torch.log(torch.bincount(y, minlength=model(x[:1]).shape[1]) / len(y))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = -torch.log_softmax(model(x) + prior, dim=1)[torch.arange(len(y)), y].mean()
        loss.backward()
        opt.step()
    return model


def zscore(x, mean, std):
    return torch.tensor((x - mean) / std, dtype=torch.float32)


def decode(counts, coh, coh_values):
    """Held-out log-likelihoods (trials x coherences) for each model, 5-fold."""
    y = np.searchsorted(coh_values, coh)
    folds = list(KFold(5, shuffle=True, random_state=0).split(counts))
    torch.manual_seed(0)
    out = {}
    for name in MODELS:
        ll = np.zeros((len(y), len(coh_values)))
        for tr, te in folds:
            mean, std = counts[tr].mean(axis=0), counts[tr].std(axis=0)
            model = train(make_model(name, counts.shape[1], coh_values),
                          zscore(counts[tr], mean, std), torch.tensor(y[tr]))
            with torch.no_grad():
                ll[te] = torch.log_softmax(model(zscore(counts[te], mean, std)), dim=1).numpy()
        out[name] = ll
    return out


def loadings(counts, coh, coh_values):
    """Fit each model on all trials and split its weights into axes x loadings (SVD)."""
    x = zscore(counts, counts.mean(axis=0), counts.std(axis=0))
    y = torch.tensor(np.searchsorted(coh_values, coh))
    torch.manual_seed(0)
    out = {}
    for name in MODELS:
        w = train(make_model(name, counts.shape[1], coh_values), x, y).weights()
        u, s, _ = np.linalg.svd(w - w.mean(axis=0), full_matrices=False)
        n_axes = {"fixed width": 1, "rank 1": 1, "rank 2": 2, "full": 3}[name]
        load = u[:, :n_axes] * s[:n_axes]
        load *= np.where(load[-1] < 0, -1, 1)          # sign is arbitrary: make +51.2% positive
        out[name] = load / np.abs(load).max(axis=0)    # scale to max 1
    return out


# ---------- behaviour ----------

def heldout_ll(features, target):
    """Log probability of what the monkey did, from a 10-fold logistic regression."""
    folds = StratifiedKFold(10, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(C=100, max_iter=2000), features, target,
                          cv=folds, method="predict_proba")
    return np.log(p[np.arange(len(target)), np.searchsorted(np.unique(target), target)])


def log_odds_right(ll, coh_values):
    p = np.exp(ll - np.logaddexp.reduce(ll, axis=1, keepdims=True))
    zero = p[:, coh_values == 0].sum(axis=1) / 2
    return np.log(p[:, coh_values > 0].sum(axis=1) + zero) - np.log(p[:, coh_values < 0].sum(axis=1) + zero)


def predictive_power(lls, coh, coh_values, choice, wager):
    """Gain in held-out log-likelihood of choice and wager from adding each
    model's likelihood, over chance and over the stimulus alone."""
    references = {"": np.zeros((len(coh), 1)),                                  # chance
                  " | stimulus": (coh[:, None] == coh_values).astype(float)}    # coherence only
    gains = {}
    for tag, ref in references.items():
        ref_choice, ref_wager = heldout_ll(ref, choice), heldout_ll(ref, wager)
        for name in MODELS:
            odds = log_odds_right(lls[name], coh_values)
            gains["choice" + tag, name] = np.mean(heldout_ll(np.column_stack([ref, odds]), choice) - ref_choice)
            gains["wager" + tag, name] = np.mean(
                heldout_ll(np.column_stack([ref, np.abs(odds), choice * odds]), wager) - ref_wager)
    return gains


# ---------- firing over time ----------

BIN = 0.02
FROM_ONSET = np.arange(-0.2, 0.8 + 1e-9, BIN)
FROM_SACCADE = np.arange(-0.8, 0.2 + 1e-9, BIN)


def firing_curves(task, spikes):
    """Each unit's mean rate per |coherence|, preferred and null direction,
    aligned to motion onset and to the saccade (end of motion)."""
    on, off = task["motion_on"].to_numpy(np.int64), task["motion_off"].to_numpy(np.int64)
    right = task["Direction"].astype(float).to_numpy() == 0
    strength = task["coherence"].abs().to_numpy()
    levels = np.unique(strength)

    # blank out time after the dots went off / before they came on
    after = on[:, None] + np.round(FROM_ONSET[:-1] * FS) >= off[:, None]
    before = off[:, None] + np.round(FROM_SACCADE[:-1] * FS) < on[:, None]
    early = (FROM_ONSET[:-1] >= 0.08) & (FROM_ONSET[:-1] < 0.33)

    def rates(s, events, bins):
        edges = events[:, None] + np.round(bins * FS).astype(np.int64)
        return np.diff(np.searchsorted(s, edges), axis=1) / BIN

    curves = []
    for s in spikes:
        onset, sacc = rates(s, on, FROM_ONSET), rates(s, off, FROM_SACCADE)
        onset[after], sacc[before] = np.nan, np.nan

        top = strength == levels.max()
        pref = right if np.nanmean(onset[top & right][:, early]) >= np.nanmean(onset[top & ~right][:, early]) else ~right
        unit = []
        for direction in [pref, ~pref]:
            rows = [(strength == 0) if lv == 0 else (direction & (strength == lv)) for lv in levels]
            unit.append([[np.nanmean(onset[r], axis=0) for r in rows],
                         [np.nanmean(sacc[r], axis=0) for r in rows]])
        curves.append(unit)
    return levels, curves      # curves: units x (pref, null) x (onset, saccade) x levels x bins


# ---------- run every session ----------

def run_session(folder):
    rec = find(folder, "Record Node */experiment*/recording*")
    task, spikes = load_task(folder, rec), load_units(rec)
    if len(spikes) < MIN_UNITS:
        raise ValueError(f"only {len(spikes)} units")
    levels, curves = firing_curves(task, spikes)

    res = {"levels": levels, "curves": curves}
    for label, max_coh in [("all", None), ("hard", HARD)]:
        t, counts = spike_counts(task, spikes, max_coh)
        if len(t) < MIN_TRIALS:
            raise ValueError(f"only {len(t)} trials")
        coh = t["coherence"].to_numpy(dtype=float)
        coh_values = np.unique(coh)
        lls = decode(counts, coh, coh_values)
        res[label] = {"coh": coh, "coh_values": coh_values, "n_units": counts.shape[1],
                      "score": {m: lls[m][np.arange(len(coh)), np.searchsorted(coh_values, coh)] for m in MODELS}}
        if label == "all":
            res["lls"] = lls
            res["loadings"] = loadings(counts, coh, coh_values)
            res["power"] = predictive_power(lls, coh, coh_values,
                                            t["choice"].to_numpy(dtype=int), t["wager"].to_numpy(dtype=int))
    return res


data_root = Path(sys.argv[1])
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

sessions = {}
for folder in sorted(f for f in data_root.iterdir() if f.is_dir() and not f.name.startswith("@")):
    saved = Path("results") / f"{folder.name}.pkl"
    if saved.exists():
        sessions[folder.name] = pickle.loads(saved.read_bytes())
        continue
    try:
        sessions[folder.name] = run_session(folder)
    except (ValueError, KeyError) as problem:
        print(f"{folder.name}: skipped ({problem})")
        continue
    saved.write_bytes(pickle.dumps(sessions[folder.name]))
    print(f"{folder.name}: done")

names = sorted(sessions)
print(f"\n{len(names)} sessions, {sum(sessions[s]['all']['n_units'] for s in names)} units")


# ---------- figures ----------

def dot_plot(ax, values, title, ylabel):
    """One gray dot per session, black = mean ± SEM."""
    for i, m in enumerate(MODELS):
        v = np.array(values[m])
        ax.plot(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, "o", color="gray", alpha=0.6)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="s", color="black", markersize=8, capsize=4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


# firing over time
levels = sessions[names[0]]["levels"]
curves = np.nanmean([unit for s in names for unit in sessions[s]["curves"]], axis=0)
colours = plt.cm.viridis(np.linspace(0, 0.9, len(levels)))
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
for row, direction in enumerate(["preferred direction", "null direction"]):
    for col, (bins, xlabel) in enumerate([(FROM_ONSET, "time from motion onset (s)"),
                                         (FROM_SACCADE, "time from saccade (s)")]):
        for i, lv in enumerate(levels):
            axes[row, col].plot(bins[:-1] + BIN / 2, curves[row, col, i], color=colours[i], label=f"{lv * 100:.1f}%")
        axes[row, col].axvline(0, color="gray", lw=0.5)
        axes[row, col].set_xlabel(xlabel)
    axes[row, 0].set_ylabel(f"{direction}\nspikes/s")
axes[0, 1].legend(title="|coherence|", fontsize=8)
fig.suptitle(f"MT/MST: {sum(len(sessions[s]['curves']) for s in names)} units")
fig.tight_layout()
fig.savefig("figures/firing_over_time.png", dpi=120)

# loading vs coherence
coh_values = sessions[names[0]]["all"]["coh_values"]
fig, panels = plt.subplots(1, len(MODELS), figsize=(16, 3.5), sharey=True)
for ax, m in zip(panels, MODELS):
    load = np.array([sessions[s]["loadings"][m] for s in names])
    for a in range(load.shape[2]):
        ax.errorbar(coh_values * 100, load[:, :, a].mean(axis=0), yerr=load[:, :, a].std(axis=0) / np.sqrt(len(names)),
                    fmt="o-", label=f"axis {a + 1}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(m)
    ax.set_xlabel("coherence (%)")
    ax.legend(fontsize=8)
panels[0].set_ylabel("loading (max 1 per session)")
fig.tight_layout()
fig.savefig("figures/loadings.png", dpi=120)

# uncertainty (entropy of the likelihood) vs |coherence|
fig, ax = plt.subplots(figsize=(5, 4))
for m in MODELS:
    per_session = []
    for s in names:
        ll = sessions[s]["lls"][m]
        p = np.exp(ll - np.logaddexp.reduce(ll, axis=1, keepdims=True))
        entropy = -np.sum(p * np.log2(p + 1e-12), axis=1)
        strength = np.abs(sessions[s]["all"]["coh"])
        per_session.append([entropy[strength == lv].mean() for lv in np.unique(strength)])
    ax.plot(np.unique(np.abs(coh_values)) * 100, np.mean(per_session, axis=0), "o-", label=m)
ax.set_xlabel("|coherence| (%)")
ax.set_ylabel("spread of the likelihood (bits)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/uncertainty.png", dpi=120)

# decoding the stimulus
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, label, title in zip(axes, ["all", "hard"], ["a) all coherences", "b) |coherence| ≤ 12.8%"]):
    chance = {s: np.log(1 / len(sessions[s][label]["coh_values"])) for s in names}
    dot_plot(ax, {m: [sessions[s][label]["score"][m].mean() - chance[s] for s in names] for m in MODELS},
             title, "log likelihood over chance")
fig.suptitle("Decoding performance of likelihood")
fig.tight_layout()
fig.savefig("figures/decoding.png", dpi=120)

# predicting choice and wager
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
panels = [("choice", "a) choice", "log likelihood over chance"),
          ("wager", "b) wager", "log likelihood over chance"),
          ("choice | stimulus", "c) choice, beyond the stimulus", "log likelihood over stimulus alone"),
          ("wager | stimulus", "d) wager, beyond the stimulus", "log likelihood over stimulus alone")]
for ax, (key, title, ylabel) in zip(axes.flat, panels):
    dot_plot(ax, {m: [sessions[s]["power"][key, m] for s in names] for m in MODELS}, title, ylabel)
fig.suptitle("Predictive power of likelihood")
fig.tight_layout()
fig.savefig("figures/predictive_power.png", dpi=120)


# ---------- numbers ----------

def summary(label, values):
    v = np.array(values)
    print(f"  {label:24s} {v.mean():+.4f} ± {v.std(ddof=1) / np.sqrt(len(v)):.4f}  "
          f"({np.sum(v > 0)}/{len(v)} sessions > 0, p = {ttest_1samp(v, 0).pvalue:.2g})")


for label in ["all", "hard"]:
    print(f"\ndecoding the stimulus ({label} coherences), held-out log-likelihood per trial:")
    for a, b in [("rank 2", "rank 1"), ("rank 2", "fixed width"), ("rank 2", "full")]:
        summary(f"{a} - {b}", [sessions[s][label]["score"][a].mean() - sessions[s][label]["score"][b].mean()
                               for s in names])

for key, _, _ in panels:
    print(f"\npredicting {key}, gain per trial:")
    for m in MODELS:
        summary(m, [sessions[s]["power"][key, m] for s in names])
print("\nfigures saved in figures/")
