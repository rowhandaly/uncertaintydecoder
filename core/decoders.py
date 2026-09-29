"""
Likelihood decoders: spike counts in, a log-likelihood for each of the
11 coherences out. All trained like Walker et al. (2020): add the log
prior, softmax, cross-entropy against the true coherence.

    FullDecoder         separate readout weights for every coherence
    LowRankDecoder      every coherence read out from the same few population axes
    FixedWidthDecoder   a bump of one fixed width; only its centre moves
"""

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold

torch.set_num_threads(1)


class FullDecoder(nn.Module):
    def __init__(self, n_neurons, n_coherences):
        super().__init__()
        self.neurons_to_log_likelihood = nn.Linear(n_neurons, n_coherences)

    def forward(self, spike_counts):
        return self.neurons_to_log_likelihood(spike_counts)

    def readout_weights(self):
        """(coherences, neurons): the weights each coherence puts on each neuron."""
        return self.neurons_to_log_likelihood.weight.detach().numpy()


class LowRankDecoder(nn.Module):
    def __init__(self, n_neurons, n_coherences, rank):
        super().__init__()
        self.neurons_to_axes = nn.Linear(n_neurons, rank, bias=False)
        self.axes_to_log_likelihood = nn.Linear(rank, n_coherences)

    def forward(self, spike_counts):
        position_on_axes = self.neurons_to_axes(spike_counts)
        return self.axes_to_log_likelihood(position_on_axes)

    def readout_weights(self):
        loadings = self.axes_to_log_likelihood.weight.detach().numpy()
        axes = self.neurons_to_axes.weight.detach().numpy()
        return loadings @ axes


class FixedWidthDecoder(nn.Module):
    def __init__(self, n_neurons, coherence_values):
        super().__init__()
        self.coherence_values = torch.tensor(coherence_values, dtype=torch.float32)
        self.neurons_to_centre = nn.Linear(n_neurons, 1)
        self.log_width = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))

    def forward(self, spike_counts):
        centre = self.neurons_to_centre(spike_counts)
        width = torch.exp(self.log_width)
        distance = self.coherence_values - centre
        return -0.5 * (distance / width) ** 2

    def readout_weights(self):
        """Expanding -(c - centre)^2 / 2 width^2, the only part that depends on
        both the coherence c and the neurons is c * centre / width^2: one axis
        (the centre's weights) with loadings proportional to c."""
        centre_weights = self.neurons_to_centre.weight.detach().numpy()[0]
        width = torch.exp(self.log_width).item()
        return np.outer(self.coherence_values.numpy(), centre_weights) / width ** 2


def train(decoder, spike_counts, coherence_index, n_steps=1000):
    """spike_counts: (trials, neurons) z-scored tensor.
    coherence_index: (trials,) which coherence was shown, 0 = most leftward."""
    n_coherences = decoder(spike_counts[:1]).shape[1]
    trials_per_coherence = torch.bincount(coherence_index, minlength=n_coherences)
    log_prior = torch.log(trials_per_coherence / len(coherence_index))

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01, weight_decay=1e-4)
    every_trial = torch.arange(len(coherence_index))
    for step in range(n_steps):
        optimizer.zero_grad()
        log_posterior = torch.log_softmax(decoder(spike_counts) + log_prior, dim=1)
        loss = -log_posterior[every_trial, coherence_index].mean()
        loss.backward()
        optimizer.step()
    return decoder


def z_score(counts, mean, std):
    return torch.tensor((counts - mean) / std, dtype=torch.float32)


def decode_held_out(make_decoder, spike_counts, coherence_index, n_folds=5):
    """Decode every trial with a decoder trained on the other trials.

    Returns the (trials, coherences) log-likelihoods, normalised per trial,
    and each trial's score: the log-likelihood of the true coherence."""
    n_trials = len(coherence_index)
    log_likelihood = np.zeros((n_trials, coherence_index.max() + 1))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    for training_trials, test_trials in folds.split(spike_counts):
        mean = spike_counts[training_trials].mean(axis=0)
        std = spike_counts[training_trials].std(axis=0)
        decoder = train(make_decoder(),
                        z_score(spike_counts[training_trials], mean, std),
                        torch.tensor(coherence_index[training_trials]))
        with torch.no_grad():
            test_output = decoder(z_score(spike_counts[test_trials], mean, std))
            log_likelihood[test_trials] = torch.log_softmax(test_output, dim=1).numpy()

    score = log_likelihood[np.arange(n_trials), coherence_index]
    return log_likelihood, score
