"""
The likelihood decoders, shared by steps 2-4.

Every decoder here does the same job: take one trial's spike counts and
return a log-likelihood for each of the 11 possible coherences,

    log L(c) = log P(spike counts | coherence = c)      (up to a constant)

and they are all trained the same way as in Walker et al. (2020):

    log posterior = log_softmax( log L + log prior )
    loss          = cross-entropy between that posterior and the true coherence

The decoders differ only in how much freedom they have:

  FullDecoder         any linear map from neurons to the 11 log-likelihoods
                      (the linear version of Walker's "full-likelihood" decoder)
  LowRankDecoder      the population is first squeezed through a few shared
                      axes, and every coherence's log-likelihood is read out
                      from those same axes (our test of shared structure)
  FixedWidthDecoder   one bump of fixed width, whose centre moves from trial
                      to trial (Walker's "fixed-uncertainty" decoder)
"""

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold


class FullDecoder(nn.Module):
    def __init__(self, n_neurons, coherence_values):
        super().__init__()
        n_coherences = len(coherence_values)
        # One weight per (neuron, coherence) pair, plus one bias per coherence.
        self.neurons_to_log_likelihood = nn.Linear(n_neurons, n_coherences)

    def forward(self, spike_counts):
        return self.neurons_to_log_likelihood(spike_counts)


class LowRankDecoder(nn.Module):
    def __init__(self, n_neurons, coherence_values, rank):
        super().__init__()
        n_coherences = len(coherence_values)
        # Step 1: project the population onto `rank` shared axes.
        self.neurons_to_axes = nn.Linear(n_neurons, rank, bias=False)
        # Step 2: each coherence's log-likelihood is a weighted sum of those axes.
        self.axes_to_log_likelihood = nn.Linear(rank, n_coherences)

    def forward(self, spike_counts):
        position_on_axes = self.neurons_to_axes(spike_counts)
        return self.axes_to_log_likelihood(position_on_axes)


class FixedWidthDecoder(nn.Module):
    def __init__(self, n_neurons, coherence_values):
        super().__init__()
        self.coherence_values = torch.tensor(coherence_values,
                                             dtype=torch.float32)
        # The centre of the bump is read out from the population...
        self.neurons_to_centre = nn.Linear(n_neurons, 1)
        # ...but its width is one number, the same on every trial.
        self.log_width = nn.Parameter(torch.tensor(np.log(0.2),
                                                   dtype=torch.float32))

    def forward(self, spike_counts):
        centre = self.neurons_to_centre(spike_counts)      # (trials, 1)
        width = torch.exp(self.log_width)
        distance = self.coherence_values - centre          # (trials, 11)
        return -0.5 * (distance / width) ** 2


def train(decoder, spike_counts, coherence_index, n_steps=1000):
    """Fit a decoder with Walker et al.'s objective.

    spike_counts:    (trials, neurons) float tensor, already z-scored
    coherence_index: (trials,) long tensor, which of the 11 coherences
                     was shown (0 = most leftward, 10 = most rightward)
    """
    # The prior is just how often each coherence was shown.
    n_coherences = decoder(spike_counts[:1]).shape[1]
    counts_per_coherence = torch.bincount(coherence_index,
                                          minlength=n_coherences)
    log_prior = torch.log(counts_per_coherence / len(coherence_index))

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01,
                                 weight_decay=1e-4)
    for step in range(n_steps):
        optimizer.zero_grad()
        log_likelihood = decoder(spike_counts)
        log_posterior = torch.log_softmax(log_likelihood + log_prior, dim=1)
        # Cross-entropy = minus the average log posterior of the true coherence.
        loss = -log_posterior[torch.arange(len(coherence_index)),
                              coherence_index].mean()
        loss.backward()
        optimizer.step()
    return decoder


def cross_validated_log_likelihoods(make_decoder, spike_counts,
                                    coherence_index, n_folds=5):
    """Decode every trial with a decoder that never saw it.

    make_decoder: a function that returns a fresh, untrained decoder
    Returns:
      log_likelihood: (trials, 11), normalised so exp() sums to 1 per trial
      score:          (trials,), held-out log probability of the true
                      coherence - higher means a better decoder
    """
    n_trials = len(coherence_index)
    n_coherences = coherence_index.max() + 1
    log_likelihood = np.zeros((n_trials, n_coherences))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    for train_trials, test_trials in folds.split(spike_counts):
        # z-score using the training trials only.
        mean = spike_counts[train_trials].mean(axis=0)
        std = spike_counts[train_trials].std(axis=0)
        train_counts = torch.tensor((spike_counts[train_trials] - mean) / std,
                                    dtype=torch.float32)
        test_counts = torch.tensor((spike_counts[test_trials] - mean) / std,
                                   dtype=torch.float32)

        decoder = train(make_decoder(),
                        train_counts,
                        torch.tensor(coherence_index[train_trials]))

        with torch.no_grad():
            test_log_likelihood = torch.log_softmax(decoder(test_counts),
                                                    dim=1).numpy()
        log_likelihood[test_trials] = test_log_likelihood

    # With a (near) uniform prior, the normalised likelihood is the posterior,
    # so the score is just the log-likelihood of the true coherence.
    score = log_likelihood[np.arange(n_trials), coherence_index]
    return log_likelihood, score
