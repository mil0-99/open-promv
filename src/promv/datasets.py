"""Synthetic process data so the library can be demonstrated without a plant.

``make_continuous_process`` builds a correlated multivariate data set from a few
hidden latent drivers and can inject a step fault into part of the run.
``make_batch_process`` builds a three-way batch data set with smooth time
trajectories and optional abnormal batches.
"""
from __future__ import annotations

import numpy as np


def make_continuous_process(
    n_samples: int = 300,
    n_variables: int = 12,
    n_latent: int = 3,
    fault_start: int | None = None,
    fault_vars=None,
    fault_magnitude: float = 4.0,
    noise: float = 0.4,
    seed: int = 0,
):
    """Return ``(X, is_fault)`` for a continuous process.

    The first ``fault_start`` rows are normal operating condition (NOC) data;
    from ``fault_start`` onward a step bias is added to ``fault_vars``.
    """
    rng = np.random.default_rng(seed)
    # latent drivers -> observed variables via a random loading matrix
    scores = rng.normal(size=(n_samples, n_latent))
    loadings = rng.normal(size=(n_latent, n_variables))
    X = scores @ loadings + noise * rng.normal(size=(n_samples, n_variables))

    is_fault = np.zeros(n_samples, dtype=bool)
    if fault_start is not None:
        if fault_vars is None:
            fault_vars = [0, 1]
        X[fault_start:, fault_vars] += fault_magnitude
        is_fault[fault_start:] = True
    return X, is_fault


def make_batch_process(
    n_batches: int = 40,
    n_variables: int = 5,
    n_time: int = 60,
    n_bad: int = 5,
    noise: float = 0.05,
    seed: int = 0,
):
    """Return ``(X3d, is_bad)`` of shape ``(n_batches, n_variables, n_time)``.

    Good batches follow a common smooth trajectory plus small batch-to-batch
    variation; bad batches have a distorted trajectory on a couple of variables.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_time)
    # archetypal trajectories per variable (mix of ramps and bells)
    base = np.vstack([
        np.sin(np.pi * t * (k + 1)) + 0.5 * t * (k + 1)
        for k in range(n_variables)
    ])

    X = np.zeros((n_batches, n_variables, n_time))
    is_bad = np.zeros(n_batches, dtype=bool)
    bad_idx = rng.choice(n_batches, size=n_bad, replace=False)

    for i in range(n_batches):
        amp = 1 + 0.05 * rng.normal(size=(n_variables, 1))   # batch gain variation
        traj = base * amp + noise * rng.normal(size=(n_variables, n_time))
        if i in bad_idx:
            # distort the second half of two variables
            traj[:2, n_time // 2:] += 0.8
            is_bad[i] = True
        X[i] = traj
    return X, is_bad
