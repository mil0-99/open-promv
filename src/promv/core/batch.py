"""Batch process data handling.

Batch processes produce three-way data: ``(n_batches, n_variables, n_time)``.
The standard MSPC approach (Nomikos & MacGregor, 1994) is to *unfold* this cube
into a two-way matrix so an ordinary PCA/PLS model can be built, giving
Multiway PCA (MPCA). Two unfolding directions are provided:

* ``batch-wise``  -> (n_batches, n_variables * n_time):  each batch is one row,
  preserving the full time trajectory. This is the form used to monitor whether
  a *completed* batch is on-aim, and to build the classic batch model.
* ``variable-wise`` -> (n_batches * n_time, n_variables): each time slice is a
  row, useful for removing the mean trajectory before batch-wise modelling.
"""
from __future__ import annotations

import numpy as np


def unfold_batchwise(X3d: np.ndarray) -> np.ndarray:
    """Unfold ``(n_batches, n_variables, n_time)`` -> ``(n_batches, n_var*n_time)``.

    Column order is variable-major: all variables at t=0, then t=1, ...
    """
    X3d = np.asarray(X3d, dtype=float)
    if X3d.ndim != 3:
        raise ValueError("Expected a 3-D array (n_batches, n_variables, n_time).")
    n_b, n_v, n_t = X3d.shape
    # move time next to variables and flatten -> (batch, time*var) variable-major
    return np.transpose(X3d, (0, 2, 1)).reshape(n_b, n_t * n_v)


def unfold_variablewise(X3d: np.ndarray) -> np.ndarray:
    """Unfold ``(n_batches, n_variables, n_time)`` -> ``(n_batches*n_time, n_var)``."""
    X3d = np.asarray(X3d, dtype=float)
    if X3d.ndim != 3:
        raise ValueError("Expected a 3-D array (n_batches, n_variables, n_time).")
    n_b, n_v, n_t = X3d.shape
    return np.transpose(X3d, (0, 2, 1)).reshape(n_b * n_t, n_v)


def refold_batchwise(X2d: np.ndarray, n_variables: int, n_time: int) -> np.ndarray:
    """Inverse of :func:`unfold_batchwise`."""
    X2d = np.asarray(X2d, dtype=float)
    n_b = X2d.shape[0]
    return np.transpose(X2d.reshape(n_b, n_time, n_variables), (0, 2, 1))


def align_to_length(batch: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a single batch ``(n_variables, n_time)`` to ``target_len``.

    A pragmatic stand-in for trajectory alignment: real batches finish at
    different times, and models need equal-length trajectories. Each variable
    is linearly interpolated onto a common time grid.
    """
    batch = np.asarray(batch, dtype=float)
    n_v, n_t = batch.shape
    old = np.linspace(0.0, 1.0, n_t)
    new = np.linspace(0.0, 1.0, target_len)
    return np.vstack([np.interp(new, old, batch[v]) for v in range(n_v)])
