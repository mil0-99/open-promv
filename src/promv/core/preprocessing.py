"""Data preprocessing: mean-centering and scaling.

Process data almost always needs centering and (usually) unit-variance
scaling before a latent-variable model is built, otherwise variables that
happen to have large engineering units dominate the model. This mirrors the
"autoscale" option found in commercial MSPC tools.
"""
from __future__ import annotations

import numpy as np


class Scaler:
    """Center and scale a data matrix, remembering the transform.

    Parameters
    ----------
    center:
        Subtract the column mean. Default ``True``.
    scale:
        Divide by the column standard deviation (unit-variance / "autoscale").
        Default ``True``.
    ddof:
        Delta degrees of freedom for the standard deviation. ``1`` gives the
        sample standard deviation (the usual choice in MSPC).
    """

    def __init__(self, center: bool = True, scale: bool = True, ddof: int = 1) -> None:
        self.center = center
        self.scale = scale
        self.ddof = ddof
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Scaler":
        X = np.asarray(X, dtype=float)
        # nan-aware so the scaler tolerates missing entries in the data
        self.mean_ = np.nanmean(X, axis=0) if self.center else np.zeros(X.shape[1])
        if self.scale:
            std = np.nanstd(X, axis=0, ddof=self.ddof)
            std[std == 0] = 1.0  # guard against constant columns
            self.std_ = std
        else:
            self.std_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fit before calling transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fit before calling inverse_transform().")
        X = np.asarray(X, dtype=float)
        return X * self.std_ + self.mean_
