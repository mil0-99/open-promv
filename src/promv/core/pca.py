"""Principal Component Analysis for process monitoring.

Implements PCA via the NIPALS algorithm, which (unlike a plain SVD) handles
missing data gracefully -- a common situation with plant historian data where
sensors drop out. The fitted model exposes the projection statistics used for
multivariate statistical process control: Hotelling's T-squared and the
squared prediction error (SPE / Q), together with their confidence limits and
variable contributions for fault diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .preprocessing import Scaler


@dataclass
class MonitoringResult:
    """Container for projection / monitoring statistics of a data set."""

    scores: np.ndarray            # T: (n_samples, n_components)
    t2: np.ndarray                # Hotelling's T^2 per sample
    spe: np.ndarray               # squared prediction error (Q) per sample
    residuals: np.ndarray         # E = X - X_hat in the scaled space
    t2_contrib: np.ndarray = field(default=None)   # per-variable T^2 contributions
    spe_contrib: np.ndarray = field(default=None)  # per-variable SPE contributions


class PCA:
    """Principal Component Analysis (NIPALS).

    Parameters
    ----------
    n_components:
        Number of latent variables (principal components) to retain.
    scaler:
        A :class:`~promv.core.preprocessing.Scaler`. If ``None`` a default
        autoscaling scaler (center + unit variance) is used.
    tol, max_iter:
        Convergence tolerance and iteration cap for the NIPALS inner loop.
    alpha:
        Significance level for the T^2 and SPE control limits (default 0.05,
        i.e. 95% limits).
    """

    def __init__(
        self,
        n_components: int = 2,
        scaler: Scaler | None = None,
        tol: float = 1e-9,
        max_iter: int = 500,
        alpha: float = 0.05,
    ) -> None:
        self.n_components = n_components
        self.scaler = scaler if scaler is not None else Scaler()
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

        # fitted attributes
        self.loadings_: np.ndarray | None = None       # P (n_features, A)
        self.scores_: np.ndarray | None = None          # T (n_samples, A)
        self.eigenvalues_: np.ndarray | None = None      # variance of each score
        self.explained_variance_ratio_: np.ndarray | None = None
        self.t2_limit_: float | None = None
        self.spe_limit_: float | None = None
        self._spe_g: float | None = None                 # SPE chi-square scaling
        self._spe_h: float | None = None                 # SPE chi-square dof
        self.n_samples_: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "PCA":
        Xs = self.scaler.fit_transform(X)
        n, m = Xs.shape
        A = self.n_components
        if A > min(n, m):
            raise ValueError("n_components cannot exceed min(n_samples, n_features).")

        T = np.zeros((n, A))
        P = np.zeros((m, A))
        E = Xs.copy()
        total_ss = np.nansum(Xs ** 2)
        explained = np.zeros(A)

        for a in range(A):
            t, p = self._nipals_component(E)
            # deflate
            E = E - np.outer(t, p)
            T[:, a] = t
            P[:, a] = p
            explained[a] = np.nansum((np.outer(t, p)) ** 2)

        self.scores_ = T
        self.loadings_ = P
        self.eigenvalues_ = np.var(T, axis=0, ddof=1)
        self.explained_variance_ratio_ = explained / total_ss
        self.n_samples_ = n

        self._fit_limits(T, E)
        return self

    def _nipals_component(self, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract one PCA component from residual matrix ``E`` (NaN-aware)."""
        n, m = E.shape
        mask = ~np.isnan(E)
        Efill = np.where(mask, E, 0.0)

        # initialise score with the column of greatest variance
        col_var = np.nanvar(E, axis=0)
        t = Efill[:, int(np.nanargmax(col_var))].copy()

        for _ in range(self.max_iter):
            # p = E' t / (t' t), summing only over observed entries
            denom_p = (mask * (t[:, None] ** 2)).sum(axis=0)
            denom_p[denom_p == 0] = np.finfo(float).eps
            p = (Efill * t[:, None]).sum(axis=0) / denom_p
            p = p / np.linalg.norm(p)

            # t = E p / (p' p), again observation-wise for missing data
            denom_t = (mask * (p[None, :] ** 2)).sum(axis=1)
            denom_t[denom_t == 0] = np.finfo(float).eps
            t_new = (Efill * p[None, :]).sum(axis=1) / denom_t

            if np.linalg.norm(t_new - t) < self.tol:
                t = t_new
                break
            t = t_new
        return t, p

    # --------------------------------------------------------------- limits
    def _fit_limits(self, T: np.ndarray, E: np.ndarray) -> None:
        n, A = T.shape

        # Hotelling's T^2 limit from the F distribution (training-set form)
        f = stats.f.ppf(1 - self.alpha, A, n - A)
        self.t2_limit_ = (A * (n - 1) * (n + 1)) / (n * (n - A)) * f

        # SPE / Q limit via the Box / Nomikos-MacGregor chi-square moment match:
        # SPE ~ g * chi^2(h) with g = v / (2*mean), h = 2*mean^2 / v
        spe = np.nansum(E ** 2, axis=1)
        mean_spe = spe.mean()
        var_spe = spe.var(ddof=1)
        if var_spe <= 0:
            self.spe_limit_ = float(mean_spe)
            self._spe_g, self._spe_h = 1.0, 1.0
        else:
            g = var_spe / (2 * mean_spe)
            h = 2 * mean_spe ** 2 / var_spe
            self._spe_g, self._spe_h = g, h
            self.spe_limit_ = g * stats.chi2.ppf(1 - self.alpha, h)

    # ----------------------------------------------------------- projection
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new observations onto the model, returning scores T."""
        self._check_fitted()
        Xs = self.scaler.transform(X)
        # least-squares projection; robust to NaNs by zero-filling residuals
        Xfill = np.where(np.isnan(Xs), 0.0, Xs)
        return Xfill @ self.loadings_

    def inverse_transform(self, T: np.ndarray) -> np.ndarray:
        """Reconstruct observations (scaled space) from scores."""
        self._check_fitted()
        return T @ self.loadings_.T

    def monitor(self, X: np.ndarray, contributions: bool = True) -> MonitoringResult:
        """Compute monitoring statistics (T^2, SPE) for new observations."""
        self._check_fitted()
        Xs = self.scaler.transform(X)
        Xfill = np.where(np.isnan(Xs), 0.0, Xs)
        T = Xfill @ self.loadings_
        X_hat = T @ self.loadings_.T
        E = Xfill - X_hat

        t2 = np.sum((T ** 2) / self.eigenvalues_, axis=1)
        spe = np.sum(E ** 2, axis=1)

        t2_contrib = spe_contrib = None
        if contributions:
            # SPE contribution is simply the squared residual per variable
            spe_contrib = E ** 2
            # T^2 contribution: sum_a (t_a / lambda_a) * p_{ja} * x_j
            # gives a per-variable share of each sample's T^2
            weighted = (T / self.eigenvalues_) @ self.loadings_.T
            t2_contrib = weighted * Xfill

        return MonitoringResult(
            scores=T, t2=t2, spe=spe, residuals=E,
            t2_contrib=t2_contrib, spe_contrib=spe_contrib,
        )

    # ------------------------------------------------------------- helpers
    def _check_fitted(self) -> None:
        if self.loadings_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
