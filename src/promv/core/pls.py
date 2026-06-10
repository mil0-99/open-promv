"""Partial Least Squares (Projection to Latent Structures) regression.

PLS relates a block of process variables ``X`` to a block of quality / outcome
variables ``Y`` through a small set of latent variables, exactly the
"predict and optimise product quality" use case that motivates tools like
Aspen ProMV. Implemented with the classic NIPALS algorithm (supports both
single- and multi-response Y).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .preprocessing import Scaler


class PLS:
    """PLS regression (NIPALS).

    Parameters
    ----------
    n_components:
        Number of latent variables to retain.
    x_scaler, y_scaler:
        Scalers for the X and Y blocks. Default to autoscaling.
    tol, max_iter:
        Inner-loop convergence controls.
    alpha:
        Significance level for X-space T^2 / SPE limits.
    """

    def __init__(
        self,
        n_components: int = 2,
        x_scaler: Scaler | None = None,
        y_scaler: Scaler | None = None,
        tol: float = 1e-10,
        max_iter: int = 500,
        alpha: float = 0.05,
    ) -> None:
        self.n_components = n_components
        self.x_scaler = x_scaler if x_scaler is not None else Scaler()
        self.y_scaler = y_scaler if y_scaler is not None else Scaler()
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

        self.x_weights_: np.ndarray | None = None     # W   (m, A)
        self.x_loadings_: np.ndarray | None = None     # P   (m, A)
        self.y_loadings_: np.ndarray | None = None     # Q   (p, A)
        self.x_scores_: np.ndarray | None = None       # T   (n, A)
        self.y_scores_: np.ndarray | None = None       # U   (n, A)
        self.coef_: np.ndarray | None = None           # B   (m, p) scaled space
        self.eigenvalues_: np.ndarray | None = None
        self.r2x_: np.ndarray | None = None
        self.r2y_: np.ndarray | None = None
        self.t2_limit_: float | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PLS":
        Xs = self.x_scaler.fit_transform(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Ys = self.y_scaler.fit_transform(Y)

        n, m = Xs.shape
        p = Ys.shape[1]
        A = self.n_components

        W = np.zeros((m, A))
        P = np.zeros((m, A))
        Q = np.zeros((p, A))
        T = np.zeros((n, A))
        U = np.zeros((n, A))

        E, F = Xs.copy(), Ys.copy()
        ssx_total, ssy_total = np.sum(E ** 2), np.sum(F ** 2)
        r2x, r2y = np.zeros(A), np.zeros(A)

        for a in range(A):
            u = F[:, np.argmax(np.var(F, axis=0))].copy()
            w = t = q = None
            for _ in range(self.max_iter):
                w = E.T @ u / (u @ u)
                w = w / np.linalg.norm(w)
                t = E @ w
                q = F.T @ t / (t @ t)
                u_new = F @ q / (q @ q)
                if np.linalg.norm(u_new - u) < self.tol:
                    u = u_new
                    break
                u = u_new
            p_load = E.T @ t / (t @ t)
            E = E - np.outer(t, p_load)
            F = F - np.outer(t, q)

            W[:, a], P[:, a], Q[:, a], T[:, a], U[:, a] = w, p_load, q, t, u
            r2x[a] = 1 - np.sum(E ** 2) / ssx_total
            r2y[a] = 1 - np.sum(F ** 2) / ssy_total

        self.x_weights_, self.x_loadings_, self.y_loadings_ = W, P, Q
        self.x_scores_, self.y_scores_ = T, U
        self.eigenvalues_ = np.var(T, axis=0, ddof=1)
        # cumulative -> per-component explained variance
        self.r2x_ = np.diff(np.concatenate([[0.0], r2x]))
        self.r2y_ = np.diff(np.concatenate([[0.0], r2y]))

        # regression coefficients in the scaled space: B = W (P'W)^-1 Q'
        self.coef_ = W @ np.linalg.inv(P.T @ W) @ Q.T

        f = stats.f.ppf(1 - self.alpha, A, n - A)
        self.t2_limit_ = (A * (n - 1) * (n + 1)) / (n * (n - A)) * f
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Xs = self.x_scaler.transform(X)
        Ys_hat = Xs @ self.coef_
        return self.y_scaler.inverse_transform(Ys_hat)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X-space scores T for new observations."""
        self._check_fitted()
        Xs = self.x_scaler.transform(X)
        return Xs @ self.x_weights_ @ np.linalg.inv(
            self.x_loadings_.T @ self.x_weights_
        )

    def vip(self) -> np.ndarray:
        """Variable Importance in Projection -- ranks X variables by relevance.

        A VIP score above ~1 marks a variable as influential for predicting Y.
        """
        self._check_fitted()
        T, W, Q = self.x_scores_, self.x_weights_, self.y_loadings_
        m, A = W.shape
        ssy = np.sum((T ** 2), axis=0) * np.sum((Q ** 2), axis=0)  # per component
        total = ssy.sum()
        vip = np.sqrt(m * ((W ** 2) @ ssy) / total)
        return vip

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
