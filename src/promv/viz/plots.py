"""Plotting helpers for multivariate process monitoring.

These mirror the standard chart set in MSPC tools: score scatter plots with the
Hotelling ellipse, loading plots, T^2 and SPE control charts, and contribution
bar charts for diagnosing which variables drive an alarm. Every function
returns a Matplotlib ``Axes`` so callers can further customise or save.
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except Exception as exc:  # pragma: no cover - matplotlib is an optional extra
    raise ImportError(
        "Plotting requires matplotlib. Install with `pip install open-promv[viz]`."
    ) from exc

from scipy import stats


def score_plot(model, comps=(0, 1), labels=None, ax=None, alpha=0.05):
    """Scatter of two score vectors with the Hotelling T^2 confidence ellipse."""
    T = model.scores_ if hasattr(model, "scores_") else model.x_scores_
    a, b = comps
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(T[:, a], T[:, b], s=30, edgecolor="k", linewidth=0.4, alpha=0.85)

    n = T.shape[0]
    s_a = np.std(T[:, a], ddof=1)
    s_b = np.std(T[:, b], ddof=1)
    f = stats.f.ppf(1 - alpha, 2, n - 2)
    radius = np.sqrt(2 * (n - 1) * (n + 1) / (n * (n - 2)) * f)
    ell = Ellipse((T[:, a].mean(), T[:, b].mean()),
                  width=2 * radius * s_a, height=2 * radius * s_b,
                  edgecolor="crimson", facecolor="none", linestyle="--", lw=1.4)
    ax.add_patch(ell)

    if labels is not None:
        for i, lab in enumerate(labels):
            ax.annotate(str(lab), (T[i, a], T[i, b]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"t[{a + 1}]")
    ax.set_ylabel(f"t[{b + 1}]")
    ax.set_title(f"Score plot ({int((1 - alpha) * 100)}% Hotelling ellipse)")
    return ax


def loading_plot(model, comps=(0, 1), var_names=None, ax=None):
    """Scatter of two loading vectors -- shows how variables relate."""
    P = model.loadings_ if hasattr(model, "loadings_") else model.x_loadings_
    a, b = comps
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P[:, a], P[:, b], s=30, color="seagreen", edgecolor="k", linewidth=0.4)
    names = var_names if var_names is not None else range(P.shape[0])
    for i, name in enumerate(names):
        ax.annotate(str(name), (P[i, a], P[i, b]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"p[{a + 1}]")
    ax.set_ylabel(f"p[{b + 1}]")
    ax.set_title("Loading plot")
    return ax


def control_chart(values, limit, ylabel="statistic", labels=None, ax=None):
    """Generic monitoring chart: values as a run sequence with a control limit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(values))
    above = values > limit
    ax.plot(x, values, "-o", ms=4, color="steelblue", lw=1)
    ax.scatter(x[above], np.asarray(values)[above], color="crimson", zorder=5,
               s=40, label="alarm")
    ax.axhline(limit, color="crimson", ls="--", lw=1.2, label="control limit")
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("observation")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{ylabel} control chart")
    return ax


def contribution_plot(contrib, var_names=None, ax=None, title="Contributions"):
    """Bar chart of per-variable contributions for a single observation."""
    contrib = np.asarray(contrib).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(contrib))
    colors = ["crimson" if c >= 0 else "steelblue" for c in contrib]
    ax.bar(x, contrib, color=colors, edgecolor="k", linewidth=0.3)
    names = var_names if var_names is not None else x
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("contribution")
    ax.set_title(title)
    return ax


def scree_plot(model, ax=None):
    """Explained-variance-per-component bar chart with cumulative overlay."""
    if hasattr(model, "explained_variance_ratio_"):
        evr = model.explained_variance_ratio_
        label = "R2X"
    else:
        evr = model.r2x_
        label = "R2X"
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    comps = np.arange(1, len(evr) + 1)
    ax.bar(comps, evr, color="slateblue", edgecolor="k", linewidth=0.3, label=label)
    ax.plot(comps, np.cumsum(evr), "-o", color="darkorange", label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance")
    ax.set_xticks(comps)
    ax.legend(fontsize=8)
    ax.set_title("Scree / explained variance")
    return ax
