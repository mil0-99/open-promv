"""Interactive Plotly charts for the open-promv dashboard.

Kept free of any Streamlit calls so the figure-building logic can be imported
and unit-tested on its own. Each function returns a ``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

# a small, calm palette
_BLUE = "#2b6cb0"
_RED = "#c5303a"
_GREEN = "#2f855a"
_GREY = "#94a3b8"


def control_chart(values, limit, name="statistic", labels=None):
    """Run chart of a monitoring statistic with a control limit; alarms in red."""
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values)) if labels is None else list(labels)
    above = values > limit

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines+markers", name=name,
        line=dict(color=_BLUE, width=1.4), marker=dict(size=5),
        hovertemplate="obs %{x}<br>" + name + " %{y:.3f}<extra></extra>",
    ))
    if above.any():
        fig.add_trace(go.Scatter(
            x=np.asarray(x)[above], y=values[above], mode="markers", name="alarm",
            marker=dict(color=_RED, size=9, line=dict(width=0.5, color="black")),
            hovertemplate="ALARM<br>obs %{x}<br>" + name + " %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=limit, line=dict(color=_RED, dash="dash", width=1.4),
                  annotation_text="control limit", annotation_position="top right")
    fig.update_layout(
        title=f"{name} control chart", xaxis_title="observation",
        yaxis_title=name, height=360, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return fig


def score_plot(scores, comps=(0, 1), labels=None, color_flags=None, alpha=0.05):
    """Scatter of two score vectors with the Hotelling T^2 confidence ellipse."""
    scores = np.asarray(scores, dtype=float)
    a, b = comps
    n = scores.shape[0]
    ta, tb = scores[:, a], scores[:, b]

    fig = go.Figure()
    if color_flags is None:
        fig.add_trace(go.Scatter(
            x=ta, y=tb, mode="markers", name="observations",
            marker=dict(size=8, color=_BLUE, line=dict(width=0.4, color="black")),
            text=labels,
            hovertemplate="%{text}<br>t[%d]=%%{x:.2f}<br>t[%d]=%%{y:.2f}<extra></extra>"
            % (a + 1, b + 1) if labels is not None else None,
        ))
    else:
        color_flags = np.asarray(color_flags, dtype=bool)
        for mask, col, nm in [(~color_flags, _BLUE, "in control"),
                              (color_flags, _RED, "alarm")]:
            if mask.any():
                txt = np.asarray(labels)[mask] if labels is not None else None
                fig.add_trace(go.Scatter(
                    x=ta[mask], y=tb[mask], mode="markers", name=nm,
                    marker=dict(size=9, color=col, line=dict(width=0.4, color="black")),
                    text=txt,
                ))

    # Hotelling ellipse for 2 components
    if n > 2:
        f = stats.f.ppf(1 - alpha, 2, n - 2)
        radius = np.sqrt(2 * (n - 1) * (n + 1) / (n * (n - 2)) * f)
        sa, sb = ta.std(ddof=1), tb.std(ddof=1)
        theta = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(
            x=ta.mean() + radius * sa * np.cos(theta),
            y=tb.mean() + radius * sb * np.sin(theta),
            mode="lines", name=f"{int((1 - alpha) * 100)}% limit",
            line=dict(color=_RED, dash="dash", width=1.3), hoverinfo="skip",
        ))

    fig.update_layout(
        title="Score plot", xaxis_title=f"t[{a + 1}]", yaxis_title=f"t[{b + 1}]",
        height=440, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    fig.add_hline(y=0, line=dict(color=_GREY, width=0.6))
    fig.add_vline(x=0, line=dict(color=_GREY, width=0.6))
    return fig


def contribution_bar(contrib, var_names=None, title="Contributions"):
    """Bar chart of per-variable contributions for one observation."""
    contrib = np.asarray(contrib, dtype=float).ravel()
    names = list(var_names) if var_names is not None else \
        [f"v{i + 1}" for i in range(len(contrib))]
    colors = [_RED if c >= 0 else _BLUE for c in contrib]
    fig = go.Figure(go.Bar(
        x=names, y=contrib, marker_color=colors,
        hovertemplate="%{x}<br>contribution %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title="variable", yaxis_title="contribution",
        height=360, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def scree_bar(explained_ratio):
    """Explained-variance bar chart with a cumulative line."""
    evr = np.asarray(explained_ratio, dtype=float)
    comps = [f"PC{i + 1}" for i in range(len(evr))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comps, y=evr, name="per component",
                         marker_color=_BLUE))
    fig.add_trace(go.Scatter(x=comps, y=np.cumsum(evr), name="cumulative",
                             mode="lines+markers", line=dict(color=_GREEN)))
    fig.update_layout(
        title="Explained variance", yaxis_title="fraction of variance",
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return fig


def parity_plot(y_true, y_pred, name="Y"):
    """Predicted-vs-actual scatter with the 45-degree line, for PLS."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="samples",
                             marker=dict(color=_BLUE, size=7,
                                         line=dict(width=0.4, color="black"))))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal",
                             line=dict(color=_RED, dash="dash")))
    fig.update_layout(
        title=f"Predicted vs actual: {name}", xaxis_title="actual",
        yaxis_title="predicted", height=400, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
    )
    return fig


def vip_bar(vip, var_names=None):
    """Variable Importance in Projection bar chart with the VIP=1 reference line."""
    vip = np.asarray(vip, dtype=float)
    names = list(var_names) if var_names is not None else \
        [f"v{i + 1}" for i in range(len(vip))]
    order = np.argsort(vip)[::-1]
    fig = go.Figure(go.Bar(
        x=[names[i] for i in order], y=vip[order],
        marker_color=[_GREEN if vip[i] >= 1 else _GREY for i in order],
        hovertemplate="%{x}<br>VIP %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line=dict(color=_RED, dash="dash"),
                  annotation_text="VIP = 1", annotation_position="top right")
    fig.update_layout(title="Variable importance (VIP)", yaxis_title="VIP",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig
