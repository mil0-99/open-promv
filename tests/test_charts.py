"""Smoke tests for the dashboard's Plotly chart helpers.

These ensure each helper returns a valid figure with data; skipped cleanly if
the optional `app` dependencies (plotly) are not installed.
"""
import os
import sys

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import charts  # noqa: E402


def test_control_chart_has_data_and_limit():
    fig = charts.control_chart(np.array([0.1, 0.2, 1.5, 0.3]), limit=1.0, name="T2")
    assert len(fig.data) >= 1
    # a horizontal control-limit line is added as a layout shape
    assert len(fig.layout.shapes) >= 1


def test_score_plot_includes_ellipse():
    scores = np.random.default_rng(0).normal(size=(50, 2))
    fig = charts.score_plot(scores, color_flags=np.zeros(50, dtype=bool))
    names = [tr.name for tr in fig.data]
    assert any("limit" in (n or "") for n in names)


def test_contribution_bar_lengths():
    fig = charts.contribution_bar(np.arange(5.0), ["a", "b", "c", "d", "e"])
    assert list(fig.data[0].x) == ["a", "b", "c", "d", "e"]
    assert len(fig.data[0].y) == 5


def test_scree_bar_has_cumulative_line():
    fig = charts.scree_bar(np.array([0.5, 0.3, 0.2]))
    assert len(fig.data) == 2  # bar + cumulative line


def test_parity_and_vip():
    rng = np.random.default_rng(1)
    fig = charts.parity_plot(rng.random(20), rng.random(20), "Y")
    assert len(fig.data) == 2  # points + ideal line
    vip_fig = charts.vip_bar(np.array([1.5, 0.4, 1.1]), ["x1", "x2", "x3"])
    assert len(vip_fig.data) == 1
