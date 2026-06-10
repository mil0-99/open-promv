"""Correctness tests for the open-promv models."""
import numpy as np
import pytest

from promv import PCA, PLS, Scaler, datasets
from promv.core import batch


# ----------------------------------------------------------------- scaler
def test_scaler_autoscale_gives_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5, scale=3, size=(100, 4))
    Xs = Scaler().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0, ddof=1), 1, atol=1e-6)


def test_scaler_inverse_roundtrip():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    sc = Scaler().fit(X)
    assert np.allclose(sc.inverse_transform(sc.transform(X)), X)


# -------------------------------------------------------------------- pca
def test_pca_loadings_orthonormal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 6))
    pca = PCA(n_components=4).fit(X)
    gram = pca.loadings_.T @ pca.loadings_
    assert np.allclose(gram, np.eye(4), atol=1e-6)


def test_pca_matches_svd_subspace():
    # PCA scores should reconstruct the centered/scaled data closely
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 5))
    pca = PCA(n_components=5).fit(X)
    Xs = pca.scaler.transform(X)
    recon = pca.inverse_transform(pca.transform(X))
    assert np.allclose(recon, Xs, atol=1e-6)


def test_pca_explained_variance_sums_to_one_full_rank():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 4))
    pca = PCA(n_components=4).fit(X)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-6)


def test_pca_handles_missing_data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 6))
    X[::7, 0] = np.nan
    X[::11, 3] = np.nan
    pca = PCA(n_components=3).fit(X)  # must not raise
    assert pca.loadings_.shape == (6, 3)
    assert np.isfinite(pca.loadings_).all()


def test_pca_detects_fault_via_spe():
    X, _ = datasets.make_continuous_process(
        n_samples=300, fault_start=200, fault_vars=[0, 1, 2],
        fault_magnitude=6.0, seed=6,
    )
    pca = PCA(n_components=3).fit(X[:150])
    res = pca.monitor(X)
    spe_alarm_fault = (res.spe[200:] > pca.spe_limit_).mean()
    spe_alarm_noc = (res.spe[:150] > pca.spe_limit_).mean()
    assert spe_alarm_fault > 0.8       # most faulty samples flagged
    assert spe_alarm_noc < 0.15        # few false alarms on NOC data


def test_pca_contributions_sum_to_spe():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 5))
    pca = PCA(n_components=2).fit(X)
    res = pca.monitor(X)
    assert np.allclose(res.spe_contrib.sum(axis=1), res.spe, atol=1e-8)


# -------------------------------------------------------------------- pls
def test_pls_recovers_linear_relationship():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 6))
    B = rng.normal(size=(6, 2))
    Y = X @ B + 0.05 * rng.normal(size=(300, 2))
    pls = PLS(n_components=6).fit(X[:200], Y[:200])
    pred = pls.predict(X[200:])
    ss_res = np.sum((Y[200:] - pred) ** 2)
    ss_tot = np.sum((Y[200:] - Y[200:].mean(0)) ** 2)
    assert 1 - ss_res / ss_tot > 0.95


def test_pls_single_response():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.01 * rng.normal(size=100)
    pls = PLS(n_components=4).fit(X, y)
    pred = pls.predict(X)
    assert pred.shape == (100, 1)


def test_vip_flags_relevant_variable():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(300, 5))
    # only variable 0 drives Y
    y = 3 * X[:, 0] + 0.01 * rng.normal(size=300)
    pls = PLS(n_components=3).fit(X, y)
    vip = pls.vip()
    assert np.argmax(vip) == 0
    assert vip[0] > 1.0


# ------------------------------------------------------------------ batch
def test_unfold_refold_roundtrip():
    X3d, _ = datasets.make_batch_process(n_batches=10, n_variables=4, n_time=30, seed=11)
    Xbw = batch.unfold_batchwise(X3d)
    assert Xbw.shape == (10, 4 * 30)
    assert np.allclose(batch.refold_batchwise(Xbw, 4, 30), X3d)


def test_variablewise_unfold_shape():
    X3d, _ = datasets.make_batch_process(n_batches=8, n_variables=3, n_time=20, seed=12)
    Xvw = batch.unfold_variablewise(X3d)
    assert Xvw.shape == (8 * 20, 3)


def test_align_to_length():
    b = np.random.default_rng(13).normal(size=(3, 47))
    aligned = batch.align_to_length(b, 60)
    assert aligned.shape == (3, 60)
