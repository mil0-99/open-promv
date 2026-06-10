# open-promv

**Open-source multivariate statistical process control (MSPC) for Python.**

`open-promv` reimplements the core analytics behind commercial process-analytics
tools such as **Aspen ProMV**: latent-variable models (PCA / PLS) that compress
hundreds of correlated process tags into a handful of interpretable components,
online-style monitoring with **Hotelling's T²** and **squared prediction error
(SPE / Q)**, **contribution plots** for fault diagnosis, and **multiway batch**
analysis. It is built on nothing heavier than NumPy and SciPy.

> This is an independent, clean-room implementation of well-published
> chemometrics methods. It is not affiliated with or derived from AspenTech, and
> "Aspen ProMV" is a trademark of Aspen Technology, Inc., used here only to
> describe the class of functionality.

## Why

Process industries record thousands of correlated sensor readings. Looking at
tags one at a time hides the real, multivariate structure. Projection methods
find the few underlying directions of variation, let you monitor a process as a
single picture, and — when something drifts — tell you *which variables* are
responsible. That is the workflow this library provides.

## Features

| Capability | open-promv | ProMV-style equivalent |
|---|---|---|
| Centering / unit-variance autoscaling | `Scaler` | data preprocessing |
| PCA via NIPALS (tolerates missing data) | `PCA` | latent-variable model |
| PLS / PLS2 regression + VIP | `PLS` | quality prediction model |
| Hotelling's T² with F-based limit | `PCA.monitor` | T² control chart |
| SPE / Q with χ²-moment limit | `PCA.monitor` | SPE control chart |
| T² and SPE variable contributions | `MonitoringResult` | contribution plots |
| Multiway batch unfolding (MPCA) | `core.batch` | batch monitoring |
| Score / loading / control / contribution charts | `promv.viz` | interactive plots |

## Install

```bash
pip install -e .          # core (numpy, scipy)
pip install -e ".[viz]"   # add matplotlib for plotting
pip install -e ".[dev]"   # add pytest for the test suite
```

Requires Python ≥ 3.9.

## Quick start

### Continuous process monitoring

```python
import numpy as np
from promv import PCA, datasets

# normal-operating-condition data to train on, then a run with a fault
X, is_fault = datasets.make_continuous_process(
    n_samples=300, n_variables=12, fault_start=200,
    fault_vars=[0, 1, 2], fault_magnitude=6.0,
)

model = PCA(n_components=3, alpha=0.01).fit(X[:150])   # fit on NOC data
res = model.monitor(X)                                  # T², SPE, contributions

alarms = (res.spe > model.spe_limit_) | (res.t2 > model.t2_limit_)
print("samples in alarm:", np.where(alarms)[0])

# diagnose: which variables drive the SPE alarm at a faulty sample?
contrib = res.spe_contrib[210]
print("top variables:", np.argsort(contrib)[::-1][:3])
```

### Quality prediction with PLS

```python
from promv import PLS

pls = PLS(n_components=4).fit(X_process, Y_quality)
y_hat = pls.predict(X_new)
important = pls.vip()        # VIP > 1 => influential process variable
```

### Batch processes (Multiway PCA)

```python
from promv import PCA, datasets
from promv.core import batch

X3d, is_bad = datasets.make_batch_process()      # (n_batches, n_vars, n_time)
X2d = batch.unfold_batchwise(X3d)                # one row per batch
mpca = PCA(n_components=3).fit(X2d[~is_bad])      # model the good batches
res = mpca.monitor(X2d)                           # flag abnormal batches
```

## Examples

Runnable scripts in [`examples/`](examples/) generate the standard MSPC chart
panel (score plot with Hotelling ellipse, T² and SPE control charts, and a
contribution bar chart):

```bash
python examples/continuous_monitoring.py   # -> continuous_monitoring.png
python examples/batch_monitoring.py        # -> batch_monitoring.png
```

## Method notes

- **NIPALS PCA/PLS** extract components iteratively and deflate the residual,
  which makes missing-data handling natural (sums run only over observed cells).
- **T² limit** uses the F-distribution form
  `T²_lim = A(n−1)(n+1) / (n(n−A)) · F_{α}(A, n−A)`.
- **SPE limit** uses the Box / Nomikos–MacGregor χ² moment match:
  `SPE ≈ g·χ²(h)` with `g = var/(2·mean)` and `h = 2·mean²/var` over the
  training residuals.
- **Batch unfolding** follows Nomikos & MacGregor (1994): the three-way cube is
  unfolded batch-wise so each completed batch is a single multivariate
  observation.

## Testing

```bash
pytest -q
```

## References

- J.E. Jackson, *A User's Guide to Principal Components*, Wiley, 1991.
- P. Nomikos & J.F. MacGregor, "Monitoring batch processes using multiway
  principal component analysis", *AIChE Journal*, 40(8), 1994.
- S. Wold, M. Sjöström, L. Eriksson, "PLS-regression: a basic tool of
  chemometrics", *Chemometrics and Intelligent Lab Systems*, 58, 2001.

## License

MIT — see [LICENSE](LICENSE).
