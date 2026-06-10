"""open-promv: open-source multivariate statistical process control.

A NumPy/SciPy implementation of the core analytics behind tools such as
Aspen ProMV -- PCA and PLS latent-variable models, Hotelling's T^2 and SPE
process monitoring with confidence limits, contribution-based fault diagnosis,
and multiway batch unfolding.
"""
from .core import Scaler, PCA, PLS, MonitoringResult, batch
from . import datasets

__version__ = "0.1.0"

__all__ = [
    "Scaler",
    "PCA",
    "PLS",
    "MonitoringResult",
    "batch",
    "datasets",
    "__version__",
]
