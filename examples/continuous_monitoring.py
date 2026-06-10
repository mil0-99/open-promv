"""End-to-end demo: build a PCA monitoring model, detect a fault, diagnose it.

Run with:  python examples/continuous_monitoring.py
Produces `continuous_monitoring.png` with the standard MSPC chart panel.
"""
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from promv import PCA, datasets
from promv import viz

# 1. Generate a continuous process with a step fault from sample 200 on
X, is_fault = datasets.make_continuous_process(
    n_samples=300, n_variables=12, n_latent=3,
    fault_start=200, fault_vars=[0, 1, 2], fault_magnitude=6.0, seed=42,
)
var_names = [f"v{i + 1}" for i in range(X.shape[1])]

# 2. Train the model on normal-operating-condition (NOC) data only
X_noc = X[:150]
model = PCA(n_components=3, alpha=0.01).fit(X_noc)
print("Explained variance per component:",
      np.round(model.explained_variance_ratio_, 3))

# 3. Monitor the whole run
res = model.monitor(X)
det = (res.spe[200:] > model.spe_limit_).mean()
print(f"SPE detection rate after fault onset: {det:.0%}")

# 4. Diagnose the fault: contributions of the first clearly-faulty sample
fault_sample = 210
contrib = res.spe_contrib[fault_sample]
top = np.argsort(contrib)[::-1][:3]
print("Top contributing variables at sample 210:",
      [var_names[i] for i in top])

# 5. Build the chart panel
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
viz.score_plot(model, comps=(0, 1), ax=axes[0, 0])
viz.control_chart(res.t2, model.t2_limit_, ylabel="Hotelling T^2", ax=axes[0, 1])
viz.control_chart(res.spe, model.spe_limit_, ylabel="SPE (Q)", ax=axes[1, 0])
viz.contribution_plot(contrib, var_names=var_names, ax=axes[1, 1],
                      title=f"SPE contributions @ sample {fault_sample}")
fig.suptitle("Continuous process monitoring (open-promv)", fontsize=14)
fig.tight_layout()
fig.savefig("continuous_monitoring.png", dpi=130)
print("Saved continuous_monitoring.png")
