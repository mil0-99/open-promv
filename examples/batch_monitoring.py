"""Batch process monitoring demo using Multiway PCA (MPCA).

Run with:  python examples/batch_monitoring.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from promv import PCA, datasets
from promv import viz
from promv.core import batch

# 1. Three-way batch data: (n_batches, n_variables, n_time)
X3d, is_bad = datasets.make_batch_process(
    n_batches=50, n_variables=5, n_time=60, n_bad=6, seed=7,
)
print(f"Batch cube shape: {X3d.shape}  |  bad batches: {np.where(is_bad)[0]}")

# 2. Unfold batch-wise so each completed batch becomes one observation
X_unfolded = batch.unfold_batchwise(X3d)

# 3. Train MPCA on the good batches only
good = ~is_bad
mpca = PCA(n_components=3, alpha=0.05).fit(X_unfolded[good])

# 4. Monitor every batch
res = mpca.monitor(X_unfolded)
flagged = (res.spe > mpca.spe_limit_) | (res.t2 > mpca.t2_limit_)
print("Flagged batches:", np.where(flagged)[0])

# 5. Plot batch-level SPE and a score plot coloured by good/bad
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
viz.control_chart(res.spe, mpca.spe_limit_, ylabel="batch SPE",
                  labels=[f"b{i}" for i in range(len(res.spe))], ax=axes[0])
ax = axes[1]
T = mpca.scores_
ax.scatter(res.scores[good, 0], res.scores[good, 1], label="good", s=35,
           edgecolor="k", linewidth=0.4)
ax.scatter(res.scores[is_bad, 0], res.scores[is_bad, 1], label="bad",
           color="crimson", s=55, marker="X", edgecolor="k")
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set_xlabel("t[1]")
ax.set_ylabel("t[2]")
ax.set_title("Batch score plot")
ax.legend()
fig.suptitle("Batch process monitoring via Multiway PCA (open-promv)")
fig.tight_layout()
fig.savefig("batch_monitoring.png", dpi=130)
print("Saved batch_monitoring.png")
