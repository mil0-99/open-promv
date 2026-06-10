"""open-promv dashboard.

An upload-a-CSV-and-go web app for multivariate process monitoring, built on
the promv library. Run with:

    streamlit run app/dashboard.py

Two modes:
  * PCA monitoring  -- fit a model on normal data, flag abnormal observations,
    diagnose them with contribution plots.
  * PLS prediction  -- relate process variables (X) to quality variables (Y),
    predict, and rank variable importance.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# make the app runnable both as an installed package and straight from a clone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from promv import PCA, PLS, datasets          # noqa: E402
from promv.core import batch as batch_utils    # noqa: E402
import charts                                   # noqa: E402

st.set_page_config(page_title="open-promv", page_icon="📈", layout="wide")


# --------------------------------------------------------------------- data
def _example_dataframe() -> pd.DataFrame:
    """A continuous process with a fault from sample 200 -- for instant trial."""
    X, fault = datasets.make_continuous_process(
        n_samples=300, n_variables=10, fault_start=200,
        fault_vars=[0, 1, 2], fault_magnitude=6.0, seed=42,
    )
    df = pd.DataFrame(X, columns=[f"sensor_{i + 1}" for i in range(X.shape[1])])
    df.insert(0, "sample_id", [f"obs_{i:03d}" for i in range(len(df))])
    return df


def _load_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


# --------------------------------------------------------------------- sidebar
st.sidebar.title("📈 open-promv")
st.sidebar.caption("Multivariate process monitoring")

mode = st.sidebar.radio(
    "Analysis", ["PCA monitoring", "PLS prediction"],
    help="PCA finds and monitors process structure. PLS predicts quality from process data.",
)

source = st.sidebar.radio("Data source", ["Upload CSV", "Use example data"])
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
    df = _load_csv(uploaded) if uploaded is not None else None
else:
    df = _example_dataframe()

if df is None:
    st.title("open-promv dashboard")
    st.info("Upload a CSV in the sidebar, or pick **Use example data** to try it now.")
    st.markdown(
        "**Expected format:** one row per observation, one column per variable. "
        "An optional text column can be used as a label/ID. Missing values are "
        "allowed — the PCA handles them."
    )
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
text_cols = [c for c in df.columns if c not in numeric_cols]

label_col = st.sidebar.selectbox(
    "Label / ID column (optional)", ["(row number)"] + text_cols,
)
labels = (df[label_col].astype(str).tolist()
          if label_col != "(row number)" else
          [str(i) for i in range(len(df))])

alpha_pct = st.sidebar.slider("Confidence level (%)", 90, 99, 95)
alpha = 1 - alpha_pct / 100


# ==================================================================== PCA mode
if mode == "PCA monitoring":
    st.title("PCA process monitoring")

    x_cols = st.multiselect(
        "Process variables (X)", numeric_cols, default=numeric_cols,
        help="The sensor / measurement columns to model.",
    )
    if len(x_cols) < 2:
        st.warning("Select at least two process variables.")
        st.stop()

    X = df[x_cols].to_numpy(dtype=float)
    n = len(X)

    c1, c2 = st.columns(2)
    with c1:
        train_mode = st.selectbox(
            "Training (normal) data",
            ["First N rows", "All rows"],
            help="Build the model on data representing normal operation.",
        )
    with c2:
        if train_mode == "First N rows":
            n_train = st.number_input("N (normal rows)", 5, n, min(150, n))
            train_idx = np.arange(int(n_train))
        else:
            train_idx = np.arange(n)

    max_comp = min(len(x_cols), len(train_idx)) - 1
    n_comp = st.slider("Number of components", 1, max(2, max_comp), min(3, max_comp))

    try:
        model = PCA(n_components=int(n_comp), alpha=alpha).fit(X[train_idx])
    except Exception as exc:  # surfacing fit errors to the user
        st.error(f"Could not fit model: {exc}")
        st.stop()

    res = model.monitor(X)
    alarms = (res.t2 > model.t2_limit_) | (res.spe > model.spe_limit_)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", n)
    m2.metric("Variance explained", f"{model.explained_variance_ratio_.sum():.0%}")
    m3.metric("Alarms", int(alarms.sum()))
    m4.metric("Alarm rate", f"{alarms.mean():.1%}")

    tabs = st.tabs(["Monitoring", "Diagnosis", "Model", "Data"])

    with tabs[0]:
        cc = st.columns(2)
        with cc[0]:
            comp_pair = (0, 1) if n_comp >= 2 else (0, 0)
            st.plotly_chart(
                charts.score_plot(res.scores, comps=comp_pair, labels=labels,
                                  color_flags=alarms, alpha=alpha),
                width='stretch',
            )
        with cc[1]:
            st.plotly_chart(
                charts.control_chart(res.t2, model.t2_limit_, name="Hotelling T²",
                                     labels=labels),
                width='stretch',
            )
        st.plotly_chart(
            charts.control_chart(res.spe, model.spe_limit_, name="SPE (Q)",
                                 labels=labels),
            width='stretch',
        )

    with tabs[1]:
        st.markdown("Pick an observation to see which variables drive its deviation.")
        flagged = [i for i in range(n) if alarms[i]]
        default_ix = flagged[0] if flagged else 0
        sel = st.selectbox(
            "Observation", range(n), index=default_ix,
            format_func=lambda i: f"{labels[i]}"
            + ("  ⚠ alarm" if alarms[i] else ""),
        )
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                charts.contribution_bar(res.spe_contrib[sel], x_cols,
                                        title=f"SPE contributions — {labels[sel]}"),
                width='stretch',
            )
        with d2:
            st.plotly_chart(
                charts.contribution_bar(res.t2_contrib[sel], x_cols,
                                        title=f"T² contributions — {labels[sel]}"),
                width='stretch',
            )
        top = np.argsort(res.spe_contrib[sel])[::-1][:3]
        st.info("Top SPE contributors: "
                + ", ".join(f"**{x_cols[i]}**" for i in top))

    with tabs[2]:
        st.plotly_chart(
            charts.scree_bar(model.explained_variance_ratio_),
            width='stretch',
        )
        st.markdown(
            f"- Components: **{n_comp}**  \n"
            f"- T² limit ({alpha_pct}%): **{model.t2_limit_:.2f}**  \n"
            f"- SPE limit ({alpha_pct}%): **{model.spe_limit_:.3f}**  \n"
            f"- Trained on **{len(train_idx)}** rows"
        )

    with tabs[3]:
        out = df.copy()
        out["T2"] = res.t2
        out["SPE"] = res.spe
        out["alarm"] = alarms
        st.dataframe(out, width='stretch', height=420)
        st.download_button(
            "Download results (CSV)",
            out.to_csv(index=False).encode(),
            file_name="promv_monitoring_results.csv",
            mime="text/csv",
        )


# ==================================================================== PLS mode
else:
    st.title("PLS quality prediction")

    c1, c2 = st.columns(2)
    with c1:
        x_cols = st.multiselect("Process variables (X)", numeric_cols,
                                default=numeric_cols[:-1])
    with c2:
        y_cols = st.multiselect("Quality variables (Y)", numeric_cols,
                                default=numeric_cols[-1:])

    overlap = set(x_cols) & set(y_cols)
    if overlap:
        st.warning(f"These columns are in both X and Y: {', '.join(overlap)}")
    if len(x_cols) < 1 or len(y_cols) < 1:
        st.warning("Select at least one X and one Y variable.")
        st.stop()

    X = df[x_cols].to_numpy(dtype=float)
    Y = df[y_cols].to_numpy(dtype=float)
    n = len(X)

    test_frac = st.slider("Hold-out test fraction", 0.0, 0.5, 0.25, 0.05)
    n_test = int(n * test_frac)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    max_comp = min(len(x_cols), len(train_idx) - 1)
    n_comp = st.slider("Number of components", 1, max(2, max_comp), min(3, max_comp))

    try:
        pls = PLS(n_components=int(n_comp), alpha=alpha).fit(X[train_idx], Y[train_idx])
    except Exception as exc:
        st.error(f"Could not fit model: {exc}")
        st.stop()

    eval_idx = test_idx if n_test > 0 else train_idx
    pred = pls.predict(X[eval_idx])
    actual = Y[eval_idx]
    ss_res = np.sum((actual - pred) ** 2, axis=0)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)

    m1, m2, m3 = st.columns(3)
    m1.metric("Train rows", len(train_idx))
    m2.metric("Eval rows", len(eval_idx))
    m3.metric("Mean R² (eval)", f"{np.nanmean(r2):.3f}")

    tabs = st.tabs(["Predictions", "Variable importance", "Data"])

    with tabs[0]:
        for j, name in enumerate(y_cols):
            st.plotly_chart(
                charts.parity_plot(actual[:, j], pred[:, j], name=name),
                width='stretch',
            )
            st.caption(f"R² for {name}: {r2[j]:.3f}")

    with tabs[1]:
        st.plotly_chart(charts.vip_bar(pls.vip(), x_cols),
                        width='stretch')
        st.caption("Variables with VIP > 1 are the most influential for predicting Y.")

    with tabs[2]:
        pred_all = pls.predict(X)
        out = df.copy()
        for j, name in enumerate(y_cols):
            out[f"{name}_pred"] = pred_all[:, j]
        st.dataframe(out, width='stretch', height=420)
        st.download_button(
            "Download predictions (CSV)",
            out.to_csv(index=False).encode(),
            file_name="promv_predictions.csv",
            mime="text/csv",
        )

st.sidebar.divider()
st.sidebar.caption("open-promv · MIT licensed")
