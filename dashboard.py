# dashboard.py
import os
from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)
import pickle
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Pneumonia Detection Dashboard")

# ----------------------------
# Parameters / default paths
# ----------------------------
OUT = Path("output")
EVAL_DIR = OUT / "cascade_eval"
STAGE1_MODEL_PATH = OUT / "stage1_model.keras"
STAGE2_MODEL_PATH = OUT / "stage2_model.keras"
META_SCALER_PATH = EVAL_DIR / "meta_scaler.pkl"
META_MODEL_PATH = EVAL_DIR / "meta_model.pkl"
META_THRESHOLD_PATH = EVAL_DIR / "meta_threshold.json"

PER_SAMPLE_VAL = EVAL_DIR / "per_sample_val.csv"
PER_SAMPLE_TEST = EVAL_DIR / "per_sample_test.csv"
PER_SAMPLE_VAL_META = EVAL_DIR / "per_sample_val_with_meta.csv"  # optional
CHOSEN_JSON = EVAL_DIR / "chosen_thresholds_and_metrics.json"
GRID_AND = EVAL_DIR / "grid_and_val.json"
GRID_OR = EVAL_DIR / "grid_or_val.json"

# ----------------------------
# Utility functions
# ----------------------------
@st.cache_resource
def load_model_tf(path):
    if not path.exists():
        return None
    with tf.device('/CPU:0'):
        return tf.keras.models.load_model(str(path), compile=False)

@st.cache_data
def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

@st.cache_data
def load_pickle(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def compute_metrics_from_preds(y_true, y_pred, probs=None):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "auc": float(roc_auc_score(y_true, probs)) if (probs is not None and len(np.unique(y_true))>1) else None
    }

def plot_confusion_matrix(cm, labels=("Neg","Pos"), title="Confusion matrix"):
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        hoverongaps=False,
        text=z, texttemplate="%{text}"
    ))
    fig.update_layout(title=title, xaxis_side="top")
    return fig

def plot_roc(y_true, probs, label="model"):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash='dash'), name="random"))
    fig.update_layout(title=f"ROC Curve - {label}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def plot_pr(y_true, probs, label="model"):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = auc(rec, prec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{label} (AP~{ap:.3f})"))
    fig.update_layout(title=f"Precision-Recall Curve - {label}", xaxis_title="Recall", yaxis_title="Precision")
    return fig

# ----------------------------
# Load artifacts
# ----------------------------
st.title("Pneumonia Detection — Evaluation Dashboard")

col_check1, col_check2 = st.columns(2)
with col_check1:
    st.subheader("Models")
    st.write("Stage-1 model path:", STAGE1_MODEL_PATH)
    st.write("Stage-2 model path:", STAGE2_MODEL_PATH)
    m1 = load_model_tf(STAGE1_MODEL_PATH)
    m2 = load_model_tf(STAGE2_MODEL_PATH)
    st.write("Stage-1 loaded:", m1 is not None)
    st.write("Stage-2 loaded:", m2 is not None)

with col_check2:
    st.subheader("Meta artifacts & CSVs")
    st.write("Meta scaler:", META_SCALER_PATH.exists())
    st.write("Meta model:", META_MODEL_PATH.exists())
    st.write("Meta threshold:", META_THRESHOLD_PATH.exists())
    st.write("per_sample_val:", PER_SAMPLE_VAL.exists())
    st.write("per_sample_test:", PER_SAMPLE_TEST.exists())
    st.write("chosen_thresholds:", CHOSEN_JSON.exists())

# Load CSVs and chosen thresholds if available
per_val = load_csv(PER_SAMPLE_VAL)
per_test = load_csv(PER_SAMPLE_TEST)
per_val_meta = load_csv(PER_SAMPLE_VAL_META)
chosen = load_json(CHOSEN_JSON) or {}
meta_scaler = load_pickle(META_SCALER_PATH)
meta_model = load_pickle(META_MODEL_PATH)
meta_thresh = load_json(META_THRESHOLD_PATH)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Display / Threshold controls")
dataset_choice = st.sidebar.selectbox("Dataset to visualize", options=["val","test"], index=1 if per_test is not None else 0)
rule_choice = st.sidebar.selectbox("Rule to use for cascade", options=["AND (seq)", "OR", "META (logistic)"], index=2 if meta_model is not None else 0)
# provide defaults from chosen json if available
default_t1 = 0.05
default_t2 = 0.4
if chosen:
    try:
        default_t1 = chosen.get("best_and_val", {}).get("T1", default_t1)
        default_t2 = chosen.get("best_and_val", {}).get("T2", default_t2)
    except Exception:
        pass

t1 = st.sidebar.slider("T1 (stage1 threshold)", 0.0, 1.0, float(default_t1), 0.01)
t2 = st.sidebar.slider("T2 (stage2 threshold)", 0.0, 1.0, float(default_t2), 0.01)
recall_target_display = st.sidebar.number_input("Recall target (for threshold tuning)", value=0.99, min_value=0.5, max_value=1.0, step=0.01)

# ----------------------------
# Select dataset frame
# ----------------------------
if dataset_choice == "val":
    if per_val is None:
        st.error("Validation per-sample CSV not found. Run cascade_eval first.")
        st.stop()
    df = per_val.copy()
else:
    if per_test is None:
        st.error("Test per-sample CSV not found. Run cascade_eval first.")
        st.stop()
    df = per_test.copy()

# Ensure probabilities exist as floats
for col in ["prob_stage1","prob_stage2","meta_prob"]:
    if col in df.columns:
        df[col] = df[col].astype(float)

# ----------------------------
# Compute predictions according to selected rule
# ----------------------------
y_true = df["true_label"].values if "true_label" in df.columns else None
p1 = df["prob_stage1"].values
p2 = df["prob_stage2"].values

if rule_choice.startswith("AND"):
    passed = p1 >= t1
    final = np.zeros_like(p1, dtype=int)
    final[passed] = (p2[passed] >= t2).astype(int)
    method_name = f"AND (T1={t1},T2={t2})"
elif rule_choice.startswith("OR"):
    final = ((p1 >= t1) | (p2 >= t2)).astype(int)
    method_name = f"OR (T1={t1},T2={t2})"
else:  # meta
    if meta_model is None or meta_scaler is None or meta_thresh is None:
        st.warning("Meta artifacts missing; falling back to AND rule.")
        passed = p1 >= t1
        final = np.zeros_like(p1, dtype=int)
        final[passed] = (p2[passed] >= t2).astype(int)
        method_name = f"AND (fallback, T1={t1},T2={t2})"
    else:
        X = np.vstack([p1,p2]).T
        Xs = meta_scaler.transform(X)
        meta_prob = meta_model.predict_proba(Xs)[:,1]
        final = (meta_prob >= meta_thresh["threshold"]).astype(int)
        method_name = f"META (th={meta_thresh['threshold']:.3f})"

# show summary metrics for selected dataset and rule
st.header(f"Summary — dataset: {dataset_choice} — rule: {method_name}")
if y_true is None:
    st.info("No ground truth labels available in CSV; only probabilities are shown.")
else:
    metrics = compute_metrics_from_preds(y_true, final, probs=(meta_prob if rule_choice.startswith("META") and 'meta_prob' in locals() else (p2 if rule_choice.startswith("AND") else np.maximum(p1,p2))))
    st.metric("Precision", f"{metrics['precision']:.3f}", delta=None)
    st.metric("Recall", f"{metrics['recall']:.3f}", delta=None)
    st.metric("F1", f"{metrics['f1']:.3f}", delta=None)
    st.write("Confusion matrix:")
    st.plotly_chart(plot_confusion_matrix(metrics["confusion_matrix"], title="Confusion Matrix"), use_container_width=True)

# ----------------------------
# Panel: Comparison of individual models
# ----------------------------
st.subheader("Model-level metrics & curves")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Stage-1 (screener)**")
    # if per-sample has stage1 preds, show simple threshold=0.5 metrics
    p1_default_pred = (p1 >= 0.5).astype(int)
    if y_true is not None:
        m1 = compute_metrics_from_preds(y_true, p1_default_pred, probs=p1)
        st.write(m1)
        st.plotly_chart(plot_roc(y_true, p1, label="Stage-1"), use_container_width=True)
        st.plotly_chart(plot_pr(y_true, p1, label="Stage-1"), use_container_width=True)
    else:
        st.write("Stage-1 probs summary")
    st.plotly_chart(px.histogram(df, x="prob_stage1", nbins=40, title="Stage-1 probability distribution"), use_container_width=True)

with col2:
    st.markdown("**Stage-2 (refiner)**")
    p2_default_pred = (p2 >= 0.5).astype(int)
    if y_true is not None:
        m2 = compute_metrics_from_preds(y_true, p2_default_pred, probs=p2)
        st.write(m2)
        st.plotly_chart(plot_roc(y_true, p2, label="Stage-2"), use_container_width=True)
        st.plotly_chart(plot_pr(y_true, p2, label="Stage-2"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="prob_stage2", nbins=40, title="Stage-2 probability distribution"), use_container_width=True)

# if meta present, show meta prob distribution
if "meta_prob" in df.columns or (meta_model is not None and meta_scaler is not None):
    st.subheader("Meta classifier (if available)")
    if "meta_prob" in df.columns:
        st.plotly_chart(px.histogram(df, x="meta_prob", nbins=40, title="Meta probability distribution"), use_container_width=True)
    elif meta_model is not None:
        # compute meta probs for currently displayed df
        X = np.vstack([p1,p2]).T
        meta_probs_display = meta_model.predict_proba(meta_scaler.transform(X))[:,1]
        st.plotly_chart(px.histogram(meta_probs_display, nbins=40, title="Meta probability distribution (computed)"), use_container_width=True)
    # show meta coefficients if available
    try:
        if meta_model is not None:
            coefs = meta_model.coef_.ravel().tolist()
            intercept = meta_model.intercept_.tolist()
            st.write("Meta logistic coefficients:", coefs, "intercept:", intercept)
            if meta_thresh is not None:
                st.write("Chosen meta threshold:", meta_thresh["threshold"])
    except Exception:
        pass

# ----------------------------
# Per-sample table and download
# ----------------------------
st.subheader("Per-sample table (filtered by rule)")
df_display = df.copy()
df_display["final_pred"] = final
# show first rows and allow search / download
st.write(f"Showing {len(df_display)} rows (dataset={dataset_choice})")
st.dataframe(df_display.head(200))

# Download filtered CSV
csv_buf = df_display.to_csv(index=False).encode('utf-8')
st.download_button("Download per-sample CSV (current view)", data=csv_buf, file_name=f"per_sample_{dataset_choice}_{method_name}.csv", mime="text/csv")

# ----------------------------
# Extra: show full grid summary if available
# ----------------------------
with st.expander("Grid search results (preview)"):
    if (EVAL_DIR / "grid_and_val.json").exists():
        ga = load_json(EVAL_DIR / "grid_and_val.json")
        st.write("AND grid sample (first 10):")
        st.write(ga[:10])
    if (EVAL_DIR / "grid_or_val.json").exists():
        gof = load_json(EVAL_DIR / "grid_or_val.json")
        st.write("OR grid sample (first 10):")
        st.write(gof[:10])

# ----------------------------
# Save chosen config button
# ----------------------------
if st.button("Save current thresholds & method as chosen"):
    out = {
        "method": method_name,
        "T1": float(t1),
        "T2": float(t2),
    }
    (EVAL_DIR / "manual_chosen.json").write_text(json.dumps(out, indent=2))
    st.success(f"Wrote manual_chosen.json with {out}")

st.markdown("---")
st.caption("Dashboard shows model-level metrics, cascade evaluation, and meta-classifier info. Use 'META' when available (recommended).")
