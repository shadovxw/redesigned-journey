# dashboard.py — minimal + per-stage evaluation before cascade
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)

st.set_page_config(layout="wide", page_title="Pneumonia - Per-stage & Cascade Metrics")

# ----------------------------
# Paths (edit if needed)
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

# ----------------------------
# Helpers / caching
# ----------------------------
@st.cache_resource
def load_model_tf(path: Path):
    if not path.exists():
        return None
    with tf.device('/CPU:0'):
        return tf.keras.models.load_model(str(path), compile=False)

@st.cache_data
def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def compute_metrics(y_true, y_pred, probs=None):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "auc": float(roc_auc_score(y_true, probs)) if (probs is not None and len(np.unique(y_true))>1) else None
    }

def plot_confusion(cm, title="Confusion"):
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Pred Normal", "Pred Pneumonia"],
        y=["True Normal", "True Pneumonia"],
        text=z, texttemplate="%{text}"
    ))
    fig.update_layout(title=title, height=320, margin=dict(t=30))
    return fig

def plot_roc(y_true, probs, title):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="random"))
    fig.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360)
    return fig

def plot_pr(y_true, probs, title):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = auc(rec, prec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP≈{ap:.3f}"))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision", height=360)
    return fig

# ----------------------------
# Load artifacts and CSVs
# ----------------------------
st.title("Pneumonia Detection — Per-stage & Cascade Metrics")

m1 = load_model_tf(STAGE1_MODEL_PATH)
m2 = load_model_tf(STAGE2_MODEL_PATH)
meta_scaler = load_pickle(META_SCALER_PATH)
meta_model = load_pickle(META_MODEL_PATH)
meta_thresh = load_json(META_THRESHOLD_PATH)

per_val = load_csv(PER_SAMPLE_VAL)
per_test = load_csv(PER_SAMPLE_TEST)

# Sidebar: choose dataset & method & thresholds (per-model and cascade)
st.sidebar.header("Options")
dataset_choice = st.sidebar.selectbox("Dataset", ["test", "val"] if per_test is not None and per_val is not None else (["val"] if per_val is not None else ["test"]))
method = st.sidebar.selectbox("Decision method", ["META (if available)", "AND", "OR"])
# per-model thresholds (for single-model evaluation)
t_model1 = st.sidebar.slider("Stage-1 threshold (single model)", 0.0, 1.0, 0.5, 0.01)
t_model2 = st.sidebar.slider("Stage-2 threshold (single model)", 0.0, 1.0, 0.5, 0.01)
# cascade thresholds
t1 = st.sidebar.slider("Cascade T1 (stage1)", 0.0, 1.0, 0.05, 0.01)
t2 = st.sidebar.slider("Cascade T2 (stage2)", 0.0, 1.0, 0.4, 0.01)

# pick dataframe
df = per_test.copy() if dataset_choice == "test" else per_val.copy()
if df is None:
    st.error("Selected dataset CSV not found. Run cascade_eval to generate per-sample CSVs.")
    st.stop()

# ensure numeric
for c in ["prob_stage1","prob_stage2","meta_prob"]:
    if c in df.columns:
        df[c] = df[c].astype(float)

y_true = df["true_label"].values if "true_label" in df.columns else None
p1 = df["prob_stage1"].values
p2 = df["prob_stage2"].values

# ----------------------------
# Single-model evaluation panels (before cascade)
# ----------------------------
st.header("Single-stage model evaluation (before cascading)")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Stage-1 (Screener) — single-model")
    st.write(f"Threshold (single-model): {t_model1:.2f}")
    # predictions and metrics for single-stage evaluator
    p1_pred = (p1 >= t_model1).astype(int)
    if y_true is not None:
        m1 = compute_metrics(y_true, p1_pred, probs=p1)
        st.metric("Precision", f"{m1['precision']:.3f}")
        st.metric("Recall", f"{m1['recall']:.3f}")
        st.metric("F1", f"{m1['f1']:.3f}")
        st.plotly_chart(plot_confusion(m1["confusion_matrix"], title="Stage-1 Confusion Matrix"), use_container_width=True)
        st.plotly_chart(plot_roc(y_true, p1, "ROC - Stage 1 (screener)"), use_container_width=True)
        st.plotly_chart(plot_pr(y_true, p1, "PR - Stage 1 (screener)"), use_container_width=True)
    else:
        st.info("No ground-truth labels; showing probability histogram.")
    st.plotly_chart(px.histogram(df, x="prob_stage1", nbins=40, title="Stage-1 probability distribution"), use_container_width=True)

with col_b:
    st.subheader("Stage-2 (Refiner) — single-model")
    st.write(f"Threshold (single-model): {t_model2:.2f}")
    p2_pred = (p2 >= t_model2).astype(int)
    if y_true is not None:
        m2 = compute_metrics(y_true, p2_pred, probs=p2)
        st.metric("Precision", f"{m2['precision']:.3f}")
        st.metric("Recall", f"{m2['recall']:.3f}")
        st.metric("F1", f"{m2['f1']:.3f}")
        st.plotly_chart(plot_confusion(m2["confusion_matrix"], title="Stage-2 Confusion Matrix"), use_container_width=True)
        st.plotly_chart(plot_roc(y_true, p2, "ROC - Stage 2 (refiner)"), use_container_width=True)
        st.plotly_chart(plot_pr(y_true, p2, "PR - Stage 2 (refiner)"), use_container_width=True)
    else:
        st.info("No ground-truth labels; showing probability histogram.")
    st.plotly_chart(px.histogram(df, x="prob_stage2", nbins=40, title="Stage-2 probability distribution"), use_container_width=True)

# ----------------------------
# Cascade evaluation (after per-stage)
# ----------------------------
st.header("Cascade evaluation (AND / OR / META)")

meta_probs = None
if method == "META (if available)":
    if meta_model is not None and meta_scaler is not None and meta_thresh is not None:
        X = np.vstack([p1,p2]).T
        Xs = meta_scaler.transform(X)
        meta_probs = meta_model.predict_proba(Xs)[:,1]
        final = (meta_probs >= meta_thresh["threshold"]).astype(int)
        method_name = f"META (th={meta_thresh['threshold']:.3f})"
    else:
        st.sidebar.warning("Meta artifacts missing — using AND fallback")
        method = "AND"
        passed = p1 >= t1
        final = np.zeros_like(p1, dtype=int)
        final[passed] = (p2[passed] >= t2).astype(int)
        method_name = f"AND (T1={t1},T2={t2})"
elif method == "AND":
    passed = p1 >= t1
    final = np.zeros_like(p1, dtype=int)
    final[passed] = (p2[passed] >= t2).astype(int)
    method_name = f"AND (T1={t1},T2={t2})"
else:  # OR
    final = ((p1 >= t1) | (p2 >= t2)).astype(int)
    method_name = f"OR (T1={t1},T2={t2})"

if y_true is None:
    st.info("No ground-truth labels present in CSV; cascade shows predictions only.")
else:
    probs_for_auc = meta_probs if meta_probs is not None else (p2 if method_name.startswith("AND") else np.maximum(p1,p2))
    metrics = compute_metrics(y_true, final, probs=probs_for_auc)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1", f"{metrics['f1']:.3f}")
    col4.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.subheader("Cascade Confusion Matrix")
    st.plotly_chart(plot_confusion(metrics["confusion_matrix"], title="Cascade Confusion Matrix"), use_container_width=True)

# ----------------------------
# Per-sample predictions (small)
# ----------------------------
st.header("Per-sample predictions (first 200 rows)")
df_display = df.copy()
df_display["stage1_pred_single"] = (p1 >= t_model1).astype(int)
df_display["stage2_pred_single"] = (p2 >= t_model2).astype(int)
df_display["final_pred"] = final
st.dataframe(df_display.head(200))

# download small csv
csv_bytes = df_display.head(200).to_csv(index=False).encode("utf-8")
st.download_button("Download shown rows (CSV)", csv_bytes, file_name=f"per_sample_{dataset_choice}_{method_name}.csv", mime="text/csv")
