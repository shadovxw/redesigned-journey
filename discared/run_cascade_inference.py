# run_cascade_inference.py
import os
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve

# ---------------- USER CONFIG ----------------
OUTPUT_DIR = "output"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 123

# Paths to pretrained models (adjust if different)
STAGE1_MODEL_PATH = os.path.join(OUTPUT_DIR, "stage1_model.h5")      # your high-recall model
STAGE2_MODEL_PATH = os.path.join(OUTPUT_DIR, "stage2_model.keras")   # your high-precision model

# threshold selection goal (try to reach this on validation)
STAGE1_TARGET_RECALL = 0.995
# ----------------------------------------------

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)



DATA_DIR = "data"
print("Using DATA_DIR =", DATA_DIR)

# --- load datasets ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=True, seed=SEED
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)

print("Classes:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# --- load pretrained models (no training) ---
if not os.path.exists(STAGE1_MODEL_PATH):
    raise FileNotFoundError(f"Stage-1 model file not found: {STAGE1_MODEL_PATH}")
if not os.path.exists(STAGE2_MODEL_PATH):
    raise FileNotFoundError(f"Stage-2 model file not found: {STAGE2_MODEL_PATH}")

print("Loading Stage-1 model from", STAGE1_MODEL_PATH)
stage1_model = tf.keras.models.load_model(STAGE1_MODEL_PATH)
print("Loading Stage-2 model from", STAGE2_MODEL_PATH)
stage2_model = tf.keras.models.load_model(STAGE2_MODEL_PATH)

# --- helper: get probs and labels from a dataset ---
def dataset_probs_labels(model, ds):
    ys = []
    ps = []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0).ravel()
        ys.extend(yb.numpy().astype(int).tolist())
        ps.extend(p.tolist())
    return np.array(ys), np.array(ps)

# --- choose threshold on validation to try to reach target recall ---
print("Computing validation predictions for threshold selection...")
y_val, p_val = dataset_probs_labels(stage1_model, val_ds)

# baseline at 0.5 (just informative)
base_preds = (p_val >= 0.5).astype(int)
print("Baseline val recall@0.5 =", float(recall_score(y_val, base_preds)))

precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
chosen_threshold = None
found = []
for i, th in enumerate(np.append(thresholds, 1.0)):
    r = recalls[i]; p = precisions[i]
    t = thresholds[i] if i < len(thresholds) else 1.0
    if r >= STAGE1_TARGET_RECALL:
        found.append((t, p, r))
if found:
    best = max(found, key=lambda x: x[1])  # maximize precision among thresholds meeting target recall
    chosen_threshold = float(best[0])
    print(f"Found threshold achieving target recall on VAL: {chosen_threshold:.4f} (val precision={best[1]:.3f}, recall={best[2]:.3f})")
else:
    # fallback: pick threshold with maximum recall (and good precision among ties)
    best_recall = max(recalls)
    idxs = [i for i, rr in enumerate(recalls) if rr == best_recall]
    best_idx = max(idxs, key=lambda i: precisions[i])
    chosen_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    print(f"No threshold reached target recall; fallback chosen_threshold={chosen_threshold:.4f} (val recall={best_recall:.3f})")

# save chosen threshold
with open(os.path.join(OUTPUT_DIR, "chosen_threshold_stage1.json"), "w") as f:
    json.dump({"chosen_threshold": chosen_threshold}, f, indent=2)

# --- helper: collect stage1 positives from a dataset (returns numpy arrays) ---
def collect_stage1_positives_from_dataset(ds, threshold):
    X = []; Y = []
    for xb, yb in ds:
        p = stage1_model.predict(xb, verbose=0).ravel()
        mask = p >= threshold
        if mask.any():
            X.extend(xb.numpy()[mask].tolist())
            Y.extend(yb.numpy().astype(int)[mask].tolist())
    return np.array(X), np.array(Y)

# (Optional) Show how many candidates stage1 finds in train/val (informative)
X_train_pos, y_train_pos = collect_stage1_positives_from_dataset(train_ds, chosen_threshold)
X_val_pos, y_val_pos = collect_stage1_positives_from_dataset(val_ds, chosen_threshold)
print(f"Stage-1 positives: train={len(X_train_pos)}, val={len(X_val_pos)}")

# --- Run cascade on TEST set and compute final metrics ---
print("Running cascade on TEST set...")
y_test_all = []; y_test_pred_all = []
stage1_probs_all = []

for xb, yb in test_ds:
    p1 = stage1_model.predict(xb, verbose=0).ravel()
    mask = p1 >= chosen_threshold
    final_preds = np.zeros_like(p1, dtype=int)
    if mask.any():
        xb_candidates = xb.numpy()[mask]
        p2 = stage2_model.predict(xb_candidates, verbose=0).ravel()
        preds2 = (p2 >= 0.5).astype(int)   # stage2 threshold fixed at 0.5 (you can tune separately)
        # assign preds2 back into final_preds
        idxs = np.where(mask)[0]
        for j, idx in enumerate(idxs):
            final_preds[idx] = int(preds2[j])
    # collect
    y_test_all.extend(yb.numpy().astype(int).tolist())
    y_test_pred_all.extend(final_preds.tolist())
    stage1_probs_all.extend(p1.tolist())

y_test_all = np.array(y_test_all); y_test_pred_all = np.array(y_test_pred_all)

acc = float(accuracy_score(y_test_all, y_test_pred_all))
prec = float(precision_score(y_test_all, y_test_pred_all, zero_division=0))
rec = float(recall_score(y_test_all, y_test_pred_all, zero_division=0))
try:
    auc = float(roc_auc_score(y_test_all, stage1_probs_all))
except Exception:
    auc = float('nan')
cm = confusion_matrix(y_test_all, y_test_pred_all).tolist()

metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "auc_stage1": auc,
    "confusion_matrix": cm,
    "stage1_model": STAGE1_MODEL_PATH,
    "stage2_model": STAGE2_MODEL_PATH,
    "chosen_threshold_stage1": float(chosen_threshold)
}

out_path = os.path.join(OUTPUT_DIR, "metrics_cascade.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved cascade metrics to", out_path)
print(json.dumps(metrics, indent=2))
