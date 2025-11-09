
import os, json, math, glob
from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


STAGE1_MODEL = "output/stage1_model.keras"
STAGE2_MODEL = "output/stage2_model.keras"
DATA_VAL = "data/val"
DATA_TEST = "data/test"
POSITIVE_DIR_NAME = "PNEUMONIA"
IMG_SIZE = (224,224)
BATCH_SIZE = 16
T1_MIN, T1_MAX, T1_STEPS = 0.00, 0.6, 31
T2_MIN, T2_MAX, T2_STEPS = 0.0, 0.99, 100
RECALL_TARGET = 0.99
OUTPUT_DIR = Path("output") / "cascade_eval"
VERBOSE = True
# ------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_files_and_labels(dataset_dir, positive_name=POSITIVE_DIR_NAME):
    files = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if len(classes) == 0:
        raise RuntimeError(f"No class subfolders found in {dataset_dir}")
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        lab = 1 if cls.upper() == positive_name.upper() else 0
        for f in sorted(glob.glob(os.path.join(cls_dir, "*"))):
            files.append(f)
            labels.append(lab)
    return files, np.array(labels, dtype=int)

def load_image_batch(paths, target_size=IMG_SIZE):
    imgs = []
    for p in paths:
        img = kimage.load_img(p, target_size=target_size, interpolation='bilinear')
        arr = kimage.img_to_array(img)
        imgs.append(arr)
    return np.stack(imgs, axis=0)

def batched_predict(model, file_paths, batch_size=BATCH_SIZE):
    probs = []
    n = len(file_paths)
    for i in range(0, n, batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch = load_image_batch(batch_paths)
        # ensure CPU inference (works if GPU unavailable too)
        with tf.device('/CPU:0'):
            preds = model.predict(batch, verbose=0)
        preds = np.array(preds).reshape(-1)
        probs.extend(preds.tolist())
    return np.array(probs, dtype=float)

def metrics_from_preds(y_true, y_pred, probs_for_auc=None):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    try:
        auc = float(roc_auc_score(y_true, probs_for_auc)) if probs_for_auc is not None else float('nan')
    except Exception:
        auc = float('nan')
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc), "auc": auc, "confusion_matrix": cm}

def evaluate_and_rule(y_true, p1, p2, T1, T2):
    passed = p1 >= T1
    final = np.zeros_like(y_true, dtype=int)
    final[passed] = (p2[passed] >= T2).astype(int)
    combined_score = np.where(passed, p2, p1)
    return metrics_from_preds(y_true, final, combined_score), final

def evaluate_or_rule(y_true, p1, p2, T1, T2):
    final = ((p1 >= T1) | (p2 >= T2)).astype(int)
    combined_score = np.maximum(p1, p2)
    return metrics_from_preds(y_true, final, combined_score), final


print("Loading models...")
with tf.device('/CPU:0'):
    stage1 = tf.keras.models.load_model(STAGE1_MODEL, compile=False)
    stage2 = tf.keras.models.load_model(STAGE2_MODEL, compile=False)
print("Done.")


print("Collecting val files...")
val_files, y_val = collect_files_and_labels(DATA_VAL)
print(f"Val: {len(val_files)} samples (positives={int(y_val.sum())})")
print("Collecting test files...")
test_files, y_test = collect_files_and_labels(DATA_TEST)
print(f"Test: {len(test_files)} samples (positives={int(y_test.sum())})")


print("Computing stage1 probabilities (val)...")
p1_val = batched_predict(stage1, val_files, batch_size=BATCH_SIZE)
print("Computing stage2 probabilities (val)...")
p2_val = batched_predict(stage2, val_files, batch_size=BATCH_SIZE)
print("Computing stage1 probabilities (test)...")
p1_test = batched_predict(stage1, test_files, batch_size=BATCH_SIZE)
print("Computing stage2 probabilities (test)...")
p2_test = batched_predict(stage2, test_files, batch_size=BATCH_SIZE)


t1_grid = np.linspace(T1_MIN, T1_MAX, int(T1_STEPS))
t2_grid = np.linspace(T2_MIN, T2_MAX, int(T2_STEPS))
print(f"Grid sizes: T1={len(t1_grid)}, T2={len(t2_grid)}")

best_and = None
best_and_metrics = None
best_or = None
best_or_metrics = None
all_results = {"and": [], "or": []}


for T1 in t1_grid:
    passed_mask = p1_val >= T1
    for T2 in t2_grid:
        metrics_and, _ = evaluate_and_rule(y_val, p1_val, p2_val, T1, T2)
        metrics_and["T1"] = float(T1); metrics_and["T2"] = float(T2)
        all_results["and"].append(metrics_and)
        # check recall constraint
        if metrics_and["recall"] >= RECALL_TARGET:
            if best_and is None or metrics_and["precision"] > best_and_metrics["precision"] or \
               (metrics_and["precision"] == best_and_metrics["precision"] and metrics_and["f1"] > best_and_metrics["f1"]):
                best_and = (T1, T2); best_and_metrics = metrics_and


if best_and is None:
    for m in all_results["and"]:
        if best_and_metrics is None or m["f1"] > best_and_metrics["f1"]:
            best_and_metrics = m
            best_and = (m["T1"], m["T2"])


for T1 in t1_grid:
    for T2 in t2_grid:
        metrics_or, _ = evaluate_or_rule(y_val, p1_val, p2_val, T1, T2)
        metrics_or["T1"] = float(T1); metrics_or["T2"] = float(T2)
        all_results["or"].append(metrics_or)
        if metrics_or["recall"] >= RECALL_TARGET:
            if best_or is None or metrics_or["precision"] > best_or_metrics["precision"] or \
               (metrics_or["precision"] == best_or_metrics["precision"] and metrics_or["f1"] > best_or_metrics["f1"]):
                best_or = (T1, T2); best_or_metrics = metrics_or

if best_or is None:
    for m in all_results["or"]:
        if best_or_metrics is None or m["f1"] > best_or_metrics["f1"]:
            best_or_metrics = m
            best_or = (m["T1"], m["T2"])

print("Best AND pair:", best_and, "metrics:")
pprint(best_and_metrics)
print("Best OR pair:", best_or, "metrics:")
pprint(best_or_metrics)


and_metrics_test, and_pred_test = evaluate_and_rule(y_test, p1_test, p2_test, best_and[0], best_and[1])
or_metrics_test, or_pred_test = evaluate_or_rule(y_test, p1_test, p2_test, best_or[0], best_or[1])
print("\nTest results (AND):")
pprint(and_metrics_test)
print("\nTest results (OR):")
pprint(or_metrics_test)


meta_metrics = None
try:
    print("\nTraining logistic meta-classifier on val probs...")
    X_val = np.vstack([p1_val, p2_val]).T
    X_test = np.vstack([p1_test, p2_test]).T
    scaler = StandardScaler().fit(X_val)
    Xv = scaler.transform(X_val)
    Xt = scaler.transform(X_test)

    lg = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    lg.fit(Xv, y_val)
    prob_meta_val = lg.predict_proba(Xv)[:,1]
    prob_meta_test = lg.predict_proba(Xt)[:,1]

    best_thresh = None
    best_m = None
    threshs = np.linspace(0.0, 0.99, 200)
    for t in threshs:
        pred = (prob_meta_val >= t).astype(int)
        m = metrics_from_preds(y_val, pred, prob_meta_val)
        m["threshold"] = float(t)
        if m["recall"] >= RECALL_TARGET:
            if best_m is None or m["precision"] > best_m["precision"] or (m["precision"]==best_m["precision"] and m["f1"]>best_m["f1"]):
                best_m = m; best_thresh = t
    if best_m is None:

        for t in threshs:
            pred = (prob_meta_val >= t).astype(int)
            m = metrics_from_preds(y_val, pred, prob_meta_val)
            m["threshold"] = float(t)
            if best_m is None or m["f1"] > best_m["f1"]:
                best_m = m; best_thresh = t

    pred_test_meta = (prob_meta_test >= best_thresh).astype(int)
    meta_metrics = metrics_from_preds(y_test, pred_test_meta, prob_meta_test)
    meta_metrics["threshold"] = float(best_thresh)
    print("Meta chosen threshold (from val):", best_thresh)
    pprint({"val_pick": best_m, "meta_on_test": meta_metrics})
except Exception as e:
    print("Meta-classifier training failed:", str(e))
    meta_metrics = None


OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


out_summary = {
    "best_and_val": {"T1": float(best_and[0]), "T2": float(best_and[1]), "metrics": best_and_metrics},
    "best_or_val": {"T1": float(best_or[0]), "T2": float(best_or[1]), "metrics": best_or_metrics},
    "and_on_test": and_metrics_test,
    "or_on_test": or_metrics_test,
    "meta_on_test": meta_metrics
}
(OUTPUT_DIR / "chosen_thresholds_and_metrics.json").write_text(json.dumps(out_summary, indent=2))


(OUTPUT_DIR / "grid_and_val.json").write_text(json.dumps(all_results["and"], indent=2))
(OUTPUT_DIR / "grid_or_val.json").write_text(json.dumps(all_results["or"], indent=2))


def save_per_sample(files, y, p1, p2, final_and=None, final_or=None, prefix="val"):
    rows = []
    for fp, yt, a, b in zip(files, y, p1, p2):
        rows.append({"filepath": fp, "true_label": int(yt), "prob_stage1": float(a), "prob_stage2": float(b),
                     "and_final": int(final_and) if final_and is not None else None,
                     "or_final": int(final_or) if final_or is not None else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"per_sample_{prefix}.csv", index=False)


_, val_and_pred = evaluate_and_rule(y_val, p1_val, p2_val, best_and[0], best_and[1])
_, val_or_pred = evaluate_or_rule(y_val, p1_val, p2_val, best_or[0], best_or[1])
save_per_sample(val_files, y_val, p1_val, p2_val, final_and=val_and_pred, final_or=val_or_pred)

save_per_sample(test_files, y_test, p1_test, p2_test, final_and=and_pred_test, final_or=or_pred_test, prefix="test")

print("Wrote outputs to", str(OUTPUT_DIR))
print("Done. Key summary:")
pprint(out_summary)
