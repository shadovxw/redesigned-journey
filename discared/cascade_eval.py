#!/usr/bin/env python3
"""
cascade_eval.py

Loads two models (stage1 = high-recall screener, stage2 = high-precision refiner),
computes probabilities on a dataset (val by default), runs a grid search over
thresholds (T1 for stage1, T2 for stage2), selects best thresholds according to
a recall constraint (or F1 if none meet the constraint), and writes results.

CPU-safe: uses tf.device('/CPU:0') for inference so it runs even without GPU.
"""

import os
import argparse
import json
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

# -----------------------
IMG_SIZE = (150, 150)   # must match training
# -----------------------

def collect_filepaths_and_labels(dataset_dir: str, positive_dir_name="PNEUMONIA") -> Tuple[List[str], np.ndarray]:
    """
    Walks dataset_dir expecting two subfolders (e.g., NORMAL, PNEUMONIA).
    Returns sorted file paths (deterministic order) and labels (0/1).
    """
    file_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not classes:
        raise ValueError(f"No class subdirectories found in {dataset_dir}")
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        # sort files for deterministic ordering
        files = sorted(glob.glob(os.path.join(cls_dir, "*")))
        for f in files:
            file_paths.append(f)
            labels.append(1 if cls.upper() == positive_dir_name.upper() else 0)
    return file_paths, np.array(labels, dtype=int)


def load_image_batch(paths: List[str], target_size=IMG_SIZE) -> np.ndarray:
    """Load a batch of images to a numpy array (dtype=float32). Don't rescale here (models typically include Rescaling)."""
    imgs = []
    for p in paths:
        img = kimage.load_img(p, target_size=target_size, interpolation='bilinear')
        arr = kimage.img_to_array(img)  # float32 by default
        imgs.append(arr)
    batch = np.stack(imgs, axis=0)
    return batch


def batched_predict(model, file_paths: List[str], batch_size=32) -> np.ndarray:
    """Predict probabilities from model for a list of file paths (returns 1-D numpy array of probs)."""
    probs = []
    n = len(file_paths)
    for i in range(0, n, batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch = load_image_batch(batch_paths)
        # force CPU usage for inference to be safe even if GPU not present
        with tf.device('/CPU:0'):
            preds = model.predict(batch, verbose=0)
        # Model outputs shape (batch,1) or (batch,)
        preds = np.array(preds).reshape(-1)
        probs.extend(preds.tolist())
    return np.array(probs, dtype=float)


def evaluate_combined(y_true: np.ndarray, probs1: np.ndarray, probs2: np.ndarray, T1: float, T2: float):
    """
    Cascade rule:
      - if probs1 >= T1 -> pass to stage2, else final prediction = 0
      - when passed: final = (probs2 >= T2)
    Returns metrics dict and final predictions array.
    """
    passed = probs1 >= T1
    final_pred = np.zeros_like(y_true, dtype=int)
    # only evaluate stage2 where passed
    final_pred[passed] = (probs2[passed] >= T2).astype(int)

    precision = precision_score(y_true, final_pred, zero_division=0)
    recall = recall_score(y_true, final_pred, zero_division=0)
    f1 = f1_score(y_true, final_pred, zero_division=0)
    acc = accuracy_score(y_true, final_pred)
    try:
        auc = roc_auc_score(y_true, probs2 * passed + probs1 * (~passed))  # rough combined score (optional)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, final_pred).tolist()

    metrics = {
        "T1": float(T1),
        "T2": float(T2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "auc_estimate": float(auc),
        "confusion_matrix": cm
    }
    return metrics, final_pred


def grid_search_thresholds(y_true, probs1, probs2, t1_grid, t2_grid, recall_target=0.98):
    """
    Search over t1_grid x t2_grid.
    Preference: maximize precision subject to recall >= recall_target.
    If no pair satisfies recall_target, pick pair with maximum F1.
    Returns best_metrics dict and best pair.
    """
    best = None
    best_metrics = None
    # store all results for diagnostics
    results = []
    for T1 in t1_grid:
        # speed: precompute passed mask for this T1
        passed_mask = probs1 >= T1
        if not passed_mask.any():
            # no samples pass to stage2 -> everything negative; still evaluate
            pass
        for T2 in t2_grid:
            metrics, _ = evaluate_combined(y_true, probs1, probs2, T1, T2)
            results.append(metrics)
            # select by recall constraint then precision
            if metrics["recall"] >= recall_target:
                if best is None or (metrics["precision"] > best_metrics["precision"]) or (
                    metrics["precision"] == best_metrics["precision"] and metrics["f1"] > best_metrics["f1"]
                ):
                    best = (T1, T2)
                    best_metrics = metrics
    # If nothing met recall constraint, fallback to best F1
    if best is None:
        # pick highest F1
        best = None
        best_metrics = None
        for m in results:
            if best_metrics is None or m["f1"] > best_metrics["f1"]:
                best_metrics = m
                best = (m["T1"], m["T2"])
    return best, best_metrics, results


def main(args):
    val_dir = args.val_dir
    if not os.path.isdir(val_dir):
        raise ValueError(f"Validation directory {val_dir} not found")

    print("Collecting val file paths and labels...")
    file_paths, y_val = collect_filepaths_and_labels(val_dir, positive_dir_name=args.positive_dir_name)
    print(f"Found {len(file_paths)} samples (positives={int(y_val.sum())}, negatives={len(y_val)-int(y_val.sum())})")

    print("Loading stage1 model from:", args.stage1_model)
    print("Loading stage2 model from:", args.stage2_model)
    with tf.device('/CPU:0'):
        stage1 = tf.keras.models.load_model(args.stage1_model, compile=False)
        stage2 = tf.keras.models.load_model(args.stage2_model, compile=False)

    print("Computing stage1 probabilities (batched)...")
    probs1 = batched_predict(stage1, file_paths, batch_size=args.batch_size)
    print("Computing stage2 probabilities (batched)...")
    probs2 = batched_predict(stage2, file_paths, batch_size=args.batch_size)

    # quick sanity: lengths must match
    assert len(probs1) == len(file_paths) == len(probs2), "Prediction length mismatch"

    # grid definition (modifiable via CLI)
    t1_grid = np.linspace(args.t1_min, args.t1_max, args.t1_steps)
    t2_grid = np.linspace(args.t2_min, args.t2_max, args.t2_steps)

    print(f"Grid search T1 in {t1_grid[0]:.3f}..{t1_grid[-1]:.3f} ({len(t1_grid)} steps), "
          f"T2 in {t2_grid[0]:.3f}..{t2_grid[-1]:.3f} ({len(t2_grid)} steps).")
    print(f"Recall target: {args.recall_target}")

    best_pair, best_metrics, all_results = grid_search_thresholds(y_val, probs1, probs2, t1_grid, t2_grid, recall_target=args.recall_target)

    print("Best pair:", best_pair)
    print("Best metrics:", json.dumps(best_metrics, indent=2))

    # Evaluate best pair and create per-sample CSV
    final_metrics, final_pred = evaluate_combined(y_val, probs1, probs2, best_pair[0], best_pair[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write thresholds + metrics
    with open(out_dir / "chosen_thresholds_and_metrics.json", "w") as f:
        json.dump({"best_pair": {"T1": best_pair[0], "T2": best_pair[1]}, "metrics": best_metrics}, f, indent=2)

    # write per-sample csv
    rows = []
    for fp, ytrue, p1, p2, p1_pred, p2_pred, finalp in zip(
        file_paths, y_val, probs1, probs2, (probs1 >= best_pair[0]).astype(int), (probs2 >= best_pair[1]).astype(int), final_pred
    ):
        rows.append({
            "filepath": fp,
            "true_label": int(ytrue),
            "prob_stage1": float(p1),
            "prob_stage2": float(p2),
            "stage1_pred": int(p1_pred),
            "stage2_pred": int(p2_pred),
            "final_pred": int(finalp)
        })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_sample_predictions_val.csv"
    df.to_csv(csv_path, index=False)

    # save all grid results (optional, large)
    all_results_path = out_dir / "grid_results_val.json"
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # final metrics
    with open(out_dir / "final_metrics_val.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Outputs written to", str(out_dir))
    print("Per-sample CSV:", str(csv_path))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2-stage cascade and search thresholds (CPU-safe).")
    parser.add_argument("--val-dir", type=str, default="data/val", help="Validation directory (two subfolders: NORMAL, PNEUMONIA).")
    parser.add_argument("--stage1-model", type=str, default="output/stage1_model.h5", help="Path to stage-1 model (high-recall).")
    parser.add_argument("--stage2-model", type=str, default="output/stage2_model.keras", help="Path to stage-2 model (high-precision).")
    parser.add_argument("--output-dir", type=str, default="output/cascade_eval", help="Directory to write outputs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (reduce if RAM limited).")
    parser.add_argument("--t1-min", type=float, default=0.0, help="Minimum T1 threshold.")
    parser.add_argument("--t1-max", type=float, default=0.6, help="Maximum T1 threshold.")
    parser.add_argument("--t1-steps", type=int, default=13, help="Number of T1 steps (linspace).")
    parser.add_argument("--t2-min", type=float, default=0.4, help="Minimum T2 threshold.")
    parser.add_argument("--t2-max", type=float, default=0.99, help="Maximum T2 threshold.")
    parser.add_argument("--t2-steps", type=int, default=60, help="Number of T2 steps (linspace).")
    parser.add_argument("--recall-target", type=float, default=0.98, help="Recall constraint when choosing thresholds.")
    parser.add_argument("--positive-dir-name", type=str, default="PNEUMONIA", help="Name of positive class folder (case-insensitive).")
    args = parser.parse_args()
    main(args)
