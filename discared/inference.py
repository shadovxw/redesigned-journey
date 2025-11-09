# inference.py -- CPU-safe folder inference using your trained cascade + meta-classifier
# Edit PARAMS below, then run: python inference.py  (or paste into a Colab cell)

import os, json, glob, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.preprocessing import StandardScaler

# -------------------- PARAMS (edit) --------------------
IMAGE_FOLDER = "inference_images"   # folder with images to score (non-recursive)
OUT_DIR = Path("output") / "cascade_eval"
STAGE1_MODEL = "output/stage1_model.keras"
STAGE2_MODEL = "output/stage2_model.keras"
META_SCALER = OUT_DIR / "meta_scaler.pkl"
META_MODEL = OUT_DIR / "meta_model.pkl"
META_THRESHOLD_JSON = OUT_DIR / "meta_threshold.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
METHOD = "meta"   # "meta", "and", or "or"
AND_T1 = 0.05     # used only for "and" or "or" when method != "meta"
AND_T2 = 0.4
# -----------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# helpers
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
        with tf.device('/CPU:0'):
            preds = model.predict(batch, verbose=0)
        preds = np.array(preds).reshape(-1)
        probs.extend(preds.tolist())
    return np.array(probs, dtype=float)

# 1) collect images
imgs = sorted([os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))])
if len(imgs) == 0:
    raise SystemExit(f"No images found in '{IMAGE_FOLDER}'. Put files there (JPEG/PNG).")

print(f"Found {len(imgs)} images to score in {IMAGE_FOLDER}")

# 2) load models
print("Loading stage1 and stage2 models (CPU)...")
stage1 = tf.keras.models.load_model(STAGE1_MODEL, compile=False)
stage2 = tf.keras.models.load_model(STAGE2_MODEL, compile=False)
print("Models loaded.")

# 3) compute probs
print("Computing stage1 probs...")
p1 = batched_predict(stage1, imgs, batch_size=BATCH_SIZE)
print("Computing stage2 probs...")
p2 = batched_predict(stage2, imgs, batch_size=BATCH_SIZE)

final_preds = None
meta_probs = None

if METHOD == "meta":
    if not META_SCALER.exists() or not META_MODEL.exists() or not META_THRESHOLD_JSON.exists():
        raise SystemExit("Meta artifacts not found. Run the meta training cell first.")
    print("Loading meta scaler & model...")
    with open(META_SCALER, "rb") as f:
        scaler = pickle.load(f)
    with open(META_MODEL, "rb") as f:
        meta_model = pickle.load(f)
    meta_probs = meta_model.predict_proba(scaler.transform(np.vstack([p1,p2]).T))[:,1]
    meta_thresh = float(json.load(open(META_THRESHOLD_JSON))["threshold"])
    final_preds = (meta_probs >= meta_thresh).astype(int)
    print(f"Using meta decision rule with threshold={meta_thresh:.4f}")
elif METHOD == "and":
    passed = p1 >= AND_T1
    final_preds = np.zeros_like(p1, dtype=int)
    final_preds[passed] = (p2[passed] >= AND_T2).astype(int)
    print(f"Using AND rule with T1={AND_T1}, T2={AND_T2}")
elif METHOD == "or":
    final_preds = ((p1 >= AND_T1) | (p2 >= AND_T2)).astype(int)
    print(f"Using OR rule with T1={AND_T1}, T2={AND_T2}")
else:
    raise SystemExit("METHOD must be one of: 'meta', 'and', 'or'")

# 4) write per-file CSV
rows = []
for fp, a, b, f in zip(imgs, p1, p2, final_preds):
    r = {"filepath": fp, "prob_stage1": float(a), "prob_stage2": float(b), "final_pred": int(f)}
    if meta_probs is not None:
        # align meta_probs
        r["meta_prob"] = float(meta_probs[len(rows)])
    rows.append(r)

df = pd.DataFrame(rows)
out_csv = OUT_DIR / f"inference_{METHOD}_results.csv"
df.to_csv(out_csv, index=False)
print("Wrote per-file results to", out_csv)

# 5) summary report (counts)
counts = df["final_pred"].value_counts().to_dict()
summary = {"total": int(len(df)), "pred_positive": int(counts.get(1,0)), "pred_negative": int(counts.get(0,0))}
(OUT_DIR / f"inference_{METHOD}_summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote summary to", OUT_DIR / f"inference_{METHOD}_summary.json")
print("Done.")
