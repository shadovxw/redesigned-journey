# inference_single.py -- classify a single X-ray image using trained cascade or meta-classifier

import os, json, pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.preprocessing import StandardScaler


IMAGE_PATH = "normal.jpeg"   
STAGE1_MODEL = "output/stage1_model.keras"
STAGE2_MODEL = "output/stage2_model.keras"
OUT_DIR = Path("output") / "cascade_eval"
META_SCALER = OUT_DIR / "meta_scaler.pkl"
META_MODEL = OUT_DIR / "meta_model.pkl"
META_THRESHOLD_JSON = OUT_DIR / "meta_threshold.json"
IMG_SIZE = (224, 224)
METHOD = "meta"    
AND_T1 = 0.0    
AND_T2 = 0.5

if not os.path.exists(IMAGE_PATH):
    raise SystemExit(f"Image not found: {IMAGE_PATH}")



def load_single_image(path, target_size=IMG_SIZE):
    img = kimage.load_img(path, target_size=target_size, interpolation='bilinear')
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_cpu(model, img_array):
    with tf.device('/CPU:0'):
        prob = model.predict(img_array, verbose=0)[0][0]
    return float(prob)


print("Loading stage1 and stage2 models (CPU)...")
stage1 = tf.keras.models.load_model(STAGE1_MODEL, compile=False)
stage2 = tf.keras.models.load_model(STAGE2_MODEL, compile=False)
print("Models loaded.\n")


img_array = load_single_image(IMAGE_PATH)

print(f"Running inference on: {IMAGE_PATH}")
p1 = predict_cpu(stage1, img_array)
p2 = predict_cpu(stage2, img_array)
print(f"Stage1 probability: {p1:.4f}")
print(f"Stage2 probability: {p2:.4f}\n")


final_pred = None
meta_prob = None

if METHOD == "meta":
    if not META_SCALER.exists() or not META_MODEL.exists() or not META_THRESHOLD_JSON.exists():
        raise SystemExit("Meta artifacts not found. Run meta training first.")
    print("Using META classifier...")
    with open(META_SCALER, "rb") as f:
        scaler = pickle.load(f)
    with open(META_MODEL, "rb") as f:
        meta_model = pickle.load(f)
    meta_thresh = float(json.load(open(META_THRESHOLD_JSON))["threshold"])
    meta_prob = meta_model.predict_proba(scaler.transform([[p1, p2]]))[:, 1][0]
    final_pred = int(meta_prob >= meta_thresh)
    print(f"Meta probability: {meta_prob:.4f} (threshold={meta_thresh:.4f})")

elif METHOD == "and":
    print("Using AND rule...")
    final_pred = int((p1 >= AND_T1) and (p2 >= AND_T2))
    print(f"Decision rule: p1>={AND_T1}, p2>={AND_T2}")

elif METHOD == "or":
    print("Using OR rule...")
    final_pred = int((p1 >= AND_T1) or (p2 >= AND_T2))
    print(f"Decision rule: p1>={AND_T1} or p2>={AND_T2}")

else:
    raise SystemExit("METHOD must be one of: 'meta', 'and', 'or'")


label = "PNEUMONIA" if final_pred == 1 else "NORMAL"

print(f"🩺 Final Prediction: {label}")
if meta_prob is not None:
    print(f"Meta confidence: {meta_prob*100:.2f}%")
else:
    print(f"Stage1={p1:.2f}, Stage2={p2:.2f}")

