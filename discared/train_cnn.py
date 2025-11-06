# train_cnn_tuned.py
"""
Train improved CNN (optionally transfer-learning) and choose a decision threshold
that tries to achieve target recall on validation while maximizing precision.
Saves final model, history, and metrics.json (including chosen threshold).
"""
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight

# ---------------- USER SETTINGS ----------------
DATA_DIR = "data"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123
OUTPUT_DIR = "output"
USE_TRANSFER_LEARNING = True   # Use MobileNetV2 features (recommended)
FREEZE_TL = True               # Freeze base model initially
TARGET_RECALL = 0.99           # desired recall on validation (try to reach this)
PNEUMONIA_BOOST = 2.0          # multiplies computed pneumonia class weight to favor recall
# ------------------------------------------------

# reproducibility (best-effort)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF verbosity

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# --- Load datasets ---
print("Loading datasets from:", train_dir, val_dir, test_dir)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=True, seed=SEED
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)

print("Class names (train):", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- class weights (help imbalance)
y_train_list = []
for _, y in train_ds.unbatch():
    y_train_list.append(int(y.numpy()))

if len(y_train_list) == 0:
    print("Warning: couldn't read train labels for class weights.")
    class_weight = None
else:
    classes = np.unique(y_train_list)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_list)
    class_weight = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
    # boost pneumonia weight (label 1) to favor recall if present
    if 1 in class_weight:
        class_weight[1] = float(class_weight[1] * PNEUMONIA_BOOST)
    print("Class weights:", class_weight)

# --- augmentation layers (applied in-model) ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

# --- build model ---
def build_model(input_shape=IMG_SIZE + (3,), use_tl=True):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    if use_tl:
        base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        if FREEZE_TL:
            base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_small_head")
    else:
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="simple_cnn_v2")
    return model

model = build_model(use_tl=USE_TRANSFER_LEARNING)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
model.summary()

# --- callbacks ---
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# --- train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[es, mc],
    class_weight=class_weight if class_weight else None
)

# save history
hist_df = pd.DataFrame(history.history)
hist_csv_path = os.path.join(OUTPUT_DIR, "history.csv")
hist_df.to_csv(hist_csv_path, index=False)

# save final model too
final_model_path = os.path.join(OUTPUT_DIR, "simple_cnn_tuned.keras")
model.save(final_model_path)

# --- choose threshold on VALIDATION set ---
def dataset_to_arrays(ds):
    """Return (y_true, y_prob) arrays for a tf.data dataset using current model."""
    y_true = []
    y_prob = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy().astype(int).tolist())
        y_prob.extend([float(p[0]) for p in probs])
    return np.array(y_true), np.array(y_prob)

print("Computing validation predictions for threshold selection...")
y_val, p_val = dataset_to_arrays(val_ds)

# baseline metrics at 0.5
y_pred_050 = (p_val >= 0.5).astype(int)
base_recall = recall_score(y_val, y_pred_050)
base_prec = precision_score(y_val, y_pred_050, zero_division=0)
print(f"Baseline val recall@0.5={base_recall:.3f}, precision@0.5={base_prec:.3f}")

# precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
# note: thresholds has length len(precisions)-1
chosen_threshold = None
chosen_method = None

# find thresholds meeting target recall (recall >= TARGET_RECALL)
# iterate over thresholds; each threshold corresponds to precisions[i], recalls[i]
# but thresholds aligns with precisions[0:-1],recalls[0:-1], so we map index -> threshold index
found = []
# build arrays aligned with thresholds
for i, th in enumerate(np.append(thresholds, 1.0)):  # append to align lengths
    r = recalls[i]
    p = precisions[i]
    if r >= TARGET_RECALL:
        # threshold for this index: if i < len(thresholds), threshold = thresholds[i], else 1.0
        thr = thresholds[i] if i < len(thresholds) else 1.0
        found.append((thr, p, r, i))

if found:
    # choose highest precision among found
    best = max(found, key=lambda x: x[1])
    chosen_threshold = float(best[0])
    chosen_method = f"recall_target_{TARGET_RECALL}"
    print(f"Found thresholds reaching target recall. Chosen threshold={chosen_threshold:.4f} with precision={best[1]:.3f}, recall={best[2]:.3f}")
else:
    # fallback: choose threshold maximizing F1 on val
    f1_scores = []
    ths = np.append(thresholds, 1.0)
    for i, th in enumerate(ths):
        # compute preds with threshold th
        preds = (p_val >= th).astype(int)
        f1 = f1_score(y_val, preds)
        f1_scores.append((th, f1))
    best_th, best_f1 = max(f1_scores, key=lambda x: x[1])
    chosen_threshold = float(best_th)
    chosen_method = "max_f1_fallback"
    print(f"No threshold reaches recall {TARGET_RECALL}. Fallback chosen_threshold={chosen_threshold:.4f} (F1={best_f1:.3f})")

# Save chosen threshold to disk
with open(os.path.join(OUTPUT_DIR, "chosen_threshold.json"), "w") as f:
    json.dump({"chosen_threshold": chosen_threshold, "method": chosen_method}, f, indent=2)

# --- Evaluate on TEST set using chosen threshold ---
print("Evaluating on TEST set using threshold:", chosen_threshold)
y_test, p_test = dataset_to_arrays(test_ds)
y_test_pred = (p_test >= chosen_threshold).astype(int)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, p_test)
except Exception:
    auc = float('nan')
cm = confusion_matrix(y_test, y_test_pred).tolist()

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "auc": float(auc),
    "confusion_matrix": cm,
    "model_path": final_model_path,
    "history_csv": hist_csv_path,
    "chosen_threshold": float(chosen_threshold),
    "threshold_selection_method": chosen_method
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model and metrics to", OUTPUT_DIR)
print(json.dumps(metrics, indent=2))

# also save train config
train_config = {
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "use_transfer_learning": USE_TRANSFER_LEARNING,
    "freeze_tl": FREEZE_TL,
    "target_recall": TARGET_RECALL,
    "pneumonia_boost": PNEUMONIA_BOOST
}
with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w") as f:
    json.dump(train_config, f, indent=2)
