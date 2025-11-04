# train_cnn.py  (improved version)
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# -------------------- USER SETTINGS --------------------
DATA_DIR = "data"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123
OUTPUT_DIR = "output"
USE_TRANSFER_LEARNING = False   # set True to use MobileNetV2 as feature extractor
FREEZE_TL = True               # if using TL, freeze base model initially
# -------------------------------------------------------

# reproducibility (best-effort)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

# --- Create tf.data datasets from directories (no rescaling here; model will rescale) ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False,
    seed=SEED
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False,
    seed=SEED
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Compute class weights from training labels (helps with imbalance) ---
y_train_list = []
for _, y in train_ds.unbatch():
    try:
        y_train_list.append(int(y.numpy()))
    except Exception:
        # fallback if something is odd
        pass

if len(y_train_list) == 0:
    print("Warning: couldn't collect labels for class weights; proceeding without them.")
    class_weight = None
else:
    classes = np.unique(y_train_list)
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_list)
    # compute_class_weight returns in order of classes array; map to dict expected by Keras
    class_weight = {int(classes[i]): float(class_weights_vals[i]) for i in range(len(classes))}
    print("Computed class_weight:", class_weight)

# --- Augmentation layers (applied in-model, only active in training) ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),   # ~ +/- 6%
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

# --- Model builder (simple CNN or transfer learning) ---
def build_model(input_shape=IMG_SIZE + (3,), use_tl=False):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # augmentation only applies in training; when using model in inference it's ignored
    x = data_augmentation(x)

    # normalize pixels 0..1
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
        model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_finetune")
    else:
        # simple but better-regularized CNN
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling2D()(x)   # much better than Flatten for generalization
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="simple_cnn_v2")

    return model

model = build_model(use_tl=USE_TRANSFER_LEARNING)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# --- Callbacks: EarlyStopping + ModelCheckpoint ---
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[es, mc],
    class_weight=class_weight  # may be None which is fine
)

# Save final model (best is already saved by checkpoint)
final_model_path = os.path.join(OUTPUT_DIR, "simple_cnn.keras")
model.save(final_model_path)

# Save history
hist_df = pd.DataFrame(history.history)
hist_csv_path = os.path.join(OUTPUT_DIR, "history.csv")
hist_df.to_csv(hist_csv_path, index=False)

# --- Evaluate on test set & compute metrics per-sample ---
y_true = []
y_pred_prob = []

for batch_imgs, batch_labels in test_ds:
    probs = model.predict(batch_imgs)
    y_true.extend(batch_labels.numpy().astype(int).tolist())
    y_pred_prob.extend([float(p[0]) for p in probs])

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_true, y_pred_prob)
except Exception:
    auc = float('nan')
cm = confusion_matrix(y_true, y_pred).tolist()

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "auc": float(auc),
    "confusion_matrix": cm,
    "model_path": final_model_path,
    "history_csv": hist_csv_path
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# save training config
train_config = {
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "use_transfer_learning": USE_TRANSFER_LEARNING,
    "freeze_tl": FREEZE_TL
}
with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w") as f:
    json.dump(train_config, f, indent=2)

print("Saved model and metrics to", OUTPUT_DIR)
print(json.dumps(metrics, indent=2))
