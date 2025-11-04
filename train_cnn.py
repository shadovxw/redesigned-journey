# train_cnn.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score

# --------- User settings ----------
DATA_DIR = "chest_xray"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 123
OUTPUT_DIR = "output"
# ----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

# Create tf.data datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=True, seed=SEED
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Simple CNN model
def build_model(input_shape=IMG_SIZE + (3,)):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')   
    ])
    return model

model = build_model()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save model and training history
model_save_path = os.path.join(OUTPUT_DIR, "simple_cnn.h5")
model.save(model_save_path)

hist_df = pd.DataFrame(history.history)
hist_csv_path = os.path.join(OUTPUT_DIR, "history.csv")
hist_df.to_csv(hist_csv_path, index=False)

# Evaluate on test set & compute metrics per-sample
y_true = []
y_pred_prob = []
filenames = []

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
except:
    auc = float('nan')
cm = confusion_matrix(y_true, y_pred).tolist()

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "auc": float(auc),
    "confusion_matrix": cm,
    "model_path": model_save_path,
    "history_csv": hist_csv_path
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model and metrics to", OUTPUT_DIR)
print(json.dumps(metrics, indent=2))
