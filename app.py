# app.py
import streamlit as st
import json
import pandas as pd
from PIL import Image
import numpy as np
import os
import tensorflow as tf

OUTPUT_DIR = "output"
TEST_DIR = "data/test"

st.set_page_config(page_title="Pneumonia Mini Experiment", layout="centered")

st.title("Pneumonia Detection — Mini Experiment")
st.markdown("Simple CNN + basic metrics dashboard")

# Load metrics
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
history_path = os.path.join(OUTPUT_DIR, "history.csv")

if not os.path.exists(metrics_path):
    st.error("No metrics found. Run `python train_cnn.py` first.")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

st.subheader("Test metrics")
cols = st.columns(4)
cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
cols[1].metric("Precision", f"{metrics['precision']:.3f}")
cols[2].metric("Recall (Sensitivity)", f"{metrics['recall']:.3f}")
cols[3].metric("AUC", f"{metrics['auc']:.3f}")

st.write("Confusion matrix (rows=true, cols=pred):")
st.write(metrics['confusion_matrix'])

# Plot training curves
if os.path.exists(history_path):
    hist = pd.read_csv(history_path)
    st.subheader("Training curves")
    st.line_chart(hist[["loss", "val_loss"]].rename(columns={"loss":"Train Loss","val_loss":"Val Loss"}))
    st.line_chart(hist[["accuracy", "val_accuracy"]].rename(columns={"accuracy":"Train Acc","val_accuracy":"Val Acc"}))

# Load model for sample predictions
model_path = metrics.get("model_path")
if model_path and os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.subheader("Sample predictions from test set")
    # collect a few sample images
    samples = []
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(TEST_DIR, label)
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            if files:
                samples.extend(files[:3])  # up to 3 per class

    if not samples:
        st.info("No test images found to display.")
    else:
        cols = st.columns(3)
        for i, img_path in enumerate(samples):
            try:
                img = Image.open(img_path).convert("RGB").resize((150,150))
                arr = np.array(img)/255.0
                prob = float(model.predict(np.expand_dims(arr, axis=0))[0][0])
                pred = "PNEUMONIA" if prob>=0.5 else "NORMAL"
                with cols[i % 3]:
                    st.image(img, caption=f"{os.path.basename(img_path)}\nPred: {pred} ({prob:.2f})", use_column_width=True)
            except Exception as e:
                st.write("Error loading", img_path, e)
else:
    st.info("Model file missing; metrics point to none or missing model.")
