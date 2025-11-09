
import json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.utils.class_weight import compute_class_weight


DATA_DIR = "data"
OUTPUT_DIR = "output"
IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 18
LR = 1e-4
USE_CLASS_WEIGHT = True
USE_FOCAL = True
FOCAL_GAMMA = 1.0
FOCAL_ALPHA = 0.75
UNFREEZE_AND_FINETUNE = True
FINETUNE_EPOCHS = 6
FINETUNE_LR = 1e-5
SEED = 123
# ------------------------------------------------

tf.random.set_seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-7
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss_val = - alpha_t * tf.pow(1.0 - p_t + eps, gamma) * tf.math.log(p_t + eps)
        return tf.reduce_mean(loss_val)
    return loss

def compute_class_weights_from_dir(train_dir):
    y = []
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir): continue
        label = 1 if cls.upper()=="PNEUMONIA" else 0
        count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir,f))])
        y += [label]*count
    if not y:
        return None
    classes = np.array([0,1])
    cw = compute_class_weight('balanced', classes=classes, y=np.array(y))
    return {int(classes[i]): float(cw[i]) for i in range(len(classes))}

def get_datasets(data_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir,"train"),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=True, seed=SEED
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir,"val"),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False, seed=SEED
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds

def build_stage1(img_size=IMG_SIZE+(3,)):
    inputs = layers.Input(shape=img_size)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.06)(x)
    x = layers.Rescaling(1./255)(x)
    base = tf.keras.applications.MobileNetV2(input_shape=img_size, include_top=False, weights='imagenet')
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="stage1_mobilenetv2")
    return model


train_ds, val_ds = get_datasets(DATA_DIR)
class_weight = compute_class_weights_from_dir(os.path.join(DATA_DIR,"train")) if USE_CLASS_WEIGHT else None
print("class_weight:", class_weight)

model = build_stage1()
loss = focal_loss(FOCAL_GAMMA, FOCAL_ALPHA) if USE_FOCAL else 'binary_crossentropy'
model.compile(optimizer=optimizers.Adam(learning_rate=LR),
              loss=loss,
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
model.summary()

ckpt_path = os.path.join(OUTPUT_DIR, "stage1_model.keras")
es = callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=6, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(ckpt_path, monitor='val_recall', mode='max', save_best_only=True, verbose=1)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[es, mc], class_weight=class_weight)


if UNFREEZE_AND_FINETUNE:
    print("Unfreezing base and finetuning...")
    base = [l for l in model.layers if isinstance(l, tf.keras.Model) or 'mobilenetv2' in l.name.lower()]

    try:
        base_model = [l for l in model.layers if len(l.weights)>0 and 'mobilenetv2' in l.name.lower()][0]
        base_model.trainable = True
    except Exception:
        for layer in model.layers:
            layer.trainable = True
    model.compile(optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=[callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=4, restore_best_weights=True)])

model.save(ckpt_path)
(Path(OUTPUT_DIR)/"history_stage1.json").write_text(json.dumps(history.history))
print("Saved stage1 to", ckpt_path)
