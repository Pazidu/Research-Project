from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# =======================
# CONFIGURATION
# =======================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 5e-5
R_RATIO = 24
FOLDS = 5

DATASET_DIR = "/content/drive/MyDrive/UMCG"
SAVE_PATH = "/content/drive/MyDrive/Models/clinical_edge_kfold"
os.makedirs(SAVE_PATH, exist_ok=True)
CLASSES = ["melanoma", "non_melanoma"]

# =======================
# CREATE DATAFRAME
# =======================
filepaths = []
labels = []

for label in CLASSES:
    class_dir = os.path.join(DATASET_DIR, label)
    for fname in os.listdir(class_dir):
        filepaths.append(os.path.join(class_dir, fname))
        labels.append(label)

df = pd.DataFrame({
    "filename": filepaths,
    "class": labels
})
print("Total images:", len(df))

# =======================
# EDGE-RGB PREPROCESS FUNCTION
# =======================
def preprocess_edge_rgb(image_path, label):
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img_rgb = tf.cast(img, tf.float32) / 255.0

    # Compute Sobel edge map
    img_batch = tf.expand_dims(img_rgb, axis=0)  # [1,H,W,3]
    sobel = tf.image.sobel_edges(img_batch)      # [1,H,W,3,2]
    edge_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))  # [1,H,W,3]
    edge_mag = tf.reduce_mean(edge_mag, axis=-1, keepdims=True)   # [1,H,W,1]
    edge_mag = tf.squeeze(edge_mag, axis=0)                        # [H,W,1]
    edge_mag = tf.clip_by_value(edge_mag, 0.0, 1.0)

    # Convert label to float
    label = tf.cast(label == "melanoma", tf.float32)

    return {"rgb_input": img_rgb, "edge_input": edge_mag}, label

# =======================
# DUAL-INPUT MODEL
# =======================
def create_dual_input_model():
    rgb_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="rgb_input")
    edge_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="edge_input")

    # RGB branch
    base_rgb = EfficientNetV2M(include_top=False, weights="imagenet", input_tensor=rgb_input)
    base_rgb.trainable = False
    for layer in base_rgb.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    x_rgb = layers.GlobalAveragePooling2D()(base_rgb.output)

    # Edge branch
    x_edge = layers.Conv2D(16, (3,3), activation="relu", padding="same")(edge_input)
    x_edge = layers.MaxPooling2D()(x_edge)
    x_edge = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x_edge)
    x_edge = layers.GlobalAveragePooling2D()(x_edge)

    # Concatenate
    x = layers.Concatenate()([x_rgb, x_edge])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# =======================
# K-FOLD TRAINING
# =======================
accuracies = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df["filename"], df["class"]), 1):
    print(f"\n===== Fold {fold} =====")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((train_df["filename"].values, train_df["class"].values))
    val_ds   = tf.data.Dataset.from_tensor_slices((val_df["filename"].values, val_df["class"].values))

    train_ds = train_ds.map(preprocess_edge_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(preprocess_edge_rgb, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = create_dual_input_model()

    LOCAL_SAVE = f"/content/best_fold{fold}.keras"
    ckpt = ModelCheckpoint(filepath=LOCAL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, reduce_lr])

    model = tf.keras.models.load_model(LOCAL_SAVE)
    loss, acc, auc = model.evaluate(val_ds)
    accuracies.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    tf.keras.backend.clear_session()  # Free RAM

print(f"\nAverage Accuracy over {FOLDS} folds: {np.mean(accuracies):.4f}")
