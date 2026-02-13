# =====================================
# Mount Drive & Imports
# =====================================
from google.colab import drive
drive.mount('/drive')

import os
import shutil
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Mixed Precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# =====================================
# Paths & Metadata
# =====================================
CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
BASE = "/content/newdata"

df = pd.read_csv(CSV_PATH)
df["label"] = df["dx"].apply(lambda x: "melanoma" if x == "mel" else "non_melanoma")

# Train/Validation/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# =====================================
# Folder Structure & Copy Images
# =====================================
for split in ["train", "valid", "test"]:
    for cls in ["melanoma", "non_melanoma"]:
        os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)

def copy_images(df, split):
    for _, row in df.iterrows():
        img = row["image_id"] + ".jpg"
        src = os.path.join(IMG_SRC, img)
        dst = os.path.join(BASE, split, row["label"], img)
        if os.path.exists(src):
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(valid_df, "valid")
copy_images(test_df, "test")
print("✅ Images copied to local SSD")

# =====================================
# Preprocessing: Edge/H-map
# =====================================
def edge_hmap_preprocess_two_inputs(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0

    # Sobel edge detection
    sobel = tf.image.sobel_edges(image)
    edge_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge_mag = tf.reduce_mean(edge_mag, axis=-1, keepdims=True)
    edge_mag = tf.clip_by_value(edge_mag, 0.0, 1.0)

    return {"rgb_input": image, "edge_input": edge_mag}, label

# =====================================
# Prepare datasets
# =====================================
def prepare_datasets(train_path, valid_path, test_path, batch_size=16):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path, image_size=(256,256), batch_size=batch_size, label_mode='categorical'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path, image_size=(256,256), batch_size=batch_size, label_mode='categorical'
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path, image_size=(256,256), batch_size=batch_size, label_mode='categorical'
    )

    # Apply dual-input preprocessing
    train_ds = train_ds.map(edge_hmap_preprocess_two_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(edge_hmap_preprocess_two_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(edge_hmap_preprocess_two_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE)
    )

train_ds, val_ds, test_ds = prepare_datasets(
    f"{BASE}/train", f"{BASE}/valid", f"{BASE}/test", batch_size=16
)

# =====================================
# Dual-Input Model
# =====================================
def create_model_edge(rgb_size=256, edge_size=256):
    # RGB branch
    rgb_input = layers.Input(shape=(rgb_size, rgb_size, 3), name="rgb_input")
    base_rgb = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(rgb_size, rgb_size, 3))
    base_rgb.trainable = False
    rgb_features = base_rgb(rgb_input)
    rgb_features = layers.GlobalAveragePooling2D()(rgb_features)

    # Edge branch
    edge_input = layers.Input(shape=(edge_size, edge_size, 1), name="edge_input")
    x_edge = layers.Conv2D(16, (3,3), activation="relu", padding="same")(edge_input)
    x_edge = layers.MaxPooling2D()(x_edge)
    x_edge = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x_edge)
    x_edge = layers.GlobalAveragePooling2D()(x_edge)

    # Concatenate RGB + Edge
    x = layers.Concatenate()([rgb_features, x_edge])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(2, activation="softmax", dtype="float32")(x)

    model = Model(inputs=[rgb_input, edge_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = create_model_edge()
model.summary()

# =====================================
# Checkpoint & Training
# =====================================
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_hmap_edge_v2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[checkpoint_best]
)

# =====================================
# Evaluation & Save
# =====================================
loss, acc = model.evaluate(test_ds)
print(f"🔥 Final Test Accuracy: {acc:.4f}")

model.save("/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_hmap_edge_v2.keras")
