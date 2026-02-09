from google.colab import drive
drive.mount('/drive')

import os
import shutil
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ===============================
# Mixed Precision
# ===============================
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# ===============================
# Paths
# ===============================
CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
BASE = "/content/newdata"

# ===============================
# Load CSV & Labels
# ===============================
df = pd.read_csv(CSV_PATH)
df["label"] = df["dx"].apply(lambda x: "melanoma" if x == "mel" else "non_melanoma")

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# ===============================
# Folder Structure
# ===============================
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

# ===============================
# Edge + H-map inspired preprocessing
# ===============================
def edge_hmap_preprocess(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0

    # Sobel edge detection
    sobel = tf.image.sobel_edges(image)
    edge_mag = tf.sqrt(
        tf.reduce_sum(tf.square(sobel), axis=-1)
    )
    edge_mag = tf.reduce_mean(edge_mag, axis=-1, keepdims=True)

    # Normalize edge map
    edge_mag = tf.clip_by_value(edge_mag, 0.0, 1.0)

    # H-map inspired weighting (emphasize borders)
    edge_mag = edge_mag * 1.5

    # Concatenate RGB + Edge → 4 channels
    image_4ch = tf.concat([image, edge_mag], axis=-1)

    return image_4ch, label

# ===============================
# Dataset Loader
# ===============================
def prepare_datasets(train_path, valid_path, test_path, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=(256, 256),
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path,
        image_size=(256, 256),
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=(256, 256),
        batch_size=batch_size,
        label_mode='categorical'
    )

    train_ds = train_ds.map(edge_hmap_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(edge_hmap_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(edge_hmap_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
    )

# ===============================
# Model (4-channel EfficientNet)
# ===============================
def create_model():
    inputs = layers.Input(shape=(256, 256, 4))

    # Convert 4 → 3 channels for EfficientNet
    x = layers.Conv2D(
        3, (1, 1), padding="same", name="rgb_adapter"
    )(inputs)

    base = EfficientNetV2S(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )

    base.trainable = False

    # Unfreeze last layers
    for layer in base.layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        2, activation="softmax", dtype="float32"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ===============================
# Training
# ===============================
batch_size = 16
EPOCHS = 25

train_ds, val_ds, test_ds = prepare_datasets(
    f"{BASE}/train",
    f"{BASE}/valid",
    f"{BASE}/test",
    batch_size
)

model = create_model()
model.summary()

CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_hmap_edge.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint]
)

# ===============================
# Evaluation
# ===============================
loss, acc = model.evaluate(test_ds)
print(f"🔥 Final Test Accuracy: {acc:.4f}")

model.save(
    "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_hmap_edge.keras"
)
