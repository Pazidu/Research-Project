from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ---------------------------
# Clear previous session (important)
# ---------------------------
tf.keras.backend.clear_session()

print("TensorFlow:", tf.__version__)

# ---------------------------
# Enable GPU memory growth
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# ---------------------------
# Enable Mixed Precision (BIG memory saver)
# ---------------------------
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision enabled")

# ---------------------------
# Paths
# ---------------------------
CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC  = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
BASE = "/content/newdata"

# Clean SSD
if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)

# ---------------------------
# Parameters (memory optimized)
# ---------------------------
AUTOTUNE = tf.data.AUTOTUNE
batchSize = 8         # 🔥 Reduced (very important)
image_size = 256      # Keep for B5

# ---------------------------
# Dataset Loader (NO CACHE)
# ---------------------------
def prepare_datasets(train_path, valid_path, test_path, batch_size, img_size):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        label_mode='categorical',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path,
        label_mode='categorical',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    # ✅ Only prefetch (NO cache)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

train_path = f"{BASE}/train"
valid_path = f"{BASE}/valid"
test_path  = f"{BASE}/test"

train_ds, val_ds, test_ds = prepare_datasets(
    train_path, valid_path, test_path, batchSize, image_size
)

# ---------------------------
# Model
# ---------------------------
def create_model():

    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Proper EfficientNet preprocessing
    x = preprocess_input(inputs)

    base = EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # 🔥 Freeze most layers (memory safe)
    for layer in base.layers[:-20]:
        layer.trainable = False

    for layer in base.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    x = base.output

    # Channel Attention (lighter version)
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1,1,2048))(se)
    se = layers.Dense(256, activation="swish", use_bias=False)(se)
    se = layers.Dense(2048, activation="sigmoid", use_bias=False)(se)
    x  = layers.Multiply()([x, se])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output must be float32 when using mixed precision
    outputs = layers.Dense(2, activation="softmax", dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

model = create_model()

# ---------------------------
# Checkpoints
# ---------------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_best = ModelCheckpoint(
    f"{CHECKPOINT_DIR}/best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

checkpoint_all = ModelCheckpoint(
    f"{CHECKPOINT_DIR}/epoch_{{epoch:02d}}.weights.h5",
    save_weights_only=True
)

# ---------------------------
# Training
# ---------------------------
history = model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
    callbacks=[checkpoint_best, checkpoint_all]
)
