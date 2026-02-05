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
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError("❌ GPU not found")
print("✅ Found GPU:", device_name)

# Google Drive (slow → only storage)
CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC  = "/drive/MyDrive/HAM10000/images"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

# Local SSD (FAST → training data)
BASE = "/content/newdata"

df = pd.read_csv(CSV_PATH)

df["label"] = df["dx"].apply(
    lambda x: "melanoma" if x == "mel" else "non_melanoma"
)

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

def make_dirs():
    for split in ["train", "valid", "test"]:
        for cls in ["melanoma", "non_melanoma"]:
            os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)

make_dirs()

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

# ==== Replace set_data with tf.data pipeline ====

AUTOTUNE = tf.data.AUTOTUNE
batchSize = 32
image_size = 256

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

    # Cache & prefetch for speed
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

train_path = f"{BASE}/train"
valid_path = f"{BASE}/valid"
test_path  = f"{BASE}/test"

train_ds, val_ds, test_ds = prepare_datasets(train_path, valid_path, test_path, batchSize, image_size)

def unfreeze_model(model, num_layers):
    for layer in model.layers[num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

def create_model():
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(1./255)(inputs)  # Normalize pixel values

    base = EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    base.trainable = False
    base = unfreeze_model(base, -100)

    x = base.output

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1,1,2048))(se)
    se = layers.Dense(85, activation="swish", use_bias=False)(se)
    se = layers.Dense(2048, activation="sigmoid", use_bias=False)(se)
    x  = layers.Multiply()([x, se])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

model = create_model()

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

history = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=[checkpoint_best, checkpoint_all]
)
