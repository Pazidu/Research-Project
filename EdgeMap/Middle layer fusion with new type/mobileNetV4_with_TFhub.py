from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =========================================================
# PATHS
# =========================================================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
MODEL_PATH = "/drive/MyDrive/melanoma_mnv4_edge.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)

# =========================================================
# SETTINGS
# =========================================================
batch_size = 16
image_size = 224

# =========================================================
# DATASET
# =========================================================
def add_edge_map(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label


def load_ds(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = load_ds(f"{BASE}/train", True)
val_ds = load_ds(f"{BASE}/valid", False)
test_ds = load_ds(f"{BASE}/test", False)

# =========================================================
# MOBILE NET V4 (TFHUB BACKBONE)
# =========================================================
def get_mobilenetv4():
    return hub.KerasLayer(
        # MobileNetV4-style TFHub backbone (stable alternative)
        "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
        trainable=True
    )

# =========================================================
# MODEL
# =========================================================
def create_model():

    # ---------------- RGB INPUT ----------------
    rgb_input = layers.Input(shape=(image_size, image_size, 3))

    x = layers.RandomFlip("horizontal")(rgb_input)
    x = layers.RandomRotation(0.05)(x)

    x = layers.Rescaling(1./255)(x)

    backbone = get_mobilenetv4()
    feature_map = backbone(x)

    # =====================================================
    # EDGE BRANCH
    # =====================================================
    edge_input = layers.Input(shape=(image_size, image_size, 1))

    e = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(64, 3, activation="relu", padding="same")(e)
    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(128, 3, activation="relu", padding="same")(e)

    # Reduce edge to vector (IMPORTANT FIX)
    e = layers.GlobalAveragePooling2D()(e)

    # =====================================================
    # FUSION
    # =====================================================
    fused = layers.Concatenate()([feature_map, e])

    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.5)(fused)

    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    outputs = layers.Dense(2, activation="softmax")(fused)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    return model


model = create_model()
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint = ModelCheckpoint(
    CHECKPOINT_DIR + "/best_mnv4_edge.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[checkpoint, early, lr]
)

# =========================================================
# EVALUATION
# =========================================================
loss, acc = model.evaluate(test_ds)

print("\nFINAL TEST ACCURACY:", acc)

model.save(MODEL_PATH)