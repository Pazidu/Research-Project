from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large  # fallback-safe MobileNetV4 style
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

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
image_size = 224   # IMPORTANT: MobileNet works better at 224

# =========================================================
# DATASET
# =========================================================
def add_edge_map(image, label):

    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

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
    return ds.map(add_edge_map).prefetch(tf.data.AUTOTUNE)

train_ds = load_ds(f"{BASE}/train", True)
val_ds = load_ds(f"{BASE}/valid", False)
test_ds = load_ds(f"{BASE}/test", False)

# =========================================================
# MODEL
# =========================================================
def create_model():

    rgb_input = layers.Input(shape=(image_size, image_size, 3))
    edge_input = layers.Input(shape=(image_size, image_size, 1))

    # ---------------- AUGMENT ----------------
    x = layers.RandomFlip("horizontal")(rgb_input)
    x = layers.RandomRotation(0.05)(x)

    x = preprocess_input(x)

    # =====================================================
    # BACKBONE (MobileNetV3Large ~ MobileNetV4 alternative)
    # =====================================================
    backbone = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    backbone.trainable = False

    feature_map = backbone.output

    # =====================================================
    # EDGE BRANCH
    # =====================================================
    e = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(64, 3, activation="relu", padding="same")(e)
    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(128, 3, activation="relu", padding="same")(e)

    # match feature size
    e = layers.Resizing(feature_map.shape[1], feature_map.shape[2])(e)
    e = layers.Conv2D(feature_map.shape[-1], 1, padding="same")(e)

    # =====================================================
    # FUSION
    # =====================================================
    fused = layers.Concatenate()([feature_map, e])

    # attention gate
    att = layers.GlobalAveragePooling2D()(fused)
    att = layers.Dense(256, activation="relu")(att)
    att = layers.Dense(fused.shape[-1], activation="sigmoid")(att)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Multiply()([fused, att])

    # =====================================================
    # CLASSIFIER
    # =====================================================
    x = layers.Dense(256, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

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