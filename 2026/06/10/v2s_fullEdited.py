from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S, ConvNeXtTiny
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =========================================================
# PATHS
# =========================================================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

MODEL_SAVE_PATH = "/drive/MyDrive/dermoscopy_sota_model.keras"

# =========================================================
# DATA
# =========================================================
if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)

batch_size = 16
image_size = 256

# =========================================================
# AUGMENTATION (LIGHT BUT EFFECTIVE)
# =========================================================
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
])

# =========================================================
# DATASET
# =========================================================
def load(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle
    )
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = load(f"{BASE}/train", True)
val_ds = load(f"{BASE}/valid", False)
test_ds = load(f"{BASE}/test", False)

# =========================================================
# LOSS (SOTA)
# =========================================================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    label_smoothing=0.05
)

# =========================================================
# MODEL
# =========================================================
def create_sota_model():

    inputs = layers.Input(shape=(image_size, image_size, 3))

    # ---------------- RGB AUGMENT ----------------
    x = augment(inputs)
    x = preprocess_input(x)

    # =====================================================
    # BACKBONE 1 - EfficientNetV2S
    # =====================================================
    eff = EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=x)

    for layer in eff.layers[:-200]:
        layer.trainable = False

    eff_feat = eff.output

    # =====================================================
    # BACKBONE 2 - ConvNeXt Tiny
    # =====================================================
    cx = ConvNeXtTiny(include_top=False, weights="imagenet", input_tensor=x)

    for layer in cx.layers[:-120]:
        layer.trainable = False

    cx_feat = cx.output

    # =====================================================
    # FEATURE FUSION
    # =====================================================
    fused = layers.Concatenate()([eff_feat, cx_feat])

    fused = layers.Conv2D(512, 1, activation="relu")(fused)

    # =====================================================
    # ATTENTION BLOCK (SE STYLE)
    # =====================================================
    se = layers.GlobalAveragePooling2D()(fused)
    se = layers.Dense(256, activation="relu")(se)
    se = layers.Dense(512, activation="sigmoid")(se)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Multiply()([fused, se])

    # =====================================================
    # CLASSIFIER HEAD
    # =====================================================
    x = layers.Dense(256, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model

model = create_sota_model()
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint = ModelCheckpoint(
    CHECKPOINT_DIR + "/best_sota.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=[checkpoint, early, lr]
)

# =========================================================
# EVALUATION
# =========================================================
loss, acc = model.evaluate(test_ds)

print("\n====================")
print("FINAL RESULTS")
print("====================")
print("Test Accuracy:", acc)