# =========================
# 1. SETUP
# =========================
from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

tf.random.set_seed(42)
np.random.seed(42)

print("TF:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# 🔥 SPEED BOOST
mixed_precision.set_global_policy("mixed_float16")


# =========================
# 2. COPY DATASET (FAST SSD)
# =========================
SRC = "/drive/MyDrive/Colab Notebooks/newdata"
DST = "/content/newdata"

if not os.path.exists(DST):
    print("Copying dataset to /content ...")
    shutil.copytree(SRC, DST)
else:
    print("Dataset already exists in /content")


# =========================
# 3. SETTINGS
# =========================
BASE = DST
IMG_SIZE = 224
BATCH = 16
EPOCHS = 25

CLASS_NAMES = None


# =========================
# 4. DATA LOADER (AUTO SPLIT)
# =========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    BASE,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    BASE,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

CLASS_NAMES = train_ds.class_names
print("Classes:", CLASS_NAMES)


# =========================
# 5. AUGMENTATION
# =========================
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# =========================
# 6. EDGE MAP (FAST + STABLE)
# =========================
def edge_map(image):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    gray = tf.image.rgb_to_grayscale(image)
    edge = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(edge), axis=-1))
    edge = tf.image.resize(edge, (IMG_SIZE, IMG_SIZE))

    return edge


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = augment(image)
    image = preprocess_input(image)

    edge = edge_map(image)

    return (image, edge), label


train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# =========================
# 7. MODEL (FAST DUAL BRANCH)
# =========================
def build_model():

    rgb_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    edge_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # RGB backbone
    base = EfficientNetV2S(include_top=False, weights="imagenet")
    x1 = base(rgb_in)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Edge branch (light CNN)
    x2 = layers.Conv2D(32, 3, padding="same", activation="relu")(edge_in)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Fusion
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(len(CLASS_NAMES), activation="softmax", dtype="float32")(x)

    return Model([rgb_in, edge_in], out)


model = build_model()


# =========================
# 8. LOSS
# =========================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=0.75
)


# =========================
# 9. COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()


# =========================
# 10. CALLBACKS
# =========================
ckpt = ModelCheckpoint(
    "/content/best_model.keras",
    monitor="val_auc",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_auc",
    patience=7,
    restore_best_weights=True
)


# =========================
# 11. TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, early]
)


# =========================
# 12. SAVE MODEL
# =========================
model.save("/content/final_model.keras")

# copy back to drive
shutil.copy("/content/final_model.keras", "/drive/MyDrive/final_model.keras")