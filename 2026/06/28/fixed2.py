# =========================
# 1. SETUP
# =========================
from google.colab import drive
drive.mount('/drive')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

tf.random.set_seed(42)
np.random.seed(42)

mixed_precision.set_global_policy("mixed_float16")


# =========================
# 2. PATH (CLEAN STRUCTURE)
# =========================
BASE = "/content/newdata"   # after copying dataset here
IMG_SIZE = 224
BATCH = 16


# =========================
# 3. COPY DATASET (FAST SSD)
# =========================
SRC = "/drive/MyDrive/Colab Notebooks/newdata"

if not os.path.exists(BASE):
    print("Copying dataset to /content ...")
    import shutil
    shutil.copytree(SRC, BASE)
else:
    print("Dataset already exists in /content")


# =========================
# 4. LOAD RAW DATA (IMPORTANT)
# =========================
train_raw = tf.keras.preprocessing.image_dataset_from_directory(
    f"{BASE}/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

val_raw = tf.keras.preprocessing.image_dataset_from_directory(
    f"{BASE}/valid",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

test_raw = tf.keras.preprocessing.image_dataset_from_directory(
    f"{BASE}/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

# Save class names BEFORE transformation (VERY IMPORTANT)
CLASS_NAMES = train_raw.class_names
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
# 6. EDGE MAP
# =========================
def edge_map(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)

    gray = tf.image.rgb_to_grayscale(img)
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


# =========================
# 7. PIPELINE (SAFE ORDER)
# =========================
train_ds = train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# =========================
# 8. MODEL (DUAL BRANCH + CLEAN)
# =========================
def build_model():

    rgb_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    edge_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    base = EfficientNetV2S(include_top=False, weights="imagenet")

    # fine-tune only last layers
    for layer in base.layers[:-100]:
        layer.trainable = False

    x1 = base(rgb_in)
    x1 = layers.GlobalAveragePooling2D()(x1)

    x2 = layers.Conv2D(32, 3, padding="same", activation="relu")(edge_in)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(len(CLASS_NAMES), activation="softmax", dtype="float32")(x)

    return Model([rgb_in, edge_in], out)


model = build_model()


# =========================
# 9. LOSS + OPTIMIZER
# =========================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=0.75
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)


# =========================
# 10. CALLBACKS (STABLE TRAINING)
# =========================
callbacks = [
    ModelCheckpoint("best_model.keras", monitor="val_auc", save_best_only=True, verbose=1),

    EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True),

    ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=3,
        verbose=1
    )
]


# =========================
# 11. TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks
)


# =========================
# 12. TEST EVALUATION
# =========================
print("\nTEST RESULTS:")
model.evaluate(test_ds)


# =========================
# 13. SAVE MODEL
# =========================
model.save("/content/final_model.keras")