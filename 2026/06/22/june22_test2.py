from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE            = "/content/newdata"
IMG_SRC         = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR  = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/efficientnetv2s_fixed.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
mixed_precision.set_global_policy("float32")

BATCH_SIZE = 16
IMAGE_SIZE = 256
EPOCHS = 30

CLASS_ORDER = ["melanoma", "non_melanoma"]


# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.1),
])


# ─────────────────────────────────────────────
# EDGE MAP
# ─────────────────────────────────────────────
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label


# ─────────────────────────────────────────────
# BALANCED TRAIN STREAM
# ─────────────────────────────────────────────
def make_class_stream(path, class_id):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE // 2,
        label_mode="categorical",
        shuffle=True,
        class_names=CLASS_ORDER,
    )

    ds = ds.filter(lambda x, y: tf.equal(tf.argmax(y[0]), class_id))
    ds = ds.repeat()

    ds = ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


mel_stream = make_class_stream(f"{BASE}/train", 0)
nonmel_stream = make_class_stream(f"{BASE}/train", 1)

balanced_ds = tf.data.Dataset.sample_from_datasets(
    [mel_stream, nonmel_stream],
    weights=[0.6, 0.4]
)

balanced_ds = balanced_ds.map(add_edge_map)
balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# VALIDATION / TEST
# ─────────────────────────────────────────────
def eval_ds(path):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=CLASS_ORDER,
    )

    ds = ds.map(add_edge_map)
    return ds.prefetch(tf.data.AUTOTUNE)


val_ds = eval_ds(f"{BASE}/valid")
test_ds = eval_ds(f"{BASE}/test")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def create_model():

    rgb_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    edge_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    base = EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=rgb_input)

    # fine-tune last layers
    for layer in base.layers[:-60]:
        layer.trainable = False

    x1 = base.output

    x2 = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Conv2D(64, 3, activation="relu", padding="same")(x2)
    x2 = layers.Conv2D(x1.shape[-1], 1, padding="same")(x2)

    fused = layers.Concatenate()([x1, x2])
    fused = layers.Conv2D(256, 3, activation="relu", padding="same")(fused)
    fused = layers.GlobalAveragePooling2D()(fused)

    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.5)(fused)

    output = layers.Dense(2, activation="softmax")(fused)

    model = tf.keras.Model([rgb_input, edge_input], output)

    # 🔥 FIXED LOSS (important change)
    loss = tf.keras.losses.CategoricalFocalCrossentropy(
        gamma=2.0,
        alpha=0.75
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(class_id=0, name="melanoma_recall"),
            tf.keras.metrics.Precision(class_id=0, name="melanoma_precision"),
            tf.keras.metrics.Recall(class_id=1, name="non_melanoma_recall"),
        ]
    )

    return model


model = create_model()
model.summary()


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    f"{CHECKPOINT_DIR}/best_model.keras",
    monitor="val_auc",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=10,
    restore_best_weights=True
)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
history = model.fit(
    balanced_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=800,
    callbacks=[checkpoint, early_stop]
)


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
print("\nTEST RESULTS:")
results = model.evaluate(test_ds)

for n, v in zip(model.metrics_names, results):
    print(n, ":", v)


# SAVE MODEL
model.save(MODEL_SAVE_PATH)