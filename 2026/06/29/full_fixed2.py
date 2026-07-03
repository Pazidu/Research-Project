from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =========================================================
# PATHS
# =========================================================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_v2.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
BATCH_SIZE = 16
IMAGE_SIZE = 256
FUSION_LAYER = "block4c_add"   # CHANGE THIS LATER FOR EXPERIMENTS
EPOCHS = 30

# =========================================================
# LIGHT AUGMENTATION (SAFE)
# =========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.02),
], name="augmentation")

# =========================================================
# DATASET
# =========================================================
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    rgb = preprocess_input(image)

    return (rgb, edge), label


def prepare_dataset(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds = prepare_dataset(f"{BASE}/valid", False)
test_ds = prepare_dataset(f"{BASE}/test", False)

# =========================================================
# MODEL
# =========================================================
def create_model():

    rgb_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    edge_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # ================= RGB BRANCH =================
    x_rgb = data_augmentation(rgb_input)
    x_rgb = preprocess_input(x_rgb)

    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # fine-tuning (safe version)
    for layer in base_model.layers[:-160]:
        layer.trainable = False
    for layer in base_model.layers[-160:]:
        layer.trainable = True

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(FUSION_LAYER).output
    )

    rgb_features = feature_extractor(x_rgb)

    # ================= EDGE BRANCH =================
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Resizing(
        rgb_features.shape[1],
        rgb_features.shape[2]
    )(x)

    x = layers.Conv2D(rgb_features.shape[-1], 1, padding="same")(x)

    # ================= FUSION =================
    fused = layers.Concatenate()([rgb_features, x])

    fused = layers.Conv2D(
        256, 3, activation="relu",
        padding="same",
        kernel_regularizer=l2(1e-5)
    )(fused)

    fused = layers.BatchNormalization()(fused)

    # ================= SIMPLE ATTENTION =================
    att = layers.GlobalAveragePooling2D()(fused)
    att = layers.Dense(256, activation="sigmoid")(att)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Multiply()([fused, att])

    # ================= CLASSIFIER =================
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-5))(fused)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


model = create_model()
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_v2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stop]
)

# =========================================================
# EVALUATION
# =========================================================
results = model.evaluate(test_ds, return_dict=True)

print("\n==================== FINAL RESULTS ====================")
print(f"Fusion Layer   : {FUSION_LAYER}")
print(f"Accuracy       : {results['accuracy']:.4f}")
print(f"AUC            : {results['auc']:.4f}")
print(f"Precision      : {results['precision']:.4f}")
print(f"Recall         : {results['recall']:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================
model.save(MODEL_SAVE_PATH)
print("Model saved!")


# Training accuracy (Epoch 20)	96.38%
# Best validation accuracy	91.51%