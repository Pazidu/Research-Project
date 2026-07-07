from google.colab import drive
drive.mount('/drive')

import os
import shutil
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =========================================================
# SEED (STABILITY)
# =========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# =========================================================
# PATHS
# =========================================================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/best_model_improved.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
batch_size = 16
image_size = 256
FUSION_LAYER = "block4c_add"

# =========================================================
# AUGMENTATION (IMPROVED BUT SAFE)
# =========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])

# =========================================================
# EDGE MAP
# =========================================================
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    rgb = preprocess_input(image)

    return (image, edge), label

# =========================================================
# DATASET
# =========================================================
def prepare_dataset(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds = prepare_dataset(f"{BASE}/valid", False)
test_ds = prepare_dataset(f"{BASE}/test", False)

# =========================================================
# LOSS (GOOD FOR IMBALANCED DATA)
# =========================================================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=[0.65, 0.35]
)

# =========================================================
# MODEL
# =========================================================
def create_model():

    rgb_input = layers.Input(shape=(image_size, image_size, 3))
    edge_input = layers.Input(shape=(image_size, image_size, 1))

    # RGB branch
    x_rgb = data_augmentation(rgb_input)
    x_rgb = preprocess_input(x_rgb)

    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )

    # 🔥 BETTER FINE-TUNING STRATEGY
    base_model.trainable = True
    for layer in base_model.layers[:-120]:
        layer.trainable = False

    fusion_layer = base_model.get_layer(FUSION_LAYER)

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=fusion_layer.output
    )

    middle_feature = feature_extractor(x_rgb)

    # Edge branch
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Resizing(
        middle_feature.shape[1],
        middle_feature.shape[2]
    )(x)

    x = layers.Conv2D(middle_feature.shape[-1], 1)(x)

    # Fusion
    fused = layers.Concatenate()([middle_feature, x])

    fused = layers.Conv2D(
        256, 3,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(1e-4)
    )(fused)

    # Attention
    att = layers.GlobalAveragePooling2D()(fused)
    att = layers.Dense(256, activation="sigmoid")(att)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Concatenate()([fused, att])

    # 🔥 CLASSIFIER (IMPROVED REGULARIZATION)
    fused = layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(fused)
    fused = layers.Dropout(0.5)(fused)

    fused = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(fused)
    fused = layers.Dropout(0.4)(fused)

    outputs = layers.Dense(2, activation="softmax")(fused)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    return model


model = create_model()
model.summary()

# =========================================================
# CALLBACKS (VERY IMPORTANT UPGRADE)
# =========================================================
checkpoint = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_improved.keras",
    monitor="val_auc",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=8,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================
class_weight = {
    0: 2.5,   # melanoma
    1: 1.0    # non_melanoma
}

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[checkpoint, early_stop, lr_scheduler],
)

# =========================================================
# EVALUATION
# =========================================================
results = model.evaluate(test_ds)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")
print("Loss, Accuracy, AUC, Precision, Recall:", results)

# =========================================================
# SAVE
# =========================================================
model.save(MODEL_SAVE_PATH)
print("Model saved!")