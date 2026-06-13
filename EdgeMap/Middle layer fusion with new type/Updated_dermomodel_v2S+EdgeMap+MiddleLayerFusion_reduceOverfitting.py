from google.colab import drive
drive.mount('/drive')

import os
import shutil
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

MODEL_SAVE_PATH = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy/efficientnetv2s_dual_branch.keras"
)

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
# LIGHT AUGMENTATION
# =========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.02),
])

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
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds = prepare_dataset(f"{BASE}/valid", False)
test_ds = prepare_dataset(f"{BASE}/test", False)

# =========================================================
# FOCAL LOSS (IMPORTANT FIX)
# =========================================================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0)

# =========================================================
# MODEL
# =========================================================
def create_dual_model():

    rgb_input = layers.Input(shape=(image_size, image_size, 3))
    edge_input = layers.Input(shape=(image_size, image_size, 1))

    # ================= RGB BRANCH =================
    x_rgb = data_augmentation(rgb_input)
    x_rgb = preprocess_input(x_rgb)

    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )

    # 🔥 stronger fine-tuning
    for layer in base_model.layers[:-160]:
        layer.trainable = False
    for layer in base_model.layers[-160:]:
        layer.trainable = True

    fusion_layer = base_model.get_layer(FUSION_LAYER)

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=fusion_layer.output
    )

    middle_feature = feature_extractor(x_rgb)

    # ================= EDGE BRANCH =================
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

    x = layers.Conv2D(middle_feature.shape[-1], 1, padding="same")(x)

    # ================= FUSION =================
    fused = layers.Concatenate()([middle_feature, x])

    fused = layers.Conv2D(
        256, 3, activation="relu",
        padding="same",
        kernel_regularizer=l2(1e-5)
    )(fused)

    # 🔥 ATTENTION BLOCK (IMPORTANT UPGRADE)
    att = layers.GlobalAveragePooling2D()(fused)
    att = layers.Dense(256, activation="sigmoid")(att)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Concatenate()([fused, att])

    # ================= CLASSIFIER =================
    fused = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(1e-5)
    )(fused)

    fused = layers.Dropout(0.3)(fused)

    outputs = layers.Dense(2, activation="softmax")(fused)

    model = tf.keras.Model(
        inputs=[rgb_input, edge_input],
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model


model = create_dual_model()
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual_{FUSION_LAYER}.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=[checkpoint_best, early_stop]
)

# =========================================================
# EVALUATION
# =========================================================
loss, acc = model.evaluate(test_ds)

print("\n==============================")
print("FINAL RESULT")
print("==============================")
print("Fusion Layer:", FUSION_LAYER)
print(f"Test Accuracy: {acc:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")