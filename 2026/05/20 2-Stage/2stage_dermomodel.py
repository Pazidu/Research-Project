from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# =========================================================
# PATHS
# =========================================================

BASE = "/content/newdata"

IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"

CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_2stage_middlefusion.keras"

# =========================================================
# COPY DATASET TO COLAB SSD
# =========================================================

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================================
# MIXED PRECISION
# =========================================================

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# =========================================================
# PARAMETERS
# =========================================================

batch_size = 16
image_size = 256
num_classes = 2

# =========================================================
# EDGE MAP FUNCTION
# =========================================================

def add_edge_map(image, label):

    image = tf.cast(image, tf.float32)

    # RGB preprocessing
    image = preprocess_input(image)

    # Convert to grayscale
    gray = tf.image.rgb_to_grayscale(image)

    # Sobel edge detection
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))

    # Normalize
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label

# =========================================================
# DATASET PREPARATION
# =========================================================

def prepare_dataset(path, shuffle):

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

train_ds = prepare_dataset(f"{BASE}/train", True)

val_ds = prepare_dataset(f"{BASE}/valid", False)

test_ds = prepare_dataset(f"{BASE}/test", False)

# =========================================================
# MODEL
# =========================================================

def create_dual_model():

    # =====================================================
    # RGB INPUT
    # =====================================================

    rgb_input = layers.Input(
        shape=(image_size, image_size, 3),
        name="rgb_input"
    )

    base = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=rgb_input
    )

    # Freeze most layers for Stage 1
    base.trainable = True

    for layer in base.layers[:-50]:
        layer.trainable = False

    # =====================================================
    # MIDDLE FEATURE MAP
    # =====================================================

    middle_layer = base.get_layer("block4c_add").output

    # =====================================================
    # EDGE INPUT
    # =====================================================

    edge_input = layers.Input(
        shape=(image_size, image_size, 1),
        name="edge_input"
    )

    x = layers.Conv2D(
        32,
        3,
        activation='relu',
        padding='same'
    )(edge_input)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        64,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        128,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        256,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)

    # =====================================================
    # RESIZE EDGE FEATURES
    # =====================================================

    target_size = middle_layer.shape[1:3]

    x = layers.Resizing(
        target_size[0],
        target_size[1]
    )(x)

    # =====================================================
    # MIDDLE FUSION
    # =====================================================

    fused = layers.Concatenate()([middle_layer, x])

    # =====================================================
    # CNN AFTER FUSION
    # =====================================================

    fused = layers.Conv2D(
        256,
        3,
        activation='relu',
        padding='same'
    )(fused)

    fused = layers.BatchNormalization()(fused)

    fused = layers.MaxPooling2D(2)(fused)

    fused = layers.Dropout(0.3)(fused)

    fused = layers.Conv2D(
        512,
        3,
        activation='relu',
        padding='same'
    )(fused)

    fused = layers.BatchNormalization()(fused)

    fused = layers.GlobalAveragePooling2D()(fused)

    fused = layers.Dropout(0.5)(fused)

    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        dtype='float32'
    )(fused)

    model = tf.keras.Model(
        inputs=[rgb_input, edge_input],
        outputs=outputs
    )

    return model

# =========================================================
# CREATE MODEL
# =========================================================

model = create_dual_model()

model.summary()

# =========================================================
# STAGE 1 COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# CALLBACKS
# =========================================================

checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_stage1.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# STAGE 1 TRAINING
# =========================================================

print("\n==============================")
print("STAGE 1 TRAINING")
print("==============================\n")

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        checkpoint_best,
        reduce_lr,
        early_stop
    ]
)

# =========================================================
# LOAD BEST STAGE 1 MODEL
# =========================================================

model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/best_stage1.keras"
)

# =========================================================
# STAGE 2 FINE-TUNING
# =========================================================

# =========================================================
# STAGE 2 FINE-TUNING
# =========================================================

print("\n==============================")
print("STAGE 2 FINE-TUNING")
print("==============================\n")

# First freeze everything
for layer in model.layers:
    layer.trainable = False

# Unfreeze ONLY last EfficientNet layers
for layer in model.layers[-40:]:
    layer.trainable = True

# Recompile with VERY LOW learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=['accuracy']
)

# =========================================================
# STAGE 2 CHECKPOINT
# =========================================================

checkpoint_stage2 = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_stage2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================================================
# STAGE 2 TRAINING
# =========================================================

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[
        checkpoint_stage2,
        reduce_lr,
        early_stop
    ]
)

# =========================================================
# LOAD BEST FINAL MODEL
# =========================================================

model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/best_stage2.keras"
)

# =========================================================
# EVALUATION
# =========================================================

print("\n==============================")
print("TEST EVALUATION")
print("==============================\n")

loss, acc = model.evaluate(test_ds)

print(f"\nFinal Test Accuracy: {acc:.4f}")

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(MODEL_SAVE_PATH)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")