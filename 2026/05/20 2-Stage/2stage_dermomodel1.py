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

MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/final_model.keras"

# =========================================================
# COPY DATASET
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
# LIGHT DATA AUGMENTATION
# =========================================================

augment = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

])

# =========================================================
# EDGE MAP FUNCTION
# =========================================================

def add_edge_map(image, label):

    image = tf.cast(image, tf.float32)

    # Augmentation
    image = augment(image)

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Grayscale
    gray = tf.image.rgb_to_grayscale(image)

    # Sobel edges
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))

    # Normalize safely
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label

# =========================================================
# DATASET
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

    # RGB input
    rgb_input = layers.Input(
        shape=(image_size, image_size, 3),
        name="rgb_input"
    )

    base = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=rgb_input
    )

    base.trainable = True

    # Freeze early layers
    for layer in base.layers[:-50]:
        layer.trainable = False

    # Middle feature map
    middle_layer = base.get_layer("block4c_add").output

    # =====================================================
    # EDGE BRANCH
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

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        64,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        128,
        3,
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        256,
        3,
        activation='relu',
        padding='same'
    )(x)

    # Resize
    target_size = middle_layer.shape[1:3]

    x = layers.Resizing(
        target_size[0],
        target_size[1]
    )(x)

    # =====================================================
    # FUSION
    # =====================================================

    fused = layers.Concatenate()([middle_layer, x])

    # =====================================================
    # POST FUSION CNN
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
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-4),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05
    ),

    metrics=['accuracy']
)

# =========================================================
# CALLBACKS
# =========================================================

checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_model.keras",
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
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# TRAIN
# =========================================================

print("\n==============================")
print("TRAINING")
print("==============================\n")

history = model.fit(

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
# LOAD BEST MODEL
# =========================================================

model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/best_model.keras"
)

# =========================================================
# TEST
# =========================================================

print("\n==============================")
print("TEST EVALUATION")
print("==============================\n")

loss, acc = model.evaluate(test_ds)

print(f"\nFinal Test Accuracy: {acc:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================

model.save(MODEL_SAVE_PATH)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")