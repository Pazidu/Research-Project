from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetV2S

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

mixed_precision.set_global_policy('mixed_float16')

# =========================================================
# PARAMETERS
# =========================================================

batch_size = 16
image_size = 224

# =========================================================
# AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# =========================================================
# EDGE FUNCTION (FIXED)
# =========================================================

def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # IMPORTANT FIX

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
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
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds = prepare_dataset(f"{BASE}/valid", False)
test_ds = prepare_dataset(f"{BASE}/test", False)

# =========================================================
# SE ATTENTION MODULE
# =========================================================

def se_block(x, ratio=8):
    filters = x.shape[-1]

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply()([x, se])

# =========================================================
# MODEL
# =========================================================

def create_dual_model():

    # =====================
    # RGB INPUT
    # =====================
    rgb_input = layers.Input(shape=(image_size, image_size, 3), name="rgb_input")

    x_rgb = data_augmentation(rgb_input)
    x_rgb = layers.Rescaling(1./255)(x_rgb)

    base = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=x_rgb
    )

    # Stage 1: freeze backbone
    base.trainable = False

    rgb_features = base.output

    # =====================
    # EDGE BRANCH
    # =====================
    edge_input = layers.Input(shape=(image_size, image_size, 1), name="edge_input")

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(edge_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # resize to match RGB
    x = layers.Resizing(rgb_features.shape[1], rgb_features.shape[2])(x)

    # =====================
    # FUSION
    # =====================
    fused = layers.Concatenate()([rgb_features, x])

    fused = layers.Conv2D(256, 3, padding='same', activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)

    # =====================
    # ATTENTION (IMPORTANT)
    # =====================
    fused = se_block(fused)

    fused = layers.Conv2D(512, 3, padding='same', activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dropout(0.5)(fused)

    outputs = layers.Dense(2, activation='softmax', dtype='float32')(fused)

    model = Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=['accuracy']
    )

    return model

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
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# CREATE MODEL
# =========================================================

model = create_dual_model()
model.summary()

# =========================================================
# STAGE 1 TRAINING (HEAD ONLY)
# =========================================================

print("\n================= STAGE 1 =================")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[checkpoint_best, reduce_lr]
)

# =========================================================
# STAGE 2 FINE TUNING
# =========================================================

print("\n================= STAGE 2 =================")

base_model = model.layers[2]  # EfficientNetV2S layer

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint_best, reduce_lr, early_stop]
)

# =========================================================
# TEST
# =========================================================

print("\n================= TEST =================")

loss, acc = model.evaluate(test_ds)
print("Final Test Accuracy:", acc)

# =========================================================
# SAVE MODEL
# =========================================================

model.save(MODEL_SAVE_PATH)
print("Model Saved!")