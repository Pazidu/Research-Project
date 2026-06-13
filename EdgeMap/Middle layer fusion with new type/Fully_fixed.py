# UNFREEZE last 30% backbone (VERY important)
# REMOVE noisy edge normalization bug
# Add attention fusion (lightweight SE block)
# Better augmentation
# Stronger regularization but controlled
# Fix learning rate schedule

from google.colab import drive
drive.mount('/drive')

import os, shutil, tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ======================
# PATHS
# ======================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
MODEL_PATH = "/drive/MyDrive/final_mnv4_edge.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ======================
# SETTINGS
# ======================
image_size = 224
batch_size = 16

# ======================
# EDGE MAP FIX (IMPORTANT)
# ======================
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)

    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))

    # FIX: stable normalization (your previous version caused instability)
    edge = (edge - tf.reduce_min(edge)) / (tf.reduce_max(edge) - tf.reduce_min(edge) + 1e-6)

    return (image, edge), label


def load_ds(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle
    )
    return ds.map(add_edge_map).prefetch(tf.data.AUTOTUNE)

train_ds = load_ds(f"{BASE}/train", True)
val_ds   = load_ds(f"{BASE}/valid", False)
test_ds  = load_ds(f"{BASE}/test", False)

# ======================
# MODEL
# ======================
def create_model():

    rgb_input = layers.Input(shape=(image_size, image_size, 3))
    edge_input = layers.Input(shape=(image_size, image_size, 1))

    # augmentation
    x = layers.RandomFlip("horizontal")(rgb_input)
    x = layers.RandomRotation(0.05)(x)
    x = preprocess_input(x)

    # backbone
    backbone = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # 🔥 IMPORTANT FIX: partial fine-tuning
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False

    feature_map = backbone.output

    # ======================
    # EDGE BRANCH
    # ======================
    e = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    e = layers.MaxPooling2D()(e)
    e = layers.Conv2D(64, 3, activation="relu", padding="same")(e)
    e = layers.MaxPooling2D()(e)
    e = layers.Conv2D(128, 3, activation="relu", padding="same")(e)

    e = layers.Resizing(feature_map.shape[1], feature_map.shape[2])(e)
    e = layers.Conv2D(feature_map.shape[-1], 1, padding="same")(e)

    # ======================
    # FUSION (SE ATTENTION)
    # ======================
    fused = layers.Concatenate()([feature_map, e])

    se = layers.GlobalAveragePooling2D()(fused)
    se = layers.Dense(128, activation="relu")(se)
    se = layers.Dense(fused.shape[-1], activation="sigmoid")(se)

    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Multiply()([fused, se])

    # ======================
    # CLASSIFIER
    # ======================
    x = layers.Dense(256, activation="relu")(fused)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    return model

model = create_model()

# ======================
# CALLBACKS
# ======================
callbacks = [
    ModelCheckpoint(
        CHECKPOINT_DIR + "/best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

# ======================
# TRAIN
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=callbacks
)

# ======================
# TEST
# ======================
loss, acc = model.evaluate(test_ds)
print("TEST ACCURACY:", acc)

model.save(MODEL_PATH)