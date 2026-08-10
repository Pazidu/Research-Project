# ==========================================================
# PART 1 - IMPORTS, SETTINGS, DATASET
# ==========================================================

from google.colab import drive
drive.mount('/drive')

import os
import random
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)

# ==========================================================
# GPU MEMORY GROWTH
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================================================
# MIXED PRECISION
# ==========================================================

mixed_precision.set_global_policy("mixed_float16")

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==========================================================
# PATHS
# ==========================================================

DATASET = "/content/newdata"

IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata_backup"

CHECKPOINT = "/drive/MyDrive/checkpoints/best_model_v2.keras"

MODEL_SAVE = (
    "/drive/MyDrive/Colab Notebooks/"
    "Models/dermoscopy/final_model_v2.keras"
)

# ==========================================================
# COPY DATASET TO COLAB
# ==========================================================

if os.path.exists(DATASET):
    shutil.rmtree(DATASET)

shutil.copytree(
    IMG_SRC,
    DATASET
)

# ==========================================================
# SETTINGS
# ==========================================================

IMG_SIZE = 300

BATCH_SIZE = 8

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 20

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([
    
    layers.RandomFlip(
        "horizontal_and_vertical"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        height_factor=(-0.10, 0.15),
        width_factor=(-0.10, 0.15)
    ),

    layers.RandomContrast(
        0.15
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

], name="augmentation")

# ==========================================================
# DATASET LOADER
# ==========================================================

def load_dataset(path, shuffle):

    ds = tf.keras.utils.image_dataset_from_directory(

        path,

        image_size=(
            IMG_SIZE,
            IMG_SIZE
        ),

        batch_size=BATCH_SIZE,

        label_mode="categorical",

        shuffle=shuffle,

        seed=SEED
    )

    class_names = ds.class_names

    return ds, class_names


# ==========================================================
# LOAD DATASETS
# ==========================================================

train_raw, class_names = load_dataset(
    DATASET + "/train",
    True
)

val_raw, _ = load_dataset(
    DATASET + "/valid",
    False
)

test_raw, _ = load_dataset(
    DATASET + "/test",
    False
)

print("\nCLASS NAMES:")
print(class_names)

# ==========================================================
# CREATE EDGE MAP
# ==========================================================

def create_edge(image):

    image = tf.cast(
        image,
        tf.float32
    )

    gray = tf.image.rgb_to_grayscale(
        image
    )

    sobel = tf.image.sobel_edges(
        gray
    )

    edge = tf.sqrt(
        tf.reduce_sum(
            tf.square(sobel),
            axis=-1
        )
    )

    # Normalize each image
    edge_max = tf.reduce_max(
        edge,
        axis=[1, 2, 3],
        keepdims=True
    )

    edge = edge / (
        edge_max + 1e-7
    )

    edge = tf.clip_by_value(
        edge,
        0.0,
        1.0
    )

    return edge


# ==========================================================
# TRAIN DATA PIPELINE
# ==========================================================

def prepare_train(images, labels):

    # Apply augmentation FIRST
    images = augmentation(
        images,
        training=True
    )

    # Create edge from the SAME augmented image
    edges = create_edge(
        images
    )

    return (
        {
            "rgb": images,
            "edge": edges
        },
        labels
    )


# ==========================================================
# VALIDATION / TEST PIPELINE
# ==========================================================

def prepare_eval(images, labels):

    # No augmentation
    edges = create_edge(
        images
    )

    return (
        {
            "rgb": images,
            "edge": edges
        },
        labels
    )


# ==========================================================
# BUILD DATASETS
# ==========================================================

train_ds = train_raw.map(
    prepare_train,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_raw.map(
    prepare_eval,
    num_parallel_calls=AUTOTUNE
)

test_ds = test_raw.map(
    prepare_eval,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)

# ==========================================================
# LOSS
# ==========================================================

loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.02
)

# ==========================================================
# MODEL CREATION
# ==========================================================

def create_model():

    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    rgb_input = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        name="rgb"
    )

    edge_input = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            1
        ),
        name="edge"
    )

    # ------------------------------------------------------
    # RGB BRANCH
    # ------------------------------------------------------

    x = preprocess_input(
        rgb_input
    )

    backbone = EfficientNetV2S(

        include_top=False,

        weights="imagenet",

        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),

        pooling=None
    )

    # Stage 1
    backbone.trainable = False

    features = backbone(
        x,
        training=False
    )

    # ------------------------------------------------------
    # EDGE BRANCH
    # ------------------------------------------------------

    e = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu"
    )(edge_input)

    e = layers.BatchNormalization()(e)

    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu"
    )(e)

    e = layers.BatchNormalization()(e)

    e = layers.MaxPooling2D()(e)

    e = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu"
    )(e)

    e = layers.BatchNormalization()(e)

    # ------------------------------------------------------
    # RESIZE EDGE FEATURES
    # ------------------------------------------------------

    e = layers.Resizing(
        features.shape[1],
        features.shape[2]
    )(e)

    e = layers.Conv2D(
        features.shape[-1],
        1,
        padding="same",
        activation="relu"
    )(e)

    # ------------------------------------------------------
    # FUSION
    # ------------------------------------------------------

    fused = layers.Concatenate()([
        features,
        e
    ])

    fused = layers.GlobalAveragePooling2D()(
        fused
    )

    fused = layers.Dense(
        512,
        activation="relu"
    )(fused)

    fused = layers.BatchNormalization()(
        fused
    )

    fused = layers.Dropout(
        0.4
    )(fused)

    output = layers.Dense(
        2,
        activation="softmax",
        dtype="float32"
    )(fused)

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = Model(
        inputs=[
            rgb_input,
            edge_input
        ],
        outputs=output
    )

    # ------------------------------------------------------
    # COMPILE
    # ------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss=loss_fn,

        metrics=[
            "accuracy",

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model, backbone


# ==========================================================
# CREATE MODEL
# ==========================================================

model, backbone = create_model()

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = ModelCheckpoint(

    filepath=CHECKPOINT,

    monitor="val_auc",

    save_best_only=True,

    mode="max",

    verbose=1
)

early_stop = EarlyStopping(

    monitor="val_auc",

    patience=7,

    mode="max",

    restore_best_weights=True,

    verbose=1
)

lr_reduce = ReduceLROnPlateau(

    monitor="val_auc",

    factor=0.5,

    patience=3,

    min_lr=1e-7,

    verbose=1
)

callbacks = [
    checkpoint,
    early_stop,
    lr_reduce
]

# ==========================================================
# IMPORTANT:
# NO CLASS WEIGHTS
# ==========================================================

print("\nTraining dataset is balanced.")

print(
    "melanoma:     7122"
)

print(
    "non_melanoma: 7122"
)

print(
    "\nClass weights DISABLED."
)

# ==========================================================
# STAGE 1
# FROZEN EFFICIENTNET
# ==========================================================

print("\n==============================")
print("STAGE 1 TRAINING")
print("==============================")

history_stage1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=callbacks
)

# ==========================================================
# STAGE 2
# FINE TUNING
# ==========================================================

print("\n==============================")
print("STAGE 2 FINE TUNING")
print("==============================")

# Unfreeze backbone
backbone.trainable = True

# Freeze most layers
for layer in backbone.layers[:-100]:

    layer.trainable = False

# Keep BatchNorm frozen
for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ----------------------------------------------------------
# RECOMPILE WITH SMALL LR
# ----------------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),

    loss=loss_fn,

    metrics=[
        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)

# ----------------------------------------------------------
# TRAIN
# ----------------------------------------------------------

history_stage2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE2,

    callbacks=callbacks
)

# ==========================================================
# LOAD BEST CHECKPOINT
# ==========================================================

print(
    "\nLoading best checkpoint..."
)

model.load_weights(
    CHECKPOINT
)

# ==========================================================
# TEST RESULTS
# ==========================================================

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

results = model.evaluate(
    test_ds,
    verbose=1
)

print(
    "\nLoss, Accuracy, AUC:"
)

print(results)

# ==========================================================
# GET TEST PREDICTIONS
# ==========================================================

y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

    y_prob.extend(
        predictions[:, 0]
    )


y_true = np.array(
    y_true
)

y_prob = np.array(
    y_prob
)

# ==========================================================
# STANDARD THRESHOLD
# ==========================================================

y_pred = (
    y_prob >= 0.5
).astype(int)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ==========================================================
# TEST AUC
# ==========================================================

test_auc = roc_auc_score(
    y_true,
    y_prob
)

print(
    "\nTest AUC:",
    test_auc
)

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    MODEL_SAVE
)

print(
    "\nMODEL SAVED SUCCESSFULLY"
)

print(
    MODEL_SAVE
)