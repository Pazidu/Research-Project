# ==========================================================
# MELANOMA DETECTION - V3
# EfficientNetV2S
# Balanced Training Dataset
# No Class Weights
# No Edge Branch
# ==========================================================


# ==========================================================
# PART 1 - IMPORTS
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# PART 2 - GPU
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
    except RuntimeError as e:
        print(e)


# ==========================================================
# MIXED PRECISION
# ==========================================================

mixed_precision.set_global_policy("mixed_float16")

print(
    "Mixed precision policy:",
    mixed_precision.global_policy()
)


# ==========================================================
# PART 3 - RANDOM SEED
# ==========================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ==========================================================
# PART 4 - PATHS
# ==========================================================

DATASET = "/content/newdata"

IMG_SRC = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "newdata_backup"
)

CHECKPOINT = (
    "/drive/MyDrive/"
    "checkpoints/"
    "best_model_v3.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "Models/"
    "dermoscopy/"
    "final_model_v3.keras"
)


# ==========================================================
# PART 5 - COPY DATASET TO COLAB
# ==========================================================

print("\n==============================")
print("COPYING DATASET")
print("==============================")

if os.path.exists(DATASET):

    print("Removing old local dataset...")

    shutil.rmtree(DATASET)


print("Copying dataset from Google Drive...")

shutil.copytree(
    IMG_SRC,
    DATASET
)

print("Dataset copied successfully.")


# ==========================================================
# PART 6 - SETTINGS
# ==========================================================

IMG_SIZE = 256

BATCH_SIZE = 8

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# PART 7 - CHECK DATASET
# ==========================================================

def count_images(folder):

    count = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                count += 1

    return count


print("\n==============================")
print("DATASET COUNTS")
print("==============================")


for split in ["train", "valid", "test"]:

    print(f"\n===== {split.upper()} =====")

    split_path = os.path.join(
        DATASET,
        split
    )

    for class_name in sorted(
        os.listdir(split_path)
    ):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if os.path.isdir(class_path):

            print(
                f"{class_name}: "
                f"{count_images(class_path)}"
            )


# ==========================================================
# PART 8 - DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        mode="horizontal_and_vertical"
    ),

    layers.RandomRotation(
        factor=0.15
    ),

    layers.RandomZoom(
        height_factor=(-0.15, 0.15),
        width_factor=(-0.15, 0.15)
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(
        factor=0.15
    )

], name="augmentation")


# ==========================================================
# PART 9 - DATASET LOADER
# ==========================================================

def load_dataset(
    path,
    shuffle=False,
    cache=False
):

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

    # EfficientNetV2 preprocessing
    ds = ds.map(
        lambda images, labels: (
            tf.cast(images, tf.float32),
            labels
        ),
        num_parallel_calls=AUTOTUNE
    )

    if cache:

        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)

    return ds, class_names


# ==========================================================
# PART 10 - LOAD DATASETS
# ==========================================================

print("\n==============================")
print("LOADING DATASETS")
print("==============================")


train_ds, class_names = load_dataset(
    DATASET + "/train",
    shuffle=True,
    cache=False
)


val_ds, val_class_names = load_dataset(
    DATASET + "/valid",
    shuffle=False,
    cache=True
)


test_ds, test_class_names = load_dataset(
    DATASET + "/test",
    shuffle=False,
    cache=True
)


# ==========================================================
# CLASS ORDER CHECK
# ==========================================================

print("\n==============================")
print("CLASS NAMES")
print("==============================")

print("Train:", class_names)
print("Valid:", val_class_names)
print("Test :", test_class_names)


if class_names != val_class_names:

    raise ValueError(
        "Train and validation class ordering differ!"
    )


if class_names != test_class_names:

    raise ValueError(
        "Train and test class ordering differ!"
    )


print("\nClass mapping:")

for i, name in enumerate(class_names):

    print(
        f"Class {i} = {name}"
    )


# ==========================================================
# EXPECTED CLASS ORDER
# ==========================================================

# image_dataset_from_directory normally sorts
# class folders alphabetically.
#
# If your folders are:
#
# melanoma
# non_melanoma
#
# then:
#
# 0 = melanoma
# 1 = non_melanoma
#
# We verify it instead of assuming.

if class_names != [
    "melanoma",
    "non_melanoma"
]:

    print("\nWARNING!")
    print(
        "Class order is not "
        "['melanoma', 'non_melanoma']"
    )

    print(
        "The evaluation code will still "
        "use the detected class order."
    )


# ==========================================================
# PART 11 - LOSS
# ==========================================================

loss_fn = tf.keras.losses.CategoricalCrossentropy()


# ==========================================================
# PART 12 - MODEL CREATION
# ==========================================================

def create_model():

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------

    inputs = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        name="image"
    )


    # ------------------------------------------------------
    # AUGMENTATION
    # ------------------------------------------------------

    x = augmentation(inputs)


    # ------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------

    x = preprocess_input(x)


    # ------------------------------------------------------
    # EFFICIENTNETV2S
    # ------------------------------------------------------

    backbone = EfficientNetV2S(

        include_top=False,

        weights="imagenet",

        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )


    # Stage 1:
    # freeze pretrained backbone

    backbone.trainable = False


    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    x = backbone(
        x,
        training=False
    )


    # ------------------------------------------------------
    # GLOBAL POOLING
    # ------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # ------------------------------------------------------
    # CLASSIFIER
    # ------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu"
    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Dropout(
        0.40
    )(x)


    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    outputs = layers.Dense(
        2,
        activation="softmax",
        dtype="float32",
        name="prediction"
    )(x)


    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetV2S_Melanoma"
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
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
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
# PART 13 - CALLBACKS
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

    patience=5,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


lr_reduce = ReduceLROnPlateau(

    monitor="val_auc",

    factor=0.5,

    patience=2,

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

print("\nClass weights: NONE")

print(
    "Training dataset is already balanced."
)


# ==========================================================
# PART 14 - STAGE 1
# ==========================================================

print("\n==============================")
print("STAGE 1 TRAINING")
print("==============================")


history_stage1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=callbacks,

    verbose=1
)


# ==========================================================
# PART 15 - STAGE 2 FINE TUNING
# ==========================================================

print("\n==============================")
print("STAGE 2 FINE TUNING")
print("==============================")


# ----------------------------------------------------------
# Unfreeze backbone
# ----------------------------------------------------------

backbone.trainable = True


# ----------------------------------------------------------
# Freeze most layers
# ----------------------------------------------------------

FINE_TUNE_LAYERS = 60


for layer in backbone.layers[
    :-FINE_TUNE_LAYERS
]:

    layer.trainable = False


# ----------------------------------------------------------
# Keep BatchNorm frozen
# ----------------------------------------------------------

for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ----------------------------------------------------------
# Count trainable layers
# ----------------------------------------------------------

trainable_count = sum(
    1
    for layer in backbone.layers
    if layer.trainable
)


print(
    "Trainable backbone layers:",
    trainable_count
)


# ==========================================================
# RECOMPILE
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),

    loss=loss_fn,

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ==========================================================
# STAGE 2 TRAINING
# ==========================================================

history_stage2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE2,

    callbacks=callbacks,

    verbose=1
)


# ==========================================================
# PART 16 - LOAD BEST CHECKPOINT
# ==========================================================

print("\n==============================")
print("LOADING BEST MODEL")
print("==============================")


model.load_weights(
    CHECKPOINT
)


print(
    "Best checkpoint loaded."
)


# ==========================================================
# PART 17 - KERAS TEST EVALUATION
# ==========================================================

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")


results = model.evaluate(
    test_ds,
    verbose=1
)


print("\nKeras results:")

for name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


# ==========================================================
# PART 18 - COLLECT TEST PREDICTIONS
# ==========================================================

print("\n==============================")
print("COLLECTING TEST PREDICTIONS")
print("==============================")


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

    # Probability of class 1
    y_prob.extend(
        predictions[:, 1]
    )


y_true = np.array(
    y_true
)

y_prob = np.array(
    y_prob
)


# ==========================================================
# DEFAULT THRESHOLD
# ==========================================================

threshold = 0.50


y_pred = (
    y_prob >= threshold
).astype(int)


# ==========================================================
# PART 19 - TEST METRICS
# ==========================================================

print("\n==============================")
print("TEST METRICS")
print("==============================")


accuracy = accuracy_score(
    y_true,
    y_pred
)


auc = roc_auc_score(
    y_true,
    y_prob
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"AUC       : {auc:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ==========================================================
# PART 20 - CONFUSION MATRIX
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
# SENSITIVITY / SPECIFICITY
# ==========================================================

tn, fp, fn, tp = cm.ravel()


sensitivity = (

    tp /
    (tp + fn)

    if (tp + fn) > 0
    else 0
)


specificity = (

    tn /
    (tn + fp)

    if (tn + fp) > 0
    else 0
)

print(
    f"\nSensitivity: "
    f"{sensitivity:.4f}"
)

print(
    f"Specificity: "
    f"{specificity:.4f}"
)

# ==========================================================
# PART 21 - CLASSIFICATION REPORT
# ==========================================================

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ==========================================================
# PART 22 - VALIDATION PREDICTIONS
# FOR THRESHOLD ANALYSIS
# ==========================================================

val_true = []
val_prob = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    val_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

    val_prob.extend(
        predictions[:, 1]
    )

val_true = np.array(
    val_true
)

val_prob = np.array(
    val_prob
)

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

best_threshold = 0.50
best_f1 = 0.0

for threshold_value in np.arange(
    0.10,
    0.91,
    0.01
):

    temp_pred = (
        val_prob >= threshold_value
    ).astype(int)

    temp_f1 = f1_score(
        val_true,
        temp_pred,
        zero_division=0
    )

    if temp_f1 > best_f1:

        best_f1 = temp_f1

        best_threshold = (
            threshold_value
        )

print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)

# ==========================================================
# TEST WITH OPTIMIZED THRESHOLD
# ==========================================================

optimized_pred = (

    y_prob >= best_threshold

).astype(int)

optimized_accuracy = accuracy_score(
    y_true,
    optimized_pred
)

optimized_precision = precision_score(
    y_true,
    optimized_pred,
    zero_division=0
)

optimized_recall = recall_score(
    y_true,
    optimized_pred,
    zero_division=0
)

optimized_f1 = f1_score(
    y_true,
    optimized_pred,
    zero_division=0
)

optimized_cm = confusion_matrix(
    y_true,
    optimized_pred
)

print(
    f"Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy  : "
    f"{optimized_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{optimized_precision:.4f}"
)

print(
    f"Recall    : "
    f"{optimized_recall:.4f}"
)

print(
    f"F1 Score  : "
    f"{optimized_f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    optimized_cm
)

# ==========================================================
# PART 23 - SAVE MODEL
# ==========================================================

os.makedirs(
    os.path.dirname(MODEL_SAVE),
    exist_ok=True
)

model.save(
    MODEL_SAVE
)

print(
    "\nMODEL SAVED SUCCESSFULLY"
)

print(
    "Path:",
    MODEL_SAVE
)

# ==========================================================
# PART 24 - FINAL SUMMARY
# ==========================================================

print(
    "Classes:",
    class_names
)

print(
    f"Test Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Test AUC: "
    f"{auc:.4f}"
)

print(
    f"Test Precision: "
    f"{precision:.4f}"
)

print(
    f"Test Recall: "
    f"{recall:.4f}"
)

print(
    f"Test F1: "
    f"{f1:.4f}"
)

print(
    f"Test Sensitivity: "
    f"{sensitivity:.4f}"
)

print(
    f"Test Specificity: "
    f"{specificity:.4f}"
)

print(
    f"Optimized Threshold: "
    f"{best_threshold:.2f}"
)
#1277