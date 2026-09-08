# ============================================================
# V29 MELANOMA CLASSIFICATION
# EfficientNetV2-S
#
# CONTROLLED EXPERIMENT:
#
# V25:
#     FINE_TUNE_FRACTION = 0.40
#
# V29:
#     FINE_TUNE_FRACTION = 0.60
#
# ALL OTHER V25 SETTINGS ARE PRESERVED.
# ============================================================


# ============================================================
# CELL 1 - INSTALL / CHECK ENVIRONMENT
# ============================================================

# !pip install -q tensorflow==2.20.0 keras==3.13.2 scikit-learn seaborn

import tensorflow as tf
import keras

print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)

print("\nGPU:")
print(tf.config.list_physical_devices("GPU"))

if tf.config.list_physical_devices("GPU"):
    print("✅ GPU available")
else:
    print("⚠️ WARNING: GPU not available")


# ============================================================
# CELL 2 - IMPORTS
# ============================================================

import os
import shutil
import random
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CELL 3 - RANDOM SEEDS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

print("Random seed:", SEED)


# ============================================================
# CELL 4 - GOOGLE DRIVE
# ============================================================

from google.colab import drive

drive.mount("/content/drive")


# ============================================================
# CELL 5 - DATASET CONFIGURATION
# ============================================================

SOURCE_DATASET = Path(
    "/content/drive/MyDrive/Colab Notebooks/newdata_backup"
)

DATASET_DIR = Path(
    "/content/newdata"
)

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

CLASS_NAMES = [
    "non_melanoma",
    "melanoma"
]

CLASS_TO_INDEX = {
    "non_melanoma": 0,
    "melanoma": 1
}

IMAGE_SIZE = (300, 300)
IMG_HEIGHT = 300
IMG_WIDTH = 300

BATCH_SIZE = 4

print("=" * 70)
print("V29 DATASET CONFIGURATION")
print("=" * 70)

print("Source dataset :", SOURCE_DATASET)
print("Local dataset  :", DATASET_DIR)
print("Image size     :", IMAGE_SIZE)
print("Batch size     :", BATCH_SIZE)
print("Classes        :", CLASS_NAMES)


# ============================================================
# CELL 6 - COPY DATASET TO LOCAL COLAB STORAGE
# ============================================================

print("=" * 70)
print("PREPARING LOCAL DATASET")
print("=" * 70)

if not SOURCE_DATASET.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_DATASET}"
    )

if DATASET_DIR.exists():
    print("Removing existing local dataset...")
    shutil.rmtree(DATASET_DIR)

print("Copying dataset from Google Drive...")

shutil.copytree(
    SOURCE_DATASET,
    DATASET_DIR
)

print("✅ Dataset copied successfully")
print(DATASET_DIR)


# ============================================================
# CELL 7 - DATASET VERIFICATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder):

    if not folder.exists():
        return []

    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def count_images(folder):

    return len(get_images(folder))


print("=" * 70)
print("DATASET COUNTS")
print("=" * 70)

dataset_counts = {}

for split_name, split_dir in [
    ("train", TRAIN_DIR),
    ("valid", VALID_DIR),
    ("test", TEST_DIR)
]:

    dataset_counts[split_name] = {}

    print(f"\n{split_name.upper()}")

    for class_name in CLASS_NAMES:

        folder = split_dir / class_name

        count = count_images(folder)

        dataset_counts[split_name][class_name] = count

        print(
            f"  {class_name:<15}: {count}"
        )


# ============================================================
# CELL 8 - EXPECTED DATASET COUNTS
# ============================================================

print("=" * 70)
print("VERIFYING DATASET")
print("=" * 70)

expected_counts = {
    "train": {
        "non_melanoma": 7122,
        "melanoma": 7122
    },
    "valid": {
        "non_melanoma": 890,
        "melanoma": 111
    },
    "test": {
        "non_melanoma": 890,
        "melanoma": 112
    }
}

verification_passed = True

for split in expected_counts:

    for class_name in expected_counts[split]:

        expected = expected_counts[split][class_name]
        actual = dataset_counts[split][class_name]

        if actual != expected:

            print(
                f"❌ {split}/{class_name}: "
                f"expected {expected}, got {actual}"
            )

            verification_passed = False

        else:

            print(
                f"✅ {split}/{class_name}: {actual}"
            )

if not verification_passed:

    print("\n⚠️ Dataset count mismatch detected.")

else:

    print("\n✅ Dataset verification passed.")


# ============================================================
# CELL 9 - DATASET SIZE SUMMARY
# ============================================================

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

for split in ["train", "valid", "test"]:

    total = sum(
        dataset_counts[split].values()
    )

    print(
        f"{split:<10}: {total} images"
    )

print()
print(
    "Training melanoma ratio:",
    dataset_counts["train"]["melanoma"] /
    dataset_counts["train"]["non_melanoma"]
)


# ============================================================
# CELL 10 - CREATE TF.DATA DATASETS
# ============================================================

print("=" * 70)
print("CREATING DATASETS")
print("=" * 70)

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class names:")
print(train_ds.class_names)


# ============================================================
# CELL 11 - DATA PIPELINE PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("✅ Dataset prefetch enabled")


# ============================================================
# CELL 12 - CHECK SAMPLE IMAGES
# ============================================================

plt.figure(figsize=(12, 8))

for images, labels in train_ds.take(1):

    for i in range(min(8, len(images))):

        ax = plt.subplot(2, 4, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        label = int(labels[i].numpy()[0])

        plt.title(
            CLASS_NAMES[label]
        )

        plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# CELL 13 - ONLINE DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),

        layers.RandomRotation(
            factor=0.08
        ),

        layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            factor=0.10
        ),

        layers.RandomBrightness(
            factor=0.08
        )

    ],
    name="v29_augmentation"
)

print("✅ Online augmentation created")


# ============================================================
# CELL 14 - WEIGHTED FOCAL LOSS
# ============================================================

NON_MELANOMA_WEIGHT = 1.0
MELANOMA_WEIGHT = 1.5

FOCAL_GAMMA = 1.0


def weighted_binary_focal_loss(
    y_true,
    y_pred
):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    # Binary cross entropy
    bce = -(
        y_true * tf.math.log(y_pred)
        +
        (1.0 - y_true)
        * tf.math.log(1.0 - y_pred)
    )

    # Focal factor
    p_t = (
        y_true * y_pred
        +
        (1.0 - y_true)
        * (1.0 - y_pred)
    )

    focal_factor = tf.pow(
        1.0 - p_t,
        FOCAL_GAMMA
    )

    # Class weights
    class_weight = (
        y_true * MELANOMA_WEIGHT
        +
        (1.0 - y_true)
        * NON_MELANOMA_WEIGHT
    )

    loss = (
        class_weight
        * focal_factor
        * bce
    )

    return tf.reduce_mean(loss)


print("=" * 70)
print("LOSS CONFIGURATION")
print("=" * 70)

print(
    "Non-melanoma weight:",
    NON_MELANOMA_WEIGHT
)

print(
    "Melanoma weight:",
    MELANOMA_WEIGHT
)

print(
    "Focal gamma:",
    FOCAL_GAMMA
)


# ============================================================
# CELL 15 - BUILD EFFICIENTNETV2-S
# ============================================================

print("=" * 70)
print("BUILDING EFFICIENTNETV2-S")
print("=" * 70)

base_model = keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)

base_model.trainable = False

inputs = keras.Input(
    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),
    name="image"
)

x = data_augmentation(
    inputs
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.Dropout(
    0.45,
    name="dropout_1"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="dense_1"
)(x)

x = layers.Dropout(
    0.35,
    name="dropout_2"
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="melanoma_probability"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="V29_EfficientNetV2S"
)

print(model.summary())


# ============================================================
# CELL 16 - MODEL PARAMETER SUMMARY
# ============================================================

total_params = model.count_params()

trainable_params = sum(
    np.prod(v.shape)
    for v in model.trainable_variables
)

non_trainable_params = sum(
    np.prod(v.shape)
    for v in model.non_trainable_variables
)

print("=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print(
    f"Total parameters     : {total_params:,}"
)

print(
    f"Trainable parameters : {trainable_params:,}"
)

print(
    f"Frozen parameters    : {non_trainable_params:,}"
)


# ============================================================
# CELL 17 - COMPILE STAGE 1
# ============================================================

STAGE1_EPOCHS = 12

STAGE1_LR = 1e-4

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),
    loss=weighted_binary_focal_loss,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="roc_auc"
        ),
        keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("=" * 70)
print("STAGE 1 CONFIGURATION")
print("=" * 70)

print(
    "Epochs:",
    STAGE1_EPOCHS
)

print(
    "Learning rate:",
    STAGE1_LR
)


# ============================================================
# CELL 18 - STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = (
    "/content/v29_stage1_best.keras"
)

stage1_callbacks = [

    keras.callbacks.ModelCheckpoint(
        stage1_checkpoint,
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

]


# ============================================================
# CELL 19 - STAGE 1 TRAINING
# ============================================================

print("=" * 70)
print("STAGE 1 TRAINING")
print("=" * 70)

history_stage1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1
)


# ============================================================
# CELL 20 - STAGE 1 PLOTS
# ============================================================

def plot_training_history(
    history,
    title_prefix
):

    history_dict = history.history

    metrics_to_plot = [
        ("loss", "val_loss"),
        ("roc_auc", "val_roc_auc"),
        ("pr_auc", "val_pr_auc"),
        ("accuracy", "val_accuracy"),
        ("precision", "val_precision"),
        ("recall", "val_recall")
    ]

    for train_metric, val_metric in metrics_to_plot:

        if train_metric not in history_dict:
            continue

        plt.figure(figsize=(8, 5))

        plt.plot(
            history_dict[train_metric],
            label=train_metric
        )

        if val_metric in history_dict:

            plt.plot(
                history_dict[val_metric],
                label=val_metric
            )

        plt.title(
            f"{title_prefix} - {train_metric}"
        )

        plt.xlabel("Epoch")
        plt.ylabel(train_metric)
        plt.legend()
        plt.grid(True)
        plt.show()


plot_training_history(
    history_stage1,
    "V29 Stage 1"
)


# ============================================================
# CELL 21 - LOAD BEST STAGE 1 MODEL
# ============================================================

if os.path.exists(stage1_checkpoint):

    model = keras.models.load_model(
        stage1_checkpoint,
        custom_objects={
            "weighted_binary_focal_loss":
                weighted_binary_focal_loss
        }
    )

    print(
        "✅ Best Stage 1 model loaded."
    )


# ============================================================
# CELL 22 - V29 CONTROLLED FINE-TUNING CHANGE
# ============================================================

# ============================================================
# V25 -> V29 CONTROLLED CHANGE
#
# V25:
#     FINE_TUNE_FRACTION = 0.40
#
# V29:
#     FINE_TUNE_FRACTION = 0.60
#
# NO OTHER EXPERIMENTAL PARAMETER IS CHANGED.
# ============================================================

FINE_TUNE_FRACTION = 0.60

print("=" * 70)
print("V29 FINE-TUNING CONFIGURATION")
print("=" * 70)

print(
    "V25 fine-tune fraction:",
    "40%"
)

print(
    "V29 fine-tune fraction:",
    f"{FINE_TUNE_FRACTION * 100:.0f}%"
)


# ============================================================
# CELL 23 - UNFREEZE FINAL 60%
# ============================================================

base_model = model.get_layer(
    "efficientnetv2-s"
)

total_base_layers = len(
    base_model.layers
)

fine_tune_from = int(
    total_base_layers
    * (1.0 - FINE_TUNE_FRACTION)
)

print("=" * 70)
print("FINE-TUNING PLAN")
print("=" * 70)

print(
    "Total EfficientNetV2-S layers:",
    total_base_layers
)

print(
    "Fine-tuning starts at layer:",
    fine_tune_from
)

print(
    "Layers being fine-tuned:",
    total_base_layers - fine_tune_from
)


# Freeze first 40%, unfreeze final 60%
for layer in base_model.layers:

    layer.trainable = False

for layer in base_model.layers[
    fine_tune_from:
]:

    # Keep BatchNormalization frozen.
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

    else:
        layer.trainable = True


# ============================================================
# CELL 24 - VERIFY TRAINABLE LAYERS
# ============================================================

trainable_base_layers = [
    layer
    for layer in base_model.layers
    if layer.trainable
]

frozen_base_layers = [
    layer
    for layer in base_model.layers
    if not layer.trainable
]

trainable_bn_layers = [
    layer
    for layer in base_model.layers
    if layer.trainable
    and isinstance(
        layer,
        layers.BatchNormalization
    )
]

print("=" * 70)
print("FINE-TUNING VERIFICATION")
print("=" * 70)

print(
    "Trainable base layers:",
    len(trainable_base_layers)
)

print(
    "Frozen base layers:",
    len(frozen_base_layers)
)

print(
    "Trainable BatchNorm layers:",
    len(trainable_bn_layers)
)

if len(trainable_bn_layers) == 0:

    print(
        "✅ BatchNorm layers remain frozen."
    )

else:

    print(
        "⚠️ WARNING: Trainable BatchNorm detected."
    )


# ============================================================
# CELL 25 - COMPILE STAGE 2
# ============================================================

STAGE2_EPOCHS = 10

STAGE2_LR = 5e-6

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),
    loss=weighted_binary_focal_loss,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="roc_auc"
        ),
        keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("=" * 70)
print("STAGE 2 CONFIGURATION")
print("=" * 70)

print(
    "Epochs:",
    STAGE2_EPOCHS
)

print(
    "Learning rate:",
    STAGE2_LR
)


# ============================================================
# CELL 26 - STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = (
    "/content/v29_best.keras"
)

stage2_callbacks = [

    keras.callbacks.ModelCheckpoint(
        stage2_checkpoint,
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )

]


# ============================================================
# CELL 27 - STAGE 2 TRAINING
# ============================================================

print("=" * 70)
print("STAGE 2 FINE-TUNING")
print("=" * 70)

history_stage2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks,
    verbose=1
)


# ============================================================
# CELL 28 - STAGE 2 PLOTS
# ============================================================

plot_training_history(
    history_stage2,
    "V29 Stage 2"
)


# ============================================================
# CELL 29 - LOAD BEST V29 MODEL
# ============================================================

if os.path.exists(stage2_checkpoint):

    model = keras.models.load_model(
        stage2_checkpoint,
        custom_objects={
            "weighted_binary_focal_loss":
                weighted_binary_focal_loss
        }
    )

    print(
        "✅ Best V29 model loaded."
    )


# ============================================================
# CELL 30 - SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    "/content/V29_EfficientNetV2S_final.keras"
)

model.save(
    FINAL_MODEL_PATH
)

print(
    "✅ Final V29 model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# CELL 31 - PREDICTION FUNCTION
# ============================================================

def predict_dataset(
    model,
    dataset
):

    probabilities = []
    labels = []

    for images, y in dataset:

        preds = model.predict(
            images,
            verbose=0
        ).reshape(-1)

        probabilities.extend(
            preds.tolist()
        )

        labels.extend(
            y.numpy().reshape(-1).tolist()
        )

    return (
        np.array(probabilities),
        np.array(labels).astype(int)
    )


# ============================================================
# CELL 32 - VALIDATION PREDICTIONS
# ============================================================

print("=" * 70)
print("VALIDATION PREDICTIONS")
print("=" * 70)

p_val, y_val = predict_dataset(
    model,
    valid_ds
)

print(
    "Validation samples:",
    len(y_val)
)

print(
    "Validation melanoma:",
    np.sum(y_val == 1)
)

print(
    "Validation non-melanoma:",
    np.sum(y_val == 0)
)


# ============================================================
# CELL 33 - TEST PREDICTIONS
# ============================================================

print("=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

p_test, y_test = predict_dataset(
    model,
    test_ds
)

print(
    "Test samples:",
    len(y_test)
)

print(
    "Test melanoma:",
    np.sum(y_test == 1)
)

print(
    "Test non-melanoma:",
    np.sum(y_test == 0)
)


# ============================================================
# CELL 34 - ROC / PR AUC
# ============================================================

val_roc_auc = roc_auc_score(
    y_val,
    p_val
)

val_pr_auc = average_precision_score(
    y_val,
    p_val
)

test_roc_auc = roc_auc_score(
    y_test,
    p_test
)

test_pr_auc = average_precision_score(
    y_test,
    p_test
)

print("=" * 70)
print("AUC RESULTS")
print("=" * 70)

print(
    f"Validation ROC-AUC : {val_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC  : {val_pr_auc:.4f}"
)

print(
    f"Test ROC-AUC       : {test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC        : {test_pr_auc:.4f}"
)


# ============================================================
# CELL 35 - ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_val,
    p_val
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {val_roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "V29 Validation ROC Curve"
)

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# CELL 36 - PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_val,
        p_val
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {val_pr_auc:.4f}"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")

plt.title(
    "V29 Validation Precision-Recall Curve"
)

plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# CELL 37 - THRESHOLD METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy":
            balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# CELL 38 - VALIDATION THRESHOLD SEARCH
# ============================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.005
)

threshold_results = []

for threshold in thresholds:

    result = calculate_metrics(
        y_val,
        p_val,
        threshold
    )

    threshold_results.append(
        result
    )

threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# CELL 39 - BEST VALIDATION F1 WITH >=70% SENSITIVITY
# ============================================================

MIN_REQUIRED_SENSITIVITY = 0.70

eligible_thresholds = (
    threshold_df[
        threshold_df["sensitivity"]
        >= MIN_REQUIRED_SENSITIVITY
    ]
)

if len(eligible_thresholds) == 0:

    raise RuntimeError(
        "No threshold satisfies "
        "minimum sensitivity requirement."
    )

best_row = (
    eligible_thresholds
    .sort_values(
        [
            "f1",
            "balanced_accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)

SELECTED_THRESHOLD = float(
    best_row["threshold"]
)

print("=" * 70)
print("V29 SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Minimum required sensitivity: "
    f"{MIN_REQUIRED_SENSITIVITY:.2%}"
)

print(
    f"Selected threshold: "
    f"{SELECTED_THRESHOLD:.3f}"
)

print(
    f"Validation F1: "
    f"{best_row['f1']:.4f}"
)

print(
    f"Validation sensitivity: "
    f"{best_row['sensitivity']:.4f}"
)

print(
    f"Validation specificity: "
    f"{best_row['specificity']:.4f}"
)


# ============================================================
# CELL 40 - VALIDATION RESULTS
# ============================================================

val_metrics = calculate_metrics(
    y_val,
    p_val,
    SELECTED_THRESHOLD
)

print("=" * 70)
print("V29 VALIDATION RESULTS")
print("=" * 70)

for key, value in val_metrics.items():

    if key == "threshold":
        print(
            f"{key:<22}: {value:.3f}"
        )

    elif key in [
        "tn",
        "fp",
        "fn",
        "tp"
    ]:
        print(
            f"{key:<22}: {int(value)}"
        )

    else:
        print(
            f"{key:<22}: {value:.4f}"
        )


# ============================================================
# CELL 41 - CONFUSION MATRIX VALIDATION
# ============================================================

val_predictions = (
    p_val >= SELECTED_THRESHOLD
).astype(int)

cm_val = confusion_matrix(
    y_val,
    val_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_val,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "V29 Validation Confusion Matrix"
)

plt.show()


# ============================================================
# CELL 42 - VALIDATION CLASSIFICATION REPORT
# ============================================================

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CELL 43 - THRESHOLDS FOR SENSITIVITY TARGETS
# ============================================================

print("=" * 70)
print("V29 VALIDATION SENSITIVITY THRESHOLDS")
print("=" * 70)

sensitivity_targets = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

sensitivity_rows = []

for target in sensitivity_targets:

    candidates = threshold_df[
        threshold_df["sensitivity"]
        >= target
    ]

    if len(candidates) == 0:

        continue

    # Highest threshold satisfying target
    selected = (
        candidates
        .sort_values(
            "threshold",
            ascending=False
        )
        .iloc[0]
    )

    sensitivity_rows.append(
        {
            "target_sensitivity":
                target,

            "threshold":
                selected["threshold"],

            "sensitivity":
                selected["sensitivity"],

            "specificity":
                selected["specificity"],

            "precision":
                selected["precision"],

            "f1":
                selected["f1"],

            "balanced_accuracy":
                selected["balanced_accuracy"]
        }
    )

sensitivity_df = pd.DataFrame(
    sensitivity_rows
)

display(
    sensitivity_df
)


# ============================================================
# CELL 44 - TEST EVALUATION
# ============================================================

print("=" * 70)
print("V29 FINAL TEST EVALUATION")
print("=" * 70)

test_metrics = calculate_metrics(
    y_test,
    p_test,
    SELECTED_THRESHOLD
)

for key, value in test_metrics.items():

    if key == "threshold":

        print(
            f"{key:<22}: {value:.3f}"
        )

    elif key in [
        "tn",
        "fp",
        "fn",
        "tp"
    ]:

        print(
            f"{key:<22}: {int(value)}"
        )

    else:

        print(
            f"{key:<22}: {value:.4f}"
        )


# ============================================================
# CELL 45 - TEST CONFUSION MATRIX
# ============================================================

test_predictions = (
    p_test >= SELECTED_THRESHOLD
).astype(int)

cm_test = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("=" * 70)
print("V29 TEST CONFUSION MATRIX")
print("=" * 70)

print(cm_test)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "V29 Test Confusion Matrix"
)

plt.show()


# ============================================================
# CELL 46 - TEST CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("V29 TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CELL 47 - TEST THRESHOLD COMPARISON
# ============================================================

test_thresholds = {
    "selected":
        SELECTED_THRESHOLD,

    "default_0.500":
        0.500,

    "best_f1":
        float(
            threshold_df
            .sort_values(
                "f1",
                ascending=False
            )
            .iloc[0]["threshold"]
        ),

    "best_balanced":
        float(
            threshold_df
            .sort_values(
                "balanced_accuracy",
                ascending=False
            )
            .iloc[0]["threshold"]
        )
}

# Add sensitivity-target thresholds
for _, row in sensitivity_df.iterrows():

    target = row[
        "target_sensitivity"
    ]

    test_thresholds[
        f"sensitivity_{int(target * 100)}"
    ] = float(
        row["threshold"]
    )


test_threshold_rows = []

for name, threshold in (
    test_thresholds.items()
):

    result = calculate_metrics(
        y_test,
        p_test,
        threshold
    )

    result["setting"] = name

    test_threshold_rows.append(
        result
    )

test_threshold_df = pd.DataFrame(
    test_threshold_rows
)

test_threshold_df = (
    test_threshold_df[
        [
            "setting",
            "threshold",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "balanced_accuracy"
        ]
    ]
)

print("=" * 70)
print("V29 TEST THRESHOLD COMPARISON")
print("=" * 70)

display(
    test_threshold_df
)


# ============================================================
# CELL 48 - TEST ROC / PR
# ============================================================

print("=" * 70)
print("V29 TEST AUC")
print("=" * 70)

print(
    f"ROC-AUC: {test_roc_auc:.4f}"
)

print(
    f"PR-AUC : {test_pr_auc:.4f}"
)


# ============================================================
# CELL 49 - PROBABILITY DISTRIBUTION
# ============================================================

melanoma_probabilities = (
    p_test[y_test == 1]
)

non_melanoma_probabilities = (
    p_test[y_test == 0]
)

print("=" * 70)
print("V29 TEST PROBABILITY ANALYSIS")
print("=" * 70)

print("\nMelanoma probabilities")

print(
    "Mean   :",
    np.mean(melanoma_probabilities)
)

print(
    "Median :",
    np.median(melanoma_probabilities)
)

print(
    "Min    :",
    np.min(melanoma_probabilities)
)

print(
    "Max    :",
    np.max(melanoma_probabilities)
)

print("\nNon-melanoma probabilities")

print(
    "Mean   :",
    np.mean(non_melanoma_probabilities)
)

print(
    "Median :",
    np.median(non_melanoma_probabilities)
)

print(
    "Min    :",
    np.min(non_melanoma_probabilities)
)

print(
    "Max    :",
    np.max(non_melanoma_probabilities)
)


# ============================================================
# CELL 50 - PROBABILITY HISTOGRAM
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    non_melanoma_probabilities,
    bins=30,
    alpha=0.6,
    label="Non-melanoma"
)

plt.hist(
    melanoma_probabilities,
    bins=30,
    alpha=0.6,
    label="Melanoma"
)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=(
        f"Threshold = "
        f"{SELECTED_THRESHOLD:.3f}"
    )
)

plt.xlabel(
    "Predicted P(melanoma)"
)

plt.ylabel("Number of images")

plt.title(
    "V29 Test Probability Distribution"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# CELL 51 - HISTORICAL EXPERIMENT COMPARISON
# ============================================================

historical_results = [

    {
        "Version": "V14",
        "ROC-AUC": 0.8721,
        "PR-AUC": 0.5213,
        "Accuracy": 0.8044,
        "Precision": 0.3385,
        "Sensitivity": 0.7857,
        "Specificity": 0.8067,
        "F1": 0.4731,
        "Balanced Accuracy": 0.7962
    },

    {
        "Version": "V16",
        "ROC-AUC": 0.8810,
        "PR-AUC": 0.5048,
        "Accuracy": 0.8483,
        "Precision": 0.3925,
        "Sensitivity": 0.6518,
        "Specificity": 0.8730,
        "F1": 0.4899,
        "Balanced Accuracy": 0.7624
    },

    {
        "Version": "V17",
        "ROC-AUC": 0.8806,
        "PR-AUC": 0.5012,
        "Accuracy": 0.8523,
        "Precision": 0.4011,
        "Sensitivity": 0.6518,
        "Specificity": 0.8775,
        "F1": 0.4966,
        "Balanced Accuracy": 0.7647
    },

    {
        "Version": "V18",
        "ROC-AUC": 0.8798,
        "PR-AUC": 0.5056,
        "Accuracy": 0.8553,
        "Precision": 0.4108,
        "Sensitivity": 0.6786,
        "Specificity": 0.8775,
        "F1": 0.5118,
        "Balanced Accuracy": 0.7780
    },

    {
        "Version": "V19",
        "ROC-AUC": 0.8821,
        "PR-AUC": 0.5081,
        "Accuracy": 0.8563,
        "Precision": 0.4149,
        "Sensitivity": 0.6964,
        "Specificity": 0.8764,
        "F1": 0.5200,
        "Balanced Accuracy": 0.7864
    },

    {
        "Version": "V20",
        "ROC-AUC": 0.8836,
        "PR-AUC": 0.5269,
        "Accuracy": 0.8453,
        "Precision": 0.3930,
        "Sensitivity": 0.7054,
        "Specificity": 0.8629,
        "F1": 0.5048,
        "Balanced Accuracy": 0.7841
    },

    {
        "Version": "V21",
        "ROC-AUC": 0.8882,
        "PR-AUC": np.nan,
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Sensitivity": np.nan,
        "Specificity": np.nan,
        "F1": np.nan,
        "Balanced Accuracy": np.nan
    },

    {
        "Version": "V22",
        "ROC-AUC": 0.8851,
        "PR-AUC": 0.5333,
        "Accuracy": 0.8413,
        "Precision": 0.3854,
        "Sensitivity": 0.7054,
        "Specificity": 0.8584,
        "F1": 0.4984,
        "Balanced Accuracy": 0.7819
    },

    {
        "Version": "V23",
        "ROC-AUC": 0.8877,
        "PR-AUC": 0.5391,
        "Accuracy": 0.8443,
        "Precision": 0.3911,
        "Sensitivity": 0.7054,
        "Specificity": 0.8618,
        "F1": 0.5032,
        "Balanced Accuracy": 0.7836
    },

    {
        "Version": "V24",
        "ROC-AUC": 0.8725,
        "PR-AUC": 0.5047,
        "Accuracy": 0.8553,
        "Precision": 0.4046,
        "Sensitivity": 0.6250,
        "Specificity": 0.8843,
        "F1": 0.4912,
        "Balanced Accuracy": 0.7546
    },

    {
        "Version": "V25",
        "ROC-AUC": 0.8910,
        "PR-AUC": 0.5526,
        "Accuracy": 0.8633,
        "Precision": 0.4302,
        "Sensitivity": 0.6875,
        "Specificity": 0.8854,
        "F1": 0.5292,
        "Balanced Accuracy": 0.7864
    },

    {
        "Version": "V26",
        "ROC-AUC": 0.8795,
        "PR-AUC": 0.5403,
        "Accuracy": 0.8214,
        "Precision": 0.3524,
        "Sensitivity": 0.7143,
        "Specificity": 0.8348,
        "F1": 0.4720,
        "Balanced Accuracy": 0.7746
    },

    {
        "Version": "V27",
        "ROC-AUC": 0.8865,
        "PR-AUC": 0.5219,
        "Accuracy": 0.8553,
        "Precision": 0.4118,
        "Sensitivity": 0.6875,
        "Specificity": 0.8764,
        "F1": 0.5151,
        "Balanced Accuracy": 0.7820
    },

    {
        "Version": "V28",
        "ROC-AUC": 0.8893,
        "PR-AUC": 0.5508,
        "Accuracy": 0.8493,
        "Precision": 0.4020,
        "Sensitivity": 0.7143,
        "Specificity": 0.8663,
        "F1": 0.5145,
        "Balanced Accuracy": 0.7903
    },

    {
        "Version": "V29",
        "ROC-AUC": test_roc_auc,
        "PR-AUC": test_pr_auc,
        "Accuracy": test_metrics["accuracy"],
        "Precision": test_metrics["precision"],
        "Sensitivity": test_metrics["sensitivity"],
        "Specificity": test_metrics["specificity"],
        "F1": test_metrics["f1"],
        "Balanced Accuracy":
            test_metrics["balanced_accuracy"]
    }

]

comparison_df = pd.DataFrame(
    historical_results
)

print("=" * 70)
print("V14 → V29 COMPARISON")
print("=" * 70)

display(
    comparison_df
)


# ============================================================
# CELL 52 - V29 VS V25
# ============================================================

v25_row = (
    comparison_df[
        comparison_df["Version"] == "V25"
    ]
    .iloc[0]
)

v29_row = (
    comparison_df[
        comparison_df["Version"] == "V29"
    ]
    .iloc[0]
)

metrics_to_compare = [
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "Balanced Accuracy"
]

difference_rows = []

for metric in metrics_to_compare:

    difference_rows.append(
        {
            "Metric": metric,
            "V25": v25_row[metric],
            "V29": v29_row[metric],
            "V29 - V25":
                v29_row[metric]
                - v25_row[metric]
        }
    )

difference_df = pd.DataFrame(
    difference_rows
)

print("=" * 70)
print("V29 VS V25")
print("=" * 70)

display(
    difference_df
)


# ============================================================
# CELL 53 - FIND BEST VERSION
# ============================================================

print("=" * 70)
print("BEST EXPERIMENT BY METRIC")
print("=" * 70)

for metric in metrics_to_compare:

    valid_rows = comparison_df[
        comparison_df[metric].notna()
    ]

    best_index = (
        valid_rows[metric]
        .idxmax()
    )

    best_version = (
        valid_rows
        .loc[best_index, "Version"]
    )

    best_value = (
        valid_rows
        .loc[best_index, metric]
    )

    print(
        f"{metric:<20}: "
        f"{best_version} "
        f"({best_value:.4f})"
    )


# ============================================================
# CELL 54 - SAVE RESULTS
# ============================================================

RESULTS_DIR = Path(
    "/content/V29_results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

comparison_df.to_csv(
    RESULTS_DIR
    / "V29_historical_comparison.csv",
    index=False
)

difference_df.to_csv(
    RESULTS_DIR
    / "V29_vs_V25.csv",
    index=False
)

test_threshold_df.to_csv(
    RESULTS_DIR
    / "V29_test_thresholds.csv",
    index=False
)

sensitivity_df.to_csv(
    RESULTS_DIR
    / "V29_sensitivity_thresholds.csv",
    index=False
)

# Save validation threshold results
threshold_df.to_csv(
    RESULTS_DIR
    / "V29_validation_thresholds.csv",
    index=False
)

# Save probabilities
pd.DataFrame(
    {
        "y_true": y_test,
        "probability": p_test,
        "prediction":
            test_predictions
    }
).to_csv(
    RESULTS_DIR
    / "V29_test_predictions.csv",
    index=False
)


# ============================================================
# CELL 55 - SAVE EXPERIMENT CONFIGURATION
# ============================================================

experiment_config = {

    "version": "V29",

    "controlled_change":
        "Fine-tuning fraction 40% -> 60%",

    "baseline":
        "V25",

    "dataset":
        str(SOURCE_DATASET),

    "image_size":
        list(IMAGE_SIZE),

    "batch_size":
        BATCH_SIZE,

    "model":
        "EfficientNetV2-S",

    "weights":
        "ImageNet",

    "non_melanoma_weight":
        NON_MELANOMA_WEIGHT,

    "melanoma_weight":
        MELANOMA_WEIGHT,

    "focal_gamma":
        FOCAL_GAMMA,

    "dropout_after_gap":
        0.45,

    "dropout_after_dense":
        0.35,

    "stage1_epochs":
        STAGE1_EPOCHS,

    "stage1_learning_rate":
        STAGE1_LR,

    "stage2_epochs":
        STAGE2_EPOCHS,

    "stage2_learning_rate":
        STAGE2_LR,

    "fine_tune_fraction":
        FINE_TUNE_FRACTION,

    "minimum_required_sensitivity":
        MIN_REQUIRED_SENSITIVITY,

    "selected_threshold":
        SELECTED_THRESHOLD,

    "validation_roc_auc":
        float(val_roc_auc),

    "validation_pr_auc":
        float(val_pr_auc),

    "test_roc_auc":
        float(test_roc_auc),

    "test_pr_auc":
        float(test_pr_auc),

    "test_accuracy":
        float(test_metrics["accuracy"]),

    "test_precision":
        float(test_metrics["precision"]),

    "test_sensitivity":
        float(test_metrics["sensitivity"]),

    "test_specificity":
        float(test_metrics["specificity"]),

    "test_f1":
        float(test_metrics["f1"]),

    "test_balanced_accuracy":
        float(
            test_metrics["balanced_accuracy"]
        )
}

with open(
    RESULTS_DIR
    / "V29_experiment_config.json",
    "w"
) as f:

    json.dump(
        experiment_config,
        f,
        indent=4
    )


# ============================================================
# CELL 56 - FINAL V29 SUMMARY
# ============================================================

print()
print("=" * 80)
print("                    V29 FINAL SUMMARY")
print("=" * 80)

print()

print(
    "Controlled change:"
)

print(
    "  V25 fine-tuning fraction = 40%"
)

print(
    "  V29 fine-tuning fraction = 60%"
)

print()

print(
    f"Selected threshold       : "
    f"{SELECTED_THRESHOLD:.3f}"
)

print(
    f"Test ROC-AUC             : "
    f"{test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC              : "
    f"{test_pr_auc:.4f}"
)

print(
    f"Test Accuracy            : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision           : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test Sensitivity         : "
    f"{test_metrics['sensitivity']:.4f}"
)

print(
    f"Test Specificity         : "
    f"{test_metrics['specificity']:.4f}"
)

print(
    f"Test F1                  : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Test Balanced Accuracy   : "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print()

print(
    "Confusion Matrix:"
)

print(
    cm_test
)

print()

print(
    "Model:"
)

print(
    FINAL_MODEL_PATH
)

print()

print(
    "Results:"
)

print(
    RESULTS_DIR
)

print()

print("=" * 80)
print("V29 COMPLETE")
print("=" * 80)