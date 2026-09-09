# ==============================================================================
# V31 - MELANOMA CLASSIFICATION
# ==============================================================================
# BASELINE : V25
# CONTROLLED CHANGE:
#     Fine-tuning learning rate:
#         V25 = 5e-6
#         V31 = 1e-5
#
# EVERYTHING ELSE IS KEPT THE SAME AS V25
#
# Architecture:
#     EfficientNetV2-S (ImageNet)
#
# Dataset:
#     train/
#         non_melanoma/
#         melanoma/
#     valid/
#         non_melanoma/
#         melanoma/
#     test/
#         non_melanoma/
#         melanoma/
#
# Class mapping:
#     0 = non_melanoma
#     1 = melanoma
#
# Output:
#     P(melanoma)
#
# Threshold:
#     Select threshold maximizing validation F1
#     subject to minimum validation sensitivity >= 70%
#
# ==============================================================================


# ==============================================================================
# CELL 1 - INSTALL / IMPORTS
# ==============================================================================

# !pip install -q tensorflow==2.20.0 keras==3.13.2

import os
import gc
import json
import math
import random
import shutil
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
import keras

from tensorflow import keras as tf_keras
from tensorflow.keras import layers

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

print("=" * 80)
print("V31 MELANOMA CLASSIFICATION")
print("=" * 80)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

print("GPU devices:")
print(tf.config.list_physical_devices("GPU"))


# ==============================================================================
# CELL 2 - REPRODUCIBILITY
# ==============================================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    keras.utils.set_random_seed(SEED)
except:
    pass

print("Random seed:", SEED)


# ==============================================================================
# CELL 3 - GPU CONFIGURATION
# ==============================================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
else:
    print("WARNING: No GPU detected.")


# ==============================================================================
# CELL 4 - CONFIGURATION
# ==============================================================================

VERSION = "V31"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

DRIVE_DATASET = "/content/drive/MyDrive/Colab Notebooks/newdata_backup"
LOCAL_DATASET = "/content/newdata"

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

MODEL_PATH = f"/content/{VERSION}_EfficientNetV2S_final.keras"
RESULTS_DIR = f"/content/{VERSION}_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------

IMG_SIZE = (300, 300)

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

BATCH_SIZE = 4

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 10

STAGE1_LR = 1e-4

# ==========================================================================
# V31 CONTROLLED CHANGE
# V25 = 5e-6
# V31 = 1e-5
# ==========================================================================

STAGE2_LR = 1e-5

# --------------------------------------------------------------------------
# Fine tuning
# --------------------------------------------------------------------------

FINE_TUNE_FRACTION = 0.40

# --------------------------------------------------------------------------
# Class weights
# --------------------------------------------------------------------------

NON_MELANOMA_WEIGHT = 1.0
MELANOMA_WEIGHT = 1.5

# --------------------------------------------------------------------------
# Focal loss
# --------------------------------------------------------------------------

FOCAL_GAMMA = 1.0

# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------

MIN_REQUIRED_SENSITIVITY = 0.70

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------

EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2

print("=" * 80)
print("V31 CONFIGURATION")
print("=" * 80)

print("Version:", VERSION)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

print("Stage 1 epochs:", STAGE1_EPOCHS)
print("Stage 1 LR:", STAGE1_LR)

print("Stage 2 epochs:", STAGE2_EPOCHS)
print("Stage 2 LR:", STAGE2_LR)

print("Fine-tune fraction:", FINE_TUNE_FRACTION)

print("Non-melanoma weight:", NON_MELANOMA_WEIGHT)
print("Melanoma weight:", MELANOMA_WEIGHT)

print("Focal gamma:", FOCAL_GAMMA)

print("Minimum sensitivity:", MIN_REQUIRED_SENSITIVITY)


# ==============================================================================
# CELL 5 - MOUNT GOOGLE DRIVE
# ==============================================================================

from google.colab import drive

drive.mount("/content/drive")


# ==============================================================================
# CELL 6 - COPY DATASET TO LOCAL COLAB STORAGE
# ==============================================================================

print("=" * 80)
print("PREPARING DATASET")
print("=" * 80)

if not os.path.exists(DRIVE_DATASET):
    raise FileNotFoundError(
        f"Dataset not found:\n{DRIVE_DATASET}"
    )

if os.path.exists(LOCAL_DATASET):
    print("Removing existing local dataset...")
    shutil.rmtree(LOCAL_DATASET)

print("Copying dataset from Google Drive...")

shutil.copytree(
    DRIVE_DATASET,
    LOCAL_DATASET
)

print("Dataset copied successfully:")
print(LOCAL_DATASET)


# ==============================================================================
# CELL 7 - VERIFY DATASET STRUCTURE
# ==============================================================================

print("=" * 80)
print("DATASET STRUCTURE")
print("=" * 80)

for split in ["train", "valid", "test"]:
    split_path = os.path.join(LOCAL_DATASET, split)

    print("\n", split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing split: {split_path}"
        )

    for class_name in ["non_melanoma", "melanoma"]:

        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"Missing class directory: {class_path}"
            )

        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ]

        print(
            f"{class_name:15s}: {len(files)} images"
        )


# ==============================================================================
# CELL 8 - COUNT DATASET
# ==============================================================================

def count_images(split_path):

    counts = {}

    for class_name in ["non_melanoma", "melanoma"]:

        class_path = os.path.join(split_path, class_name)

        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ]

        counts[class_name] = len(files)

    return counts


train_counts = count_images(
    os.path.join(LOCAL_DATASET, "train")
)

valid_counts = count_images(
    os.path.join(LOCAL_DATASET, "valid")
)

test_counts = count_images(
    os.path.join(LOCAL_DATASET, "test")
)

print("=" * 80)
print("DATASET COUNTS")
print("=" * 80)

print("Train:", train_counts)
print("Valid:", valid_counts)
print("Test :", test_counts)


# ==============================================================================
# CELL 9 - CREATE TF.DATASETS
# ==============================================================================

print("=" * 80)
print("CREATING DATASETS")
print("=" * 80)

train_dir = os.path.join(LOCAL_DATASET, "train")
valid_dir = os.path.join(LOCAL_DATASET, "valid")
test_dir = os.path.join(LOCAL_DATASET, "test")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["non_melanoma", "melanoma"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["non_melanoma", "melanoma"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["non_melanoma", "melanoma"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class names:")
print(train_ds.class_names)


# ==============================================================================
# CELL 10 - DATASET PERFORMANCE
# ==============================================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Datasets prepared.")


# ==============================================================================
# CELL 11 - ONLINE DATA AUGMENTATION
# ==============================================================================

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
        ),
    ],
    name="online_augmentation"
)

print("Online augmentation:")
data_augmentation.summary()


# ==============================================================================
# CELL 12 - VISUALIZE AUGMENTATION
# ==============================================================================

for images, labels in train_ds.take(1):

    plt.figure(figsize=(12, 8))

    for i in range(min(8, len(images))):

        augmented = data_augmentation(
            tf.expand_dims(images[i], 0),
            training=True
        )

        plt.subplot(2, 4, i + 1)
        plt.imshow(
            tf.cast(
                tf.clip_by_value(
                    augmented[0],
                    0,
                    255
                ),
                tf.uint8
            )
        )

        plt.title(
            "melanoma"
            if int(labels[i].numpy()) == 1
            else "non-melanoma"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    break


# ==============================================================================
# CELL 13 - FOCAL LOSS
# ==============================================================================

def binary_focal_loss(
    gamma=2.0,
    positive_weight=1.0,
    negative_weight=1.0
):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        # Positive class
        positive_loss = (
            -positive_weight
            * y_true
            * tf.pow(1.0 - y_pred, gamma)
            * tf.math.log(y_pred)
        )

        # Negative class
        negative_loss = (
            -negative_weight
            * (1.0 - y_true)
            * tf.pow(y_pred, gamma)
            * tf.math.log(1.0 - y_pred)
        )

        return tf.reduce_mean(
            positive_loss + negative_loss
        )

    return loss


loss_fn = binary_focal_loss(
    gamma=FOCAL_GAMMA,
    positive_weight=MELANOMA_WEIGHT,
    negative_weight=NON_MELANOMA_WEIGHT
)

print("Focal loss configured.")
print("Gamma:", FOCAL_GAMMA)
print("Positive weight:", MELANOMA_WEIGHT)
print("Negative weight:", NON_MELANOMA_WEIGHT)


# ==============================================================================
# CELL 14 - BUILD EFFICIENTNETV2-S BACKBONE
# ==============================================================================

print("=" * 80)
print("BUILDING EFFICIENTNETV2-S")
print("=" * 80)

base_model = keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

base_model.trainable = False

print(
    "Backbone layers:",
    len(base_model.layers)
)


# ==============================================================================
# CELL 15 - V25 CLASSIFIER HEAD
# ==============================================================================

inputs = keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="image"
)

x = data_augmentation(inputs)

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
    name="dense_256"
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
    name="V31_EfficientNetV2S"
)

print("Model created.")


# ==============================================================================
# CELL 16 - MODEL SUMMARY
# ==============================================================================

model.summary()


# ==============================================================================
# CELL 17 - COMPILE STAGE 1
# ==============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),
    loss=loss_fn,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.AUC(
            name="roc_auc",
            curve="ROC"
        ),

        keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)

print("=" * 80)
print("STAGE 1 COMPILED")
print("=" * 80)
print("Learning rate:", STAGE1_LR)


# ==============================================================================
# CELL 18 - CALLBACKS STAGE 1
# ==============================================================================

stage1_callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        os.path.join(
            RESULTS_DIR,
            "stage1_training.csv"
        )
    )
]


# ==============================================================================
# CELL 19 - STAGE 1 TRAINING
# ==============================================================================

print("=" * 80)
print("STAGE 1 TRAINING")
print("=" * 80)

history_stage1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1
)


# ==============================================================================
# CELL 20 - STAGE 1 HISTORY
# ==============================================================================

history1_df = pd.DataFrame(
    history_stage1.history
)

history1_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "stage1_history.csv"
    ),
    index=False
)

print(history1_df.tail())


# ==============================================================================
# CELL 21 - UNFREEZE FINAL 40% OF BACKBONE
# ==============================================================================

print("=" * 80)
print("PREPARING FINE-TUNING")
print("=" * 80)

base_model.trainable = True

total_layers = len(base_model.layers)

fine_tune_start = int(
    total_layers * (1.0 - FINE_TUNE_FRACTION)
)

print("Total backbone layers:", total_layers)
print("Fine-tune start index:", fine_tune_start)
print("Fine-tune fraction:", FINE_TUNE_FRACTION)


# ==============================================================================
# CELL 22 - FREEZE EARLY LAYERS / BATCHNORM
# ==============================================================================

for index, layer in enumerate(base_model.layers):

    if index < fine_tune_start:
        layer.trainable = False

    else:

        # Keep BatchNorm frozen
        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

        else:
            layer.trainable = True


# Verify
trainable_count = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)

non_trainable_count = sum(
    1
    for layer in base_model.layers
    if not layer.trainable
)

print("Trainable backbone layers:", trainable_count)
print("Frozen backbone layers:", non_trainable_count)


# ==============================================================================
# CELL 23 - COMPILE STAGE 2
# ==============================================================================

# ==========================================================================
# V31 CONTROLLED CHANGE:
#
# V25:
#     Stage 2 LR = 5e-6
#
# V31:
#     Stage 2 LR = 1e-5
#
# ==========================================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),
    loss=loss_fn,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.AUC(
            name="roc_auc",
            curve="ROC"
        ),

        keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)

print("=" * 80)
print("STAGE 2 COMPILED")
print("=" * 80)

print("V31 fine-tuning learning rate:", STAGE2_LR)


# ==============================================================================
# CELL 24 - CALLBACKS STAGE 2
# ==============================================================================

stage2_callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        os.path.join(
            RESULTS_DIR,
            "stage2_training.csv"
        )
    )
]


# ==============================================================================
# CELL 25 - STAGE 2 FINE-TUNING
# ==============================================================================

print("=" * 80)
print("STAGE 2 FINE-TUNING")
print("=" * 80)

print("Fine-tuning fraction:", FINE_TUNE_FRACTION)
print("Fine-tuning LR:", STAGE2_LR)

history_stage2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks,
    verbose=1
)


# ==============================================================================
# CELL 26 - COMBINE TRAINING HISTORY
# ==============================================================================

history2_df = pd.DataFrame(
    history_stage2.history
)

history2_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "stage2_history.csv"
    ),
    index=False
)

combined_history = {}

for key in set(
    history_stage1.history.keys()
).union(
    history_stage2.history.keys()
):

    combined_history[key] = (
        history_stage1.history.get(key, [])
        +
        history_stage2.history.get(key, [])
    )

combined_history_df = pd.DataFrame(
    combined_history
)

combined_history_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "combined_history.csv"
    ),
    index=False
)

print(combined_history_df.tail())


# ==============================================================================
# CELL 27 - PLOT TRAINING LOSS
# ==============================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    combined_history_df["loss"],
    label="Train Loss"
)

plt.plot(
    combined_history_df["val_loss"],
    label="Validation Loss"
)

plt.axvline(
    x=len(history_stage1.history["loss"]) - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.title("V31 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()


# ==============================================================================
# CELL 28 - PLOT ROC-AUC
# ==============================================================================

if (
    "roc_auc" in combined_history_df.columns
    and
    "val_roc_auc" in combined_history_df.columns
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        combined_history_df["roc_auc"],
        label="Train ROC-AUC"
    )

    plt.plot(
        combined_history_df["val_roc_auc"],
        label="Validation ROC-AUC"
    )

    plt.axvline(
        x=len(history_stage1.history["loss"]) - 1,
        linestyle="--",
        label="Fine-tuning starts"
    )

    plt.title("V31 ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True)

    plt.show()


# ==============================================================================
# CELL 29 - PLOT PR-AUC
# ==============================================================================

if (
    "pr_auc" in combined_history_df.columns
    and
    "val_pr_auc" in combined_history_df.columns
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        combined_history_df["pr_auc"],
        label="Train PR-AUC"
    )

    plt.plot(
        combined_history_df["val_pr_auc"],
        label="Validation PR-AUC"
    )

    plt.axvline(
        x=len(history_stage1.history["loss"]) - 1,
        linestyle="--",
        label="Fine-tuning starts"
    )

    plt.title("V31 PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(True)

    plt.show()


# ==============================================================================
# CELL 30 - SAVE FINAL MODEL
# ==============================================================================

print("=" * 80)
print("SAVING V31 MODEL")
print("=" * 80)

model.save(
    MODEL_PATH
)

print("Model saved:")
print(MODEL_PATH)


# ==============================================================================
# CELL 31 - VALIDATION PREDICTIONS
# ==============================================================================

print("=" * 80)
print("VALIDATION PREDICTIONS")
print("=" * 80)

valid_probabilities = model.predict(
    valid_ds,
    verbose=1
).ravel()

valid_labels = np.concatenate(
    [
        labels.numpy().ravel()
        for _, labels in valid_ds
    ]
).astype(int)

print("Validation samples:", len(valid_labels))
print(
    "Validation melanoma:",
    int(np.sum(valid_labels == 1))
)
print(
    "Validation non-melanoma:",
    int(np.sum(valid_labels == 0))
)


# ==============================================================================
# CELL 32 - TEST PREDICTIONS
# ==============================================================================

print("=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)

test_probabilities = model.predict(
    test_ds,
    verbose=1
).ravel()

test_labels = np.concatenate(
    [
        labels.numpy().ravel()
        for _, labels in test_ds
    ]
).astype(int)

print("Test samples:", len(test_labels))
print(
    "Test melanoma:",
    int(np.sum(test_labels == 1))
)
print(
    "Test non-melanoma:",
    int(np.sum(test_labels == 0))
)


# ==============================================================================
# CELL 33 - AUC RESULTS
# ==============================================================================

valid_roc_auc = roc_auc_score(
    valid_labels,
    valid_probabilities
)

valid_pr_auc = average_precision_score(
    valid_labels,
    valid_probabilities
)

test_roc_auc = roc_auc_score(
    test_labels,
    test_probabilities
)

test_pr_auc = average_precision_score(
    test_labels,
    test_probabilities
)

print("=" * 80)
print("V31 AUC RESULTS")
print("=" * 80)

print(
    f"Validation ROC-AUC : {valid_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC  : {valid_pr_auc:.4f}"
)

print(
    f"Test ROC-AUC       : {test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC        : {test_pr_auc:.4f}"
)


# ==============================================================================
# CELL 34 - THRESHOLD SEARCH FUNCTION
# ==============================================================================

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

    balanced = balanced_accuracy_score(
        y_true,
        predictions
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ==============================================================================
# CELL 35 - SELECT THRESHOLD
# ==============================================================================

thresholds = np.round(
    np.arange(
        0.01,
        1.00,
        0.005
    ),
    3
)

threshold_results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        valid_labels,
        valid_probabilities,
        threshold
    )

    if (
        metrics["sensitivity"]
        >= MIN_REQUIRED_SENSITIVITY
    ):
        threshold_results.append(
            metrics
        )

if len(threshold_results) == 0:

    raise RuntimeError(
        "No threshold satisfies the minimum required sensitivity."
    )

threshold_df = pd.DataFrame(
    threshold_results
)

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

SELECTED_THRESHOLD = float(
    best_threshold_row["threshold"]
)

print("=" * 80)
print("V31 SELECTED THRESHOLD")
print("=" * 80)

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
    f"{best_threshold_row['f1']:.4f}"
)

print(
    f"Validation sensitivity: "
    f"{best_threshold_row['sensitivity']:.4f}"
)

print(
    f"Validation specificity: "
    f"{best_threshold_row['specificity']:.4f}"
)


# ==============================================================================
# CELL 36 - VALIDATION RESULTS
# ==============================================================================

valid_selected_metrics = calculate_metrics(
    valid_labels,
    valid_probabilities,
    SELECTED_THRESHOLD
)

print("=" * 80)
print("V31 VALIDATION RESULTS")
print("=" * 80)

for key, value in valid_selected_metrics.items():

    if isinstance(value, float):

        print(
            f"{key:22s}: {value:.4f}"
        )

    else:

        print(
            f"{key:22s}: {value}"
        )


# ==============================================================================
# CELL 37 - VALIDATION CLASSIFICATION REPORT
# ==============================================================================

valid_predictions = (
    valid_probabilities
    >= SELECTED_THRESHOLD
).astype(int)

print(
    classification_report(
        valid_labels,
        valid_predictions,
        target_names=[
            "non_melanoma",
            "melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


# ==============================================================================
# CELL 38 - VALIDATION SENSITIVITY THRESHOLDS
# ==============================================================================

TARGET_SENSITIVITIES = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

sensitivity_threshold_rows = []

for target in TARGET_SENSITIVITIES:

    candidates = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            valid_labels,
            valid_probabilities,
            threshold
        )

        if metrics["sensitivity"] >= target:
            candidates.append(metrics)

    if len(candidates) > 0:

        # Choose the highest threshold that still
        # satisfies the target sensitivity.
        candidates_df = pd.DataFrame(
            candidates
        )

        selected_row = candidates_df.loc[
            candidates_df["threshold"].idxmax()
        ]

        sensitivity_threshold_rows.append(
            {
                "target_sensitivity": target,
                "threshold": selected_row["threshold"],
                "sensitivity": selected_row["sensitivity"],
                "specificity": selected_row["specificity"],
                "precision": selected_row["precision"],
                "f1": selected_row["f1"],
                "balanced_accuracy":
                    selected_row["balanced_accuracy"]
            }
        )

sensitivity_threshold_df = pd.DataFrame(
    sensitivity_threshold_rows
)

print("=" * 80)
print("V31 VALIDATION SENSITIVITY THRESHOLDS")
print("=" * 80)

display(
    sensitivity_threshold_df
)


# ==============================================================================
# CELL 39 - TEST EVALUATION
# ==============================================================================

test_selected_metrics = calculate_metrics(
    test_labels,
    test_probabilities,
    SELECTED_THRESHOLD
)

print("=" * 80)
print("V31 FINAL TEST EVALUATION")
print("=" * 80)

for key, value in test_selected_metrics.items():

    if isinstance(value, float):

        print(
            f"{key:22s}: {value:.4f}"
        )

    else:

        print(
            f"{key:22s}: {value}"
        )


# ==============================================================================
# CELL 40 - TEST CLASSIFICATION REPORT
# ==============================================================================

test_predictions = (
    test_probabilities
    >= SELECTED_THRESHOLD
).astype(int)

print("=" * 80)
print("V31 TEST CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "non_melanoma",
            "melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


# ==============================================================================
# CELL 41 - TEST THRESHOLD COMPARISON
# ==============================================================================

threshold_settings = {
    "selected": SELECTED_THRESHOLD,
    "default_0.500": 0.500,
    "best_f1": None,
    "best_balanced": None,
    "sensitivity_70": None,
    "sensitivity_75": None,
    "sensitivity_80": None,
    "sensitivity_85": None,
    "sensitivity_90": None,
    "sensitivity_95": None
}

# Best test F1
test_all_threshold_metrics = []

for threshold in thresholds:

    metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold
    )

    test_all_threshold_metrics.append(
        metrics
    )

test_all_threshold_df = pd.DataFrame(
    test_all_threshold_metrics
)

best_f1_test = test_all_threshold_df.loc[
    test_all_threshold_df["f1"].idxmax()
]

best_balanced_test = test_all_threshold_df.loc[
    test_all_threshold_df[
        "balanced_accuracy"
    ].idxmax()
]

threshold_settings["best_f1"] = float(
    best_f1_test["threshold"]
)

threshold_settings["best_balanced"] = float(
    best_balanced_test["threshold"]
)

# Validation sensitivity threshold settings
for target in TARGET_SENSITIVITIES:

    candidates = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            valid_labels,
            valid_probabilities,
            threshold
        )

        if metrics["sensitivity"] >= target:
            candidates.append(metrics)

    if len(candidates) > 0:

        candidates_df = pd.DataFrame(
            candidates
        )

        row = candidates_df.loc[
            candidates_df["threshold"].idxmax()
        ]

        key = f"sensitivity_{int(target * 100)}"

        threshold_settings[key] = float(
            row["threshold"]
        )


threshold_comparison_rows = []

for setting, threshold in threshold_settings.items():

    metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold
    )

    row = {
        "setting": setting
    }

    row.update(metrics)

    threshold_comparison_rows.append(
        row
    )

test_threshold_comparison = pd.DataFrame(
    threshold_comparison_rows
)

print("=" * 80)
print("V31 TEST THRESHOLD COMPARISON")
print("=" * 80)

display(
    test_threshold_comparison[
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


# ==============================================================================
# CELL 42 - TEST ROC CURVE
# ==============================================================================

fpr, tpr, roc_thresholds = roc_curve(
    test_labels,
    test_probabilities
)

plt.figure(figsize=(8, 7))

plt.plot(
    fpr,
    tpr,
    label=f"V31 ROC-AUC = {test_roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "V31 Test ROC Curve"
)

plt.legend()
plt.grid(True)

plt.show()


# ==============================================================================
# CELL 43 - TEST PRECISION-RECALL CURVE
# ==============================================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        test_labels,
        test_probabilities
    )
)

plt.figure(figsize=(8, 7))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"V31 PR-AUC = {test_pr_auc:.4f}"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")

plt.title(
    "V31 Test Precision-Recall Curve"
)

plt.legend()
plt.grid(True)

plt.show()


# ==============================================================================
# CELL 44 - CONFUSION MATRIX
# ==============================================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "non_melanoma",
        "melanoma"
    ],
    yticklabels=[
        "non_melanoma",
        "melanoma"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    f"V31 Test Confusion Matrix\n"
    f"Threshold = {SELECTED_THRESHOLD:.3f}"
)

plt.show()


# ==============================================================================
# CELL 45 - PROBABILITY DISTRIBUTION
# ==============================================================================

melanoma_probabilities = test_probabilities[
    test_labels == 1
]

non_melanoma_probabilities = test_probabilities[
    test_labels == 0
]

print("=" * 80)
print("V31 TEST PROBABILITY ANALYSIS")
print("=" * 80)

print(
    "Melanoma probabilities"
)

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

print()

print(
    "Non-melanoma probabilities"
)

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


# ==============================================================================
# CELL 46 - PROBABILITY HISTOGRAM
# ==============================================================================

plt.figure(figsize=(10, 6))

plt.hist(
    non_melanoma_probabilities,
    bins=50,
    alpha=0.6,
    label="Non-melanoma"
)

plt.hist(
    melanoma_probabilities,
    bins=50,
    alpha=0.6,
    label="Melanoma"
)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Threshold = {SELECTED_THRESHOLD:.3f}"
)

plt.xlabel(
    "Predicted probability of melanoma"
)

plt.ylabel("Number of images")

plt.title(
    "V31 Test Probability Distribution"
)

plt.legend()
plt.grid(True)

plt.show()


# ==============================================================================
# CELL 47 - SAVE PREDICTIONS
# ==============================================================================

validation_predictions_df = pd.DataFrame(
    {
        "true_label": valid_labels,
        "melanoma_probability":
            valid_probabilities,
        "prediction":
            valid_predictions
    }
)

test_predictions_df = pd.DataFrame(
    {
        "true_label": test_labels,
        "melanoma_probability":
            test_probabilities,
        "prediction":
            test_predictions
    }
)

validation_predictions_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "validation_predictions.csv"
    ),
    index=False
)

test_predictions_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),
    index=False
)

print("Predictions saved.")


# ==============================================================================
# CELL 48 - CREATE V31 RESULTS
# ==============================================================================

final_results = {

    "version": VERSION,

    "controlled_change": (
        "V25 fine-tuning learning rate "
        "5e-6 -> V31 1e-5"
    ),

    "configuration": {

        "image_size": IMG_SIZE,

        "batch_size": BATCH_SIZE,

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

        "non_melanoma_weight":
            NON_MELANOMA_WEIGHT,

        "melanoma_weight":
            MELANOMA_WEIGHT,

        "focal_gamma":
            FOCAL_GAMMA,

        "minimum_required_sensitivity":
            MIN_REQUIRED_SENSITIVITY
    },

    "validation": {

        "samples":
            int(len(valid_labels)),

        "melanoma":
            int(np.sum(valid_labels == 1)),

        "non_melanoma":
            int(np.sum(valid_labels == 0)),

        "roc_auc":
            float(valid_roc_auc),

        "pr_auc":
            float(valid_pr_auc),

        "selected_threshold":
            float(SELECTED_THRESHOLD),

        "accuracy":
            float(
                valid_selected_metrics["accuracy"]
            ),

        "precision":
            float(
                valid_selected_metrics["precision"]
            ),

        "sensitivity":
            float(
                valid_selected_metrics["sensitivity"]
            ),

        "specificity":
            float(
                valid_selected_metrics["specificity"]
            ),

        "f1":
            float(
                valid_selected_metrics["f1"]
            ),

        "balanced_accuracy":
            float(
                valid_selected_metrics[
                    "balanced_accuracy"
                ]
            ),

        "tn":
            int(valid_selected_metrics["tn"]),

        "fp":
            int(valid_selected_metrics["fp"]),

        "fn":
            int(valid_selected_metrics["fn"]),

        "tp":
            int(valid_selected_metrics["tp"])
    },

    "test": {

        "samples":
            int(len(test_labels)),

        "melanoma":
            int(np.sum(test_labels == 1)),

        "non_melanoma":
            int(np.sum(test_labels == 0)),

        "roc_auc":
            float(test_roc_auc),

        "pr_auc":
            float(test_pr_auc),

        "selected_threshold":
            float(SELECTED_THRESHOLD),

        "accuracy":
            float(
                test_selected_metrics["accuracy"]
            ),

        "precision":
            float(
                test_selected_metrics["precision"]
            ),

        "sensitivity":
            float(
                test_selected_metrics["sensitivity"]
            ),

        "specificity":
            float(
                test_selected_metrics["specificity"]
            ),

        "f1":
            float(
                test_selected_metrics["f1"]
            ),

        "balanced_accuracy":
            float(
                test_selected_metrics[
                    "balanced_accuracy"
                ]
            ),

        "tn":
            int(test_selected_metrics["tn"]),

        "fp":
            int(test_selected_metrics["fp"]),

        "fn":
            int(test_selected_metrics["fn"]),

        "tp":
            int(test_selected_metrics["tp"])
    },

    "model_path": MODEL_PATH,

    "results_directory": RESULTS_DIR
}


# ==============================================================================
# CELL 49 - JSON SERIALIZATION FIX
# ==============================================================================

def make_json_serializable(obj):

    if isinstance(obj, dict):

        return {
            str(key):
            make_json_serializable(value)
            for key, value in obj.items()
        }

    elif isinstance(obj, (list, tuple)):

        return [
            make_json_serializable(value)
            for value in obj
        ]

    elif isinstance(obj, np.integer):

        return int(obj)

    elif isinstance(obj, np.floating):

        return float(obj)

    elif isinstance(obj, np.ndarray):

        return obj.tolist()

    elif isinstance(obj, np.bool_):

        return bool(obj)

    elif isinstance(
        obj,
        (float, int, str, bool)
    ) or obj is None:

        return obj

    else:

        return str(obj)


final_results = make_json_serializable(
    final_results
)


# ==============================================================================
# CELL 50 - SAVE JSON
# ==============================================================================

results_json_path = os.path.join(
    RESULTS_DIR,
    "V31_final_results.json"
)

with open(
    results_json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_results,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    "Results saved:",
    results_json_path
)


# ==============================================================================
# CELL 51 - CREATE V14-V31 COMPARISON
# ==============================================================================

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
        "ROC-AUC": 0.8895,
        "PR-AUC": 0.5757,
        "Accuracy": 0.8653,
        "Precision": 0.4294,
        "Sensitivity": 0.6250,
        "Specificity": 0.8955,
        "F1": 0.5091,
        "Balanced Accuracy": 0.7603
    },

    {
        "Version": "V30",
        "ROC-AUC": 0.8618,
        "PR-AUC": 0.4658,
        "Accuracy": 0.8224,
        "Precision": 0.3486,
        "Sensitivity": 0.6786,
        "Specificity": 0.8404,
        "F1": 0.4606,
        "Balanced Accuracy": 0.7595
    },

    {
        "Version": "V31",
        "ROC-AUC": float(test_roc_auc),
        "PR-AUC": float(test_pr_auc),
        "Accuracy": float(
            test_selected_metrics["accuracy"]
        ),
        "Precision": float(
            test_selected_metrics["precision"]
        ),
        "Sensitivity": float(
            test_selected_metrics["sensitivity"]
        ),
        "Specificity": float(
            test_selected_metrics["specificity"]
        ),
        "F1": float(
            test_selected_metrics["f1"]
        ),
        "Balanced Accuracy": float(
            test_selected_metrics[
                "balanced_accuracy"
            ]
        )
    }
]

comparison_df = pd.DataFrame(
    historical_results
)

comparison_path = os.path.join(
    RESULTS_DIR,
    "V14_V31_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("=" * 80)
print("V14-V31 COMPARISON")
print("=" * 80)

display(comparison_df)


# ==============================================================================
# CELL 52 - V31 VS V25
# ==============================================================================

v25_row = comparison_df[
    comparison_df["Version"] == "V25"
].iloc[0]

v31_row = comparison_df[
    comparison_df["Version"] == "V31"
].iloc[0]

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

v25_v31_comparison = []

for metric in metrics_to_compare:

    v25_value = float(v25_row[metric])
    v31_value = float(v31_row[metric])

    v25_v31_comparison.append(
        {
            "Metric": metric,
            "V25": v25_value,
            "V31": v31_value,
            "V31 - V25":
                v31_value - v25_value
        }
    )

v25_v31_df = pd.DataFrame(
    v25_v31_comparison
)

print("=" * 80)
print("V31 VS V25")
print("=" * 80)

display(v25_v31_df)


# ==============================================================================
# CELL 53 - BEST EXPERIMENT BY METRIC
# ==============================================================================

print("=" * 80)
print("BEST EXPERIMENT BY METRIC")
print("=" * 80)

for metric in metrics_to_compare:

    metric_df = comparison_df[
        comparison_df[metric].notna()
    ]

    best_index = metric_df[metric].idxmax()

    best_version = metric_df.loc[
        best_index,
        "Version"
    ]

    best_value = metric_df.loc[
        best_index,
        metric
    ]

    print(
        f"{metric:20s}: "
        f"{best_version} "
        f"({best_value:.4f})"
    )


# ==============================================================================
# CELL 54 - ACCURACY ERROR COUNT
# ==============================================================================

test_total = len(test_labels)

test_correct = int(
    round(
        test_selected_metrics["accuracy"]
        * test_total
    )
)

test_errors = (
    test_total
    - test_correct
)

required_correct_for_96 = math.ceil(
    0.96 * test_total
)

maximum_errors_for_96 = (
    test_total
    - required_correct_for_96
)

print("=" * 80)
print("V31 ACCURACY ANALYSIS")
print("=" * 80)

print(
    "Test samples:",
    test_total
)

print(
    "Correct:",
    test_correct
)

print(
    "Errors:",
    test_errors
)

print(
    "Correct required for 96%:",
    required_correct_for_96
)

print(
    "Maximum errors allowed for 96%:",
    maximum_errors_for_96
)


# ==============================================================================
# CELL 55 - FINAL SUMMARY
# ==============================================================================

print()
print("=" * 80)
print("V31 FINAL SUMMARY")
print("=" * 80)

print(
    "Controlled change:"
)

print(
    "V25 fine-tuning LR = 5e-6"
)

print(
    "V31 fine-tuning LR = 1e-5"
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
    f"{test_selected_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision           : "
    f"{test_selected_metrics['precision']:.4f}"
)

print(
    f"Test Sensitivity         : "
    f"{test_selected_metrics['sensitivity']:.4f}"
)

print(
    f"Test Specificity         : "
    f"{test_selected_metrics['specificity']:.4f}"
)

print(
    f"Test F1                  : "
    f"{test_selected_metrics['f1']:.4f}"
)

print(
    f"Test Balanced Accuracy   : "
    f"{test_selected_metrics['balanced_accuracy']:.4f}"
)

print()

print(
    "Confusion Matrix:"
)

print(cm)

print()

print(
    "Model:",
    MODEL_PATH
)

print(
    "Results:",
    RESULTS_DIR
)

print()
print("=" * 80)
print("V31 COMPLETE")
print("=" * 80)


# ==============================================================================
# CELL 56 - CLEANUP
# ==============================================================================

gc.collect()

try:
    tf.keras.backend.clear_session()
except:
    pass

print("Cleanup completed.")