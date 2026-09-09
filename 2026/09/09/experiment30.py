# !pip install -q tensorflow==2.20.0 keras==3.13.2

import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

import os
import shutil
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import tensorflow as tf
import keras

from tensorflow import data as tf_data
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

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

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

print("Seed:", SEED)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Memory growth setting skipped:", e)
else:
    print("WARNING: No GPU detected.")

SOURCE_DATASET = "/content/drive/MyDrive/Colab Notebooks/newdata_backup"
DATASET_DIR = "/content/newdata"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

print("Source dataset:", SOURCE_DATASET)
print("Local dataset :", DATASET_DIR)

if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

print("Copying dataset from Google Drive...")

shutil.copytree(SOURCE_DATASET, DATASET_DIR)

print("Dataset copied successfully.")

for split in ["train", "valid", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    print("\n", "=" * 60)
    print(split.upper())
    print("=" * 60)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing directory: {split_path}")

    classes = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ])

    print("Classes:", classes)

    for cls in classes:
        cls_path = os.path.join(split_path, cls)

        count = len([
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ])

        print(f"{cls}: {count}")

IMG_SIZE = (300, 300)
BATCH_SIZE = 4

CLASS_NAMES = ["non_melanoma", "melanoma"]

# ------------------------------------------------------------
# CLASS WEIGHTS — SAME AS V25
# ------------------------------------------------------------

NON_MELANOMA_WEIGHT = 1.0
MELANOMA_WEIGHT = 1.5

# ------------------------------------------------------------
# FOCAL LOSS — SAME AS V25
# ------------------------------------------------------------

FOCAL_GAMMA = 1.0

# ------------------------------------------------------------
# TRAINING — SAME AS V25
# ------------------------------------------------------------

STAGE1_EPOCHS = 12
STAGE1_LR = 1e-4

STAGE2_EPOCHS = 10
STAGE2_LR = 5e-6

# ------------------------------------------------------------
# FINE-TUNING — SAME AS V25
# ------------------------------------------------------------

FINE_TUNE_FRACTION = 0.40

# ------------------------------------------------------------
# THRESHOLD POLICY — SAME AS V25
# ------------------------------------------------------------

MIN_REQUIRED_SENSITIVITY = 0.70

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Melanoma weight:", MELANOMA_WEIGHT)
print("Focal gamma:", FOCAL_GAMMA)
print("Fine-tune fraction:", FINE_TUNE_FRACTION)

MODEL_PATH = "/content/V30_EfficientNetV2S_final.keras"
RESULTS_DIR = "/content/V30_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Model path:", MODEL_PATH)
print("Results directory:", RESULTS_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:")
print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Dataset pipelines ready.")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal_and_vertical",
            seed=SEED
        ),

        layers.RandomRotation(
            0.08,
            seed=SEED
        ),

        layers.RandomZoom(
            0.10,
            seed=SEED
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            seed=SEED
        ),

        layers.RandomContrast(
            0.10,
            seed=SEED
        ),

        layers.RandomBrightness(
            0.08,
            seed=SEED
        ),
    ],
    name="online_augmentation"
)

print(data_augmentation)

def weighted_binary_focal_loss(
    non_melanoma_weight=1.0,
    melanoma_weight=1.5,
    gamma=1.0
):
    def loss_fn(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        # Binary cross entropy
        bce = -(
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        # Focal weighting
        p_t = (
            y_true * y_pred
            +
            (1.0 - y_true) * (1.0 - y_pred)
        )

        focal_weight = tf.pow(
            1.0 - p_t,
            gamma
        )

        # Class weighting
        class_weight = (
            y_true * melanoma_weight
            +
            (1.0 - y_true) * non_melanoma_weight
        )

        loss = class_weight * focal_weight * bce

        return tf.reduce_mean(loss)

    return loss_fn


loss_function = weighted_binary_focal_loss(
    non_melanoma_weight=NON_MELANOMA_WEIGHT,
    melanoma_weight=MELANOMA_WEIGHT,
    gamma=FOCAL_GAMMA
)

print("Weighted focal loss created.")

def build_v30_model():

    inputs = keras.Input(
        shape=(*IMG_SIZE, 3),
        name="image"
    )

    # --------------------------------------------------------
    # ONLINE AUGMENTATION
    # --------------------------------------------------------

    x = data_augmentation(inputs)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    x = preprocess_input(x)

    # --------------------------------------------------------
    # EFFICIENTNETV2-S
    # --------------------------------------------------------

    backbone = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    backbone.trainable = False

    x = backbone.output

    # --------------------------------------------------------
    # V30 CLASSIFIER HEAD
    #
    # CONTROLLED CHANGE FROM V25:
    #
    # V25:
    # GAP
    # Dropout(0.45)
    # Dense(256)
    # Dropout(0.35)
    #
    # V30:
    # GAP
    # BatchNormalization
    # Dropout(0.40)
    # Dense(128)
    # Dropout(0.30)
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.BatchNormalization(
        name="classifier_batch_norm"
    )(x)

    x = layers.Dropout(
        0.40,
        name="classifier_dropout_1"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="classifier_dense"
    )(x)

    x = layers.Dropout(
        0.30,
        name="classifier_dropout_2"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="melanoma_probability"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="V30_EfficientNetV2S"
    )

    return model, backbone


model, backbone = build_v30_model()

model.summary()


trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.trainable_variables
])

non_trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.non_trainable_variables
])

print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", non_trainable_params)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),
    loss=loss_function,
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

print("Stage 1 compiled.")

stage1_checkpoint = os.path.join(
    RESULTS_DIR,
    "V30_stage1_best.keras"
)

stage1_callbacks = [

    callbacks.ModelCheckpoint(
        stage1_checkpoint,
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Stage 1 callbacks ready.")

# -----------------------------------------------------
# stage 1 training

print("=" * 70)
print("V30 STAGE 1 — FROZEN BACKBONE")
print("=" * 70)

history_stage1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1
)

# -----------------------------------------------------

def plot_training_history(history, title_prefix):

    history_dict = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    if "pr_auc" in history_dict:

        plt.figure(figsize=(8, 5))
        plt.plot(
            history_dict["pr_auc"],
            label="Training PR-AUC"
        )
        plt.plot(
            history_dict["val_pr_auc"],
            label="Validation PR-AUC"
        )
        plt.title(f"{title_prefix} PR-AUC")
        plt.xlabel("Epoch")
        plt.ylabel("PR-AUC")
        plt.legend()
        plt.grid(True)
        plt.show()


plot_training_history(
    history_stage1,
    "V30 Stage 1"
)

print("=" * 70)
print("CONFIGURING V30 FINE-TUNING")
print("=" * 70)

total_layers = len(backbone.layers)

fine_tune_start = int(
    total_layers * (1.0 - FINE_TUNE_FRACTION)
)

print("Total backbone layers:", total_layers)
print("Fine-tuning starts at:", fine_tune_start)
print("Fine-tuning fraction:", FINE_TUNE_FRACTION)

# First freeze everything
for layer in backbone.layers:
    layer.trainable = False

# Unfreeze final 40%
for layer in backbone.layers[fine_tune_start:]:
    layer.trainable = True

# Keep BatchNormalization layers frozen
for layer in backbone.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_backbone_layers = sum(
    layer.trainable
    for layer in backbone.layers
)

print(
    "Trainable backbone layers:",
    trainable_backbone_layers
)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),
    loss=loss_function,
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

print("Stage 2 compiled.")


stage2_checkpoint = os.path.join(
    RESULTS_DIR,
    "V30_stage2_best.keras"
)

stage2_callbacks = [

    callbacks.ModelCheckpoint(
        stage2_checkpoint,
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

print("Stage 2 callbacks ready.")

# -----------------------------------------------------
#training stage 2

print("=" * 70)
print("V30 STAGE 2 — FINE TUNING")
print("=" * 70)

history_stage2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks,
    verbose=1
)

# -----------------------------------------------------

plot_training_history(
    history_stage2,
    "V30 Stage 2"
)

print("=" * 70)
print("LOADING BEST V30 CHECKPOINT")
print("=" * 70)

if os.path.exists(stage2_checkpoint):
    best_checkpoint = stage2_checkpoint
    print("Using Stage 2 best checkpoint.")

elif os.path.exists(stage1_checkpoint):
    best_checkpoint = stage1_checkpoint
    print("Stage 2 checkpoint unavailable.")
    print("Using Stage 1 best checkpoint.")

else:
    best_checkpoint = None
    print("No checkpoint found. Using current model.")


if best_checkpoint is not None:

    model = keras.models.load_model(
        best_checkpoint,
        custom_objects={
            "loss_fn": loss_function
        }
    )

    print("Loaded:", best_checkpoint)

else:
    print("Current model retained.")

model.save(MODEL_PATH)

print("Final model saved:")
print(MODEL_PATH)

def get_predictions(model, dataset):

    probabilities = model.predict(
        dataset,
        verbose=1
    ).reshape(-1)

    labels = np.concatenate([
        y.numpy()
        for _, y in dataset
    ]).astype(int)

    return probabilities, labels

print("=" * 70)
print("VALIDATION PREDICTIONS")
print("=" * 70)

p_val, y_val = get_predictions(
    model,
    valid_ds
)

print("Validation samples:", len(y_val))
print("Validation melanoma:", np.sum(y_val == 1))
print("Validation non-melanoma:", np.sum(y_val == 0))


print("=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

p_test, y_test = get_predictions(
    model,
    test_ds
)

print("Test samples:", len(y_test))
print("Test melanoma:", np.sum(y_test == 1))
print("Test non-melanoma:", np.sum(y_test == 0))

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
print("V30 AUC RESULTS")
print("=" * 70)

print(f"Validation ROC-AUC : {val_roc_auc:.4f}")
print(f"Validation PR-AUC  : {val_pr_auc:.4f}")
print(f"Test ROC-AUC       : {test_roc_auc:.4f}")
print(f"Test PR-AUC        : {test_pr_auc:.4f}")

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

    balanced_acc = balanced_accuracy_score(
        y_true,
        predictions
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

thresholds = np.arange(
    0.01,
    1.00,
    0.005
)

threshold_results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        y_val,
        p_val,
        threshold
    )

    if (
        metrics["sensitivity"]
        >= MIN_REQUIRED_SENSITIVITY
    ):
        threshold_results.append(metrics)

if not threshold_results:

    raise RuntimeError(
        "No threshold satisfies the minimum sensitivity requirement."
    )

threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

SELECTED_THRESHOLD = float(
    best_row["threshold"]
)

print("=" * 70)
print("V30 SELECTED THRESHOLD")
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

val_metrics = calculate_metrics(
    y_val,
    p_val,
    SELECTED_THRESHOLD
)

print("=" * 70)
print("V30 VALIDATION RESULTS")
print("=" * 70)

for key, value in val_metrics.items():

    if isinstance(value, float):
        print(
            f"{key:22s}: {value:.4f}"
        )
    else:
        print(
            f"{key:22s}: {value}"
        )

val_predictions = (
    p_val >= SELECTED_THRESHOLD
).astype(int)

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)

TARGET_SENSITIVITIES = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

sensitivity_results = []

for target in TARGET_SENSITIVITIES:

    candidates = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_val,
            p_val,
            threshold
        )

        if metrics["sensitivity"] >= target:
            candidates.append(metrics)

    if candidates:

        # Choose highest threshold that still
        # satisfies the target sensitivity.
        candidate_df = pd.DataFrame(candidates)

        row = candidate_df.loc[
            candidate_df["threshold"].idxmax()
        ].copy()

        row["target_sensitivity"] = target

        sensitivity_results.append(row)

sensitivity_df = pd.DataFrame(
    sensitivity_results
)

sensitivity_df = sensitivity_df[
    [
        "target_sensitivity",
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy"
    ]
]

print("=" * 70)
print("V30 VALIDATION SENSITIVITY THRESHOLDS")
print("=" * 70)

display(sensitivity_df)

fpr_val, tpr_val, _ = roc_curve(
    y_val,
    p_val
)

fpr_test, tpr_test, _ = roc_curve(
    y_test,
    p_test
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_val,
    tpr_val,
    label=f"Validation AUC = {val_roc_auc:.4f}"
)

plt.plot(
    fpr_test,
    tpr_test,
    label=f"Test AUC = {test_roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("V30 ROC Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "V30_ROC_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

precision_val, recall_val, _ = precision_recall_curve(
    y_val,
    p_val
)

precision_test, recall_test, _ = precision_recall_curve(
    y_test,
    p_test
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_val,
    precision_val,
    label=f"Validation PR-AUC = {val_pr_auc:.4f}"
)

plt.plot(
    recall_test,
    precision_test,
    label=f"Test PR-AUC = {test_pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("V30 Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "V30_PR_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

test_metrics = calculate_metrics(
    y_test,
    p_test,
    SELECTED_THRESHOLD
)

print("=" * 70)
print("V30 FINAL TEST EVALUATION")
print("=" * 70)

for key, value in test_metrics.items():

    if isinstance(value, float):
        print(
            f"{key:22s}: {value:.4f}"
        )
    else:
        print(
            f"{key:22s}: {value}"
        )

test_predictions = (
    p_test >= SELECTED_THRESHOLD
).astype(int)

test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("=" * 70)
print("V30 TEST CONFUSION MATRIX")
print("=" * 70)

print(test_cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    test_cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("V30 Test Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "V30_confusion_matrix.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("=" * 70)
print("V30 TEST CLASSIFICATION REPORT")
print("=" * 70)

classification_report_text = classification_report(
    y_test,
    test_predictions,
    target_names=CLASS_NAMES,
    digits=4
)

print(classification_report_text)

best_f1_threshold = float(
    threshold_df.loc[
        threshold_df["f1"].idxmax(),
        "threshold"
    ]
)

best_balanced_threshold = float(
    threshold_df.loc[
        threshold_df["balanced_accuracy"].idxmax(),
        "threshold"
    ]
)

test_threshold_settings = {
    "selected": SELECTED_THRESHOLD,
    "default_0.500": 0.500,
    "best_f1": best_f1_threshold,
    "best_balanced": best_balanced_threshold
}

# Add sensitivity target thresholds
for _, row in sensitivity_df.iterrows():

    target = row["target_sensitivity"]

    test_threshold_settings[
        f"sensitivity_{int(target * 100)}"
    ] = float(row["threshold"])

comparison_rows = []

for setting, threshold in test_threshold_settings.items():

    metrics = calculate_metrics(
        y_test,
        p_test,
        threshold
    )

    metrics["setting"] = setting

    comparison_rows.append(metrics)

test_threshold_comparison = pd.DataFrame(
    comparison_rows
)

test_threshold_comparison = test_threshold_comparison[
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

print("=" * 70)
print("V30 TEST THRESHOLD COMPARISON")
print("=" * 70)

display(test_threshold_comparison)

print("=" * 70)
print("V30 TEST AUC")
print("=" * 70)

print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"PR-AUC : {test_pr_auc:.4f}")

melanoma_probabilities = p_test[
    y_test == 1
]

non_melanoma_probabilities = p_test[
    y_test == 0
]

print("=" * 70)
print("V30 TEST PROBABILITY ANALYSIS")
print("=" * 70)

print("\nMelanoma probabilities")

print(
    f"Mean   : "
    f"{np.mean(melanoma_probabilities):.6f}"
)

print(
    f"Median : "
    f"{np.median(melanoma_probabilities):.6f}"
)

print(
    f"Min    : "
    f"{np.min(melanoma_probabilities):.6f}"
)

print(
    f"Max    : "
    f"{np.max(melanoma_probabilities):.6f}"
)

print("\nNon-melanoma probabilities")

print(
    f"Mean   : "
    f"{np.mean(non_melanoma_probabilities):.6f}"
)

print(
    f"Median : "
    f"{np.median(non_melanoma_probabilities):.6f}"
)

print(
    f"Min    : "
    f"{np.min(non_melanoma_probabilities):.6f}"
)

print(
    f"Max    : "
    f"{np.max(non_melanoma_probabilities):.6f}"
)

plt.figure(figsize=(9, 6))

plt.hist(
    non_melanoma_probabilities,
    bins=40,
    alpha=0.6,
    label="Non-melanoma"
)

plt.hist(
    melanoma_probabilities,
    bins=40,
    alpha=0.6,
    label="Melanoma"
)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Threshold = {SELECTED_THRESHOLD:.3f}"
)

plt.xlabel("Predicted melanoma probability")
plt.ylabel("Number of images")
plt.title("V30 Test Probability Distribution")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "V30_probability_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()


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
        "ROC-AUC": test_roc_auc,
        "PR-AUC": test_pr_auc,
        "Accuracy": test_metrics["accuracy"],
        "Precision": test_metrics["precision"],
        "Sensitivity": test_metrics["sensitivity"],
        "Specificity": test_metrics["specificity"],
        "F1": test_metrics["f1"],
        "Balanced Accuracy": test_metrics["balanced_accuracy"]
    }
]

historical_df = pd.DataFrame(
    historical_results
)

print("=" * 70)
print("V14 → V30 COMPARISON")
print("=" * 70)

display(historical_df)

v25_results = {
    "ROC-AUC": 0.8910,
    "PR-AUC": 0.5526,
    "Accuracy": 0.8633,
    "Precision": 0.4302,
    "Sensitivity": 0.6875,
    "Specificity": 0.8854,
    "F1": 0.5292,
    "Balanced Accuracy": 0.7864
}

v30_results = {
    "ROC-AUC": test_roc_auc,
    "PR-AUC": test_pr_auc,
    "Accuracy": test_metrics["accuracy"],
    "Precision": test_metrics["precision"],
    "Sensitivity": test_metrics["sensitivity"],
    "Specificity": test_metrics["specificity"],
    "F1": test_metrics["f1"],
    "Balanced Accuracy": test_metrics["balanced_accuracy"]
}

comparison_rows = []

for metric in v25_results:

    comparison_rows.append({
        "Metric": metric,
        "V25": v25_results[metric],
        "V30": v30_results[metric],
        "V30 - V25":
            v30_results[metric]
            - v25_results[metric]
    })

v30_vs_v25_df = pd.DataFrame(
    comparison_rows
)

print("=" * 70)
print("V30 VS V25")
print("=" * 70)

display(v30_vs_v25_df)

metric_columns = [
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "Balanced Accuracy"
]

print("=" * 70)
print("BEST EXPERIMENT BY METRIC")
print("=" * 70)

for metric in metric_columns:

    valid_rows = historical_df[
        historical_df[metric].notna()
    ]

    best_idx = valid_rows[metric].idxmax()

    best_version = historical_df.loc[
        best_idx,
        "Version"
    ]

    best_value = historical_df.loc[
        best_idx,
        metric
    ]

    print(
        f"{metric:20s}: "
        f"{best_version} ({best_value:.4f})"
    )

historical_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "V30_historical_comparison.csv"
    ),
    index=False
)

v30_vs_v25_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "V30_vs_V25.csv"
    ),
    index=False
)

sensitivity_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "V30_sensitivity_thresholds.csv"
    ),
    index=False
)

test_threshold_comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "V30_test_threshold_comparison.csv"
    ),
    index=False
)

print("CSV results saved.")

final_results = {
    "version": "V30",

    "controlled_change": {
        "baseline": "V25",
        "change": "Classifier head redesign",
        "v25_head": [
            "GlobalAveragePooling2D",
            "Dropout(0.45)",
            "Dense(256, relu)",
            "Dropout(0.35)",
            "Dense(1, sigmoid)"
        ],
        "v30_head": [
            "GlobalAveragePooling2D",
            "BatchNormalization",
            "Dropout(0.40)",
            "Dense(128, relu)",
            "Dropout(0.30)",
            "Dense(1, sigmoid)"
        ]
    },

    "configuration": {
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "non_melanoma_weight": NON_MELANOMA_WEIGHT,
        "melanoma_weight": MELANOMA_WEIGHT,
        "focal_gamma": FOCAL_GAMMA,
        "fine_tune_fraction": FINE_TUNE_FRACTION,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage1_learning_rate": STAGE1_LR,
        "stage2_epochs": STAGE2_EPOCHS,
        "stage2_learning_rate": STAGE2_LR,
        "minimum_required_sensitivity":
            MIN_REQUIRED_SENSITIVITY
    },

    "threshold": SELECTED_THRESHOLD,

    "validation": {
        "roc_auc": val_roc_auc,
        "pr_auc": val_pr_auc,
        **val_metrics
    },

    "test": {
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        **test_metrics
    },

    "confusion_matrix": test_cm.tolist(),

    "model_path": MODEL_PATH,

    "results_directory": RESULTS_DIR
}

json_path = os.path.join(
    RESULTS_DIR,
    "V30_results.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )

print("JSON saved:")
print(json_path)


print()
print("=" * 80)
print("                    V30 FINAL SUMMARY")
print("=" * 80)

print()

print("Controlled change:")
print("  V25 classifier head → V30 redesigned regularized head")

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

print("Confusion Matrix:")
print(test_cm)

print()

print("Model:")
print(MODEL_PATH)

print()

print("Results:")
print(RESULTS_DIR)

print()
print("=" * 80)
print("V30 COMPLETE")
print("=" * 80)

