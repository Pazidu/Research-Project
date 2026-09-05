# ================================================================
# V27 - OFFLINE MELANOMA AUGMENTATION EXPERIMENT
# ================================================================
#
# CONTROLLED CHANGE FROM PREVIOUS EXPERIMENTS:
#
#   Training melanoma images:
#       Original: 7,122
#       Augmented: 7,122
#       Total: 14,244
#
#   Non-melanoma:
#       7,122 unchanged
#
# VALIDATION AND TEST ARE NEVER AUGMENTED.
#
# Original dataset:
#     /content/newdata
#
# Augmented dataset:
#     /content/newdata_offline_aug
#
# Label mapping:
#     0 = non_melanoma
#     1 = melanoma
#
# Model output:
#     P(melanoma)
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

import os
import random
import shutil
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

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


# ================================================================
# 2. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

print("=" * 70)
print("V27 EXPERIMENT")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("Seed:", SEED)


# ================================================================
# 3. PATHS
# ================================================================

SOURCE_DATASET = Path("/content/newdata")

V27_DATASET = Path(
    "/content/newdata_offline_aug"
)

MODEL_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Models/dermoscopy"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V27_MODEL_PATH = (
    MODEL_DIR /
    "efficientnet_v27_final.keras"
)


# ================================================================
# 4. CONFIGURATION
# ================================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

# ------------------------------------------------
# OFFLINE AUGMENTATION
# ------------------------------------------------

# 2x total melanoma training images.
#
# 7,122 original
# +7,122 augmented
# =14,244 total

AUGMENT_MULTIPLIER = 2


# ------------------------------------------------
# MELANOMA CLASS
# ------------------------------------------------

NON_MELANOMA = 0
MELANOMA = 1


# ------------------------------------------------
# THRESHOLD REQUIREMENT
# ------------------------------------------------

MIN_REQUIRED_SENSITIVITY = 0.70


# ------------------------------------------------
# DATASET SETTINGS
# ------------------------------------------------

CLASS_NAMES = [
    "non_melanoma",
    "melanoma"
]

SPLITS = [
    "train",
    "valid",
    "test"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ================================================================
# 5. GPU CHECK
# ================================================================

print()
print("=" * 70)
print("GPU INFORMATION")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if gpus:
    print("GPU available.")
else:
    print("WARNING: GPU not detected.")


# ================================================================
# 6. CHECK SOURCE DATASET
# ================================================================

print()
print("=" * 70)
print("CHECKING SOURCE DATASET")
print("=" * 70)

if not SOURCE_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE_DATASET}"
    )

for split in SPLITS:

    split_path = SOURCE_DATASET / split

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing dataset split: {split_path}"
        )

print(
    "Source dataset:",
    SOURCE_DATASET
)


# ================================================================
# 7. HELPER FUNCTIONS
# ================================================================

def get_images(folder):

    if not folder.exists():
        return []

    return sorted([
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


def count_images(folder):

    return len(
        get_images(folder)
    )


def get_class_counts(base_dir):

    results = {}

    for split in SPLITS:

        results[split] = {}

        for class_name in CLASS_NAMES:

            folder = (
                base_dir /
                split /
                class_name
            )

            results[split][class_name] = (
                count_images(folder)
            )

    return results


def print_class_counts(
    base_dir,
    title
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    counts = get_class_counts(
        base_dir
    )

    total_dataset = 0

    for split in SPLITS:

        print()
        print(split.upper())

        split_total = 0

        for class_name in CLASS_NAMES:

            count = counts[
                split
            ][class_name]

            split_total += count

            print(
                f"  {class_name:<15}: "
                f"{count:>6}"
            )

        print(
            f"  {'TOTAL':<15}: "
            f"{split_total:>6}"
        )

        total_dataset += split_total

    print()
    print(
        f"TOTAL DATASET: "
        f"{total_dataset}"
    )

    return counts


# ================================================================
# 8. ORIGINAL DATASET COUNTS
# ================================================================

original_counts = print_class_counts(
    SOURCE_DATASET,
    "ORIGINAL DATASET"
)


# ================================================================
# 9. GET TRAINING MELANOMA IMAGES
# ================================================================

SOURCE_TRAIN_MELANOMA = (
    SOURCE_DATASET /
    "train" /
    "melanoma"
)

melanoma_images = get_images(
    SOURCE_TRAIN_MELANOMA
)

original_melanoma_count = (
    len(melanoma_images)
)

print()
print("=" * 70)
print("ORIGINAL MELANOMA TRAINING DATA")
print("=" * 70)

print(
    "Original melanoma images:",
    original_melanoma_count
)

if original_melanoma_count == 0:
    raise RuntimeError(
        "No melanoma training images found."
    )


# ================================================================
# 10. OFFLINE AUGMENTATION FUNCTION
# ================================================================

def augment_melanoma_image(image):

    img = image.copy()

    # ------------------------------------------------------------
    # Horizontal flip
    # ------------------------------------------------------------

    if random.random() < 0.50:

        img = ImageOps.mirror(img)


    # ------------------------------------------------------------
    # Vertical flip
    # ------------------------------------------------------------

    if random.random() < 0.25:

        img = ImageOps.flip(img)


    # ------------------------------------------------------------
    # Small rotation
    # ------------------------------------------------------------

    if random.random() < 0.70:

        angle = random.uniform(
            -20,
            20
        )

        # Use edge-based fill rather than introducing
        # a strong artificial border.

        img = img.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False
        )


    # ------------------------------------------------------------
    # Small brightness variation
    # ------------------------------------------------------------

    if random.random() < 0.50:

        factor = random.uniform(
            0.90,
            1.10
        )

        img = ImageEnhance.Brightness(
            img
        ).enhance(
            factor
        )


    # ------------------------------------------------------------
    # Small contrast variation
    # ------------------------------------------------------------

    if random.random() < 0.50:

        factor = random.uniform(
            0.90,
            1.10
        )

        img = ImageEnhance.Contrast(
            img
        ).enhance(
            factor
        )


    # ------------------------------------------------------------
    # Small color variation
    # ------------------------------------------------------------

    if random.random() < 0.35:

        factor = random.uniform(
            0.92,
            1.08
        )

        img = ImageEnhance.Color(
            img
        ).enhance(
            factor
        )


    # ------------------------------------------------------------
    # Very mild sharpness variation
    # ------------------------------------------------------------

    if random.random() < 0.15:

        factor = random.uniform(
            0.90,
            1.10
        )

        img = ImageEnhance.Sharpness(
            img
        ).enhance(
            factor
        )


    # ------------------------------------------------------------
    # Very mild blur
    # ------------------------------------------------------------

    if random.random() < 0.08:

        radius = random.uniform(
            0.2,
            0.5
        )

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius
            )
        )


    return img


# ================================================================
# 11. CREATE V27 DATASET
# ================================================================

print()
print("=" * 70)
print("CREATING V27 DATASET")
print("=" * 70)

if V27_DATASET.exists():

    print(
        "Removing previous V27 dataset..."
    )

    shutil.rmtree(
        V27_DATASET
    )

V27_DATASET.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 12. COPY VALIDATION AND TEST UNCHANGED
# ================================================================

print()
print("=" * 70)
print("COPYING VALID AND TEST SETS")
print("=" * 70)

for split in [
    "valid",
    "test"
]:

    source_split = (
        SOURCE_DATASET /
        split
    )

    output_split = (
        V27_DATASET /
        split
    )

    shutil.copytree(
        source_split,
        output_split
    )

    print(
        f"✓ {split} copied unchanged"
    )


# ================================================================
# 13. COPY TRAINING NON-MELANOMA
# ================================================================

print()
print("=" * 70)
print("COPYING TRAINING NON-MELANOMA")
print("=" * 70)

source_non_melanoma = (
    SOURCE_DATASET /
    "train" /
    "non_melanoma"
)

output_non_melanoma = (
    V27_DATASET /
    "train" /
    "non_melanoma"
)

output_non_melanoma.mkdir(
    parents=True,
    exist_ok=True
)

non_melanoma_images = get_images(
    source_non_melanoma
)

for image_path in non_melanoma_images:

    shutil.copy2(
        image_path,
        output_non_melanoma /
        image_path.name
    )

print(
    "Copied:",
    len(non_melanoma_images),
    "non-melanoma images"
)


# ================================================================
# 14. COPY ORIGINAL MELANOMA
# ================================================================

print()
print("=" * 70)
print("COPYING ORIGINAL MELANOMA")
print("=" * 70)

output_melanoma = (
    V27_DATASET /
    "train" /
    "melanoma"
)

output_melanoma.mkdir(
    parents=True,
    exist_ok=True
)

for image_path in melanoma_images:

    shutil.copy2(
        image_path,
        output_melanoma /
        image_path.name
    )

print(
    "Copied:",
    original_melanoma_count,
    "original melanoma images"
)


# ================================================================
# 15. GENERATE AUGMENTED MELANOMA
# ================================================================

target_melanoma_count = (
    original_melanoma_count *
    AUGMENT_MULTIPLIER
)

augmented_to_generate = (
    target_melanoma_count -
    original_melanoma_count
)

print()
print("=" * 70)
print("OFFLINE MELANOMA AUGMENTATION")
print("=" * 70)

print(
    "Original melanoma:",
    original_melanoma_count
)

print(
    "Target melanoma:",
    target_melanoma_count
)

print(
    "Augmented images:",
    augmented_to_generate
)


generated = 0

while generated < augmented_to_generate:

    source_path = random.choice(
        melanoma_images
    )

    try:

        with Image.open(
            source_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            augmented = (
                augment_melanoma_image(
                    image
                )
            )

            generated += 1

            output_name = (
                f"{source_path.stem}"
                f"_offline_aug_"
                f"{generated:05d}.jpg"
            )

            output_path = (
                output_melanoma /
                output_name
            )

            augmented.save(
                output_path,
                format="JPEG",
                quality=95
            )

    except Exception as e:

        generated -= 1

        print(
            "Augmentation error:",
            source_path.name,
            e
        )

    if (
        generated % 500 == 0
        or generated ==
        augmented_to_generate
    ):

        print(
            f"Generated "
            f"{generated}/"
            f"{augmented_to_generate}"
        )


# ================================================================
# 16. FINAL DATASET COUNTS
# ================================================================

v27_counts = print_class_counts(
    V27_DATASET,
    "V27 DATASET AFTER OFFLINE AUGMENTATION"
)


# ================================================================
# 17. VERIFY VALIDATION AND TEST
# ================================================================

print()
print("=" * 70)
print("VERIFYING VALID AND TEST")
print("=" * 70)

verification_passed = True

for split in [
    "valid",
    "test"
]:

    for class_name in CLASS_NAMES:

        original_folder = (
            SOURCE_DATASET /
            split /
            class_name
        )

        new_folder = (
            V27_DATASET /
            split /
            class_name
        )

        original_files = sorted(
            p.name
            for p in get_images(
                original_folder
            )
        )

        new_files = sorted(
            p.name
            for p in get_images(
                new_folder
            )
        )

        if original_files != new_files:

            verification_passed = False

            print(
                f"❌ CHANGED: "
                f"{split}/{class_name}"
            )

        else:

            print(
                f"✓ UNCHANGED: "
                f"{split}/{class_name}"
                f" ({len(new_files)} images)"
            )


if not verification_passed:

    raise RuntimeError(
        "Validation/test verification failed."
    )

print()
print(
    "✅ Validation and test verified unchanged."
)


# ================================================================
# 18. CREATE TF.DATASETS
# ================================================================

print()
print("=" * 70)
print("LOADING V27 DATASETS")
print("=" * 70)

train_dir = (
    V27_DATASET /
    "train"
)

valid_dir = (
    V27_DATASET /
    "valid"
)

test_dir = (
    V27_DATASET /
    "test"
)


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# 19. PREFETCH
# ================================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

valid_ds = valid_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)


# ================================================================
# 20. BUILD EFFICIENTNET
# ================================================================

print()
print("=" * 70)
print("BUILDING EFFICIENTNETB0")
print("=" * 70)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

base_model.trainable = False


# ================================================================
# 21. MODEL
# ================================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

x = preprocess_input(
    inputs
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ================================================================
# 22. COMPILE
# ================================================================

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


model.summary()


# ================================================================
# 23. CALLBACKS
# ================================================================

checkpoint_path = (
    MODEL_DIR /
    "efficientnet_v27_best.keras"
)


early_stopping = callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=6,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


checkpoint = callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ================================================================
# 24. TRAIN
# ================================================================

print()
print("=" * 70)
print("V27 TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)


# ================================================================
# 25. LOAD BEST MODEL
# ================================================================

print()
print("=" * 70)
print("LOADING BEST V27 MODEL")
print("=" * 70)

if checkpoint_path.exists():

    model = tf.keras.models.load_model(
        checkpoint_path
    )

    print(
        "Loaded:",
        checkpoint_path
    )


# ================================================================
# 26. PREDICTION HELPER
# ================================================================

def get_predictions(dataset):

    y_true = []
    y_prob = []

    for images, labels in dataset:

        probabilities = (
            model.predict(
                images,
                verbose=0
            ).reshape(-1)
        )

        y_prob.extend(
            probabilities
        )

        y_true.extend(
            labels.numpy()
            .reshape(-1)
            .astype(int)
        )

    return (
        np.array(y_true),
        np.array(y_prob)
    )


# ================================================================
# 27. VALIDATION PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("V27 VALIDATION PREDICTIONS")
print("=" * 70)

y_valid, p_valid = get_predictions(
    valid_ds
)

print(
    "Validation samples:",
    len(y_valid)
)

print(
    "Validation melanoma:",
    int(np.sum(y_valid == 1))
)

print(
    "Validation non-melanoma:",
    int(np.sum(y_valid == 0))
)


# ================================================================
# 28. VALIDATION ROC / PR
# ================================================================

valid_roc_auc = roc_auc_score(
    y_valid,
    p_valid
)

valid_pr_auc = average_precision_score(
    y_valid,
    p_valid
)

print()
print("=" * 70)
print("V27 VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"Validation ROC-AUC: "
    f"{valid_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC: "
    f"{valid_pr_auc:.4f}"
)


# ================================================================
# 29. THRESHOLD METRICS
# ================================================================

def calculate_threshold_metrics(
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

    youden = (
        sensitivity +
        specificity -
        1
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        "youden_j": youden,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp)
    }


# ================================================================
# 30. VALIDATION THRESHOLD SEARCH
# ================================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.005
)

validation_threshold_results = []

for threshold in thresholds:

    result = calculate_threshold_metrics(
        y_valid,
        p_valid,
        threshold
    )

    validation_threshold_results.append(
        result
    )


# ------------------------------------------------
# Best accuracy
# ------------------------------------------------

best_accuracy = max(
    validation_threshold_results,
    key=lambda x: x["accuracy"]
)


# ------------------------------------------------
# Best F1
# ------------------------------------------------

best_f1 = max(
    validation_threshold_results,
    key=lambda x: x["f1"]
)


# ------------------------------------------------
# Best balanced accuracy
# ------------------------------------------------

best_balanced = max(
    validation_threshold_results,
    key=lambda x: x[
        "balanced_accuracy"
    ]
)


# ------------------------------------------------
# Best Youden J
# ------------------------------------------------

best_youden = max(
    validation_threshold_results,
    key=lambda x: x["youden_j"]
)


print()
print("=" * 70)
print("V27 VALIDATION THRESHOLD SEARCH")
print("=" * 70)

print()
print("Best accuracy:")
print(best_accuracy)

print()
print("Best F1:")
print(best_f1)

print()
print("Best balanced accuracy:")
print(best_balanced)

print()
print("Best Youden J:")
print(best_youden)


# ================================================================
# 31. SENSITIVITY-CONSTRAINED SEARCH
# ================================================================

print()
print("=" * 70)
print("V27 SENSITIVITY-CONSTRAINED THRESHOLD SEARCH")
print("=" * 70)

sensitivity_targets = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

sensitivity_results = {}

for target in sensitivity_targets:

    eligible = [
        x
        for x in validation_threshold_results
        if x["sensitivity"] >= target
    ]

    if eligible:

        best = max(
            eligible,
            key=lambda x: x["f1"]
        )

        sensitivity_results[
            target
        ] = best

        print(
            f"{int(target*100)}% sensitivity -> "
            f"threshold={best['threshold']:.3f}, "
            f"sensitivity={best['sensitivity']:.4f}, "
            f"specificity={best['specificity']:.4f}, "
            f"precision={best['precision']:.4f}, "
            f"F1={best['f1']:.4f}, "
            f"balanced={best['balanced_accuracy']:.4f}"
        )


# ================================================================
# 32. FINAL THRESHOLD
# ================================================================

eligible_final = [
    x
    for x in validation_threshold_results
    if x["sensitivity"]
    >= MIN_REQUIRED_SENSITIVITY
]


if not eligible_final:

    raise RuntimeError(
        "No threshold satisfies "
        "minimum sensitivity requirement."
    )


selected_threshold_result = max(
    eligible_final,
    key=lambda x: x["f1"]
)

FINAL_THRESHOLD = (
    selected_threshold_result[
        "threshold"
    ]
)


print()
print("=" * 70)
print("SELECTING V27 FINAL THRESHOLD")
print("=" * 70)

print(
    f"Minimum required sensitivity: "
    f"{MIN_REQUIRED_SENSITIVITY}"
)

print(
    f"Selected threshold: "
    f"{FINAL_THRESHOLD:.3f}"
)

print(
    "Reason: maximum validation F1 "
    "subject to validation sensitivity >= 70%"
)

print()
print(
    "Validation performance at selected threshold:"
)

for key, value in (
    selected_threshold_result.items()
):

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ================================================================
# 33. TEST PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("V27 TEST PREDICTIONS")
print("=" * 70)

y_test, p_test = get_predictions(
    test_ds
)

print(
    "Test samples:",
    len(y_test)
)

print(
    "Test melanoma:",
    int(np.sum(y_test == 1))
)

print(
    "Test non-melanoma:",
    int(np.sum(y_test == 0))
)


# ================================================================
# 34. TEST ROC / PR
# ================================================================

test_roc_auc = roc_auc_score(
    y_test,
    p_test
)

test_pr_auc = average_precision_score(
    y_test,
    p_test
)

print()
print("=" * 70)
print("V27 TEST ROC / PR")
print("=" * 70)

print(
    f"Test ROC-AUC: "
    f"{test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC: "
    f"{test_pr_auc:.4f}"
)


# ================================================================
# 35. FINAL TEST RESULTS
# ================================================================

final_test = calculate_threshold_metrics(
    y_test,
    p_test,
    FINAL_THRESHOLD
)

print()
print("=" * 70)
print("FINAL V27 TEST RESULTS")
print("=" * 70)

print(
    f"Test ROC-AUC: "
    f"{test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC: "
    f"{test_pr_auc:.4f}"
)

print(
    f"Threshold: "
    f"{FINAL_THRESHOLD:.3f}"
)

print(
    f"Accuracy: "
    f"{final_test['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{final_test['precision']:.4f}"
)

print(
    f"Sensitivity: "
    f"{final_test['sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{final_test['specificity']:.4f}"
)

print(
    f"F1 Score: "
    f"{final_test['f1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{final_test['balanced_accuracy']:.4f}"
)


# ================================================================
# 36. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    (
        p_test >= FINAL_THRESHOLD
    ).astype(int),
    labels=[0, 1]
)

print()
print("=" * 70)
print("V27 CONFUSION MATRIX")
print("=" * 70)

print(cm)

tn, fp, fn, tp = cm.ravel()

print()
print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)


# ================================================================
# 37. CLASSIFICATION REPORT
# ================================================================

print()
print("=" * 70)
print("V27 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        (
            p_test >= FINAL_THRESHOLD
        ).astype(int),
        target_names=[
            "non_melanoma",
            "melanoma"
        ],
        digits=4,
        zero_division=0
    )
)


# ================================================================
# 38. TEST THRESHOLD COMPARISON
# ================================================================

print()
print("=" * 70)
print("V27 TEST THRESHOLD COMPARISON")
print("=" * 70)


threshold_comparisons = {

    "v27_selected":
        FINAL_THRESHOLD,

    "v27_best_f1":
        best_f1["threshold"],

    "v27_best_balanced":
        best_balanced["threshold"],

    "v27_best_youden":
        best_youden["threshold"],

    "default_0.50":
        0.50
}


for target, threshold in (
    threshold_comparisons.items()
):

    result = calculate_threshold_metrics(
        y_test,
        p_test,
        threshold
    )

    print()
    print("-" * 50)
    print(target)

    print(
        f"Threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{result['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{result['specificity']:.4f}"
    )

    print(
        f"F1: "
        f"{result['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{result['balanced_accuracy']:.4f}"
    )


# ================================================================
# 39. SENSITIVITY TEST THRESHOLD COMPARISON
# ================================================================

for target in sensitivity_targets:

    if target not in sensitivity_results:
        continue

    validation_result = (
        sensitivity_results[target]
    )

    threshold = (
        validation_result["threshold"]
    )

    result = calculate_threshold_metrics(
        y_test,
        p_test,
        threshold
    )

    print()
    print("-" * 50)

    print(
        f"sensitivity_{int(target*100)}%"
    )

    print(
        f"Threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{result['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{result['specificity']:.4f}"
    )

    print(
        f"F1: "
        f"{result['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{result['balanced_accuracy']:.4f}"
    )


# ================================================================
# 40. PROBABILITY ANALYSIS
# ================================================================

print()
print("=" * 70)
print("V27 PROBABILITY ANALYSIS")
print("=" * 70)

melanoma_probabilities = (
    p_test[y_test == 1]
)

non_melanoma_probabilities = (
    p_test[y_test == 0]
)

print()
print("Melanoma probability:")

print(
    "Mean:",
    round(
        float(
            np.mean(
                melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Median:",
    round(
        float(
            np.median(
                melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Min:",
    round(
        float(
            np.min(
                melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Max:",
    round(
        float(
            np.max(
                melanoma_probabilities
            )
        ),
        4
    )
)


print()
print("Non-melanoma probability:")

print(
    "Mean:",
    round(
        float(
            np.mean(
                non_melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Median:",
    round(
        float(
            np.median(
                non_melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Min:",
    round(
        float(
            np.min(
                non_melanoma_probabilities
            )
        ),
        4
    )
)

print(
    "Max:",
    round(
        float(
            np.max(
                non_melanoma_probabilities
            )
        ),
        4
    )
)


# ================================================================
# 41. ROC CURVE
# ================================================================

fpr, tpr, _ = roc_curve(
    y_test,
    p_test
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"V27 ROC-AUC = {test_roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "V27 ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# 42. PRECISION-RECALL CURVE
# ================================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_test,
        p_test
    )
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"V27 PR-AUC = {test_pr_auc:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "V27 Precision-Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# 43. TRAINING HISTORY
# ================================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["auc"],
    label="Train AUC"
)

plt.plot(
    history.history["val_auc"],
    label="Validation AUC"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "AUC"
)

plt.title(
    "V27 Training History"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# 44. SAVE FINAL MODEL
# ================================================================

print()
print("=" * 70)
print("SAVING FINAL V27 MODEL")
print("=" * 70)

model.save(
    V27_MODEL_PATH
)

print(
    "V27 MODEL SAVED:"
)

print(
    V27_MODEL_PATH
)


# ================================================================
# 45. SAVE V27 RESULTS
# ================================================================

results = {

    "version": "V27",

    "dataset":
        str(V27_DATASET),

    "source_dataset":
        str(SOURCE_DATASET),

    "label_mapping": {
        "0": "non_melanoma",
        "1": "melanoma"
    },

    "model_output":
        "P(melanoma)",

    "offline_augmentation":
        True,

    "augmentation_multiplier":
        AUGMENT_MULTIPLIER,

    "original_melanoma_count":
        original_melanoma_count,

    "target_melanoma_count":
        target_melanoma_count,

    "augmented_melanoma_count":
        augmented_to_generate,

    "final_threshold":
        float(FINAL_THRESHOLD),

    "threshold_reason":
        "maximum validation F1 subject to validation sensitivity >= 70%",

    "validation_roc_auc":
        float(valid_roc_auc),

    "validation_pr_auc":
        float(valid_pr_auc),

    "test_roc_auc":
        float(test_roc_auc),

    "test_pr_auc":
        float(test_pr_auc),

    "test_accuracy":
        float(final_test["accuracy"]),

    "test_precision":
        float(final_test["precision"]),

    "test_sensitivity":
        float(final_test["sensitivity"]),

    "test_specificity":
        float(final_test["specificity"]),

    "test_f1":
        float(final_test["f1"]),

    "test_balanced_accuracy":
        float(
            final_test[
                "balanced_accuracy"
            ]
        ),

    "confusion_matrix": [
        [
            int(tn),
            int(fp)
        ],
        [
            int(fn),
            int(tp)
        ]
    ]
}


results_path = (
    MODEL_DIR /
    "efficientnet_v27_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ================================================================
# 46. FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("V27 EXPERIMENT COMPLETE")
print("=" * 70)

print()

print("DATASET:")
print(V27_DATASET)

print()

print("LABEL MAPPING:")
print("0 = non_melanoma")
print("1 = melanoma")

print()

print("MODEL OUTPUT:")
print("P(melanoma)")

print()

print("OFFLINE AUGMENTATION:")
print("Melanoma training set = 2x")

print()

print("TRAINING COUNTS:")

print(
    "Non-melanoma:",
    v27_counts["train"]["non_melanoma"]
)

print(
    "Melanoma:",
    v27_counts["train"]["melanoma"]
)

print()

print("FINAL THRESHOLD:")
print(
    f"{FINAL_THRESHOLD:.4f}"
)

print()

print("VALIDATION ROC-AUC:")
print(
    f"{valid_roc_auc:.4f}"
)

print()

print("VALIDATION PR-AUC:")
print(
    f"{valid_pr_auc:.4f}"
)

print()

print("TEST ROC-AUC:")
print(
    f"{test_roc_auc:.4f}"
)

print()

print("TEST PR-AUC:")
print(
    f"{test_pr_auc:.4f}"
)

print()

print("TEST ACCURACY:")
print(
    f"{final_test['accuracy']:.4f}"
)

print()

print("TEST PRECISION:")
print(
    f"{final_test['precision']:.4f}"
)

print()

print("TEST SENSITIVITY:")
print(
    f"{final_test['sensitivity']:.4f}"
)

print()

print("TEST SPECIFICITY:")
print(
    f"{final_test['specificity']:.4f}"
)

print()

print("TEST F1:")
print(
    f"{final_test['f1']:.4f}"
)

print()

print("TEST BALANCED ACCURACY:")
print(
    f"{final_test['balanced_accuracy']:.4f}"
)

print()

print("CONFUSION MATRIX:")
print(cm)

print()

print("TRUE NEGATIVE:")
print(tn)

print()

print("FALSE POSITIVE:")
print(fp)

print()

print("FALSE NEGATIVE:")
print(fn)

print()

print("TRUE POSITIVE:")
print(tp)

print()

print("MODEL:")
print(V27_MODEL_PATH)

print()

print("RESULTS:")
print(results_path)

print()
print("=" * 70)

