# ================================================================
# V12 MELANOMA CLASSIFICATION
# EfficientNetV2-S
# Based on V11 results
# ================================================================


# ================================================================
# CELL 1 - IMPORTS
# ================================================================

import os
import json
import random
import warnings
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from tensorflow import keras as tf_keras

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

warnings.filterwarnings("ignore")

print("=" * 70)
print("V12 MELANOMA CLASSIFICATION")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

print("=" * 70)


# ================================================================
# CELL 2 - MOUNT GOOGLE DRIVE
# ================================================================

from google.colab import drive

print("=" * 70)
print("MOUNTING GOOGLE DRIVE")
print("=" * 70)

drive.mount("/content/drive")

print("\nGoogle Drive mounted successfully.")

print("\nMyDrive contents:")

for item in os.listdir("/content/drive/MyDrive")[:30]:
    print(" -", item)


# ================================================================
# CELL 3 - REPRODUCIBILITY
# ================================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Random seed:", SEED)


# ================================================================
# CELL 4 - GPU CHECK
# ================================================================

print("\n" + "=" * 70)
print("GPU INFORMATION")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print(" ", gpu)

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except Exception as e:

        print("Memory growth warning:", e)

else:

    print("WARNING: No GPU detected.")

    print(
        "Go to Runtime > Change runtime type > GPU."
    )


# ================================================================
# CELL 5 - DATASET CONFIGURATION
# ================================================================

print("\n" + "=" * 70)
print("DATASET CONFIGURATION")
print("=" * 70)


# ------------------------------------------------
# IMPORTANT:
# THIS IS YOUR ACTUAL DATASET LOCATION
# ------------------------------------------------

IMG_SRC = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/newdata_backup"
)


# ------------------------------------------------
# LOCAL COLAB DATASET
# ------------------------------------------------

DATASET = "/content/newdata"


# ------------------------------------------------
# DATASET ROOT
# ------------------------------------------------

DATASET_ROOT = DATASET


# ------------------------------------------------
# CLASS NAMES
# ------------------------------------------------

CLASS_NAMES = [
    "non_melanoma",
    "melanoma"
]


# ------------------------------------------------
# IMAGE SETTINGS
# ------------------------------------------------

IMG_SIZE = 224

BATCH_SIZE = 8

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------

STAGE1_EPOCHS = 12

STAGE2_EPOCHS = 10


STAGE1_LR = 1e-4

STAGE2_LR = 5e-6

MIN_LR = 1e-7


# ------------------------------------------------
# V12 CHANGE
# ------------------------------------------------
# V11:
# MELANOMA_WEIGHT = 1.5
#
# V12:
# MELANOMA_WEIGHT = 1.25
#
# Reason:
# V11 had 240 false positives at threshold 0.07.
# ------------------------------------------------

NON_MELANOMA_WEIGHT = 1.0

MELANOMA_WEIGHT = 1.25


# ------------------------------------------------
# FINE-TUNING
# ------------------------------------------------

FINE_TUNE_FRACTION = 0.35


# ------------------------------------------------
# V12 AUGMENTATION SETTINGS
# ------------------------------------------------

ROTATION_FACTOR = 0.05

ZOOM_FACTOR = 0.08

TRANSLATION_FACTOR = 0.04

CONTRAST_FACTOR = 0.08


print("Dataset source:")
print(IMG_SRC)

print("\nLocal dataset:")
print(DATASET)

print("\nClass names:")
print(CLASS_NAMES)

print("\nImage size:", IMG_SIZE)

print("Batch size:", BATCH_SIZE)

print("Non-melanoma weight:", NON_MELANOMA_WEIGHT)

print("Melanoma weight:", MELANOMA_WEIGHT)

print("\nV12 strategy:")
print("Reduce melanoma weighting to reduce false-positive bias.")


# ================================================================
# CELL 6 - DATASET STRUCTURE VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("VALIDATING DATASET SOURCE")
print("=" * 70)


def has_required_dataset_structure(path):

    required_splits = [
        "train",
        "valid",
        "test"
    ]

    required_classes = [
        "melanoma",
        "non_melanoma"
    ]

    if not os.path.isdir(path):

        return False

    for split in required_splits:

        split_path = os.path.join(
            path,
            split
        )

        if not os.path.isdir(split_path):

            return False

        for class_name in required_classes:

            class_path = os.path.join(
                split_path,
                class_name
            )

            if not os.path.isdir(class_path):

                return False

    return True


print("Checking:")

print(IMG_SRC)


if not os.path.exists(IMG_SRC):

    raise FileNotFoundError(

        "\nDataset source was NOT found:\n\n"
        f"{IMG_SRC}\n\n"
        "Please check your Google Drive path."

    )


if not has_required_dataset_structure(IMG_SRC):

    print("\nDataset source exists, but structure is incorrect.")

    print("\nExpected structure:")

    print(
        "newdata_backup/\n"
        "├── train/\n"
        "│   ├── melanoma/\n"
        "│   └── non_melanoma/\n"
        "├── valid/\n"
        "│   ├── melanoma/\n"
        "│   └── non_melanoma/\n"
        "└── test/\n"
        "    ├── melanoma/\n"
        "    └── non_melanoma/"
    )

    raise FileNotFoundError(
        "\nRequired train/valid/test/class directories were not found."
    )


print("\nDataset structure verified successfully.")


# ================================================================
# CELL 7 - COPY DATASET TO LOCAL COLAB STORAGE
# ================================================================

print("\n" + "=" * 70)
print("SETTING UP LOCAL DATASET")
print("=" * 70)


print("Drive source:")
print(IMG_SRC)

print("\nLocal destination:")
print(DATASET)


if os.path.exists(DATASET):

    print("\nRemoving previous local dataset...")

    shutil.rmtree(DATASET)


print("\nCopying dataset to Colab local storage...")

shutil.copytree(
    IMG_SRC,
    DATASET
)


if not has_required_dataset_structure(DATASET):

    raise FileNotFoundError(
        "Copied dataset structure is invalid."
    )


print("\nDataset copied successfully.")

print("Local dataset verified.")


# ================================================================
# CELL 8 - DATASET PATHS
# ================================================================

TRAIN_DIR = os.path.join(
    DATASET,
    "train"
)

VALID_DIR = os.path.join(
    DATASET,
    "valid"
)

TEST_DIR = os.path.join(
    DATASET,
    "test"
)


print("\n" + "=" * 70)
print("FINAL DATASET PATHS")
print("=" * 70)

print("Source:")
print(IMG_SRC)

print("\nLocal dataset:")
print(DATASET)

print("\nTrain:")
print(TRAIN_DIR)

print("\nValid:")
print(VALID_DIR)

print("\nTest:")
print(TEST_DIR)


# ================================================================
# CELL 9 - OUTPUT DIRECTORIES
# ================================================================

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/checkpoints"
)

MODEL_DIR = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/Models/dermoscopy"
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_NAME = "efficientnet_v12"


BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    MODEL_NAME + "_best.keras"
)


FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_final.keras"
)


THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_threshold.txt"
)


RESULTS_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_results.json"
)


ROC_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_roc_data.json"
)


PR_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_pr_data.json"
)


CM_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_confusion_matrix.json"
)


REPORT_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_classification_report.txt"
)


THRESHOLD_TABLE_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME + "_threshold_analysis.csv"
)


print("\n" + "=" * 70)
print("V12 OUTPUT PATHS")
print("=" * 70)

print("Checkpoint:")
print(CHECKPOINT_DIR)

print("\nModel:")
print(FINAL_MODEL_PATH)

print("\nResults:")
print(RESULTS_PATH)


# ================================================================
# CELL 10 - CHECK DATASET DIRECTORIES
# ================================================================

print("\n" + "=" * 70)
print("CHECKING DATASET DIRECTORIES")
print("=" * 70)


for directory in [

    TRAIN_DIR,

    VALID_DIR,

    TEST_DIR

]:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Dataset directory not found:\n{directory}"
        )

    print("OK:", directory)


print("\nDataset structure:")


for split in [

    "train",

    "valid",

    "test"

]:

    split_path = os.path.join(
        DATASET,
        split
    )

    print("\n" + split.upper())

    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            split_path,
            class_name
        )

        print(
            " ",
            class_name,
            "->",
            class_path
        )


# ================================================================
# CELL 11 - IMAGE COUNTS
# ================================================================

def count_images(directory):

    counts = {

        "melanoma": 0,

        "non_melanoma": 0

    }

    extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"

    }

    for class_name in counts:

        class_dir = os.path.join(
            directory,
            class_name
        )

        if not os.path.exists(class_dir):

            continue

        for root, _, files in os.walk(
            class_dir
        ):

            for file in files:

                if Path(
                    file
                ).suffix.lower() in extensions:

                    counts[
                        class_name
                    ] += 1

    return counts


train_counts = count_images(
    TRAIN_DIR
)

valid_counts = count_images(
    VALID_DIR
)

test_counts = count_images(
    TEST_DIR
)


print("\n" + "=" * 70)
print("DATASET COUNTS")
print("=" * 70)


for name, counts in [

    ("TRAIN", train_counts),

    ("VALIDATION", valid_counts),

    ("TEST", test_counts)

]:

    print("\n" + name)

    print(
        "  melanoma:",
        counts["melanoma"]
    )

    print(
        "  non_melanoma:",
        counts["non_melanoma"]
    )

    print(
        "  total:",
        counts["melanoma"] +
        counts["non_melanoma"]
    )


# ================================================================
# CELL 12 - CREATE TF DATASETS
# ================================================================

print("\n" + "=" * 70)
print("CREATING TF.DATA DATASETS")
print("=" * 70)


train_ds = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="binary",

    class_names=CLASS_NAMES,

    image_size=(

        IMG_SIZE,

        IMG_SIZE

    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED

)


valid_ds = tf.keras.utils.image_dataset_from_directory(

    VALID_DIR,

    labels="inferred",

    label_mode="binary",

    class_names=CLASS_NAMES,

    image_size=(

        IMG_SIZE,

        IMG_SIZE

    ),

    batch_size=BATCH_SIZE,

    shuffle=False

)


test_ds = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="binary",

    class_names=CLASS_NAMES,

    image_size=(

        IMG_SIZE,

        IMG_SIZE

    ),

    batch_size=BATCH_SIZE,

    shuffle=False

)


print("\nClass mapping:")

print("0 = non_melanoma")

print("1 = melanoma")

print("\nModel output:")

print("P(melanoma)")


# ================================================================
# CELL 13 - PREFETCH
# ================================================================

train_ds = train_ds.prefetch(
    AUTOTUNE
)

valid_ds = valid_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)


print(
    "Dataset prefetching enabled."
)


# ================================================================
# CELL 14 - V12 DATA AUGMENTATION
# ================================================================

print("\n" + "=" * 70)
print("V12 DATA AUGMENTATION")
print("=" * 70)


data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=ROTATION_FACTOR
        ),

        tf.keras.layers.RandomZoom(
            height_factor=ZOOM_FACTOR,
            width_factor=ZOOM_FACTOR
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=TRANSLATION_FACTOR,
            width_factor=TRANSLATION_FACTOR
        ),

        tf.keras.layers.RandomContrast(
            factor=CONTRAST_FACTOR
        )

    ],

    name="v12_augmentation"

)


print("Augmentation configured.")

print(
    "Rotation:",
    ROTATION_FACTOR
)

print(
    "Zoom:",
    ZOOM_FACTOR
)

print(
    "Translation:",
    TRANSLATION_FACTOR
)

print(
    "Contrast:",
    CONTRAST_FACTOR
)


# ================================================================
# CELL 15 - MELANOMA WEIGHTED BCE
# ================================================================

@tf.keras.utils.register_keras_serializable()
class MelanomaWeightedBinaryCrossentropy(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        non_melanoma_weight=1.0,
        melanoma_weight=1.25,
        name="melanoma_weighted_bce",
        **kwargs
    ):

        super().__init__(
            name=name,
            **kwargs
        )

        self.non_melanoma_weight = float(
            non_melanoma_weight
        )

        self.melanoma_weight = float(
            melanoma_weight
        )


    def call(
        self,
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


        bce = -(

            y_true *
            tf.math.log(y_pred)

            +

            (1.0 - y_true) *
            tf.math.log(1.0 - y_pred)

        )


        weights = (

            y_true *
            self.melanoma_weight

            +

            (1.0 - y_true) *
            self.non_melanoma_weight

        )


        return bce * weights


    def get_config(self):

        config = super().get_config()

        config.update({

            "non_melanoma_weight":
                self.non_melanoma_weight,

            "melanoma_weight":
                self.melanoma_weight

        })

        return config


# ================================================================
# CELL 16 - METRICS
# ================================================================

def create_metrics():

    return [

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        ),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )

    ]


print("Metrics configured.")


# ================================================================
# CELL 17 - CREATE EFFICIENTNETV2-S
# ================================================================

print("\n" + "=" * 70)
print("CREATING V12 MODEL")
print("=" * 70)


inputs = tf.keras.Input(

    shape=(

        IMG_SIZE,

        IMG_SIZE,

        3

    ),

    name="image"

)


x = data_augmentation(
    inputs
)


backbone = tf.keras.applications.EfficientNetV2S(

    include_top=False,

    weights="imagenet",

    input_shape=(

        IMG_SIZE,

        IMG_SIZE,

        3

    ),

    pooling=None,

    include_preprocessing=True,

    name="efficientnetv2-s"

)


backbone.trainable = False


x = backbone(

    x,

    training=False

)


x = tf.keras.layers.GlobalAveragePooling2D(

    name="global_average_pooling"

)(x)


x = tf.keras.layers.Dropout(

    0.35,

    name="dropout_1"

)(x)


x = tf.keras.layers.Dense(

    128,

    activation="swish",

    name="classifier_dense"

)(x)


x = tf.keras.layers.Dropout(

    0.25,

    name="dropout_2"

)(x)


outputs = tf.keras.layers.Dense(

    1,

    activation="sigmoid",

    name="melanoma_probability"

)(x)


model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs,

    name="efficientnet_v12"

)


print("\nV12 model created.")

print("Backbone:", backbone.name)

print(
    "Backbone trainable:",
    backbone.trainable
)


# ================================================================
# CELL 18 - MODEL SUMMARY
# ================================================================

model.summary()


# ================================================================
# CELL 19 - COMPILE STAGE 1
# ================================================================

print("\n" + "=" * 70)
print("COMPILING V12 STAGE 1")
print("=" * 70)


loss_fn = MelanomaWeightedBinaryCrossentropy(

    non_melanoma_weight=
        NON_MELANOMA_WEIGHT,

    melanoma_weight=
        MELANOMA_WEIGHT

)


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=
            STAGE1_LR

    ),

    loss=loss_fn,

    metrics=create_metrics()

)


print("Stage 1 compiled.")

print(
    "Learning rate:",
    STAGE1_LR
)


# ================================================================
# CELL 20 - STAGE 1 CALLBACKS
# ================================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    BEST_MODEL_PATH,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=MIN_LR,

    verbose=1

)


early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=5,

    restore_best_weights=True,

    verbose=1

)


print("Stage 1 callbacks ready.")


# ================================================================
# CELL 21 - STAGE 1 TRAINING
# ================================================================

print("\n" + "=" * 70)
print("V12 STAGE 1 - FROZEN EFFICIENTNETV2-S")
print("=" * 70)


print(
    "Melanoma weight:",
    MELANOMA_WEIGHT
)

print(
    "Backbone trainable:",
    backbone.trainable
)


history_stage1 = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=STAGE1_EPOCHS,

    callbacks=[

        checkpoint,

        reduce_lr,

        early_stop

    ],

    verbose=1

)


# ================================================================
# CELL 22 - LOAD BEST STAGE 1
# ================================================================

print("\n" + "=" * 70)
print("LOADING BEST V12 STAGE 1 MODEL")
print("=" * 70)


best_stage1_model = tf.keras.models.load_model(

    BEST_MODEL_PATH,

    compile=False

)


print(
    "Best Stage 1 model loaded."
)


backbone = best_stage1_model.get_layer(

    "efficientnetv2-s"

)


model = best_stage1_model


print(
    "Backbone:",
    backbone.name
)


# ================================================================
# CELL 23 - FINE-TUNING SETUP
# ================================================================

print("\n" + "=" * 70)
print("PREPARING V12 FINE-TUNING")
print("=" * 70)


backbone.trainable = True


total_layers = len(
    backbone.layers
)


fine_tune_from = int(

    total_layers *
    (1.0 - FINE_TUNE_FRACTION)

)


print(
    "Total backbone layers:",
    total_layers
)

print(
    "Fine-tuning from layer:",
    fine_tune_from
)


for layer in backbone.layers:

    layer.trainable = False


for layer in backbone.layers[
    fine_tune_from:
]:

    if isinstance(

        layer,

        tf.keras.layers.BatchNormalization

    ):

        layer.trainable = False

    else:

        layer.trainable = True


trainable_count = sum(

    1

    for layer in backbone.layers

    if layer.trainable

)


print(
    "Trainable backbone layers:",
    trainable_count
)


# ================================================================
# CELL 24 - COMPILE STAGE 2
# ================================================================

print("\n" + "=" * 70)
print("COMPILING V12 STAGE 2")
print("=" * 70)


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=
            STAGE2_LR

    ),

    loss=MelanomaWeightedBinaryCrossentropy(

        non_melanoma_weight=
            NON_MELANOMA_WEIGHT,

        melanoma_weight=
            MELANOMA_WEIGHT

    ),

    metrics=create_metrics()

)


print("Stage 2 compiled.")

print(
    "Learning rate:",
    STAGE2_LR
)


# ================================================================
# CELL 25 - STAGE 2 CALLBACKS
# ================================================================

checkpoint_stage2 = tf.keras.callbacks.ModelCheckpoint(

    BEST_MODEL_PATH,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)


reduce_lr_stage2 = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=MIN_LR,

    verbose=1

)


early_stop_stage2 = tf.keras.callbacks.EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1

)


print("Stage 2 callbacks ready.")


# ================================================================
# CELL 26 - STAGE 2 TRAINING
# ================================================================

print("\n" + "=" * 70)
print("V12 STAGE 2 - FINE-TUNING")
print("=" * 70)


print(
    "Trainable backbone layers:",
    trainable_count
)

print(
    "Fine-tuning learning rate:",
    STAGE2_LR
)


history_stage2 = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=[

        checkpoint_stage2,

        reduce_lr_stage2,

        early_stop_stage2

    ],

    verbose=1

)


# ================================================================
# CELL 27 - LOAD ABSOLUTE BEST V12 MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING ABSOLUTE BEST V12 MODEL")
print("=" * 70)


best_model = tf.keras.models.load_model(

    BEST_MODEL_PATH,

    compile=False

)


print(
    "Absolute best V12 model loaded."
)


# ================================================================
# CELL 28 - PREDICTION FUNCTION
# ================================================================

def get_predictions(
    model,
    dataset
):

    probabilities = []

    labels = []


    for images, batch_labels in dataset:

        batch_predictions = model.predict(

            images,

            verbose=0

        )


        probabilities.extend(

            batch_predictions.reshape(-1)

        )


        labels.extend(

            batch_labels.numpy().reshape(-1)

        )


    probabilities = np.asarray(

        probabilities,

        dtype=np.float32

    )


    labels = np.asarray(

        labels,

        dtype=np.int32

    )


    return labels, probabilities


# ================================================================
# CELL 29 - VALIDATION PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("V12 VALIDATION PREDICTIONS")
print("=" * 70)


y_val, p_val = get_predictions(

    best_model,

    valid_ds

)


print(
    "Validation samples:",
    len(y_val)
)

print(
    "Validation melanoma:",
    int(np.sum(y_val == 1))
)

print(
    "Validation non-melanoma:",
    int(np.sum(y_val == 0))
)


# ================================================================
# CELL 30 - VALIDATION ROC / PR
# ================================================================

val_roc_auc = roc_auc_score(

    y_val,

    p_val

)


val_pr_auc = average_precision_score(

    y_val,

    p_val

)


print("\n" + "=" * 70)
print("V12 VALIDATION PERFORMANCE")
print("=" * 70)


print(
    "Validation ROC-AUC:",
    f"{val_roc_auc:.4f}"
)

print(
    "Validation PR-AUC:",
    f"{val_pr_auc:.4f}"
)


# ================================================================
# CELL 31 - THRESHOLD EVALUATION FUNCTION
# ================================================================

def evaluate_threshold(

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


# ================================================================
# CELL 32 - DENSE VALIDATION THRESHOLD SEARCH
# ================================================================

print("\n" + "=" * 70)
print("V12 VALIDATION THRESHOLD SEARCH")
print("=" * 70)


# V11 used 0.01 increments.
# V12 uses 0.005 increments.

thresholds = np.arange(

    0.005,

    1.000,

    0.005

)


threshold_results = []


for threshold in thresholds:

    result = evaluate_threshold(

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


# ------------------------------------------------
# BEST ACCURACY
# ------------------------------------------------

best_accuracy_row = threshold_df.loc[

    threshold_df["accuracy"].idxmax()

]


# ------------------------------------------------
# BEST F1
# ------------------------------------------------

best_f1_row = threshold_df.loc[

    threshold_df["f1"].idxmax()

]


# ------------------------------------------------
# BEST BALANCED ACCURACY
# ------------------------------------------------

best_balanced_row = threshold_df.loc[

    threshold_df[
        "balanced_accuracy"
    ].idxmax()

]


# ------------------------------------------------
# BEST YOUDEN J
# ------------------------------------------------

threshold_df["youden_j"] = (

    threshold_df["sensitivity"]

    +

    threshold_df["specificity"]

    - 1.0

)


best_youden_row = threshold_df.loc[

    threshold_df["youden_j"].idxmax()

]


print("\nBest accuracy:")

print(
    best_accuracy_row.to_dict()
)


print("\nBest F1:")

print(
    best_f1_row.to_dict()
)


print("\nBest balanced accuracy:")

print(
    best_balanced_row.to_dict()
)


print("\nBest Youden J:")

print(
    best_youden_row.to_dict()
)


# ================================================================
# CELL 33 - SENSITIVITY-CONSTRAINED SEARCH
# ================================================================

print("\n" + "=" * 70)
print("V12 SENSITIVITY-CONSTRAINED THRESHOLD SEARCH")
print("=" * 70)


sensitivity_targets = [

    0.70,

    0.75,

    0.80,

    0.85,

    0.90,

    0.95

]


sensitivity_thresholds = {}


for target in sensitivity_targets:

    eligible = threshold_df[

        threshold_df[
            "sensitivity"
        ] >= target

    ]


    if len(eligible) == 0:

        print(

            f"{int(target * 100)}% sensitivity "
            "-> no threshold found"

        )

        continue


    # ------------------------------------------------
    # Among thresholds meeting the target sensitivity,
    # select highest specificity.
    #
    # If specificity ties, use highest F1.
    # ------------------------------------------------

    max_specificity = eligible[
        "specificity"
    ].max()


    eligible_specificity = eligible[

        eligible[
            "specificity"
        ] == max_specificity

    ]


    best = eligible_specificity.loc[

        eligible_specificity[
            "f1"
        ].idxmax()

    ]


    sensitivity_thresholds[
        int(target * 100)
    ] = best.to_dict()


    print(

        f"{int(target * 100)}% sensitivity -> "

        f"threshold={best['threshold']:.3f}, "

        f"sensitivity={best['sensitivity']:.4f}, "

        f"specificity={best['specificity']:.4f}, "

        f"precision={best['precision']:.4f}, "

        f"F1={best['f1']:.4f}, "

        f"balanced={best['balanced_accuracy']:.4f}"

    )


# ================================================================
# CELL 34 - SELECT V12 FINAL THRESHOLD
# ================================================================

print("\n" + "=" * 70)
print("SELECTING V12 FINAL THRESHOLD")
print("=" * 70)


# ------------------------------------------------
# PRIMARY V12 OBJECTIVE:
#
# Maximize F1 while requiring at least 70%
# validation sensitivity.
#
# This is deliberately different from V11's
# maximum balanced-accuracy threshold.
# ------------------------------------------------

MIN_REQUIRED_SENSITIVITY = 0.70


eligible_f1 = threshold_df[

    threshold_df[
        "sensitivity"
    ] >= MIN_REQUIRED_SENSITIVITY

]


if len(eligible_f1) == 0:

    print(
        "WARNING: No threshold satisfies "
        "minimum sensitivity."
    )

    FINAL_THRESHOLD = float(
        best_balanced_row["threshold"]
    )

    THRESHOLD_REASON = (
        "fallback to validation balanced accuracy"
    )

else:

    best_f1_constrained = eligible_f1.loc[

        eligible_f1[
            "f1"
        ].idxmax()

    ]


    FINAL_THRESHOLD = float(

        best_f1_constrained[
            "threshold"
        ]

    )


    THRESHOLD_REASON = (

        "maximum validation F1 "
        "subject to validation sensitivity >= 70%"

    )


print(
    "\nMinimum required sensitivity:",
    MIN_REQUIRED_SENSITIVITY
)


print(
    "Selected threshold:",
    f"{FINAL_THRESHOLD:.3f}"
)


print(
    "Reason:",
    THRESHOLD_REASON
)


selected_validation_metrics = evaluate_threshold(

    y_val,

    p_val,

    FINAL_THRESHOLD

)


print("\nValidation performance at selected threshold:")

for key, value in selected_validation_metrics.items():

    if key != "threshold":

        print(
            f"{key}: {value:.4f}"
        )


# ================================================================
# CELL 35 - TEST PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("V12 TEST PREDICTIONS")
print("=" * 70)


y_test, p_test = get_predictions(

    best_model,

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
# CELL 36 - TEST ROC / PR
# ================================================================

test_roc_auc = roc_auc_score(

    y_test,

    p_test

)


test_pr_auc = average_precision_score(

    y_test,

    p_test

)


print("\n" + "=" * 70)
print("V12 TEST ROC / PR")
print("=" * 70)


print(
    "Test ROC-AUC:",
    f"{test_roc_auc:.4f}"
)


print(
    "Test PR-AUC:",
    f"{test_pr_auc:.4f}"
)


# ================================================================
# CELL 37 - FINAL TEST RESULTS
# ================================================================

final_result = evaluate_threshold(

    y_test,

    p_test,

    FINAL_THRESHOLD

)


print("\n" + "=" * 70)
print("FINAL V12 TEST RESULTS")
print("=" * 70)


print(
    "Test ROC-AUC:",
    f"{test_roc_auc:.4f}"
)

print(
    "Test PR-AUC:",
    f"{test_pr_auc:.4f}"
)

print(
    "Threshold:",
    f"{FINAL_THRESHOLD:.3f}"
)

print(
    "Accuracy:",
    f"{final_result['accuracy']:.4f}"
)

print(
    "Precision:",
    f"{final_result['precision']:.4f}"
)

print(
    "Sensitivity:",
    f"{final_result['sensitivity']:.4f}"
)

print(
    "Specificity:",
    f"{final_result['specificity']:.4f}"
)

print(
    "F1 Score:",
    f"{final_result['f1']:.4f}"
)

print(
    "Balanced Accuracy:",
    f"{final_result['balanced_accuracy']:.4f}"
)


# ================================================================
# CELL 38 - CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    y_test,

    (
        p_test >= FINAL_THRESHOLD
    ).astype(int),

    labels=[0, 1]

)


print("\n" + "=" * 70)
print("V12 CONFUSION MATRIX")
print("=" * 70)

print(cm)


tn, fp, fn, tp = cm.ravel()


print("\nTrue Negative :", tn)

print("False Positive:", fp)

print("False Negative:", fn)

print("True Positive :", tp)


# ================================================================
# CELL 39 - CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("V12 CLASSIFICATION REPORT")
print("=" * 70)


y_test_pred = (

    p_test >= FINAL_THRESHOLD

).astype(int)


report = classification_report(

    y_test,

    y_test_pred,

    labels=[0, 1],

    target_names=[

        "non_melanoma",

        "melanoma"

    ],

    digits=4,

    zero_division=0

)


print(report)


# ================================================================
# CELL 40 - COMPLETE TEST THRESHOLD COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("V12 TEST THRESHOLD COMPARISON")
print("=" * 70)


comparison_thresholds = {

    "v12_selected":
        FINAL_THRESHOLD,

    "v12_best_f1":
        float(
            best_f1_row[
                "threshold"
            ]
        ),

    "v12_best_balanced":
        float(
            best_balanced_row[
                "threshold"
            ]
        ),

    "v12_best_youden":
        float(
            best_youden_row[
                "threshold"
            ]
        ),

    "default_0.50":
        0.50,

    "sensitivity_70%":
        sensitivity_thresholds.get(
            70,
            {}
        ).get(
            "threshold",
            None
        ),

    "sensitivity_75%":
        sensitivity_thresholds.get(
            75,
            {}
        ).get(
            "threshold",
            None
        ),

    "sensitivity_80%":
        sensitivity_thresholds.get(
            80,
            {}
        ).get(
            "threshold",
            None
        ),

    "sensitivity_85%":
        sensitivity_thresholds.get(
            85,
            {}
        ).get(
            "threshold",
            None
        ),

    "sensitivity_90%":
        sensitivity_thresholds.get(
            90,
            {}
        ).get(
            "threshold",
            None
        ),

    "sensitivity_95%":
        sensitivity_thresholds.get(
            95,
            {}
        ).get(
            "threshold",
            None
        )

}


comparison_results = {}


comparison_rows = []


for name, threshold in comparison_thresholds.items():

    if threshold is None:

        continue


    result = evaluate_threshold(

        y_test,

        p_test,

        threshold

    )


    comparison_results[name] = result

    comparison_rows.append(result)


    print("\n" + "-" * 50)

    print(name)

    print(
        "Threshold:",
        f"{threshold:.3f}"
    )

    print(
        "Accuracy:",
        f"{result['accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{result['precision']:.4f}"
    )

    print(
        "Sensitivity:",
        f"{result['sensitivity']:.4f}"
    )

    print(
        "Specificity:",
        f"{result['specificity']:.4f}"
    )

    print(
        "F1:",
        f"{result['f1']:.4f}"
    )

    print(
        "Balanced Accuracy:",
        f"{result['balanced_accuracy']:.4f}"
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


# ================================================================
# CELL 41 - PROBABILITY ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("V12 PROBABILITY ANALYSIS")
print("=" * 70)


melanoma_probs = p_test[
    y_test == 1
]


non_melanoma_probs = p_test[
    y_test == 0
]


probability_analysis = {

    "melanoma": {

        "mean":
            float(
                np.mean(
                    melanoma_probs
                )
            ),

        "median":
            float(
                np.median(
                    melanoma_probs
                )
            ),

        "min":
            float(
                np.min(
                    melanoma_probs
                )
            ),

        "max":
            float(
                np.max(
                    melanoma_probs
                )
            )

    },

    "non_melanoma": {

        "mean":
            float(
                np.mean(
                    non_melanoma_probs
                )
            ),

        "median":
            float(
                np.median(
                    non_melanoma_probs
                )
            ),

        "min":
            float(
                np.min(
                    non_melanoma_probs
                )
            ),

        "max":
            float(
                np.max(
                    non_melanoma_probs
                )
            )

    }

}


print("\nMelanoma probability:")

print(
    "Mean:",
    f"{probability_analysis['melanoma']['mean']:.4f}"
)

print(
    "Median:",
    f"{probability_analysis['melanoma']['median']:.4f}"
)

print(
    "Min:",
    f"{probability_analysis['melanoma']['min']:.4f}"
)

print(
    "Max:",
    f"{probability_analysis['melanoma']['max']:.4f}"
)


print("\nNon-melanoma probability:")

print(
    "Mean:",
    f"{probability_analysis['non_melanoma']['mean']:.4f}"
)

print(
    "Median:",
    f"{probability_analysis['non_melanoma']['median']:.4f}"
)

print(
    "Min:",
    f"{probability_analysis['non_melanoma']['min']:.4f}"
)

print(
    "Max:",
    f"{probability_analysis['non_melanoma']['max']:.4f}"
)


# ================================================================
# CELL 42 - ROC CURVE
# ================================================================

fpr, tpr, roc_thresholds = roc_curve(

    y_test,

    p_test

)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    fpr,

    tpr,

    label=f"V12 ROC-AUC = {test_roc_auc:.4f}"

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
    "V12 - ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# CELL 43 - PRECISION RECALL CURVE
# ================================================================

precision_curve, recall_curve, pr_thresholds = (

    precision_recall_curve(

        y_test,

        p_test

    )

)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    recall_curve,

    precision_curve,

    label=f"V12 PR-AUC = {test_pr_auc:.4f}"

)


plt.xlabel(
    "Recall / Sensitivity"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "V12 - Precision Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# CELL 44 - CONFUSION MATRIX PLOT
# ================================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(

    f"V12 Confusion Matrix\n"
    f"Threshold = {FINAL_THRESHOLD:.3f}"

)


plt.colorbar()


plt.xticks(

    [0, 1],

    [

        "non_melanoma",

        "melanoma"

    ],

    rotation=45

)


plt.yticks(

    [0, 1],

    [

        "non_melanoma",

        "melanoma"

    ]

)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )


plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()

plt.show()


# ================================================================
# CELL 45 - THRESHOLD CURVES
# ================================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    threshold_df["threshold"],

    threshold_df["sensitivity"],

    label="Sensitivity"

)


plt.plot(

    threshold_df["threshold"],

    threshold_df["specificity"],

    label="Specificity"

)


plt.plot(

    threshold_df["threshold"],

    threshold_df["balanced_accuracy"],

    label="Balanced Accuracy"

)


plt.plot(

    threshold_df["threshold"],

    threshold_df["f1"],

    label="F1"

)


plt.axvline(

    FINAL_THRESHOLD,

    linestyle="--",

    label=f"Selected = {FINAL_THRESHOLD:.3f}"

)


plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "V12 Threshold Analysis"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ================================================================
# CELL 46 - SAVE ROC DATA
# ================================================================

roc_data = {

    "fpr":
        fpr.tolist(),

    "tpr":
        tpr.tolist(),

    "thresholds":
        roc_thresholds.tolist(),

    "roc_auc":
        float(test_roc_auc)

}


with open(
    ROC_PATH,
    "w"
) as f:

    json.dump(

        roc_data,

        f,

        indent=4

    )


print(
    "ROC data saved:"
)

print(ROC_PATH)


# ================================================================
# CELL 47 - SAVE PR DATA
# ================================================================

pr_data = {

    "precision":
        precision_curve.tolist(),

    "recall":
        recall_curve.tolist(),

    "thresholds":
        pr_thresholds.tolist(),

    "pr_auc":
        float(test_pr_auc)

}


with open(
    PR_PATH,
    "w"
) as f:

    json.dump(

        pr_data,

        f,

        indent=4

    )


print(
    "PR data saved:"
)

print(PR_PATH)


# ================================================================
# CELL 48 - SAVE CONFUSION MATRIX
# ================================================================

cm_data = {

    "matrix":
        cm.tolist(),

    "labels": [

        "non_melanoma",

        "melanoma"

    ],

    "threshold":
        float(FINAL_THRESHOLD)

}


with open(
    CM_PATH,
    "w"
) as f:

    json.dump(

        cm_data,

        f,

        indent=4

    )


print(
    "Confusion matrix saved:"
)

print(CM_PATH)


# ================================================================
# CELL 49 - SAVE CLASSIFICATION REPORT
# ================================================================

with open(
    REPORT_PATH,
    "w"
) as f:

    f.write(report)


print(
    "Classification report saved:"
)

print(REPORT_PATH)


# ================================================================
# CELL 50 - SAVE FINAL THRESHOLD
# ================================================================

with open(
    THRESHOLD_PATH,
    "w"
) as f:

    f.write(
        str(FINAL_THRESHOLD)
    )


print(
    "Threshold saved:"
)

print(THRESHOLD_PATH)


# ================================================================
# CELL 51 - SAVE THRESHOLD ANALYSIS
# ================================================================

threshold_df.to_csv(

    THRESHOLD_TABLE_PATH,

    index=False

)


print(
    "Threshold analysis saved:"
)

print(
    THRESHOLD_TABLE_PATH
)


# ================================================================
# CELL 52 - EXPERIMENT SUMMARY
# ================================================================

experiment_summary = {

    "experiment":
        "V12",

    "based_on":
        "V11",

    "model":
        "EfficientNetV2-S",

    "tensorflow_version":
        tf.__version__,

    "keras_version":
        keras.__version__,

    "image_size":
        IMG_SIZE,

    "batch_size":
        BATCH_SIZE,

    "seed":
        SEED,

    "dataset_source":
        IMG_SRC,

    "dataset_root":
        DATASET_ROOT,

    "label_mapping": {

        "0":
            "non_melanoma",

        "1":
            "melanoma"

    },

    "model_output":
        "P(melanoma)",

    "loss":
        "MelanomaWeightedBinaryCrossentropy",

    "non_melanoma_weight":
        NON_MELANOMA_WEIGHT,

    "melanoma_weight":
        MELANOMA_WEIGHT,

    "v11_melanoma_weight":
        1.5,

    "v12_melanoma_weight":
        MELANOMA_WEIGHT,

    "stage1_epochs":
        STAGE1_EPOCHS,

    "stage2_epochs":
        STAGE2_EPOCHS,

    "stage1_learning_rate":
        STAGE1_LR,

    "stage2_learning_rate":
        STAGE2_LR,

    "fine_tune_fraction":
        FINE_TUNE_FRACTION,

    "train_counts":
        train_counts,

    "validation_counts":
        valid_counts,

    "test_counts":
        test_counts,

    "validation_roc_auc":
        float(val_roc_auc),

    "validation_pr_auc":
        float(val_pr_auc),

    "validation_best_accuracy":
        best_accuracy_row.to_dict(),

    "validation_best_f1":
        best_f1_row.to_dict(),

    "validation_best_balanced_accuracy":
        best_balanced_row.to_dict(),

    "validation_best_youden":
        best_youden_row.to_dict(),

    "minimum_required_sensitivity":
        MIN_REQUIRED_SENSITIVITY,

    "selected_threshold":
        float(FINAL_THRESHOLD),

    "threshold_reason":
        THRESHOLD_REASON,

    "selected_validation_metrics":
        selected_validation_metrics,

    "test_roc_auc":
        float(test_roc_auc),

    "test_pr_auc":
        float(test_pr_auc),

    "test_metrics":
        final_result,

    "threshold_comparison":
        comparison_results,

    "confusion_matrix":
        cm.tolist(),

    "probability_analysis":
        probability_analysis

}


# ================================================================
# CELL 53 - SAVE EXPERIMENT SUMMARY
# ================================================================

with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(

        experiment_summary,

        f,

        indent=4

    )


print(
    "Experiment summary saved:"
)

print(RESULTS_PATH)


# ================================================================
# CELL 54 - SAVE FINAL MODEL
# ================================================================

print("\n" + "=" * 70)
print("SAVING FINAL V12 MODEL")
print("=" * 70)


best_model.save(
    FINAL_MODEL_PATH
)


print(
    "V12 MODEL SAVED:"
)

print(FINAL_MODEL_PATH)


# ================================================================
# CELL 55 - V11 VS V12 COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("V11 VS V12")
print("=" * 70)


v11_results = {

    "ROC-AUC": 0.8707,

    "PR-AUC": 0.4921,

    "Accuracy": 0.7415,

    "Precision": 0.2793,

    "Sensitivity": 0.8304,

    "Specificity": 0.7303,

    "F1": 0.4180,

    "Balanced Accuracy": 0.7803

}


v12_results = {

    "ROC-AUC":
        test_roc_auc,

    "PR-AUC":
        test_pr_auc,

    "Accuracy":
        final_result["accuracy"],

    "Precision":
        final_result["precision"],

    "Sensitivity":
        final_result["sensitivity"],

    "Specificity":
        final_result["specificity"],

    "F1":
        final_result["f1"],

    "Balanced Accuracy":
        final_result["balanced_accuracy"]

}


comparison = pd.DataFrame({

    "V11": v11_results,

    "V12": v12_results

})


comparison["Change"] = (

    comparison["V12"]

    -

    comparison["V11"]

)


print(comparison.round(4))


# ================================================================
# CELL 56 - FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("V12 EXPERIMENT COMPLETE")
print("=" * 70)


print("\nDATASET:")

print(DATASET_ROOT)


print("\nLABEL MAPPING:")

print("0 = non_melanoma")

print("1 = melanoma")


print("\nMODEL OUTPUT:")

print("P(melanoma)")


print("\nV12 MELANOMA WEIGHT:")

print(MELANOMA_WEIGHT)


print("\nFINAL THRESHOLD:")

print(
    f"{FINAL_THRESHOLD:.4f}"
)


print("\nTHRESHOLD REASON:")

print(
    THRESHOLD_REASON
)


print("\nVALIDATION ROC-AUC:")

print(
    f"{val_roc_auc:.4f}"
)


print("\nVALIDATION PR-AUC:")

print(
    f"{val_pr_auc:.4f}"
)


print("\nTEST ROC-AUC:")

print(
    f"{test_roc_auc:.4f}"
)


print("\nTEST PR-AUC:")

print(
    f"{test_pr_auc:.4f}"
)


print("\nTEST ACCURACY:")

print(
    f"{final_result['accuracy']:.4f}"
)


print("\nTEST PRECISION:")

print(
    f"{final_result['precision']:.4f}"
)


print("\nTEST SENSITIVITY:")

print(
    f"{final_result['sensitivity']:.4f}"
)


print("\nTEST SPECIFICITY:")

print(
    f"{final_result['specificity']:.4f}"
)


print("\nTEST F1:")

print(
    f"{final_result['f1']:.4f}"
)


print("\nTEST BALANCED ACCURACY:")

print(
    f"{final_result['balanced_accuracy']:.4f}"
)


print("\nCONFUSION MATRIX:")

print(cm)


print("\nTRUE NEGATIVE:")

print(tn)


print("\nFALSE POSITIVE:")

print(fp)


print("\nFALSE NEGATIVE:")

print(fn)


print("\nTRUE POSITIVE:")

print(tp)


print("\nFINAL MODEL:")

print(FINAL_MODEL_PATH)


print("\nTHRESHOLD:")

print(THRESHOLD_PATH)


print("\nSUMMARY:")

print(RESULTS_PATH)


print("\nROC DATA:")

print(ROC_PATH)


print("\nPR DATA:")

print(PR_PATH)


print("\nCONFUSION MATRIX DATA:")

print(CM_PATH)


print("\nCLASSIFICATION REPORT:")

print(REPORT_PATH)


print("\nTHRESHOLD ANALYSIS:")

print(THRESHOLD_TABLE_PATH)


print("\n")
print("=" * 70)
print("V12 FINISHED SUCCESSFULLY")
print("=" * 70)