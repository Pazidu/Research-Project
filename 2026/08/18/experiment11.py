# ================================================================
# V11 - MELANOMA CLASSIFICATION
# EfficientNetV2-S
# Google Colab
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
print("V11 MELANOMA CLASSIFICATION")
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

DRIVE_ROOT = "/content/drive/MyDrive"

print("\nDrive root:")
print(DRIVE_ROOT)

print("\nMyDrive contents:")

for item in os.listdir(DRIVE_ROOT)[:30]:
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

print("\nRandom seed:", SEED)


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

        print(
            "Memory growth warning:",
            e
        )

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
# ACTUAL DATASET LOCATION IN GOOGLE DRIVE
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
# CLASS NAMES
# IMPORTANT:
# 0 = non_melanoma
# 1 = melanoma
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
# MELANOMA WEIGHT
# ------------------------------------------------

NON_MELANOMA_WEIGHT = 1.0

MELANOMA_WEIGHT = 1.5


# ------------------------------------------------
# FINE-TUNING
# ------------------------------------------------

FINE_TUNE_FRACTION = 0.35


print("Drive root:")
print(DRIVE_ROOT)

print("\nGoogle Drive dataset source:")
print(IMG_SRC)

print("\nLocal Colab dataset:")
print(DATASET)

print("\nClass names:")
print(CLASS_NAMES)

print("\nImage size:", IMG_SIZE)

print("Batch size:", BATCH_SIZE)

print("Melanoma weight:", MELANOMA_WEIGHT)


# ================================================================
# CELL 6 - DATASET STRUCTURE CHECK
# ================================================================

print("\n" + "=" * 70)
print("CHECKING SOURCE DATASET")
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

        "Please make sure your Google Drive contains:\n\n"

        "MyDrive/\n"
        "└── Colab Notebooks/\n"
        "    └── newdata_backup/\n"
        "        ├── train/\n"
        "        │   ├── melanoma/\n"
        "        │   └── non_melanoma/\n"
        "        ├── valid/\n"
        "        │   ├── melanoma/\n"
        "        │   └── non_melanoma/\n"
        "        └── test/\n"
        "            ├── melanoma/\n"
        "            └── non_melanoma/\n"
    )


print("Source directory exists.")


if not has_required_dataset_structure(IMG_SRC):

    raise FileNotFoundError(

        "\nDataset exists, but its structure is incorrect.\n\n"

        "Expected structure:\n"

        "newdata_backup/\n"
        "├── train/\n"
        "│   ├── melanoma/\n"
        "│   └── non_melanoma/\n"
        "├── valid/\n"
        "│   ├── melanoma/\n"
        "│   └── non_melanoma/\n"
        "└── test/\n"
        "    ├── melanoma/\n"
        "    └── non_melanoma/\n"
    )


print("Dataset structure verified.")


# ================================================================
# CELL 7 - COPY DATASET TO LOCAL COLAB STORAGE
# ================================================================

print("\n" + "=" * 70)
print("SETTING UP LOCAL DATASET")
print("=" * 70)


print("Source:")
print(IMG_SRC)

print("\nDestination:")
print(DATASET)


if os.path.exists(DATASET):

    print("\nRemoving previous local dataset...")

    shutil.rmtree(DATASET)


print("\nCopying dataset from Google Drive...")
print("This may take some time depending on dataset size.")


shutil.copytree(
    IMG_SRC,
    DATASET
)


print("\nDataset copied successfully.")


if not has_required_dataset_structure(DATASET):

    raise FileNotFoundError(

        "Copied dataset does not have the required structure:\n"
        f"{DATASET}"

    )


print("Local dataset structure verified.")


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

print("\n" + "=" * 70)
print("CREATING OUTPUT DIRECTORIES")
print("=" * 70)


CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/Models/dermoscopy/checkpoints"
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


MODEL_NAME = "efficientnet_v11"


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


print("Checkpoint directory:")
print(CHECKPOINT_DIR)

print("\nModel directory:")
print(MODEL_DIR)


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


print("\nDataset directory structure:")


for split in [

    "train",
    "valid",
    "test"

]:

    split_path = os.path.join(
        DATASET,
        split
    )

    print(f"\n{split.upper()}:")


    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            split_path,
            class_name
        )

        print(
            "  ",
            class_name,
            "->",
            class_path
        )


# ================================================================
# CELL 11 - DATASET COUNTS
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

            print(
                "WARNING: missing class:",
                class_dir
            )

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


print("\nTRAIN")

print(
    "melanoma:",
    train_counts["melanoma"]
)

print(
    "non_melanoma:",
    train_counts["non_melanoma"]
)


print("\nVALID")

print(
    "melanoma:",
    valid_counts["melanoma"]
)

print(
    "non_melanoma:",
    valid_counts["non_melanoma"]
)


print("\nTEST")

print(
    "melanoma:",
    test_counts["melanoma"]
)

print(
    "non_melanoma:",
    test_counts["non_melanoma"]
)


# ================================================================
# CELL 12 - CREATE DATASETS
# ================================================================

print("\n" + "=" * 70)
print("CREATING TENSORFLOW DATASETS")
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


print("\nClass names:")
print(CLASS_NAMES)


print("\nIMPORTANT LABEL MAPPING")

print("0 = non_melanoma")

print("1 = melanoma")

print("MODEL OUTPUT = P(melanoma)")


# ================================================================
# CELL 13 - PERFORMANCE SETTINGS
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
# CELL 14 - DATA AUGMENTATION
# ================================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.08
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        )

    ],

    name="v11_augmentation"

)


print(
    "Data augmentation created."
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
        melanoma_weight=1.5,
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


print(
    "Metrics configured."
)


# ================================================================
# CELL 17 - CREATE EFFICIENTNETV2-S
# ================================================================

print("\n" + "=" * 70)
print("CREATING V11 MODEL")
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

    name="efficientnet_v11"

)


print(
    "\nModel created successfully."
)

print(
    "Model:",
    model.name
)

print(
    "Backbone:",
    backbone.name
)


# ================================================================
# CELL 18 - MODEL SUMMARY
# ================================================================

model.summary()


# ================================================================
# CELL 19 - COMPILE STAGE 1
# ================================================================

print("\n" + "=" * 70)
print("COMPILING STAGE 1")
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


print(
    "Stage 1 compiled."
)


# ================================================================
# CELL 20 - CALLBACKS STAGE 1
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


print(
    "Stage 1 callbacks ready."
)


# ================================================================
# CELL 21 - STAGE 1 TRAINING
# ================================================================

print("\n")
print("=" * 70)
print("STAGE 1 - FROZEN EFFICIENTNETV2-S")
print("=" * 70)


print(
    "Melanoma loss weight:",
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
# CELL 22 - LOAD BEST STAGE 1 MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING BEST STAGE 1 MODEL")
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


print(
    "Backbone found:",
    backbone.name
)


model = best_stage1_model


# ================================================================
# CELL 23 - FINE-TUNING SETUP
# ================================================================

print("\n" + "=" * 70)
print("PREPARING FINE-TUNING")
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


# ------------------------------------------------
# Freeze everything
# ------------------------------------------------

for layer in backbone.layers:

    layer.trainable = False


# ------------------------------------------------
# Unfreeze final portion
# ------------------------------------------------

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
print("COMPILING STAGE 2")
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


print(
    "Stage 2 compiled."
)


# ================================================================
# CELL 25 - FINE-TUNING CALLBACKS
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


print(
    "Stage 2 callbacks ready."
)


# ================================================================
# CELL 26 - STAGE 2 TRAINING
# ================================================================

print("\n")
print("=" * 70)
print("STAGE 2 - FINE-TUNING EFFICIENTNETV2-S")
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
# CELL 27 - LOAD ABSOLUTE BEST V11 MODEL
# ================================================================

print("\n")
print("=" * 70)
print("LOADING ABSOLUTE BEST V11 MODEL")
print("=" * 70)


best_model = tf.keras.models.load_model(

    BEST_MODEL_PATH,

    compile=False

)


print(
    "Absolute best V11 model loaded."
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

print("\n")
print("=" * 70)
print("VALIDATION PREDICTIONS")
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


print(
    "\nValidation ROC-AUC:",
    round(val_roc_auc, 4)
)


print(
    "Validation PR-AUC:",
    round(val_pr_auc, 4)
)


# ================================================================
# CELL 31 - THRESHOLD EVALUATION
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

        "threshold":
            float(threshold),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "balanced_accuracy":
            float(balanced),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)

    }


# ================================================================
# CELL 32 - VALIDATION THRESHOLD SEARCH
# ================================================================

print("\n")
print("=" * 70)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 70)


thresholds = np.arange(

    0.01,

    1.00,

    0.01

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


best_accuracy_row = threshold_df.loc[

    threshold_df["accuracy"].idxmax()

]


best_f1_row = threshold_df.loc[

    threshold_df["f1"].idxmax()

]


best_balanced_row = threshold_df.loc[

    threshold_df["balanced_accuracy"].idxmax()

]


print("\nBest accuracy threshold:")

print(
    round(
        best_accuracy_row["threshold"],
        2
    )
)

print(
    "Validation accuracy:",
    round(
        best_accuracy_row["accuracy"],
        4
    )
)


print("\nBest F1 threshold:")

print(
    round(
        best_f1_row["threshold"],
        2
    )
)

print(
    "Validation F1:",
    round(
        best_f1_row["f1"],
        4
    )
)


print("\nBest balanced accuracy threshold:")

print(
    round(
        best_balanced_row["threshold"],
        2
    )
)

print(
    "Validation balanced accuracy:",
    round(
        best_balanced_row["balanced_accuracy"],
        4
    )
)


# ================================================================
# CELL 33 - SENSITIVITY TARGET SEARCH
# ================================================================

print("\n")
print("=" * 70)
print("SENSITIVITY TARGET THRESHOLDS")
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
        threshold_df["sensitivity"] >= target
    ]


    if len(eligible) == 0:

        print(

            f"{int(target * 100)}% sensitivity "
            "-> no threshold found"

        )

        continue


    best = eligible.loc[

        eligible["specificity"].idxmax()

    ]


    sensitivity_thresholds[
        int(target * 100)
    ] = best.to_dict()


    print(

        f"{int(target * 100)}% sensitivity -> "

        f"threshold={best['threshold']:.2f}, "

        f"sensitivity={best['sensitivity']:.4f}, "

        f"specificity={best['specificity']:.4f}, "

        f"precision={best['precision']:.4f}, "

        f"F1={best['f1']:.4f}, "

        f"balanced={best['balanced_accuracy']:.4f}"

    )


# ================================================================
# CELL 34 - SELECT FINAL VALIDATION THRESHOLD
# ================================================================

FINAL_THRESHOLD = float(

    best_balanced_row[
        "threshold"
    ]

)


THRESHOLD_REASON = (

    "validation balanced accuracy optimization"

)


print("\n")
print("=" * 70)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 70)


print(
    "Final threshold:",
    round(FINAL_THRESHOLD, 2)
)


print(
    "Reason:",
    THRESHOLD_REASON
)


# ================================================================
# CELL 35 - TEST PREDICTIONS
# ================================================================

print("\n")
print("=" * 70)
print("TEST PREDICTIONS")
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


print(
    "\nTest ROC-AUC:",
    round(test_roc_auc, 4)
)


print(
    "Test PR-AUC:",
    round(test_pr_auc, 4)
)


# ================================================================
# CELL 37 - FINAL TEST METRICS
# ================================================================

final_result = evaluate_threshold(

    y_test,

    p_test,

    FINAL_THRESHOLD

)


print("\n")
print("=" * 70)
print("FINAL V11 TEST RESULTS")
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
    f"{FINAL_THRESHOLD:.2f}"
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


print("\nConfusion Matrix:")
print(cm)


# ================================================================
# CELL 39 - CLASSIFICATION REPORT
# ================================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
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
# CELL 40 - THRESHOLD COMPARISON
# ================================================================

print("\n")
print("=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)


comparison_thresholds = {

    "balanced_validation":
        FINAL_THRESHOLD,

    "f1_validation":
        float(
            best_f1_row["threshold"]
        ),

    "accuracy_validation":
        float(
            best_accuracy_row["threshold"]
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
        )

}


comparison_results = {}


for name, threshold in comparison_thresholds.items():

    if threshold is None:
        continue


    result = evaluate_threshold(

        y_test,

        p_test,

        threshold

    )


    comparison_results[name] = result


    print("\n")
    print(name)

    print(
        "Threshold:",
        f"{threshold:.2f}"
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


# ================================================================
# CELL 41 - PROBABILITY ANALYSIS
# ================================================================

print("\n")
print("=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)


melanoma_probs = p_test[
    y_test == 1
]


non_melanoma_probs = p_test[
    y_test == 0
]


print("\nMelanoma probability:")


print(
    "Mean:",
    round(
        float(
            np.mean(
                melanoma_probs
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
                melanoma_probs
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
                melanoma_probs
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
                melanoma_probs
            )
        ),
        4
    )
)


print("\nNon-melanoma probability:")


print(
    "Mean:",
    round(
        float(
            np.mean(
                non_melanoma_probs
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
                non_melanoma_probs
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
                non_melanoma_probs
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
                non_melanoma_probs
            )
        ),
        4
    )
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

    label=f"V11 ROC-AUC = {test_roc_auc:.4f}"

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
    "V11 - ROC Curve"
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

    label=f"V11 PR-AUC = {test_pr_auc:.4f}"

)


plt.xlabel(
    "Recall / Sensitivity"
)

plt.ylabel(
    "Precision"
)


plt.title(
    "V11 - Precision Recall Curve"
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

    f"V11 Confusion Matrix\n"
    f"Threshold = {FINAL_THRESHOLD:.2f}"

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
    figsize=(9, 6)
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

    label=f"Selected = {FINAL_THRESHOLD:.2f}"

)


plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Score"
)


plt.title(
    "V11 Threshold Analysis"
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

print(
    ROC_PATH
)


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

print(
    PR_PATH
)


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

print(
    CM_PATH
)


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

print(
    REPORT_PATH
)


# ================================================================
# CELL 50 - SAVE THRESHOLD
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

print(
    THRESHOLD_PATH
)


# ================================================================
# CELL 51 - EXPERIMENT SUMMARY
# ================================================================

experiment_summary = {

    "experiment":
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
        DATASET,

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

    "validation_best_accuracy_threshold":
        float(
            best_accuracy_row["threshold"]
        ),

    "validation_best_accuracy":
        float(
            best_accuracy_row["accuracy"]
        ),

    "validation_best_f1_threshold":
        float(
            best_f1_row["threshold"]
        ),

    "validation_best_f1":
        float(
            best_f1_row["f1"]
        ),

    "validation_best_balanced_threshold":
        float(
            best_balanced_row["threshold"]
        ),

    "validation_best_balanced_accuracy":
        float(
            best_balanced_row[
                "balanced_accuracy"
            ]
        ),

    "selected_threshold":
        float(FINAL_THRESHOLD),

    "threshold_reason":
        THRESHOLD_REASON,

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

    "probability_analysis": {

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

}


# ================================================================
# CELL 52 - SAVE EXPERIMENT SUMMARY
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

print(
    RESULTS_PATH
)


# ================================================================
# CELL 53 - SAVE FINAL MODEL
# ================================================================

print("\n")
print("=" * 70)
print("SAVING FINAL V11 MODEL")
print("=" * 70)


best_model.save(
    FINAL_MODEL_PATH
)


print(
    "MODEL SAVED:"
)

print(
    FINAL_MODEL_PATH
)


# ================================================================
# CELL 54 - FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("EXPERIMENT V11 COMPLETE")
print("=" * 70)


print("\nDATASET SOURCE:")

print(
    IMG_SRC
)


print("\nLOCAL DATASET:")

print(
    DATASET
)


print("\nFINAL LABEL MAPPING:")

print(
    "0 = non_melanoma"
)

print(
    "1 = melanoma"
)


print("\nMODEL OUTPUT:")

print(
    "P(melanoma)"
)


print("\nFINAL THRESHOLD:")

print(
    f"{FINAL_THRESHOLD:.4f}"
)


print("\nVALIDATION ROC-AUC:")

print(
    f"{val_roc_auc:.4f}"
)


print("\nVALIDATION PR-AUC:")

print(
    f"{val_pr_auc:.4f}"
)


print("\nFINAL TEST ROC-AUC:")

print(
    f"{test_roc_auc:.4f}"
)


print("\nFINAL TEST PR-AUC:")

print(
    f"{test_pr_auc:.4f}"
)


print("\nFINAL TEST ACCURACY:")

print(
    f"{final_result['accuracy']:.4f}"
)


print("\nFINAL TEST PRECISION:")

print(
    f"{final_result['precision']:.4f}"
)


print("\nFINAL TEST SENSITIVITY:")

print(
    f"{final_result['sensitivity']:.4f}"
)


print("\nFINAL TEST SPECIFICITY:")

print(
    f"{final_result['specificity']:.4f}"
)


print("\nFINAL TEST F1:")

print(
    f"{final_result['f1']:.4f}"
)


print("\nFINAL TEST BALANCED ACCURACY:")

print(
    f"{final_result['balanced_accuracy']:.4f}"
)


print("\nCONFUSION MATRIX:")

print(cm)


print("\nFINAL MODEL:")

print(
    FINAL_MODEL_PATH
)


print("\nTHRESHOLD:")

print(
    THRESHOLD_PATH
)


print("\nSUMMARY:")

print(
    RESULTS_PATH
)


print("\nROC DATA:")

print(
    ROC_PATH
)


print("\nPR DATA:")

print(
    PR_PATH
)


print("\nCONFUSION MATRIX DATA:")

print(
    CM_PATH
)


print("\nCLASSIFICATION REPORT:")

print(
    REPORT_PATH
)


print("\n")
print("=" * 70)
print("V11 FINISHED SUCCESSFULLY")
print("=" * 70)