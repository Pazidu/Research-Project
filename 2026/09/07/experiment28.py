# ================================================================
# V28 MELANOMA CLASSIFICATION
# ================================================================
# Controlled experiment:
#   V25 -> V28: melanoma loss weight 1.50 -> 1.00
#
# Everything else is intentionally preserved from V25:
#   - EfficientNetV2-S / ImageNet weights
#   - 300x300 input
#   - batch size 4
#   - online augmentation
#   - weighted focal loss, gamma=1.0
#   - 12 frozen + up to 10 fine-tuning epochs
#   - stage 1 LR 1e-4
#   - stage 2 LR 5e-6
#   - final 40% fine-tuning, BatchNorm frozen
#   - validation threshold selection: max F1 subject to >=70% sensitivity
#   - validation/test sets remain unchanged
#
# V28 HYPOTHESIS:
# Reducing melanoma loss emphasis may reduce false positives and improve
# precision/specificity/overall accuracy, because V25/V27 showed many
# false-positive melanoma predictions.
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
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("V28 MELANOMA CLASSIFICATION")
print("=" * 70)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("=" * 70)

# ================================================================
# CELL 2 - MOUNT GOOGLE DRIVE
# ================================================================

from google.colab import drive

drive.mount("/content/drive")

print("\nGoogle Drive mounted successfully.")

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
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth warning:", e)
else:
    print("WARNING: No GPU detected.")
    print("Go to Runtime > Change runtime type > GPU.")

# ================================================================
# CELL 5 - V28 DATASET CONFIGURATION
# ================================================================

IMG_SRC = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/newdata_backup"
)

DATASET = "/content/newdata"
DATASET_ROOT = DATASET

CLASS_NAMES = ["non_melanoma", "melanoma"]

IMG_SIZE = 300
BATCH_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 10

STAGE1_LR = 1e-4
STAGE2_LR = 5e-6
MIN_LR = 1e-7

# ================================================================
# V28 CONTROLLED CHANGE
# V25: 1.50
# V28: 1.00
# ================================================================
NON_MELANOMA_WEIGHT = 1.0
MELANOMA_WEIGHT = 1.00
FOCAL_GAMMA = 1.0

FINE_TUNE_FRACTION = 0.40
MIN_REQUIRED_SENSITIVITY = 0.70

print("Drive dataset source:")
print(IMG_SRC)
print("\nLocal Colab dataset:")
print(DATASET)
print("\nClass names:")
print(CLASS_NAMES)
print("\nImage size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Non-melanoma weight:", NON_MELANOMA_WEIGHT)
print("Melanoma weight:", MELANOMA_WEIGHT)
print("Focal gamma:", FOCAL_GAMMA)
print("Minimum required sensitivity:", MIN_REQUIRED_SENSITIVITY)
print("\nCONTROLLED CHANGE FROM V25:")
print("Melanoma weight: 1.50 -> 1.00")

# ================================================================
# CELL 6 - DATASET STRUCTURE CHECK
# ================================================================

print("\n" + "=" * 70)
print("CHECKING SOURCE DATASET")
print("=" * 70)


def has_required_dataset_structure(path):
    required_splits = ["train", "valid", "test"]
    required_classes = ["melanoma", "non_melanoma"]

    if not os.path.isdir(path):
        return False

    for split in required_splits:
        split_path = os.path.join(path, split)
        if not os.path.isdir(split_path):
            return False
        for class_name in required_classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                return False
    return True


print("Checking:")
print(IMG_SRC)

if not os.path.exists(IMG_SRC):
    raise FileNotFoundError(
        "Dataset source was not found:\n\n"
        f"{IMG_SRC}\n\n"
        "Expected structure:\n"
        "newdata_backup/\n"
        "  train/melanoma/\n"
        "  train/non_melanoma/\n"
        "  valid/melanoma/\n"
        "  valid/non_melanoma/\n"
        "  test/melanoma/\n"
        "  test/non_melanoma/"
    )

if not has_required_dataset_structure(IMG_SRC):
    raise FileNotFoundError(
        "Dataset exists, but the required directory structure was not found.\n\n"
        f"Dataset: {IMG_SRC}"
    )

print("\nSource dataset structure verified.")

# ================================================================
# CELL 7 - COPY DATASET TO LOCAL COLAB STORAGE
# ================================================================

print("\n" + "=" * 70)
print("COPYING DATASET TO LOCAL COLAB STORAGE")
print("=" * 70)

if os.path.exists(DATASET):
    print("Removing existing local dataset...")
    shutil.rmtree(DATASET)

print("\nFROM:")
print(IMG_SRC)
print("\nTO:")
print(DATASET)

shutil.copytree(IMG_SRC, DATASET)

if not has_required_dataset_structure(DATASET):
    raise FileNotFoundError(
        "The copied local dataset does not have the required directory structure.\n\n"
        f"Dataset: {DATASET}"
    )

print("\nLocal dataset copied successfully.")
print("Local dataset structure verified.")

# ================================================================
# CELL 8 - DATASET PATHS
# ================================================================

TRAIN_DIR = os.path.join(DATASET, "train")
VALID_DIR = os.path.join(DATASET, "valid")
TEST_DIR = os.path.join(DATASET, "test")

print("=" * 70)
print("FINAL DATASET PATHS")
print("=" * 70)
print("Source:", IMG_SRC)
print("\nLocal dataset:", DATASET)
print("\nTrain:", TRAIN_DIR)
print("\nValid:", VALID_DIR)
print("\nTest:", TEST_DIR)

# ================================================================
# CELL 9 - OUTPUT DIRECTORIES
# ================================================================

CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
MODEL_DIR = (
    "/content/drive/MyDrive/"
    "Colab Notebooks/Models/dermoscopy"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "efficientnet_v28"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME + "_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_final.keras")
THRESHOLD_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_threshold.txt")
RESULTS_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_results.json")
ROC_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_roc_data.json")
PR_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_pr_data.json")
CM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_confusion_matrix.json")
REPORT_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_classification_report.txt")
THRESHOLD_RESULTS_PATH = os.path.join(
    MODEL_DIR, MODEL_NAME + "_threshold_results.csv"
)

print("=" * 70)
print("V28 OUTPUT PATHS")
print("=" * 70)
print("Checkpoint:", BEST_MODEL_PATH)
print("\nFinal model:", FINAL_MODEL_PATH)
print("\nThreshold:", THRESHOLD_PATH)
print("\nResults:", RESULTS_PATH)

# ================================================================
# CELL 10 - CHECK DATASET DIRECTORIES
# ================================================================

print("\n" + "=" * 70)
print("CHECKING DATASET DIRECTORIES")
print("=" * 70)

for directory in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Dataset directory not found:\n{directory}")
    print("OK:", directory)

# ================================================================
# CELL 11 - DATASET COUNTS
# ================================================================

def count_images(directory):
    counts = {"melanoma": 0, "non_melanoma": 0}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in counts:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print("WARNING: missing class:", class_dir)
            continue

        for root, _, files in os.walk(class_dir):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    counts[class_name] += 1
    return counts


train_counts = count_images(TRAIN_DIR)
valid_counts = count_images(VALID_DIR)
test_counts = count_images(TEST_DIR)

print("\n" + "=" * 70)
print("V28 DATASET COUNTS")
print("=" * 70)

for name, counts in [
    ("TRAIN", train_counts),
    ("VALID", valid_counts),
    ("TEST", test_counts),
]:
    total = counts["melanoma"] + counts["non_melanoma"]
    print(f"\n{name}")
    print("melanoma:", counts["melanoma"])
    print("non_melanoma:", counts["non_melanoma"])
    print("total:", total)

# ================================================================
# CELL 12 - CREATE DATASETS
# ================================================================

print("\n" + "=" * 70)
print("CREATING DATASETS")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nClass names:", CLASS_NAMES)
print("IMPORTANT LABEL MAPPING")
print("0 = non_melanoma")
print("1 = melanoma")
print("MODEL OUTPUT = P(melanoma)")

# ================================================================
# CELL 13 - PERFORMANCE SETTINGS
# ================================================================

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Dataset prefetching enabled.")

# ================================================================
# CELL 14 - V28 DATA AUGMENTATION
# ================================================================

print("\n" + "=" * 70)
print("CREATING V28 DATA AUGMENTATION")
print("=" * 70)

# Preserved unchanged from V25.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=0.08),
        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
        ),
        tf.keras.layers.RandomContrast(factor=0.10),
        tf.keras.layers.RandomBrightness(factor=0.08),
    ],
    name="v28_augmentation",
)

print("V28 augmentation created.")
print("Online augmentation is preserved from V25.")

# ================================================================
# CELL 15 - V28 WEIGHTED FOCAL LOSS
# ================================================================

@tf.keras.utils.register_keras_serializable()
class MelanomaWeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        non_melanoma_weight=1.0,
        melanoma_weight=1.00,
        gamma=FOCAL_GAMMA,
        name="melanoma_weighted_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.non_melanoma_weight = float(non_melanoma_weight)
        self.melanoma_weight = float(melanoma_weight)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        p_t = (
            y_true * y_pred
            + (1.0 - y_true) * (1.0 - y_pred)
        )

        focal_factor = tf.pow(1.0 - p_t, self.gamma)

        class_weights = (
            y_true * self.melanoma_weight
            + (1.0 - y_true) * self.non_melanoma_weight
        )

        return bce * focal_factor * class_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "non_melanoma_weight": self.non_melanoma_weight,
            "melanoma_weight": self.melanoma_weight,
            "gamma": self.gamma,
        })
        return config


print("V28 focal gamma:", FOCAL_GAMMA)
print("V28 melanoma weight:", MELANOMA_WEIGHT)
print("V25 melanoma weight: 1.50")
print("V28 melanoma weight: 1.00")

# ================================================================
# CELL 16 - METRICS
# ================================================================

def create_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


print("Metrics configured.")

# ================================================================
# CELL 17 - CREATE V28 MODEL
# ================================================================

print("\n" + "=" * 70)
print("CREATING V28 MODEL")
print("=" * 70)

inputs = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="image",
)

x = data_augmentation(inputs)

backbone = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    include_preprocessing=True,
    name="efficientnetv2-s",
)

backbone.trainable = False
x = backbone(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = tf.keras.layers.Dropout(0.45, name="dropout_1")(x)
x = tf.keras.layers.Dense(128, activation="swish", name="classifier_dense")(x)
x = tf.keras.layers.Dropout(0.35, name="dropout_2")(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="melanoma_probability",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="efficientnet_v28",
)

print("\nModel created successfully.")
print("Model:", model.name)
print("Backbone:", backbone.name)
print("Melanoma weight:", MELANOMA_WEIGHT)
print("Focal gamma:", FOCAL_GAMMA)

# ================================================================
# CELL 18 - MODEL SUMMARY
# ================================================================

model.summary()

# ================================================================
# CELL 19 - COMPILE STAGE 1
# ================================================================

print("\n" + "=" * 70)
print("COMPILING V28 STAGE 1")
print("=" * 70)

loss_fn = MelanomaWeightedFocalLoss(
    non_melanoma_weight=NON_MELANOMA_WEIGHT,
    melanoma_weight=MELANOMA_WEIGHT,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
    loss=loss_fn,
    metrics=create_metrics(),
)

print("Stage 1 compiled.")
print("Learning rate:", STAGE1_LR)

# ================================================================
# CELL 20 - STAGE 1 CALLBACKS
# ================================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=MIN_LR,
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

print("Stage 1 callbacks ready.")

# ================================================================
# CELL 21 - STAGE 1 TRAINING
# ================================================================

print("\n" + "=" * 70)
print("V28 STAGE 1 - FROZEN EFFICIENTNETV2-S")
print("=" * 70)
print("Melanoma loss weight:", MELANOMA_WEIGHT)
print("Backbone trainable:", backbone.trainable)
print("Stage 1 epochs:", STAGE1_EPOCHS)

history_stage1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1,
)

# ================================================================
# CELL 22 - LOAD BEST STAGE 1 MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING BEST V28 STAGE 1 MODEL")
print("=" * 70)

best_stage1_model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False,
)

backbone = best_stage1_model.get_layer("efficientnetv2-s")
model = best_stage1_model

print("Best Stage 1 model loaded.")
print("Backbone found:", backbone.name)

# ================================================================
# CELL 23 - FINE-TUNING SETUP
# ================================================================

print("\n" + "=" * 70)
print("PREPARING V28 FINE-TUNING")
print("=" * 70)

backbone.trainable = True
total_layers = len(backbone.layers)
fine_tune_from = int(total_layers * (1.0 - FINE_TUNE_FRACTION))

print("Total backbone layers:", total_layers)
print("Fine-tuning from layer:", fine_tune_from)

for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[fine_tune_from:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum(
    1 for layer in backbone.layers if layer.trainable
)

print("Trainable backbone layers:", trainable_count)

# ================================================================
# CELL 24 - COMPILE STAGE 2
# ================================================================

print("\n" + "=" * 70)
print("COMPILING V28 STAGE 2")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
    loss=MelanomaWeightedFocalLoss(
        non_melanoma_weight=NON_MELANOMA_WEIGHT,
        melanoma_weight=MELANOMA_WEIGHT,
    ),
    metrics=create_metrics(),
)

print("Stage 2 compiled.")
print("Learning rate:", STAGE2_LR)

# ================================================================
# CELL 25 - STAGE 2 CALLBACKS
# ================================================================

checkpoint_stage2 = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

reduce_lr_stage2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=MIN_LR,
    verbose=1,
)

early_stop_stage2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

print("Stage 2 callbacks ready.")

# ================================================================
# CELL 26 - STAGE 2 TRAINING
# ================================================================

print("\n" + "=" * 70)
print("V28 STAGE 2 - FINE-TUNING EFFICIENTNETV2-S")
print("=" * 70)
print("Trainable backbone layers:", trainable_count)
print("Fine-tuning learning rate:", STAGE2_LR)
print("Stage 2 epochs:", STAGE2_EPOCHS)

history_stage2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=[checkpoint_stage2, reduce_lr_stage2, early_stop_stage2],
    verbose=1,
)

# ================================================================
# CELL 27 - LOAD ABSOLUTE BEST V28 MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING ABSOLUTE BEST V28 MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False,
)

print("Absolute best V28 model loaded.")

# ================================================================
# CELL 28 - PREDICTION FUNCTION
# ================================================================

def get_predictions(model, dataset):
    probabilities = []
    labels = []

    for images, batch_labels in dataset:
        batch_predictions = model.predict(images, verbose=0)
        probabilities.extend(batch_predictions.reshape(-1))
        labels.extend(batch_labels.numpy().reshape(-1))

    return (
        np.asarray(probabilities, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )

# ================================================================
# CELL 29 - V28 VALIDATION PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("V28 VALIDATION PREDICTIONS")
print("=" * 70)

# Keep the return order explicit: probabilities first, labels second.
p_val, y_val = get_predictions(best_model, valid_ds)

print("Validation samples:", len(y_val))
print("Validation melanoma:", int(np.sum(y_val == 1)))
print("Validation non-melanoma:", int(np.sum(y_val == 0)))

# ================================================================
# CELL 30 - V28 VALIDATION PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("V28 VALIDATION PERFORMANCE")
print("=" * 70)

val_roc_auc = roc_auc_score(y_val, p_val)
val_pr_auc = average_precision_score(y_val, p_val)

print("Validation ROC-AUC:", f"{val_roc_auc:.4f}")
print("Validation PR-AUC:", f"{val_pr_auc:.4f}")

# ================================================================
# CELL 31 - THRESHOLD EVALUATION
# ================================================================

def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    sensitivity = recall_score(y_true, predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, predictions, zero_division=0)
    balanced = balanced_accuracy_score(y_true, predictions)
    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
        "youden_j": float(youden_j),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# ================================================================
# CELL 32 - V28 VALIDATION THRESHOLD SEARCH
# ================================================================

print("\n" + "=" * 70)
print("V28 VALIDATION THRESHOLD SEARCH")
print("=" * 70)

thresholds = np.arange(0.005, 1.000, 0.005)
threshold_results = [
    evaluate_threshold(y_val, p_val, threshold)
    for threshold in thresholds
]
threshold_df = pd.DataFrame(threshold_results)

best_accuracy_row = threshold_df.loc[threshold_df["accuracy"].idxmax()]
best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_balanced_row = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]
best_youden_row = threshold_df.loc[threshold_df["youden_j"].idxmax()]

print("\nBest accuracy:")
print(best_accuracy_row.to_dict())
print("\nBest F1:")
print(best_f1_row.to_dict())
print("\nBest balanced accuracy:")
print(best_balanced_row.to_dict())
print("\nBest Youden J:")
print(best_youden_row.to_dict())

# ================================================================
# CELL 33 - V28 SENSITIVITY-CONSTRAINED SEARCH
# ================================================================

print("\n" + "=" * 70)
print("V28 SENSITIVITY-CONSTRAINED THRESHOLD SEARCH")
print("=" * 70)

sensitivity_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
sensitivity_thresholds = {}

for target in sensitivity_targets:
    eligible = threshold_df[
        threshold_df["sensitivity"] >= target
    ]

    if len(eligible) == 0:
        print(
            f"{int(target * 100)}% sensitivity -> no threshold found"
        )
        continue

    best = eligible.loc[eligible["specificity"].idxmax()]
    sensitivity_thresholds[int(target * 100)] = best.to_dict()

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
# CELL 34 - SELECT FINAL V28 THRESHOLD
# ================================================================

print("\n" + "=" * 70)
print("SELECTING V28 FINAL THRESHOLD")
print("=" * 70)

minimum_sensitivity = MIN_REQUIRED_SENSITIVITY

eligible_f1 = threshold_df[
    threshold_df["sensitivity"] >= minimum_sensitivity
]

if len(eligible_f1) == 0:
    raise RuntimeError(
        "No validation threshold satisfies "
        f"sensitivity >= {minimum_sensitivity:.2f}"
    )

best_constrained_f1 = eligible_f1.loc[eligible_f1["f1"].idxmax()]
FINAL_THRESHOLD = float(best_constrained_f1["threshold"])
THRESHOLD_REASON = (
    "maximum validation F1 subject to validation sensitivity >= 70%"
)
selected_validation_result = best_constrained_f1.to_dict()

print("Minimum required sensitivity:", minimum_sensitivity)
print("Selected threshold:", f"{FINAL_THRESHOLD:.3f}")
print("Reason:", THRESHOLD_REASON)
print("\nValidation performance at selected threshold:")

for key in [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
]:
    print(f"{key}:", f"{selected_validation_result[key]:.4f}")

# ================================================================
# CELL 35 - V28 TEST PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("V28 TEST PREDICTIONS")
print("=" * 70)

p_test, y_test = get_predictions(best_model, test_ds)

print("Test samples:", len(y_test))
print("Test melanoma:", int(np.sum(y_test == 1)))
print("Test non-melanoma:", int(np.sum(y_test == 0)))

# ================================================================
# CELL 36 - V28 TEST ROC / PR
# ================================================================

print("\n" + "=" * 70)
print("V28 TEST ROC / PR")
print("=" * 70)

test_roc_auc = roc_auc_score(y_test, p_test)
test_pr_auc = average_precision_score(y_test, p_test)

print("Test ROC-AUC:", f"{test_roc_auc:.4f}")
print("Test PR-AUC:", f"{test_pr_auc:.4f}")

# ================================================================
# CELL 37 - FINAL V28 TEST RESULTS
# ================================================================

final_result = evaluate_threshold(y_test, p_test, FINAL_THRESHOLD)

print("\n" + "=" * 70)
print("FINAL V28 TEST RESULTS")
print("=" * 70)
print("Test ROC-AUC:", f"{test_roc_auc:.4f}")
print("Test PR-AUC:", f"{test_pr_auc:.4f}")
print("Threshold:", f"{FINAL_THRESHOLD:.3f}")
print("Accuracy:", f"{final_result['accuracy']:.4f}")
print("Precision:", f"{final_result['precision']:.4f}")
print("Sensitivity:", f"{final_result['sensitivity']:.4f}")
print("Specificity:", f"{final_result['specificity']:.4f}")
print("F1 Score:", f"{final_result['f1']:.4f}")
print("Balanced Accuracy:", f"{final_result['balanced_accuracy']:.4f}")

# ================================================================
# CELL 38 - V28 CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    (p_test >= FINAL_THRESHOLD).astype(int),
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print("V28 CONFUSION MATRIX")
print("=" * 70)
print(cm)
print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

# ================================================================
# CELL 39 - V28 CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("V28 CLASSIFICATION REPORT")
print("=" * 70)

y_test_pred = (p_test >= FINAL_THRESHOLD).astype(int)

report = classification_report(
    y_test,
    y_test_pred,
    labels=[0, 1],
    target_names=["non_melanoma", "melanoma"],
    digits=4,
    zero_division=0,
)

print(report)

# ================================================================
# CELL 40 - V28 TEST THRESHOLD COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("V28 TEST THRESHOLD COMPARISON")
print("=" * 70)

comparison_thresholds = {
    "v28_selected": FINAL_THRESHOLD,
    "v28_best_f1": float(best_f1_row["threshold"]),
    "v28_best_balanced": float(best_balanced_row["threshold"]),
    "v28_best_youden": float(best_youden_row["threshold"]),
    "default_0.50": 0.50,
}

for target in [70, 75, 80, 85, 90, 95]:
    comparison_thresholds[f"sensitivity_{target}%"] = (
        sensitivity_thresholds.get(target, {}).get("threshold", None)
    )

comparison_results = {}

for name, threshold in comparison_thresholds.items():
    if threshold is None:
        continue

    result = evaluate_threshold(y_test, p_test, threshold)
    comparison_results[name] = result

    print("\n" + "-" * 50)
    print(name)
    print("Threshold:", f"{threshold:.3f}")
    print("Accuracy:", f"{result['accuracy']:.4f}")
    print("Precision:", f"{result['precision']:.4f}")
    print("Sensitivity:", f"{result['sensitivity']:.4f}")
    print("Specificity:", f"{result['specificity']:.4f}")
    print("F1:", f"{result['f1']:.4f}")
    print("Balanced Accuracy:", f"{result['balanced_accuracy']:.4f}")

# ================================================================
# CELL 41 - V28 PROBABILITY ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("V28 PROBABILITY ANALYSIS")
print("=" * 70)

melanoma_probs = p_test[y_test == 1]
non_melanoma_probs = p_test[y_test == 0]

print("\nMelanoma probability:")
print("Mean:", round(float(np.mean(melanoma_probs)), 4))
print("Median:", round(float(np.median(melanoma_probs)), 4))
print("Min:", round(float(np.min(melanoma_probs)), 4))
print("Max:", round(float(np.max(melanoma_probs)), 4))

print("\nNon-melanoma probability:")
print("Mean:", round(float(np.mean(non_melanoma_probs)), 4))
print("Median:", round(float(np.median(non_melanoma_probs)), 4))
print("Min:", round(float(np.min(non_melanoma_probs)), 4))
print("Max:", round(float(np.max(non_melanoma_probs)), 4))

# ================================================================
# CELL 42 - V28 ROC CURVE
# ================================================================

fpr, tpr, roc_thresholds = roc_curve(y_test, p_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"V28 ROC-AUC = {test_roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("V28 - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================================================================
# CELL 43 - V28 PRECISION RECALL CURVE
# ================================================================

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
    y_test,
    p_test,
)

plt.figure(figsize=(8, 6))
plt.plot(
    recall_curve,
    precision_curve,
    label=f"V28 PR-AUC = {test_pr_auc:.4f}",
)
plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("V28 - Precision Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================================================================
# CELL 44 - V28 CONFUSION MATRIX PLOT
# ================================================================

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title(
    f"V28 Confusion Matrix\nThreshold = {FINAL_THRESHOLD:.3f}"
)
plt.colorbar()
plt.xticks([0, 1], ["non_melanoma", "melanoma"], rotation=45)
plt.yticks([0, 1], ["non_melanoma", "melanoma"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# ================================================================
# CELL 45 - V28 THRESHOLD CURVES
# ================================================================

plt.figure(figsize=(9, 6))
plt.plot(threshold_df["threshold"], threshold_df["sensitivity"], label="Sensitivity")
plt.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity")
plt.plot(
    threshold_df["threshold"],
    threshold_df["balanced_accuracy"],
    label="Balanced Accuracy",
)
plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
plt.axvline(
    FINAL_THRESHOLD,
    linestyle="--",
    label=f"Selected = {FINAL_THRESHOLD:.3f}",
)
plt.axhline(
    MIN_REQUIRED_SENSITIVITY,
    linestyle=":",
    label=f"Minimum Sensitivity = {MIN_REQUIRED_SENSITIVITY:.2f}",
)
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("V28 Threshold Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================================================================
# CELL 46 - SAVE ROC DATA
# ================================================================

roc_data = {
    "experiment": "V28",
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": roc_thresholds.tolist(),
    "roc_auc": float(test_roc_auc),
}

with open(ROC_PATH, "w") as f:
    json.dump(roc_data, f, indent=4)

print("ROC data saved:")
print(ROC_PATH)

# ================================================================
# CELL 47 - SAVE PR DATA
# ================================================================

pr_data = {
    "experiment": "V28",
    "precision": precision_curve.tolist(),
    "recall": recall_curve.tolist(),
    "thresholds": pr_thresholds.tolist(),
    "pr_auc": float(test_pr_auc),
}

with open(PR_PATH, "w") as f:
    json.dump(pr_data, f, indent=4)

print("PR data saved:")
print(PR_PATH)

# ================================================================
# CELL 48 - SAVE CONFUSION MATRIX
# ================================================================

cm_data = {
    "experiment": "V28",
    "matrix": cm.tolist(),
    "labels": ["non_melanoma", "melanoma"],
    "threshold": float(FINAL_THRESHOLD),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
}

with open(CM_PATH, "w") as f:
    json.dump(cm_data, f, indent=4)

print("Confusion matrix saved:")
print(CM_PATH)

# ================================================================
# CELL 49 - SAVE CLASSIFICATION REPORT
# ================================================================

with open(REPORT_PATH, "w") as f:
    f.write(report)

print("Classification report saved:")
print(REPORT_PATH)

# ================================================================
# CELL 50 - SAVE FINAL THRESHOLD
# ================================================================

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(FINAL_THRESHOLD))

print("Threshold saved:")
print(THRESHOLD_PATH)

# ================================================================
# CELL 51 - SAVE THRESHOLD RESULTS
# ================================================================

threshold_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)

print("Threshold results saved:")
print(THRESHOLD_RESULTS_PATH)

# ================================================================
# CELL 52 - V28 EXPERIMENT SUMMARY
# ================================================================

experiment_summary = {
    "experiment": "V28",
    "controlled_change": "Melanoma loss weight 1.50 -> 1.00",
    "model": "EfficientNetV2-S",
    "tensorflow_version": tf.__version__,
    "keras_version": keras.__version__,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "dataset_source": IMG_SRC,
    "dataset_root": DATASET_ROOT,
    "label_mapping": {
        "0": "non_melanoma",
        "1": "melanoma",
    },
    "model_output": "P(melanoma)",
    "augmentation": [
        "RandomFlip(horizontal_and_vertical)",
        "RandomRotation(0.08)",
        "RandomZoom(0.10)",
        "RandomTranslation(0.05)",
        "RandomContrast(0.10)",
        "RandomBrightness(0.08)",
    ],
    "loss": "MelanomaWeightedFocalLoss",
    "focal_gamma": FOCAL_GAMMA,
    "non_melanoma_weight": NON_MELANOMA_WEIGHT,
    "melanoma_weight": MELANOMA_WEIGHT,
    "stage1_epochs": STAGE1_EPOCHS,
    "stage2_epochs": STAGE2_EPOCHS,
    "stage1_learning_rate": STAGE1_LR,
    "stage2_learning_rate": STAGE2_LR,
    "fine_tune_fraction": FINE_TUNE_FRACTION,
    "train_counts": train_counts,
    "validation_counts": valid_counts,
    "test_counts": test_counts,
    "validation_roc_auc": float(val_roc_auc),
    "validation_pr_auc": float(val_pr_auc),
    "validation_best_accuracy": best_accuracy_row.to_dict(),
    "validation_best_f1": best_f1_row.to_dict(),
    "validation_best_balanced": best_balanced_row.to_dict(),
    "validation_best_youden": best_youden_row.to_dict(),
    "minimum_required_sensitivity": float(MIN_REQUIRED_SENSITIVITY),
    "selected_threshold": float(FINAL_THRESHOLD),
    "threshold_reason": THRESHOLD_REASON,
    "selected_validation_result": selected_validation_result,
    "test_roc_auc": float(test_roc_auc),
    "test_pr_auc": float(test_pr_auc),
    "test_metrics": final_result,
    "threshold_comparison": comparison_results,
    "confusion_matrix": cm.tolist(),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "probability_analysis": {
        "melanoma": {
            "mean": float(np.mean(melanoma_probs)),
            "median": float(np.median(melanoma_probs)),
            "min": float(np.min(melanoma_probs)),
            "max": float(np.max(melanoma_probs)),
        },
        "non_melanoma": {
            "mean": float(np.mean(non_melanoma_probs)),
            "median": float(np.median(non_melanoma_probs)),
            "min": float(np.min(non_melanoma_probs)),
            "max": float(np.max(non_melanoma_probs)),
        },
    },
}

# ================================================================
# CELL 53 - SAVE EXPERIMENT SUMMARY
# ================================================================

with open(RESULTS_PATH, "w") as f:
    json.dump(experiment_summary, f, indent=4)

print("Experiment summary saved:")
print(RESULTS_PATH)

# ================================================================
# CELL 54 - SAVE FINAL V28 MODEL
# ================================================================

print("\n" + "=" * 70)
print("SAVING FINAL V28 MODEL")
print("=" * 70)

best_model.save(FINAL_MODEL_PATH)

print("V28 MODEL SAVED:")
print(FINAL_MODEL_PATH)

# ================================================================
# CELL 55 - PERFORMANCE COMPARISON V14 TO V28
# ================================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON - V14 TO V28")
print("=" * 70)

v14_metrics = {
    "ROC-AUC": 0.8721,
    "PR-AUC": 0.5213,
    "Accuracy": 0.8044,
    "Precision": 0.3385,
    "Sensitivity": 0.7857,
    "Specificity": 0.8067,
    "F1": 0.4731,
    "Balanced Accuracy": 0.7962,
}

v16_metrics = {
    "ROC-AUC": 0.8810,
    "PR-AUC": 0.5048,
    "Accuracy": 0.8483,
    "Precision": 0.3925,
    "Sensitivity": 0.6518,
    "Specificity": 0.8730,
    "F1": 0.4899,
    "Balanced Accuracy": 0.7624,
}

v17_metrics = {
    "ROC-AUC": 0.8806,
    "PR-AUC": 0.5012,
    "Accuracy": 0.8523,
    "Precision": 0.4011,
    "Sensitivity": 0.6518,
    "Specificity": 0.8775,
    "F1": 0.4966,
    "Balanced Accuracy": 0.7647,
}

v18_metrics = {
    "ROC-AUC": 0.8798,
    "PR-AUC": 0.5056,
    "Accuracy": 0.8553,
    "Precision": 0.4108,
    "Sensitivity": 0.6786,
    "Specificity": 0.8775,
    "F1": 0.5118,
    "Balanced Accuracy": 0.7780,
}

v19_metrics = {
    "ROC-AUC": 0.8821,
    "PR-AUC": 0.5081,
    "Accuracy": 0.8563,
    "Precision": 0.4149,
    "Sensitivity": 0.6964,
    "Specificity": 0.8764,
    "F1": 0.5200,
    "Balanced Accuracy": 0.7864,
}

v20_metrics = {
    "ROC-AUC": 0.8836,
    "PR-AUC": 0.5269,
    "Accuracy": 0.8453,
    "Precision": 0.3930,
    "Sensitivity": 0.7054,
    "Specificity": 0.8629,
    "F1": 0.5048,
    "Balanced Accuracy": 0.7841,
}

# V21 has only a reliably retained ROC-AUC from the experiment history.
v21_metrics = {
    "ROC-AUC": 0.8882,
    "PR-AUC": np.nan,
    "Accuracy": np.nan,
    "Precision": np.nan,
    "Sensitivity": np.nan,
    "Specificity": np.nan,
    "F1": np.nan,
    "Balanced Accuracy": np.nan,
}

v22_metrics = {
    "ROC-AUC": 0.8851,
    "PR-AUC": 0.5333,
    "Accuracy": 0.8413,
    "Precision": 0.3854,
    "Sensitivity": 0.7054,
    "Specificity": 0.8584,
    "F1": 0.4984,
    "Balanced Accuracy": 0.7819,
}

v23_metrics = {
    "ROC-AUC": 0.8877,
    "PR-AUC": 0.5391,
    "Accuracy": 0.8443,
    "Precision": 0.3911,
    "Sensitivity": 0.7054,
    "Specificity": 0.8618,
    "F1": 0.5032,
    "Balanced Accuracy": 0.7836,
}

v24_metrics = {
    "ROC-AUC": 0.8725,
    "PR-AUC": 0.5047,
    "Accuracy": 0.8553,
    "Precision": 0.4046,
    "Sensitivity": 0.6250,
    "Specificity": 0.8843,
    "F1": 0.4912,
    "Balanced Accuracy": 0.7546,
}

v25_metrics = {
    "ROC-AUC": 0.8910,
    "PR-AUC": 0.5526,
    "Accuracy": 0.8633,
    "Precision": 0.4302,
    "Sensitivity": 0.6875,
    "Specificity": 0.8854,
    "F1": 0.5292,
    "Balanced Accuracy": 0.7864,
}

v26_metrics = {
    "ROC-AUC": 0.8795,
    "PR-AUC": 0.5403,
    "Accuracy": 0.8214,
    "Precision": 0.3524,
    "Sensitivity": 0.7143,
    "Specificity": 0.8348,
    "F1": 0.4720,
    "Balanced Accuracy": 0.7746,
}

v27_metrics = {
    "ROC-AUC": 0.8865,
    "PR-AUC": 0.5219,
    "Accuracy": 0.8553,
    "Precision": 0.4118,
    "Sensitivity": 0.6875,
    "Specificity": 0.8764,
    "F1": 0.5151,
    "Balanced Accuracy": 0.7820,
}

v28_metrics = {
    "ROC-AUC": float(test_roc_auc),
    "PR-AUC": float(test_pr_auc),
    "Accuracy": float(final_result["accuracy"]),
    "Precision": float(final_result["precision"]),
    "Sensitivity": float(final_result["sensitivity"]),
    "Specificity": float(final_result["specificity"]),
    "F1": float(final_result["f1"]),
    "Balanced Accuracy": float(final_result["balanced_accuracy"]),
}

comparison_df = pd.DataFrame({
    "V14": v14_metrics,
    "V16": v16_metrics,
    "V17": v17_metrics,
    "V18": v18_metrics,
    "V19": v19_metrics,
    "V20": v20_metrics,
    "V21": v21_metrics,
    "V22": v22_metrics,
    "V23": v23_metrics,
    "V24": v24_metrics,
    "V25": v25_metrics,
    "V26": v26_metrics,
    "V27": v27_metrics,
    "V28": v28_metrics,
})

comparison_df["V28 - V25"] = comparison_df["V28"] - comparison_df["V25"]
comparison_df["V28 - V27"] = comparison_df["V28"] - comparison_df["V27"]

print("\nFULL PERFORMANCE COMPARISON")
print("=" * 80)
print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 80)
print("V28 IMPROVEMENT / CHANGE FROM V25")
print("=" * 80)
print(
    (comparison_df["V28"] - comparison_df["V25"]).to_string(
        float_format=lambda x: f"{x:+.4f}"
    )
)

print("\n" + "=" * 80)
print("V28 IMPROVEMENT / CHANGE FROM V27")
print("=" * 80)
print(
    (comparison_df["V28"] - comparison_df["V27"]).to_string(
        float_format=lambda x: f"{x:+.4f}"
    )
)

# ================================================================
# CELL 56 - FINAL V28 SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("V28 EXPERIMENT COMPLETE")
print("=" * 70)

print("\nDATASET:")
print(DATASET_ROOT)

print("\nLABEL MAPPING:")
print("0 = non_melanoma")
print("1 = melanoma")

print("\nMODEL OUTPUT:")
print("P(melanoma)")

print("\nV28 CONTROLLED CHANGE:")
print("Melanoma loss weight: 1.50 -> 1.00")

print("\nFINAL THRESHOLD:")
print(f"{FINAL_THRESHOLD:.4f}")

print("\nTHRESHOLD REASON:")
print(THRESHOLD_REASON)

print("\nVALIDATION ROC-AUC:")
print(f"{val_roc_auc:.4f}")

print("\nVALIDATION PR-AUC:")
print(f"{val_pr_auc:.4f}")

print("\nTEST ROC-AUC:")
print(f"{test_roc_auc:.4f}")

print("\nTEST PR-AUC:")
print(f"{test_pr_auc:.4f}")

for label, key in [
    ("TEST ACCURACY", "accuracy"),
    ("TEST PRECISION", "precision"),
    ("TEST SENSITIVITY", "sensitivity"),
    ("TEST SPECIFICITY", "specificity"),
    ("TEST F1", "f1"),
    ("TEST BALANCED ACCURACY", "balanced_accuracy"),
]:
    print(f"\n{label}:")
    print(f"{final_result[key]:.4f}")

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
print("\nTHRESHOLD RESULTS:")
print(THRESHOLD_RESULTS_PATH)

print("\n" + "=" * 70)
print("V28 FINISHED SUCCESSFULLY")
print("=" * 70)
