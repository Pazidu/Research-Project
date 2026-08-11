# ============================================================
# MELANOMA CLASSIFICATION - NEXT RUN
# EfficientNetV2S
# RAM-SAFE VERSION
# ============================================================

# ============================================================
# 1. CLEAR PREVIOUS TENSORFLOW MEMORY
# ============================================================

import gc
import os
import random
import shutil

gc.collect()

# ============================================================
# 2. IMPORTS
# ============================================================

from google.colab import drive
drive.mount('/drive')

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
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 3. GPU MEMORY SETTINGS
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============================================================
# 4. MIXED PRECISION
# ============================================================

mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# 5. RANDOM SEED
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 6. PATHS
# ============================================================

DATASET = "/content/newdata"

IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata_backup"

CHECKPOINT = (
    "/drive/MyDrive/checkpoints/"
    "efficientnet_v4_best.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/Colab Notebooks/"
    "Models/dermoscopy/"
    "efficientnet_v4_final.keras"
)

# ============================================================
# 7. COPY DATASET TO COLAB
# ============================================================

if os.path.exists(DATASET):
    print("Removing old local dataset...")
    shutil.rmtree(DATASET)

print("Copying dataset to Colab...")

shutil.copytree(
    IMG_SRC,
    DATASET
)

print("Dataset copied.")

# ============================================================
# 8. SETTINGS
# ============================================================

IMG_SIZE = 256

# Keep this low to reduce RAM/VRAM usage
BATCH_SIZE = 8

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 8

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# 9. CHECK DATASET COUNTS
# ============================================================

def count_images(folder):

    count = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                count += 1

    return count


print("\n========================================")
print("DATASET COUNTS")
print("========================================")

for split in ["train", "valid", "test"]:

    print("\n", split.upper())

    for cls in ["melanoma", "non_melanoma"]:

        path = os.path.join(
            DATASET,
            split,
            cls
        )

        count = count_images(path)

        print(
            f"{cls}: {count}"
        )

# ============================================================
# 10. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.10
        ),

    ],

    name="augmentation"
)

# ============================================================
# 11. DATASET LOADER
# ============================================================

def load_dataset(
    path,
    shuffle
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

    # IMPORTANT:
    # Do not cache the dataset.
    # Caching the whole dataset can consume RAM.

    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds, class_names


# ============================================================
# 12. LOAD DATA
# ============================================================

train_ds, class_names = load_dataset(

    DATASET + "/train",

    True
)

val_ds, _ = load_dataset(

    DATASET + "/valid",

    False
)

test_ds, _ = load_dataset(

    DATASET + "/test",

    False
)

print(
    "\nClass names:",
    class_names
)

# ============================================================
# 13. MODEL
# ============================================================

def create_model():

    inputs = layers.Input(

        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        ),

        name="image"
    )

    # -----------------------------
    # AUGMENTATION
    # -----------------------------

    x = augmentation(inputs)

    # -----------------------------
    # PREPROCESS
    # -----------------------------

    x = preprocess_input(x)

    # -----------------------------
    # EFFICIENTNET
    # -----------------------------

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
    # Freeze pretrained backbone

    backbone.trainable = False

    x = backbone(
        x,
        training=False
    )

    # -----------------------------
    # CLASSIFICATION HEAD
    # -----------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(

        2,

        activation="softmax",

        dtype="float32",

        name="prediction"
    )(x)

    model = Model(

        inputs=inputs,

        outputs=outputs
    )

    return model, backbone


# ============================================================
# 14. CREATE MODEL
# ============================================================

model, backbone = create_model()

model.summary()

# ============================================================
# 15. COMPILE STAGE 1
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(),

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        ),

    ]
)

# ============================================================
# 16. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=CHECKPOINT,

    monitor="val_auc",

    save_best_only=True,

    mode="max",

    verbose=1
)

early_stop = EarlyStopping(

    monitor="val_auc",

    patience=4,

    mode="max",

    restore_best_weights=True,

    verbose=1
)

lr_reduce = ReduceLROnPlateau(

    monitor="val_auc",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    mode="max",

    verbose=1
)

callbacks = [

    checkpoint,

    early_stop,

    lr_reduce

]

# ============================================================
# 17. IMPORTANT:
# NO CLASS WEIGHTS
# ============================================================

# Training dataset is balanced:
#
# melanoma       = 7122
# non_melanoma   = 7122
#
# Therefore:

class_weight = None

# ============================================================
# 18. STAGE 1
# ============================================================

print("\n")
print("==========================================")
print("STAGE 1 - FROZEN EFFICIENTNET")
print("==========================================")

history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE1_EPOCHS,

    callbacks=callbacks,

    class_weight=class_weight
)

# ============================================================
# 19. LOAD BEST STAGE 1
# ============================================================

print("\nLoading best Stage 1 model...")

model.load_weights(
    CHECKPOINT
)

# ============================================================
# 20. CLEAR SOME MEMORY
# ============================================================

gc.collect()

# ============================================================
# 21. STAGE 2 FINE-TUNING
# ============================================================

print("\n")
print("==========================================")
print("STAGE 2 - FINE TUNING")
print("==========================================")

backbone.trainable = True

# Freeze most layers
# Only last ~60 layers train

for layer in backbone.layers[:-60]:

    layer.trainable = False

# IMPORTANT:
# Keep BatchNormalization frozen
# during fine tuning.

for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False

# ============================================================
# 22. COMPILE STAGE 2
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=5e-6
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(),

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        ),

    ]
)

# ============================================================
# 23. STAGE 2 TRAIN
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=callbacks,

    class_weight=None
)

# ============================================================
# 24. LOAD ABSOLUTE BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_weights(
    CHECKPOINT
)

# ============================================================
# 25. VALIDATION PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION PREDICTIONS")
print("==========================================")

val_probs = []
val_true = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    # Probability of melanoma
    val_probs.extend(
        predictions[:, 0]
    )

    val_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

val_probs = np.array(
    val_probs
)

val_true = np.array(
    val_true
)

# ============================================================
# 26. VALIDATION AUC
# ============================================================

val_auc = roc_auc_score(

    val_true,

    val_probs
)

print(
    "Validation ROC-AUC:",
    round(val_auc, 4)
)

# ============================================================
# 27. THRESHOLD SEARCH
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION THRESHOLD SEARCH")
print("==========================================")

best_threshold = 0.50
best_f1 = 0.0

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    pred = (
        val_probs >= threshold
    ).astype(int)

    score = f1_score(

        val_true,

        pred,

        zero_division=0
    )

    if score > best_f1:

        best_f1 = score

        best_threshold = threshold


print(
    "Best threshold:",
    round(best_threshold, 2)
)

print(
    "Validation F1:",
    round(best_f1, 4)
)

# ============================================================
# 28. TEST PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("TEST PREDICTIONS")
print("==========================================")

test_probs = []
test_true = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    test_probs.extend(
        predictions[:, 0]
    )

    test_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

test_probs = np.array(
    test_probs
)

test_true = np.array(
    test_true
)

# ============================================================
# 29. TEST ROC-AUC
# ============================================================

test_auc = roc_auc_score(

    test_true,

    test_probs
)

print(
    "Test ROC-AUC:",
    round(test_auc, 4)
)

# ============================================================
# 30. DEFAULT THRESHOLD
# ============================================================

test_pred_default = (

    test_probs >= 0.50

).astype(int)

# ============================================================
# 31. OPTIMIZED THRESHOLD
# ============================================================

test_pred = (

    test_probs >= best_threshold

).astype(int)

# ============================================================
# 32. METRICS
# ============================================================

accuracy = accuracy_score(

    test_true,

    test_pred
)

precision = precision_score(

    test_true,

    test_pred,

    zero_division=0
)

recall = recall_score(

    test_true,

    test_pred,

    zero_division=0
)

f1 = f1_score(

    test_true,

    test_pred,

    zero_division=0
)

# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_true,

    test_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = (

    tn /
    (tn + fp)

    if (tn + fp) > 0
    else 0
)

sensitivity = (

    tp /
    (tp + fn)

    if (tp + fn) > 0
    else 0
)

# ============================================================
# 34. FINAL RESULTS
# ============================================================

print("\n")
print("==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(
    "Test ROC-AUC      :",
    round(test_auc, 4)
)

print(
    "Threshold         :",
    round(best_threshold, 2)
)

print(
    "Accuracy          :",
    round(accuracy, 4)
)

print(
    "Precision         :",
    round(precision, 4)
)

print(
    "Recall/Sensitivity:",
    round(sensitivity, 4)
)

print(
    "Specificity       :",
    round(specificity, 4)
)

print(
    "F1 Score          :",
    round(f1, 4)
)

print("\nConfusion Matrix:")

print(cm)

# ============================================================
# 35. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(

    classification_report(

        test_true,

        test_pred,

        target_names=[

            "melanoma",

            "non_melanoma"

        ],

        zero_division=0
    )
)

# ============================================================
# 36. DEFAULT 0.50 THRESHOLD RESULTS
# ============================================================

print("\n")
print("==========================================")
print("DEFAULT THRESHOLD = 0.50")
print("==========================================")

default_accuracy = accuracy_score(

    test_true,

    test_pred_default
)

default_precision = precision_score(

    test_true,

    test_pred_default,

    zero_division=0
)

default_recall = recall_score(

    test_true,

    test_pred_default,

    zero_division=0
)

default_f1 = f1_score(

    test_true,

    test_pred_default,

    zero_division=0
)

print(
    "Accuracy :",
    round(default_accuracy, 4)
)

print(
    "Precision:",
    round(default_precision, 4)
)

print(
    "Recall   :",
    round(default_recall, 4)
)

print(
    "F1       :",
    round(default_f1, 4)
)

# ============================================================
# 37. SAVE MODEL
# ============================================================

print("\nSaving final model...")

model.save(
    MODEL_SAVE
)

print(
    "MODEL SAVED:"
)

print(
    MODEL_SAVE
)

# ============================================================
# 38. CLEANUP
# ============================================================

gc.collect()

print("\n")
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")