# ==========================================================
# MELANOMA DETECTION - NEXT EXPERIMENT
# EfficientNetV2S | 320x320 | BALANCED DATA
# ==========================================================

# ==========================================================
# 1. IMPORTS
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
# 2. GPU
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU detected:", gpus)

    except RuntimeError as e:
        print(e)

else:
    print("WARNING: GPU NOT DETECTED")


# ==========================================================
# 3. MIXED PRECISION
# ==========================================================

mixed_precision.set_global_policy("mixed_float16")

print("Mixed precision policy:",
      mixed_precision.global_policy())


# ==========================================================
# 4. RANDOM SEED
# ==========================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ==========================================================
# 5. PATHS
# ==========================================================

DATASET = "/content/newdata"

IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata_backup"

CHECKPOINT = (
    "/drive/MyDrive/checkpoints/"
    "best_model_v4.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/Colab Notebooks/"
    "Models/dermoscopy/"
    "final_model_v4.keras"
)


# ==========================================================
# 6. COPY DATASET TO COLAB
# ==========================================================

print("\n==============================")
print("COPYING DATASET")
print("==============================")

if os.path.exists(DATASET):
    shutil.rmtree(DATASET)

shutil.copytree(
    IMG_SRC,
    DATASET
)

print("Dataset copied successfully.")


# ==========================================================
# 7. SETTINGS
# ==========================================================

IMG_SIZE = 320

BATCH_SIZE = 8

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# 8. DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        mode="horizontal_and_vertical"
    ),

    layers.RandomRotation(
        factor=0.10
    ),

    layers.RandomZoom(
        height_factor=(-0.10, 0.15),
        width_factor=(-0.10, 0.15)
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(
        factor=0.15
    ),

], name="augmentation")


# ==========================================================
# 9. DATASET LOADER
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

    # Performance optimization
    ds = ds.cache()

    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds, class_names


# ==========================================================
# 10. LOAD DATASETS
# ==========================================================

print("\n==============================")
print("LOADING DATASETS")
print("==============================")


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


print("\nClass names:")
print(class_names)


# ==========================================================
# 11. PRINT DATASET COUNTS
# ==========================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Train:")
print("Melanoma      : 7122")
print("Non-melanoma  : 7122")

print("\nValidation:")
print("Melanoma      : 111")
print("Non-melanoma  : 890")

print("\nTest:")
print("Melanoma      : 112")
print("Non-melanoma  : 890")


# ==========================================================
# 12. LOSS
# ==========================================================

loss_fn = tf.keras.losses.CategoricalCrossentropy()


# ==========================================================
# 13. MODEL CREATION
# ==========================================================

def create_model():

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

    # ------------------------------------------------------
    # STAGE 1
    # Freeze backbone
    # ------------------------------------------------------

    backbone.trainable = False

    features = backbone(
        x,
        training=False
    )

    # ------------------------------------------------------
    # CLASSIFICATION HEAD
    # ------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(
        features
    )

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        512,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    output = layers.Dense(
        2,
        activation="softmax",
        dtype="float32",
        name="prediction"
    )(x)

    model = Model(
        inputs=inputs,
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

            tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.AUC(
                name="auc",
                curve="ROC",
                num_thresholds=200
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
# 14. CREATE MODEL
# ==========================================================

model, backbone = create_model()

model.summary()


# ==========================================================
# 15. CALLBACKS
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

    mode="max",

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

print("\nClass weights:")
print("NONE")

print(
    "\nTraining dataset is balanced:"
    "\nMelanoma = 7122"
    "\nNon-melanoma = 7122"
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

    callbacks=callbacks,

    verbose=1
)


# ==========================================================
# LOAD BEST STAGE 1 MODEL
# ==========================================================

print("\nLoading best Stage 1 checkpoint...")

model.load_weights(
    CHECKPOINT
)


# ==========================================================
# STAGE 2
# FINE TUNING
# ==========================================================

print("\n==============================")
print("STAGE 2 FINE TUNING")
print("==============================")


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


print(
    "\nTrainable backbone layers:"
)

print(
    sum(
        layer.trainable
        for layer in backbone.layers
    )
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

        tf.keras.metrics.CategoricalAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC",
            num_thresholds=200
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
# LOAD BEST MODEL
# ==========================================================

print("\n==============================")
print("LOADING BEST MODEL")
print("==============================")


model.load_weights(
    CHECKPOINT
)

print("Best model loaded.")


# ==========================================================
# STANDARD TEST EVALUATION
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
# COLLECT TEST PREDICTIONS
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

    # Probability of melanoma
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
# IMPORTANT
# CLASS INDEX CHECK
# ==========================================================

print("\nClass mapping:")

for i, name in enumerate(
    class_names
):

    print(
        i,
        "->",
        name
    )


# ==========================================================
# ROC AUC
# ==========================================================

test_auc = roc_auc_score(
    y_true,
    y_prob
)


print(
    "\nTest ROC AUC:",
    round(test_auc, 4)
)


# ==========================================================
# DEFAULT THRESHOLD
# ==========================================================

threshold = 0.50


y_pred = (
    y_prob >= threshold
).astype(int)


# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
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


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================")
print("TEST METRICS - THRESHOLD 0.50")
print("==============================")

print(
    f"Accuracy  : {accuracy:.4f}"
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

print(
    f"AUC       : {test_auc:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(

        y_true,

        y_pred,

        target_names=[
            "melanoma",
            "non_melanoma"
        ],

        zero_division=0
    )
)


# ==========================================================
# SENSITIVITY / SPECIFICITY
# ==========================================================

tn, fp, fn, tp = cm.ravel()


sensitivity = tp / (
    tp + fn
) if (tp + fn) > 0 else 0


specificity = tn / (
    tn + fp
) if (tn + fp) > 0 else 0


print(
    "Sensitivity:",
    round(sensitivity, 4)
)

print(
    "Specificity:",
    round(specificity, 4)
)


# ==========================================================
# VALIDATION THRESHOLD SEARCH
# ==========================================================

print("\n==============================")
print("VALIDATION THRESHOLD SEARCH")
print("==============================")


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
        predictions[:, 0]
    )


val_true = np.array(
    val_true
)

val_prob = np.array(
    val_prob
)


# ==========================================================
# SEARCH THRESHOLDS
# ==========================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_threshold = 0.50
best_f1 = 0.0


for t in thresholds:

    pred = (
        val_prob >= t
    ).astype(int)

    score = f1_score(
        val_true,
        pred,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = t


print(
    "\nBest validation threshold:",
    round(best_threshold, 2)
)

print(
    "Best validation F1:",
    round(best_f1, 4)
)


# ==========================================================
# TEST WITH VALIDATION THRESHOLD
# ==========================================================

print("\n==============================")
print("TEST WITH VALIDATION THRESHOLD")
print("==============================")


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
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {optimized_accuracy:.4f}"
)

print(
    f"Precision : {optimized_precision:.4f}"
)

print(
    f"Recall    : {optimized_recall:.4f}"
)

print(
    f"F1 Score  : {optimized_f1:.4f}"
)


print("\nConfusion Matrix:")

print(
    optimized_cm
)


# ==========================================================
# SAVE MODEL
# ==========================================================

print("\n==============================")
print("SAVING FINAL MODEL")
print("==============================")


model.save(
    MODEL_SAVE
)


print(
    "\nMODEL SAVED SUCCESSFULLY:"
)

print(
    MODEL_SAVE
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n========================================")
print("FINAL SUMMARY")
print("========================================")

print(
    "Image size:",
    IMG_SIZE
)

print(
    "Training images:",
    7122 * 2
)

print(
    "Stage 1 epochs:",
    len(history_stage1.history["loss"])
)

print(
    "Stage 2 epochs:",
    len(history_stage2.history["loss"])
)

print(
    f"\nTest Accuracy @ 0.50: "
    f"{accuracy:.4f}"
)

print(
    f"Test AUC: "
    f"{test_auc:.4f}"
)

print(
    f"Test F1 @ 0.50: "
    f"{f1:.4f}"
)

print(
    f"\nValidation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Test Accuracy @ optimized threshold: "
    f"{optimized_accuracy:.4f}"
)

print(
    f"Test F1 @ optimized threshold: "
    f"{optimized_f1:.4f}"
)

print(
    "\n========================================"
)