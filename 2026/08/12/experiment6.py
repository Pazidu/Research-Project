# ============================================================
# MELANOMA CLASSIFICATION - EXPERIMENT V6
#
# EfficientNetV2S
# Balanced training dataset: 7122 / 7122
#
# Based on V5 results:
# Validation ROC-AUC : ~0.857
# Test ROC-AUC       : ~0.859
#
# V6 CHANGES:
# - Keep 288x288
# - Keep Focal Loss
# - Keep strong augmentation
# - Keep last 80 layers fine-tuning
# - Correct melanoma probability handling
# - Validation-only threshold selection
# - F1 threshold
# - Balanced accuracy threshold
# - Sensitivity-target thresholds
# - PR-AUC
# - ROC-AUC
# - Detailed melanoma metrics
# - Probability distribution analysis
# - RAM-safe pipeline
# ============================================================


# ============================================================
# 1. CLEAR MEMORY
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

drive.mount("/drive")


import numpy as np
import tensorflow as tf

from tensorflow.keras import (
    layers,
    Model,
    mixed_precision
)

from tensorflow.keras.applications import EfficientNetV2S

from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input
)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# 3. GPU
# ============================================================

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


# ============================================================
# 4. MIXED PRECISION
# ============================================================

mixed_precision.set_global_policy(
    "mixed_float16"
)


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

IMG_SRC = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "newdata_backup"
)

CHECKPOINT = (
    "/drive/MyDrive/checkpoints/"
    "efficientnet_v6_best.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "Models/"
    "dermoscopy/"
    "efficientnet_v6_final.keras"
)


# ============================================================
# 7. COPY DATASET TO LOCAL COLAB STORAGE
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

IMG_SIZE = 288

BATCH_SIZE = 8

STAGE1_EPOCHS = 12

STAGE2_EPOCHS = 10

FINE_TUNE_LAYERS = 80

AUTOTUNE = tf.data.AUTOTUNE

# IMPORTANT:
# Training dataset is already balanced.
#
# melanoma     = 7122
# non_melanoma = 7122
#
# Therefore DO NOT use class_weight.

CLASS_WEIGHT = None


# ============================================================
# 9. CHECK DATASET
# ============================================================

def count_images(folder):

    count = 0

    if not os.path.exists(folder):

        return 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):

                count += 1

    return count


print("\n============================================")
print("DATASET COUNTS")
print("============================================")


for split in [
    "train",
    "valid",
    "test"
]:

    print("\n", split.upper())

    for cls in [
        "melanoma",
        "non_melanoma"
    ]:

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
            height_factor=(-0.12, 0.12),
            width_factor=(-0.12, 0.12)
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.12
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

        label_mode="binary",

        shuffle=shuffle,

        seed=SEED
    )

    class_names = ds.class_names

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

val_ds, val_class_names = load_dataset(
    DATASET + "/valid",
    False
)

test_ds, test_class_names = load_dataset(
    DATASET + "/test",
    False
)


print(
    "\nClass names:",
    class_names
)

print(
    "Validation class names:",
    val_class_names
)

print(
    "Test class names:",
    test_class_names
)


# ============================================================
# 13. VERIFY CLASS ORDER
# ============================================================

if class_names != [
    "melanoma",
    "non_melanoma"
]:

    raise ValueError(
        "Unexpected class order: "
        + str(class_names)
    )


print("\n============================================")
print("CLASS ORDER VERIFIED")
print("============================================")

print("Class 0 = melanoma")
print("Class 1 = non_melanoma")


# ============================================================
# 14. FOCAL LOSS
# ============================================================

class BinaryFocalLoss(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        gamma=2.0,
        alpha=0.5,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.gamma = gamma

        self.alpha = alpha


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
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true)
            *
            tf.math.log(1.0 - y_pred)
        )


        p_t = (
            y_true * y_pred
            +
            (1.0 - y_true)
            *
            (1.0 - y_pred)
        )


        alpha_factor = (
            y_true * self.alpha
            +
            (1.0 - y_true)
            *
            (1.0 - self.alpha)
        )


        focal_weight = (
            alpha_factor
            *
            tf.pow(
                1.0 - p_t,
                self.gamma
            )
        )


        loss = (
            focal_weight
            *
            bce
        )


        return tf.reduce_mean(loss)


loss_fn = BinaryFocalLoss(
    gamma=2.0,
    alpha=0.5
)


# ============================================================
# 15. MODEL
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


    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    x = augmentation(
        inputs
    )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    x = preprocess_input(
        x
    )


    # --------------------------------------------------------
    # EFFICIENTNETV2S
    # --------------------------------------------------------

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
    # Freeze backbone

    backbone.trainable = False


    x = backbone(
        x,
        training=False
    )


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(
        x
    )


    x = layers.BatchNormalization()(
        x
    )


    x = layers.Dense(
        256,
        activation="relu"
    )(x)


    x = layers.Dropout(
        0.35
    )(x)


    # sigmoid:
    #
    # 1 = non_melanoma
    # 0 = melanoma
    #
    # Therefore:
    #
    # melanoma probability =
    # 1 - sigmoid output

    outputs = layers.Dense(

        1,

        activation="sigmoid",

        dtype="float32",

        name="prediction"

    )(x)


    model = Model(

        inputs=inputs,

        outputs=outputs
    )


    return model, backbone


# ============================================================
# 16. CREATE MODEL
# ============================================================

model, backbone = create_model()

model.summary()


# ============================================================
# 17. STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4
    ),

    loss=loss_fn,

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        )

    ]
)


# ============================================================
# 18. CALLBACKS
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
# 19. STAGE 1
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

    class_weight=CLASS_WEIGHT

)


# ============================================================
# 20. LOAD BEST STAGE 1
# ============================================================

print(
    "\nLoading best Stage 1 model..."
)


model.load_weights(
    CHECKPOINT
)


gc.collect()


# ============================================================
# 21. STAGE 2 FINE-TUNING
# ============================================================

print("\n")
print("==========================================")
print("STAGE 2 - FINE TUNING")
print("==========================================")


backbone.trainable = True


# Freeze all except final 80 layers

for layer in backbone.layers:

    layer.trainable = False


for layer in backbone.layers[-FINE_TUNE_LAYERS:]:

    layer.trainable = True


# ------------------------------------------------------------
# KEEP ALL BATCH NORMALIZATION FROZEN
# ------------------------------------------------------------

for layer in backbone.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


print(
    "\nTrainable backbone layers:"
)


trainable_count = sum(

    1
    for layer in backbone.layers
    if layer.trainable

)


print(
    trainable_count
)


# ============================================================
# 22. COMPILE STAGE 2
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=5e-6
    ),

    loss=loss_fn,

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        )

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

    class_weight=CLASS_WEIGHT

)


# ============================================================
# 24. LOAD ABSOLUTE BEST MODEL
# ============================================================

print(
    "\nLoading absolute best model..."
)


model.load_weights(
    CHECKPOINT
)


# ============================================================
# 25. GET MELANOMA PROBABILITY
# ============================================================

def get_predictions(
    dataset
):

    probabilities = []

    true_labels = []


    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )


        # Model output:
        #
        # 1 = non_melanoma
        #
        # Therefore:
        #
        # melanoma probability =
        # 1 - prediction

        melanoma_probability = (
            1.0 - predictions[:, 0]
        )


        probabilities.extend(
            melanoma_probability
        )


        true_labels.extend(
            labels.numpy()
            .astype(int)
            .reshape(-1)
        )


    return (

        np.array(
            probabilities
        ),

        np.array(
            true_labels
        )

    )


# ============================================================
# 26. VALIDATION PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION PREDICTIONS")
print("==========================================")


val_probs, val_true = get_predictions(
    val_ds
)


# Original labels:
#
# 0 = melanoma
# 1 = non_melanoma
#
# Convert to:
#
# 1 = melanoma
# 0 = non_melanoma

val_melanoma_true = (
    val_true == 0
).astype(int)


print(
    "Validation samples:",
    len(val_true)
)

print(
    "Validation melanoma:",
    np.sum(val_melanoma_true)
)

print(
    "Validation non-melanoma:",
    np.sum(
        1 - val_melanoma_true
    )
)


# ============================================================
# 27. VALIDATION ROC-AUC
# ============================================================

val_auc = roc_auc_score(

    val_melanoma_true,

    val_probs
)


# ============================================================
# 28. VALIDATION PR-AUC
# ============================================================

val_pr_auc = average_precision_score(

    val_melanoma_true,

    val_probs
)


print(
    "\nValidation ROC-AUC:",
    round(val_auc, 4)
)

print(
    "Validation PR-AUC :",
    round(val_pr_auc, 4)
)


# ============================================================
# 29. THRESHOLD SEARCH
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION THRESHOLD SEARCH")
print("==========================================")


best_accuracy = 0.0
best_accuracy_threshold = 0.50


best_f1 = 0.0
best_f1_threshold = 0.50


best_balanced = 0.0
best_balanced_threshold = 0.50


# Sensitivity targets

sensitivity_targets = {

    0.80: None,

    0.85: None,

    0.90: None

}


threshold_results = []


for threshold in np.arange(

    0.05,

    0.96,

    0.01

):

    pred = (

        val_probs >= threshold

    ).astype(int)


    acc = accuracy_score(

        val_melanoma_true,

        pred
    )


    precision = precision_score(

        val_melanoma_true,

        pred,

        zero_division=0
    )


    sensitivity = recall_score(

        val_melanoma_true,

        pred,

        zero_division=0
    )


    specificity = recall_score(

        1 - val_melanoma_true,

        1 - pred,

        zero_division=0
    )


    f1 = f1_score(

        val_melanoma_true,

        pred,

        zero_division=0
    )


    balanced = (

        sensitivity
        +
        specificity

    ) / 2


    threshold_results.append(

        (

            threshold,
            acc,
            precision,
            sensitivity,
            specificity,
            f1,
            balanced

        )

    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if acc > best_accuracy:

        best_accuracy = acc

        best_accuracy_threshold = threshold


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_f1_threshold = threshold


    # --------------------------------------------------------
    # Balanced
    # --------------------------------------------------------

    if balanced > best_balanced:

        best_balanced = balanced

        best_balanced_threshold = threshold


# ============================================================
# 30. SENSITIVITY TARGET THRESHOLDS
# ============================================================

for target in sensitivity_targets.keys():

    candidates = [

        row
        for row in threshold_results

        if row[3] >= target

    ]


    if len(candidates) > 0:

        # Among thresholds meeting
        # sensitivity target,
        # choose highest specificity.

        candidates.sort(

            key=lambda x: x[4],

            reverse=True

        )


        sensitivity_targets[target] = (
            candidates[0][0],
            candidates[0][3],
            candidates[0][4],
            candidates[0][5]
        )


# ============================================================
# 31. PRINT THRESHOLD RESULTS
# ============================================================

print(
    "Best accuracy threshold:",
    round(
        best_accuracy_threshold,
        2
    )
)

print(
    "Validation accuracy:",
    round(
        best_accuracy,
        4
    )
)


print(
    "\nBest F1 threshold:",
    round(
        best_f1_threshold,
        2
    )
)

print(
    "Validation F1:",
    round(
        best_f1,
        4
    )
)


print(
    "\nBest balanced threshold:",
    round(
        best_balanced_threshold,
        2
    )
)

print(
    "Validation balanced:",
    round(
        best_balanced,
        4
    )
)


print(
    "\nSensitivity target thresholds:"
)


for target, result in sensitivity_targets.items():

    if result is None:

        print(
            f"{target:.0%}: No threshold found"
        )

    else:

        threshold_value = result[0]

        sensitivity_value = result[1]

        specificity_value = result[2]

        f1_value = result[3]


        print(

            f"{target:.0%} sensitivity -> "
            f"threshold={threshold_value:.2f}, "
            f"sensitivity={sensitivity_value:.4f}, "
            f"specificity={specificity_value:.4f}, "
            f"F1={f1_value:.4f}"

        )


# ============================================================
# 32. SELECT FINAL THRESHOLD
# ============================================================

# IMPORTANT:
#
# DO NOT select threshold based on TEST data.
#
# For melanoma detection we choose
# F1 threshold from validation.
#
# This is preferable to the accuracy threshold
# because V5 showed that the accuracy threshold
# produced very low melanoma sensitivity.

FINAL_THRESHOLD = best_f1_threshold


print("\n")
print("==========================================")
print("SELECTED VALIDATION THRESHOLD")
print("==========================================")

print(
    "Final threshold:",
    round(
        FINAL_THRESHOLD,
        2
    )
)

print(
    "Reason: validation F1 optimization"
)


# ============================================================
# 33. TEST PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("TEST PREDICTIONS")
print("==========================================")


test_probs, test_true = get_predictions(
    test_ds
)


test_melanoma_true = (
    test_true == 0
).astype(int)


print(
    "Test samples:",
    len(test_true)
)

print(
    "Test melanoma:",
    np.sum(test_melanoma_true)
)

print(
    "Test non-melanoma:",
    np.sum(
        1 - test_melanoma_true
    )
)


# ============================================================
# 34. TEST ROC-AUC
# ============================================================

test_auc = roc_auc_score(

    test_melanoma_true,

    test_probs
)


# ============================================================
# 35. TEST PR-AUC
# ============================================================

test_pr_auc = average_precision_score(

    test_melanoma_true,

    test_probs
)


print(
    "\nTest ROC-AUC:",
    round(test_auc, 4)
)

print(
    "Test PR-AUC :",
    round(test_pr_auc, 4)
)


# ============================================================
# 36. FINAL TEST PREDICTIONS
# ============================================================

test_pred = (

    test_probs >= FINAL_THRESHOLD

).astype(int)


# ============================================================
# 37. METRICS
# ============================================================

accuracy = accuracy_score(

    test_melanoma_true,

    test_pred
)


precision = precision_score(

    test_melanoma_true,

    test_pred,

    zero_division=0
)


recall = recall_score(

    test_melanoma_true,

    test_pred,

    zero_division=0
)


f1 = f1_score(

    test_melanoma_true,

    test_pred,

    zero_division=0
)


balanced = balanced_accuracy_score(

    test_melanoma_true,

    test_pred
)


# ============================================================
# 38. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_melanoma_true,

    test_pred
)


tn, fp, fn, tp = cm.ravel()


specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0

)


sensitivity = (

    tp / (tp + fn)

    if (tp + fn) > 0

    else 0

)


# ============================================================
# 39. FINAL TEST RESULTS
# ============================================================

print("\n")
print("==========================================")
print("FINAL TEST RESULTS")
print("==========================================")


print(
    "Test ROC-AUC       :",
    round(test_auc, 4)
)


print(
    "Test PR-AUC        :",
    round(test_pr_auc, 4)
)


print(
    "Threshold          :",
    round(FINAL_THRESHOLD, 2)
)


print(
    "Accuracy           :",
    round(accuracy, 4)
)


print(
    "Precision          :",
    round(precision, 4)
)


print(
    "Sensitivity        :",
    round(sensitivity, 4)
)


print(
    "Specificity        :",
    round(specificity, 4)
)


print(
    "F1 Score           :",
    round(f1, 4)
)


print(
    "Balanced Accuracy  :",
    round(balanced, 4)
)


print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# 40. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")


print(

    classification_report(

        test_melanoma_true,

        test_pred,

        target_names=[

            "non_melanoma",

            "melanoma"

        ],

        zero_division=0

    )

)


# ============================================================
# 41. COMPARE MULTIPLE THRESHOLDS ON TEST
#
# IMPORTANT:
# This section is ONLY for reporting.
#
# Threshold was selected using VALIDATION.
# ============================================================

print("\n")
print("==========================================")
print("TEST THRESHOLD COMPARISON")
print("==========================================")


thresholds_to_report = [

    (
        "Accuracy threshold",
        best_accuracy_threshold
    ),

    (
        "F1 threshold",
        best_f1_threshold
    ),

    (
        "Balanced threshold",
        best_balanced_threshold
    ),

    (
        "0.50 default",
        0.50

    )

]


for name, threshold_value in thresholds_to_report:

    pred = (

        test_probs >= threshold_value

    ).astype(int)


    acc = accuracy_score(

        test_melanoma_true,

        pred
    )


    prec = precision_score(

        test_melanoma_true,

        pred,

        zero_division=0
    )


    rec = recall_score(

        test_melanoma_true,

        pred,

        zero_division=0
    )


    f1_score_value = f1_score(

        test_melanoma_true,

        pred,

        zero_division=0
    )


    spec = recall_score(

        1 - test_melanoma_true,

        1 - pred,

        zero_division=0
    )


    print("\n", name)

    print(
        "Threshold:",
        round(threshold_value, 2)
    )

    print(
        "Accuracy:",
        round(acc, 4)
    )

    print(
        "Precision:",
        round(prec, 4)
    )

    print(
        "Recall:",
        round(rec, 4)
    )

    print(
        "Specificity:",
        round(spec, 4)
    )

    print(
        "F1:",
        round(f1_score_value, 4)
    )


# ============================================================
# 42. MELANOMA PROBABILITY DISTRIBUTION
# ============================================================

print("\n")
print("==========================================")
print("PROBABILITY DISTRIBUTION")
print("==========================================")


melanoma_probs = test_probs[
    test_melanoma_true == 1
]


non_melanoma_probs = test_probs[
    test_melanoma_true == 0
]


print(
    "\nMelanoma probability:"
)

print(
    "Mean:",
    round(
        np.mean(melanoma_probs),
        4
    )
)

print(
    "Median:",
    round(
        np.median(melanoma_probs),
        4
    )
)

print(
    "Min:",
    round(
        np.min(melanoma_probs),
        4
    )
)

print(
    "Max:",
    round(
        np.max(melanoma_probs),
        4
    )
)


print(
    "\nNon-melanoma probability:"
)

print(
    "Mean:",
    round(
        np.mean(non_melanoma_probs),
        4
    )
)

print(
    "Median:",
    round(
        np.median(non_melanoma_probs),
        4
    )
)

print(
    "Min:",
    round(
        np.min(non_melanoma_probs),
        4
    )
)

print(
    "Max:",
    round(
        np.max(non_melanoma_probs),
        4
    )
)


# ============================================================
# 43. SAVE MODEL
# ============================================================

os.makedirs(

    os.path.dirname(
        MODEL_SAVE
    ),

    exist_ok=True
)


print(
    "\nSaving final model..."
)


model.save(
    MODEL_SAVE
)


print(
    "\nMODEL SAVED:"
)

print(
    MODEL_SAVE
)


# ============================================================
# 44. SAVE THRESHOLD INFORMATION
# ============================================================

THRESHOLD_SAVE = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "Models/"
    "dermoscopy/"
    "efficientnet_v6_threshold.txt"
)


with open(
    THRESHOLD_SAVE,
    "w"
) as f:

    f.write(
        "EfficientNetV2S V6\n"
    )

    f.write(
        f"Validation ROC-AUC: {val_auc:.6f}\n"
    )

    f.write(
        f"Validation PR-AUC: {val_pr_auc:.6f}\n"
    )

    f.write(
        f"Best F1 threshold: "
        f"{best_f1_threshold:.4f}\n"
    )

    f.write(
        f"Best balanced threshold: "
        f"{best_balanced_threshold:.4f}\n"
    )

    f.write(
        f"Best accuracy threshold: "
        f"{best_accuracy_threshold:.4f}\n"
    )

    f.write(
        f"Selected final threshold: "
        f"{FINAL_THRESHOLD:.4f}\n"
    )

    f.write(
        f"Test ROC-AUC: {test_auc:.6f}\n"
    )

    f.write(
        f"Test PR-AUC: {test_pr_auc:.6f}\n"
    )

    f.write(
        f"Test F1: {f1:.6f}\n"
    )

    f.write(
        f"Test sensitivity: "
        f"{sensitivity:.6f}\n"
    )

    f.write(
        f"Test specificity: "
        f"{specificity:.6f}\n"
    )


print(
    "\nTHRESHOLD INFO SAVED:"
)

print(
    THRESHOLD_SAVE
)


# ============================================================
# 45. CLEANUP
# ============================================================

gc.collect()


print("\n")
print("==========================================")
print("EXPERIMENT V6 COMPLETE")
print("==========================================")