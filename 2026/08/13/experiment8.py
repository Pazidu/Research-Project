# ============================================================
# 1. CLEAR MEMORY
# ============================================================

import gc
import os
import random
import shutil
import json

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
    balanced_accuracy_score
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
    "/drive/MyDrive/Colab Notebooks/newdata_backup"
)

CHECKPOINT = (
    "/drive/MyDrive/checkpoints/"
    "efficientnet_v8_best.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy/efficientnet_v8_final.keras"
)

THRESHOLD_SAVE = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy/efficientnet_v8_threshold.txt"
)

SUMMARY_SAVE = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy/efficientnet_v8_results.json"
)


# ============================================================
# 7. COPY DATASET TO LOCAL COLAB STORAGE
# ============================================================

if os.path.exists(DATASET):

    print(
        "Removing old local dataset..."
    )

    shutil.rmtree(DATASET)


print(
    "Copying dataset to Colab..."
)


shutil.copytree(
    IMG_SRC,
    DATASET
)


print(
    "Dataset copied."
)


# ============================================================
# 8. SETTINGS
# ============================================================

IMG_SIZE = 288

BATCH_SIZE = 8

STAGE1_EPOCHS = 12

STAGE2_EPOCHS = 10

AUTOTUNE = tf.data.AUTOTUNE


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


print("\n")
print("============================================")
print("DATASET COUNTS")
print("============================================")


for split in [

    "train",

    "valid",

    "test"

]:

    print(
        "\n",
        split.upper()
    )

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

            height_factor=(
                -0.12,
                0.12
            ),

            width_factor=(
                -0.12,
                0.12
            )

        ),

        layers.RandomTranslation(

            height_factor=0.05,

            width_factor=0.05

        ),

        layers.RandomContrast(
            0.12
        )

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


    print(
        "Original class order:",
        class_names
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    #
    # image_dataset_from_directory sorts alphabetically:
    #
    # melanoma = 0
    # non_melanoma = 1
    # --------------------------------------------------------

    if class_names != [

        "melanoma",

        "non_melanoma"

    ]:

        raise ValueError(

            f"Unexpected class order: "
            f"{class_names}"

        )


    # --------------------------------------------------------
    # CHANGE LABEL SEMANTICS
    #
    # Original:
    #
    # melanoma = 0
    # non_melanoma = 1
    #
    # V8:
    #
    # non_melanoma = 0
    # melanoma = 1
    # --------------------------------------------------------

    ds = ds.map(

        lambda images, labels: (

            images,

            1.0 - labels

        ),

        num_parallel_calls=AUTOTUNE

    )


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
# IMPORTANT MODEL CLASS ORDER
# ============================================================

print("\n")
print("==========================================")
print("IMPORTANT MODEL LABEL ORDER")
print("==========================================")

print(
    "0 = non_melanoma"
)

print(
    "1 = melanoma"
)


# ============================================================
# 13. FOCAL LOSS
# ============================================================

class BinaryFocalLoss(
    tf.keras.losses.Loss
):

    def __init__(

        self,

        gamma=2.0,

        alpha=0.75,

        **kwargs

    ):

        super().__init__(
            **kwargs
        )

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


        epsilon = (
            tf.keras.backend.epsilon()
        )


        y_pred = tf.clip_by_value(

            y_pred,

            epsilon,

            1.0 - epsilon

        )


        bce = -(

            y_true
            *
            tf.math.log(y_pred)

            +

            (1.0 - y_true)
            *
            tf.math.log(
                1.0 - y_pred
            )

        )


        p_t = (

            y_true
            *
            y_pred

            +

            (1.0 - y_true)
            *
            (1.0 - y_pred)

        )


        # ----------------------------------------------------
        # alpha = 0.75
        #
        # Positive class = melanoma
        #
        # Therefore melanoma receives higher weight.
        # ----------------------------------------------------

        alpha_factor = (

            y_true
            *
            self.alpha

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


        return tf.reduce_mean(
            loss
        )


loss_fn = BinaryFocalLoss(

    gamma=2.0,

    alpha=0.75

)


# ============================================================
# 14. MODEL
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


    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # OUTPUT
    #
    # 0 = non_melanoma
    # 1 = melanoma
    #
    # sigmoid output = melanoma probability
    # --------------------------------------------------------

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
# 15. CREATE MODEL
# ============================================================

model, backbone = create_model()

model.summary()


# ============================================================
# 16. METRICS
# ============================================================

def get_metrics():

    return [

        "accuracy",

        tf.keras.metrics.AUC(

            name="auc",

            curve="ROC"

        ),

        tf.keras.metrics.AUC(

            name="pr_auc",

            curve="PR"

        )

    ]


# ============================================================
# 17. STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4

    ),

    loss=loss_fn,

    metrics=get_metrics()

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

print(
    "=========================================="
)

print(
    "STAGE 1 - FROZEN EFFICIENTNET"
)

print(
    "=========================================="
)


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE1_EPOCHS,

    callbacks=callbacks

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
# 21. STAGE 2 - FINE TUNING
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "STAGE 2 - FINE TUNING"
)

print(
    "=========================================="
)


backbone.trainable = True


# ------------------------------------------------------------
# Freeze most layers
# ------------------------------------------------------------

for layer in backbone.layers[:-80]:

    layer.trainable = False


# ------------------------------------------------------------
# KEEP BATCH NORMALIZATION FROZEN
# ------------------------------------------------------------

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

    loss=loss_fn,

    metrics=get_metrics()

)


# ============================================================
# 23. STAGE 2
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=callbacks

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
#
# V8 IMPORTANT CHANGE:
#
# Model output directly represents:
#
#     P(melanoma)
#
# because:
#
#     0 = non_melanoma
#     1 = melanoma
#
# Therefore DO NOT use:
#
#     1 - predictions
#
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


        # ----------------------------------------------------
        # DIRECT MELANOMA PROBABILITY
        # ----------------------------------------------------

        melanoma_probability = (

            predictions[:, 0]

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

        np.array(probabilities),

        np.array(true_labels)

    )


# ============================================================
# 26. VALIDATION PREDICTIONS
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "VALIDATION PREDICTIONS"
)

print(
    "=========================================="
)


val_probs, val_true = get_predictions(

    val_ds

)


# ------------------------------------------------------------
# Since:
#
# 0 = non_melanoma
# 1 = melanoma
#
# melanoma_true is simply val_true.
# ------------------------------------------------------------

val_melanoma_true = (

    val_true == 1

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

print(
    "=========================================="
)

print(
    "VALIDATION THRESHOLD SEARCH"
)

print(
    "=========================================="
)


best_accuracy = -1

best_accuracy_threshold = 0.50


best_f1 = -1

best_f1_threshold = 0.50


best_balanced = -1

best_balanced_threshold = 0.50


# ------------------------------------------------------------
# Sensitivity targets
# ------------------------------------------------------------

sensitivity_targets = [

    0.80,

    0.85,

    0.90

]


sensitivity_results = {}


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


    f1 = f1_score(

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


    balanced = (

        sensitivity
        +
        specificity

    ) / 2


    # --------------------------------------------------------
    # BEST ACCURACY
    # --------------------------------------------------------

    if acc > best_accuracy:

        best_accuracy = acc

        best_accuracy_threshold = threshold


    # --------------------------------------------------------
    # BEST F1
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_f1_threshold = threshold


    # --------------------------------------------------------
    # BEST BALANCED ACCURACY
    # --------------------------------------------------------

    if balanced > best_balanced:

        best_balanced = balanced

        best_balanced_threshold = threshold


    # --------------------------------------------------------
    # SENSITIVITY TARGET
    # --------------------------------------------------------

    for target in sensitivity_targets:

        key = int(
            target * 100
        )


        if (

            sensitivity >= target

            and key not in sensitivity_results

        ):

            sensitivity_results[key] = {

                "threshold": float(
                    threshold
                ),

                "sensitivity": float(
                    sensitivity
                ),

                "specificity": float(
                    specificity
                ),

                "f1": float(
                    f1
                )

            }


# ============================================================
# 30. PRINT THRESHOLD RESULTS
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


for target in sensitivity_targets:

    key = int(
        target * 100
    )


    if key in sensitivity_results:

        result = sensitivity_results[key]


        print(

            f"{key}% sensitivity -> "

            f"threshold={result['threshold']:.2f}, "

            f"sensitivity={result['sensitivity']:.4f}, "

            f"specificity={result['specificity']:.4f}, "

            f"F1={result['f1']:.4f}"

        )

    else:

        print(

            f"{key}% sensitivity -> "

            "not reached"

        )


# ============================================================
# 31. SELECT FINAL THRESHOLD
# ============================================================
#
# V8 uses validation F1 optimization.
#
# The TEST set is NOT used for selecting the threshold.
#
# This prevents test-set leakage.
#
# If your research requires a minimum sensitivity target,
# you can instead select the corresponding sensitivity
# threshold from sensitivity_results.
# ============================================================

FINAL_THRESHOLD = best_f1_threshold


THRESHOLD_REASON = (
    "validation F1 optimization"
)


print("\n")

print(
    "=========================================="
)

print(
    "SELECTED VALIDATION THRESHOLD"
)

print(
    "=========================================="
)


print(

    "Final threshold:",

    round(

        FINAL_THRESHOLD,

        2

    )

)


print(

    "Reason:",

    THRESHOLD_REASON

)


# ============================================================
# 32. TEST PREDICTIONS
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "TEST PREDICTIONS"
)

print(
    "=========================================="
)


test_probs, test_true = get_predictions(

    test_ds

)


# ------------------------------------------------------------
# V8:
#
# 0 = non_melanoma
# 1 = melanoma
# ------------------------------------------------------------

test_melanoma_true = (

    test_true == 1

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
# 33. TEST ROC-AUC
# ============================================================

test_auc = roc_auc_score(

    test_melanoma_true,

    test_probs

)


# ============================================================
# 34. TEST PR-AUC
# ============================================================

test_pr_auc = average_precision_score(

    test_melanoma_true,

    test_probs

)


print(

    "\nTest ROC-AUC:",

    round(

        test_auc,

        4

    )

)


print(

    "Test PR-AUC :",

    round(

        test_pr_auc,

        4

    )

)


# ============================================================
# 35. FUNCTION FOR THRESHOLD METRICS
# ============================================================

def calculate_metrics(

    y_true,

    probs,

    threshold

):

    pred = (

        probs >= threshold

    ).astype(int)


    accuracy = accuracy_score(

        y_true,

        pred

    )


    precision = precision_score(

        y_true,

        pred,

        zero_division=0

    )


    sensitivity = recall_score(

        y_true,

        pred,

        zero_division=0

    )


    f1 = f1_score(

        y_true,

        pred,

        zero_division=0

    )


    balanced = balanced_accuracy_score(

        y_true,

        pred

    )


    cm = confusion_matrix(

        y_true,

        pred

    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

    else:

        tn = 0
        fp = 0
        fn = 0
        tp = 0


    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0

    )


    return {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1
        ),

        "balanced_accuracy": float(
            balanced
        ),

        "confusion_matrix": cm.tolist()

    }


# ============================================================
# 36. FINAL TEST RESULTS
# ============================================================

final_metrics = calculate_metrics(

    test_melanoma_true,

    test_probs,

    FINAL_THRESHOLD

)


print("\n")

print(
    "=========================================="
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=========================================="
)


print(

    "Test ROC-AUC       :",

    round(

        test_auc,

        4

    )

)


print(

    "Test PR-AUC        :",

    round(

        test_pr_auc,

        4

    )

)


print(

    "Threshold          :",

    round(

        FINAL_THRESHOLD,

        2

    )

)


print(

    "Accuracy           :",

    round(

        final_metrics["accuracy"],

        4

    )

)


print(

    "Precision          :",

    round(

        final_metrics["precision"],

        4

    )

)


print(

    "Sensitivity        :",

    round(

        final_metrics["sensitivity"],

        4

    )

)


print(

    "Specificity        :",

    round(

        final_metrics["specificity"],

        4

    )

)


print(

    "F1 Score           :",

    round(

        final_metrics["f1"],

        4

    )

)


print(

    "Balanced Accuracy  :",

    round(

        final_metrics["balanced_accuracy"],

        4

    )

)


# ============================================================
# 37. CONFUSION MATRIX
# ============================================================

cm = np.array(

    final_metrics[
        "confusion_matrix"
    ]

)


print(
    "\nConfusion Matrix:"
)


print(cm)


# ============================================================
# 38. CLASSIFICATION REPORT
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=========================================="
)


print(

    classification_report(

        test_melanoma_true,

        (

            test_probs
            >=
            FINAL_THRESHOLD

        ).astype(int),

        target_names=[

            "non_melanoma",

            "melanoma"

        ],

        zero_division=0

    )

)


# ============================================================
# 39. TEST WITH MULTIPLE THRESHOLDS
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "THRESHOLD COMPARISON"
)

print(
    "=========================================="
)


thresholds_to_compare = {

    "accuracy":
        best_accuracy_threshold,

    "f1":
        best_f1_threshold,

    "balanced":
        best_balanced_threshold,

    "default_0.50":
        0.50

}


# ------------------------------------------------------------
# Add sensitivity thresholds
# ------------------------------------------------------------

for target in sensitivity_targets:

    key = int(
        target * 100
    )


    if key in sensitivity_results:

        thresholds_to_compare[

            f"sensitivity_{key}%"

        ] = sensitivity_results[

            key

        ][

            "threshold"

        ]


# ------------------------------------------------------------
# Evaluate each threshold on TEST
# ------------------------------------------------------------

for name, threshold_value in (

    thresholds_to_compare.items()

):

    metrics = calculate_metrics(

        test_melanoma_true,

        test_probs,

        threshold_value

    )


    print(
        "\n",
        name
    )


    print(

        "Threshold:",

        round(

            threshold_value,

            2

        )

    )


    print(

        "Accuracy:",

        round(

            metrics["accuracy"],

            4

        )

    )


    print(

        "Precision:",

        round(

            metrics["precision"],

            4

        )

    )


    print(

        "Sensitivity:",

        round(

            metrics["sensitivity"],

            4

        )

    )


    print(

        "Specificity:",

        round(

            metrics["specificity"],

            4

        )

    )


    print(

        "F1:",

        round(

            metrics["f1"],

            4

        )

    )


    print(

        "Balanced Accuracy:",

        round(

            metrics["balanced_accuracy"],

            4

        )

    )


# ============================================================
# 40. PROBABILITY DISTRIBUTION
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "PROBABILITY ANALYSIS"
)

print(
    "=========================================="
)


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


print(
    "\nNon-melanoma probability:"
)


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


# ============================================================
# 41. SAVE MODEL
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
# 42. SAVE THRESHOLD
# ============================================================

threshold_info = {

    "experiment":
        "V8",

    "selected_threshold":
        float(
            FINAL_THRESHOLD
        ),

    "reason":
        THRESHOLD_REASON,

    "validation_f1":
        float(
            best_f1
        ),

    "validation_f1_threshold":
        float(
            best_f1_threshold
        ),

    "validation_accuracy":
        float(
            best_accuracy
        ),

    "validation_accuracy_threshold":
        float(
            best_accuracy_threshold
        ),

    "validation_balanced_accuracy":
        float(
            best_balanced
        ),

    "validation_balanced_threshold":
        float(
            best_balanced_threshold
        ),

    "label_mapping": {

        "0":
            "non_melanoma",

        "1":
            "melanoma"

    }

}


with open(

    THRESHOLD_SAVE,

    "w"

) as f:

    for key, value in (
        threshold_info.items()
    ):

        f.write(

            f"{key}: {value}\n"

        )


print(
    "\nTHRESHOLD INFO SAVED:"
)


print(
    THRESHOLD_SAVE
)


# ============================================================
# 43. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment":
        "V8",

    "model":
        "EfficientNetV2S",

    "label_mapping": {

        "0":
            "non_melanoma",

        "1":
            "melanoma"

    },

    "image_size":
        IMG_SIZE,

    "batch_size":
        BATCH_SIZE,

    "focal_loss": {

        "gamma":
            2.0,

        "alpha":
            0.75

    },

    "stage1_epochs":
        STAGE1_EPOCHS,

    "stage2_epochs":
        STAGE2_EPOCHS,

    "validation": {

        "samples":
            int(
                len(val_true)
            ),

        "melanoma":
            int(
                np.sum(
                    val_melanoma_true
                )
            ),

        "non_melanoma":
            int(
                np.sum(
                    1 -
                    val_melanoma_true
                )
            ),

        "roc_auc":
            float(
                val_auc
            ),

        "pr_auc":
            float(
                val_pr_auc
            ),

        "best_accuracy_threshold":
            float(
                best_accuracy_threshold
            ),

        "best_accuracy":
            float(
                best_accuracy
            ),

        "best_f1_threshold":
            float(
                best_f1_threshold
            ),

        "best_f1":
            float(
                best_f1
            ),

        "best_balanced_threshold":
            float(
                best_balanced_threshold
            ),

        "best_balanced_accuracy":
            float(
                best_balanced
            ),

        "sensitivity_targets":
            sensitivity_results

    },


    "test": {

        "samples":
            int(
                len(test_true)
            ),

        "melanoma":
            int(
                np.sum(
                    test_melanoma_true
                )
            ),

        "non_melanoma":
            int(
                np.sum(
                    1 -
                    test_melanoma_true
                )
            ),

        "roc_auc":
            float(
                test_auc
            ),

        "pr_auc":
            float(
                test_pr_auc
            ),

        "selected_threshold":
            float(
                FINAL_THRESHOLD
            ),

        "threshold_reason":
            THRESHOLD_REASON,

        "accuracy":
            final_metrics[
                "accuracy"
            ],

        "precision":
            final_metrics[
                "precision"
            ],

        "sensitivity":
            final_metrics[
                "sensitivity"
            ],

        "specificity":
            final_metrics[
                "specificity"
            ],

        "f1":
            final_metrics[
                "f1"
            ],

        "balanced_accuracy":
            final_metrics[
                "balanced_accuracy"
            ],

        "confusion_matrix":
            final_metrics[
                "confusion_matrix"
            ]

    },


    "probability_analysis": {

        "melanoma_mean":
            float(
                np.mean(
                    melanoma_probs
                )
            ),

        "melanoma_median":
            float(
                np.median(
                    melanoma_probs
                )
            ),

        "melanoma_min":
            float(
                np.min(
                    melanoma_probs
                )
            ),

        "melanoma_max":
            float(
                np.max(
                    melanoma_probs
                )
            ),

        "non_melanoma_mean":
            float(
                np.mean(
                    non_melanoma_probs
                )
            ),

        "non_melanoma_median":
            float(
                np.median(
                    non_melanoma_probs
                )
            ),

        "non_melanoma_min":
            float(
                np.min(
                    non_melanoma_probs
                )
            ),

        "non_melanoma_max":
            float(
                np.max(
                    non_melanoma_probs
                )
            )

    }

}


with open(

    SUMMARY_SAVE,

    "w"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )


print(
    "\nEXPERIMENT SUMMARY SAVED:"
)


print(
    SUMMARY_SAVE
)


# ============================================================
# 44. CLEANUP
# ============================================================

gc.collect()


print("\n")

print(
    "=========================================="
)

print(
    "EXPERIMENT V8 COMPLETE"
)

print(
    "=========================================="
)


print("\n")

print(
    "FINAL LABEL MAPPING:"
)

print(
    "0 = non_melanoma"
)

print(
    "1 = melanoma"
)

print("\n")

print(
    "FINAL MELANOMA PROBABILITY:"
)

print(
    "model output directly = P(melanoma)"
)

print("\n")

print(
    "FINAL THRESHOLD:"
)

print(
    round(
        FINAL_THRESHOLD,
        4
    )
)

print("\n")

print(
    "FINAL TEST ROC-AUC:"
)

print(
    round(
        test_auc,
        4
    )
)

print("\n")

print(
    "FINAL TEST PR-AUC:"
)

print(
    round(
        test_pr_auc,
        4
    )
)

print("\n")

print(
    "FINAL TEST SENSITIVITY:"
)

print(
    round(
        final_metrics["sensitivity"],
        4
    )
)

print("\n")

print(
    "FINAL TEST SPECIFICITY:"
)

print(
    round(
        final_metrics["specificity"],
        4
    )
)