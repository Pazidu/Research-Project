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
# 3. VERSION CHECK
# ============================================================

print("\n")
print("==========================================")
print("ENVIRONMENT")
print("==========================================")

print(
    "TensorFlow:",
    tf.__version__
)

print(
    "Keras:",
    tf.keras.__version__
)


# ============================================================
# 4. GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print(
    "\nGPUs:",
    gpus
)


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
# 5. MIXED PRECISION
# ============================================================

mixed_precision.set_global_policy(
    "mixed_float16"
)

print(
    "\nMixed precision policy:",
    mixed_precision.global_policy()
)


# ============================================================
# 6. RANDOM SEED
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 7. PATHS
# ============================================================

DATASET = "/content/newdata"

IMG_SRC = (
    "/drive/MyDrive/Colab Notebooks/newdata_backup"
)


# ------------------------------------------------------------
# V9 CHECKPOINT
# ------------------------------------------------------------

CHECKPOINT_DIR = (
    "/drive/MyDrive/checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


CHECKPOINT = (
    CHECKPOINT_DIR
    +
    "/efficientnet_v9_best.keras"
)


# ------------------------------------------------------------
# V9 MODEL
# ------------------------------------------------------------

MODEL_DIR = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_SAVE = (
    MODEL_DIR
    +
    "/efficientnet_v9_final.keras"
)


THRESHOLD_SAVE = (
    MODEL_DIR
    +
    "/efficientnet_v9_threshold.txt"
)


SUMMARY_SAVE = (
    MODEL_DIR
    +
    "/efficientnet_v9_results.json"
)


# ============================================================
# 8. SETTINGS
# ============================================================

IMG_SIZE = 288

BATCH_SIZE = 8

STAGE1_EPOCHS = 12

STAGE2_EPOCHS = 10

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------
# V9 FINE-TUNING
# ------------------------------------------------------------

FINE_TUNE_LAYERS = 80


# ------------------------------------------------------------
# V9 FOCAL LOSS
# ------------------------------------------------------------

FOCAL_GAMMA = 2.0

FOCAL_ALPHA = 0.75


# ============================================================
# 9. COPY DATASET TO LOCAL COLAB STORAGE
# ============================================================

print("\n")
print("==========================================")
print("DATASET PREPARATION")
print("==========================================")


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
    "Dataset copied successfully."
)


# ============================================================
# 10. IMAGE COUNTER
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


# ============================================================
# 11. DATASET COUNTS
# ============================================================

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


        count = count_images(
            path
        )


        print(
            f"{cls}: {count}"
        )


# ============================================================
# 12. DATA AUGMENTATION
# ============================================================
#
# Keep augmentation realistic for dermoscopy.
#
# Horizontal/vertical flip:
# safe for lesion orientation.
#
# Rotation:
# safe because lesion orientation is not clinically important.
#
# Zoom/translation:
# improves spatial robustness.
#
# Contrast:
# helps with lighting/camera variation.
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
# 13. DATASET LOADER
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
    # --------------------------------------------------------

    if class_names != [

        "melanoma",
        "non_melanoma"

    ]:

        raise ValueError(

            "Unexpected class order: "
            + str(class_names)

        )


    # --------------------------------------------------------
    # IMPORTANT LABEL CONVERSION
    #
    # Directory order:
    #
    # melanoma     = 0
    # non_melanoma = 1
    #
    # Convert to:
    #
    # non_melanoma = 0
    # melanoma     = 1
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
# 14. LOAD DATASETS
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
# 15. IMPORTANT LABEL MAPPING
# ============================================================

print("\n")
print("==========================================")
print("IMPORTANT LABEL MAPPING")
print("==========================================")


print(
    "0 = non_melanoma"
)

print(
    "1 = melanoma"
)

print(
    "MODEL OUTPUT = P(melanoma)"
)


# ============================================================
# 16. BINARY FOCAL LOSS
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
        # Positive class = melanoma
        #
        # alpha = 0.75
        #
        # melanoma gets higher importance.
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

    gamma=FOCAL_GAMMA,

    alpha=FOCAL_ALPHA

)


# ============================================================
# 17. TEST EFFICIENTNETV2S BEFORE BUILDING V9
# ============================================================
#
# This protects against the previous:
#
# KeyError: 'efficientnetv2s'
#
# problem.
#
# Your current environment has already successfully passed
# this test.
# ============================================================

print("\n")
print("==========================================")
print("TESTING EFFICIENTNETV2S")
print("==========================================")


test_backbone = EfficientNetV2S(

    include_top=False,

    weights="imagenet",

    input_shape=(

        IMG_SIZE,
        IMG_SIZE,
        3

    )

)


print(
    "SUCCESS: EfficientNetV2S created"
)


print(
    "Model name:",
    test_backbone.name
)


del test_backbone

gc.collect()


# ============================================================
# 18. CREATE MODEL
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
    # EFFICIENTNETV2-S
    # --------------------------------------------------------

    print("\nLoading EfficientNetV2S ImageNet weights...")


    backbone = EfficientNetV2S(

        include_top=False,

        weights="imagenet",

        input_shape=(

            IMG_SIZE,
            IMG_SIZE,
            3

        )

    )


    print(
        "SUCCESS: ImageNet weights loaded"
    )


    print(
        "Model name:",
        backbone.name
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
    # sigmoid = P(melanoma)
    # --------------------------------------------------------

    outputs = layers.Dense(

        1,

        activation="sigmoid",

        dtype="float32",

        name="prediction"

    )(x)


    model = Model(

        inputs=inputs,

        outputs=outputs,

        name="efficientnet_v2s_melanoma_v9"

    )


    return model, backbone


# ============================================================
# 19. CREATE V9 MODEL
# ============================================================

print("\n")
print("==========================================")
print("CREATING V9 MODEL")
print("==========================================")


model, backbone = create_model()


model.summary()


# ============================================================
# 20. METRICS
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
# 21. STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4

    ),

    loss=loss_fn,

    metrics=get_metrics()

)


# ============================================================
# 22. CALLBACKS
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
# 23. STAGE 1
# ============================================================

print("\n")
print("==========================================")
print("STAGE 1 - FROZEN EFFICIENTNETV2-S")
print("==========================================")


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE1_EPOCHS,

    callbacks=callbacks

)


# ============================================================
# 24. CHECKPOINT EXISTS
# ============================================================

print("\n")
print("Checking Stage 1 checkpoint...")


if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(

        "Stage 1 checkpoint was not created:\n"
        + CHECKPOINT

    )


print(
    "Checkpoint found:"
)

print(
    CHECKPOINT
)


# ============================================================
# 25. LOAD BEST STAGE 1
# ============================================================

print("\n")
print("Loading best Stage 1 model...")


model.load_weights(
    CHECKPOINT
)


print(
    "Best Stage 1 model loaded."
)


gc.collect()


# ============================================================
# 26. STAGE 2 - FINE TUNING
# ============================================================

print("\n")
print("==========================================")
print("STAGE 2 - FINE TUNING")
print("==========================================")


backbone.trainable = True


# ------------------------------------------------------------
# Freeze most layers
# ------------------------------------------------------------

for layer in backbone.layers[:-FINE_TUNE_LAYERS]:

    layer.trainable = False


# ------------------------------------------------------------
# Keep BatchNormalization frozen
# ------------------------------------------------------------

for layer in backbone.layers:

    if isinstance(

        layer,

        layers.BatchNormalization

    ):

        layer.trainable = False


# ------------------------------------------------------------
# Print trainable layers
# ------------------------------------------------------------

trainable_count = sum(

    1

    for layer in backbone.layers

    if layer.trainable

)


print(
    "Trainable backbone layers:",
    trainable_count
)


# ============================================================
# 27. COMPILE STAGE 2
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=5e-6

    ),

    loss=loss_fn,

    metrics=get_metrics()

)


# ============================================================
# 28. STAGE 2
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=callbacks

)


# ============================================================
# 29. CHECK FINAL CHECKPOINT
# ============================================================

if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(

        "Best checkpoint does not exist:\n"
        + CHECKPOINT

    )


# ============================================================
# 30. LOAD ABSOLUTE BEST MODEL
# ============================================================

print("\n")
print("Loading absolute best V9 model...")


model.load_weights(
    CHECKPOINT
)


print(
    "Absolute best model loaded."
)


# ============================================================
# 31. GET PREDICTIONS
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
# 32. VALIDATION PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION PREDICTIONS")
print("==========================================")


val_probs, val_true = get_predictions(

    val_ds

)


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
# 33. VALIDATION ROC-AUC
# ============================================================

val_auc = roc_auc_score(

    val_melanoma_true,

    val_probs

)


# ============================================================
# 34. VALIDATION PR-AUC
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
# 35. VALIDATION THRESHOLD SEARCH
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION THRESHOLD SEARCH")
print("==========================================")


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


# ------------------------------------------------------------
# Search thresholds
# ------------------------------------------------------------

for threshold in np.arange(

    0.01,

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
    # SENSITIVITY TARGETS
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

                "threshold":
                    float(threshold),

                "sensitivity":
                    float(sensitivity),

                "specificity":
                    float(specificity),

                "f1":
                    float(f1)

            }


# ============================================================
# 36. PRINT THRESHOLD RESULTS
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
# 37. SELECT FINAL THRESHOLD
# ============================================================
#
# IMPORTANT:
#
# The test set is NOT used to select threshold.
#
# V9 default:
# validation F1 optimization.
#
# For a medical screening application, you may eventually
# prefer a sensitivity-oriented threshold.
# ============================================================

FINAL_THRESHOLD = best_f1_threshold


THRESHOLD_REASON = (
    "validation F1 optimization"
)


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

    "Reason:",

    THRESHOLD_REASON

)


# ============================================================
# 38. TEST PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("TEST PREDICTIONS")
print("==========================================")


test_probs, test_true = get_predictions(

    test_ds

)


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
# 39. TEST ROC-AUC
# ============================================================

test_auc = roc_auc_score(

    test_melanoma_true,

    test_probs

)


# ============================================================
# 40. TEST PR-AUC
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

    "Test PR-AUC:",

    round(

        test_pr_auc,

        4

    )

)


# ============================================================
# 41. METRIC FUNCTION
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

        "confusion_matrix":
            cm.tolist()

    }


# ============================================================
# 42. FINAL TEST RESULTS
# ============================================================

final_metrics = calculate_metrics(

    test_melanoma_true,

    test_probs,

    FINAL_THRESHOLD

)


print("\n")
print("==========================================")
print("FINAL TEST RESULTS")
print("==========================================")


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
# 43. CONFUSION MATRIX
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
# 44. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")


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
# 45. THRESHOLD COMPARISON
# ============================================================

print("\n")
print("==========================================")
print("THRESHOLD COMPARISON")
print("==========================================")


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
# 46. PROBABILITY ANALYSIS
# ============================================================

print("\n")
print("==========================================")
print("PROBABILITY ANALYSIS")
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
# 47. SAVE MODEL
# ============================================================

print("\n")
print("==========================================")
print("SAVING MODEL")
print("==========================================")


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
# 48. SAVE THRESHOLD INFORMATION
# ============================================================

threshold_info = {

    "experiment":
        "V9",

    "model":
        "EfficientNetV2S",

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

    },

    "model_output":
        "P(melanoma)"

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
# 49. SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment":
        "V9",

    "model":
        "EfficientNetV2S",

    "tensorflow_version":
        tf.__version__,

    "keras_version":
        tf.keras.__version__,

    "label_mapping": {

        "0":
            "non_melanoma",

        "1":
            "melanoma"

    },

    "model_output":
        "P(melanoma)",

    "image_size":
        IMG_SIZE,

    "batch_size":
        BATCH_SIZE,

    "fine_tune_layers":
        FINE_TUNE_LAYERS,

    "focal_loss": {

        "gamma":
            FOCAL_GAMMA,

        "alpha":
            FOCAL_ALPHA

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
# 50. CLEANUP
# ============================================================

gc.collect()


# ============================================================
# 51. FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================")
print("EXPERIMENT V9 COMPLETE")
print("==========================================")


print("\n")
print("FINAL LABEL MAPPING:")

print(
    "0 = non_melanoma"
)

print(
    "1 = melanoma"
)


print("\n")
print("FINAL MELANOMA PROBABILITY:")

print(
    "model output directly = P(melanoma)"
)


print("\n")
print("FINAL THRESHOLD:")

print(

    round(

        FINAL_THRESHOLD,

        4

    )

)


print("\n")
print("FINAL TEST ROC-AUC:")

print(

    round(

        test_auc,

        4

    )

)


print("\n")
print("FINAL TEST PR-AUC:")

print(

    round(

        test_pr_auc,

        4

    )

)


print("\n")
print("FINAL TEST SENSITIVITY:")

print(

    round(

        final_metrics["sensitivity"],

        4

    )

)


print("\n")
print("FINAL TEST SPECIFICITY:")

print(

    round(

        final_metrics["specificity"],

        4

    )

)


print("\n")
print("FINAL TEST F1:")

print(

    round(

        final_metrics["f1"],

        4

    )

)


print("\n")
print("FINAL TEST BALANCED ACCURACY:")

print(

    round(

        final_metrics["balanced_accuracy"],

        4

    )

)


print("\n")
print("MODEL:")

print(
    MODEL_SAVE
)


print("\n")
print("THRESHOLD:")

print(
    THRESHOLD_SAVE
)


print("\n")
print("SUMMARY:")

print(
    SUMMARY_SAVE
)


print("\n")
print("==========================================")
print("V9 FINISHED SUCCESSFULLY")
print("==========================================")