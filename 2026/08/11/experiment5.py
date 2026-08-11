# ============================================================
# MELANOMA CLASSIFICATION - EXPERIMENT V5
#
# EfficientNetV2S
# Balanced training dataset: 7122 / 7122
# Focal Loss
# Strong medical-image augmentation
# RAM-SAFE
# Threshold optimization
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
    f1_score,
    roc_curve
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
    "efficientnet_v5_best.keras"
)

MODEL_SAVE = (
    "/drive/MyDrive/"
    "Colab Notebooks/"
    "Models/dermoscopy/"
    "efficientnet_v5_final.keras"
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
# IMPORTANT
# ============================================================

print(
    "\nIMPORTANT CLASS ORDER:"
)

print(
    "0 = melanoma"
)

print(
    "1 = non_melanoma"
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


        # Standard binary cross entropy

        bce = -(
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true)
            *
            tf.math.log(1.0 - y_pred)
        )


        # Focal weighting

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


    # Stage 1

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


    # IMPORTANT:
    # sigmoid output = probability of class 1
    #
    # BUT our class 0 is melanoma.
    #
    # Therefore later we will convert:
    #
    # melanoma probability = 1 - sigmoid_output
    #


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
# 16. STAGE 1 COMPILE
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
# 17. CALLBACKS
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

    callbacks=callbacks

)


# ============================================================
# 19. LOAD BEST STAGE 1
# ============================================================

print(
    "\nLoading best Stage 1 model..."
)


model.load_weights(
    CHECKPOINT
)


gc.collect()


# ============================================================
# 20. STAGE 2
# ============================================================

print("\n")
print("==========================================")
print("STAGE 2 - FINE TUNING")
print("==========================================")


backbone.trainable = True


# Freeze most layers

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
# 21. COMPILE STAGE 2
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
# 22. STAGE 2 TRAIN
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=callbacks

)


# ============================================================
# 23. LOAD ABSOLUTE BEST MODEL
# ============================================================

print(
    "\nLoading absolute best model..."
)


model.load_weights(
    CHECKPOINT
)


# ============================================================
# 24. FUNCTION TO GET MELANOMA PROBABILITY
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


        # sigmoid output represents
        # probability of class 1
        #
        # class 1 = non_melanoma
        #
        # Therefore:
        #
        # melanoma probability = 1 - prediction

        melanoma_probability = (
            1.0 - predictions[:, 0]
        )


        probabilities.extend(
            melanoma_probability
        )


        true_labels.extend(
            labels.numpy().astype(int).reshape(-1)
        )


    return (
        np.array(probabilities),
        np.array(true_labels)
    )


# ============================================================
# 25. VALIDATION PREDICTIONS
# ============================================================

print("\n")
print("==========================================")
print("VALIDATION PREDICTIONS")
print("==========================================")


val_probs, val_true = get_predictions(
    val_ds
)


print(
    "Validation samples:",
    len(val_true)
)


# ============================================================
# 26. VALIDATION AUC
# ============================================================

val_auc = roc_auc_score(

    val_true,

    # true melanoma label is 0
    # therefore invert labels

    1 - val_true,

    sample_weight=None
)

# Correct ROC-AUC using melanoma as positive class

val_auc = roc_auc_score(

    (val_true == 0).astype(int),

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


# melanoma = positive

val_melanoma_true = (
    val_true == 0
).astype(int)


best_accuracy = 0.0

best_accuracy_threshold = 0.50


best_f1 = 0.0

best_f1_threshold = 0.50


best_balanced = 0.0

best_balanced_threshold = 0.50


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


    balanced_score = (
        sensitivity + specificity
    ) / 2


    if acc > best_accuracy:

        best_accuracy = acc

        best_accuracy_threshold = threshold


    if f1 > best_f1:

        best_f1 = f1

        best_f1_threshold = threshold


    if balanced_score > best_balanced:

        best_balanced = balanced_score

        best_balanced_threshold = threshold


print(
    "Best accuracy threshold:",
    round(best_accuracy_threshold, 2)
)

print(
    "Validation accuracy:",
    round(best_accuracy, 4)
)


print(
    "\nBest F1 threshold:",
    round(best_f1_threshold, 2)
)

print(
    "Validation F1:",
    round(best_f1, 4)
)


print(
    "\nBest balanced threshold:",
    round(best_balanced_threshold, 2)
)

print(
    "Validation balanced score:",
    round(best_balanced, 4)
)


# ============================================================
# 28. TEST PREDICTIONS
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


# ============================================================
# 29. TEST AUC
# ============================================================

test_auc = roc_auc_score(

    test_melanoma_true,

    test_probs
)


print(
    "Test ROC-AUC:",
    round(test_auc, 4)
)


# ============================================================
# 30. TEST WITH ACCURACY-OPTIMIZED THRESHOLD
# ============================================================

threshold = best_accuracy_threshold


test_pred = (
    test_probs >= threshold
).astype(int)


# ============================================================
# 31. METRICS
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


# ============================================================
# 32. CONFUSION MATRIX
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
# 33. FINAL RESULTS
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
    round(threshold, 2)
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
    "Sensitivity       :",
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


print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# 34. CLASSIFICATION REPORT
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
# 35. TEST WITH F1 THRESHOLD
# ============================================================

print("\n")
print("==========================================")
print("TEST WITH F1-OPTIMIZED THRESHOLD")
print("==========================================")


f1_threshold_pred = (

    test_probs >= best_f1_threshold

).astype(int)


f1_acc = accuracy_score(

    test_melanoma_true,

    f1_threshold_pred
)


f1_precision = precision_score(

    test_melanoma_true,

    f1_threshold_pred,

    zero_division=0
)


f1_recall = recall_score(

    test_melanoma_true,

    f1_threshold_pred,

    zero_division=0
)


f1_value = f1_score(

    test_melanoma_true,

    f1_threshold_pred,

    zero_division=0
)


print(
    "Threshold:",
    round(best_f1_threshold, 2)
)


print(
    "Accuracy:",
    round(f1_acc, 4)
)


print(
    "Precision:",
    round(f1_precision, 4)
)


print(
    "Recall:",
    round(f1_recall, 4)
)


print(
    "F1:",
    round(f1_value, 4)
)


# ============================================================
# 36. SAVE MODEL
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
# 37. CLEANUP
# ============================================================

gc.collect()


print("\n")
print("==========================================")
print("EXPERIMENT V5 COMPLETE")
print("==========================================")