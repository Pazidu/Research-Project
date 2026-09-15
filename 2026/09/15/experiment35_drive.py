# =============================================================================
# V35 - EfficientNetV2-S + Channel + Spatial Attention
# =============================================================================
#
# CONTROLLED EXPERIMENT
#
# V31:
#   EfficientNetV2-S
#   GAP
#   Dropout(0.45)
#   Dense(256, ReLU)
#   Dropout(0.35)
#   Dense(1, sigmoid)
#
# V35:
#   EfficientNetV2-S
#   CHANNEL ATTENTION (SE BLOCK)
#   SPATIAL ATTENTION (7x7)      <-- ONLY CONTROLLED CHANGE FROM V33
#   GAP
#   Dropout(0.45)
#   Dense(256, ReLU)
#   Dropout(0.35)
#   Dense(1, sigmoid)
#
# Everything else is kept identical to V33.
# =============================================================================


# =============================================================================
# CELL 1 - INSTALL / IMPORTS
# =============================================================================

# !pip -q install seaborn

import os
import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))


# =============================================================================
# CELL 2 - REPRODUCIBILITY
# =============================================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Random seed:", SEED)


# =============================================================================
# CELL 3 - V35 CONFIGURATION
# =============================================================================

VERSION = "V35"

# -------------------------
# Dataset
# -------------------------

DRIVE_DATASET = "/content/drive/MyDrive/Colab Notebooks/newdata_backup"
LOCAL_DATASET = "/content/newdata"

# -------------------------
# Image / batch
# -------------------------

IMG_SIZE = (300, 300)
BATCH_SIZE = 4

# -------------------------
# Training
# -------------------------

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 10

STAGE1_LR = 1e-4
STAGE2_LR = 1e-5

# Same as V33
FINE_TUNE_FRACTION = 0.60

# -------------------------
# Loss
# -------------------------

NON_MELANOMA_WEIGHT = 1.0
MELANOMA_WEIGHT = 1.5

FOCAL_GAMMA = 1.0

# -------------------------
# Threshold policy
# -------------------------

MIN_REQUIRED_SENSITIVITY = 0.70

# -------------------------
# V35 attention
# -------------------------

# SE reduction ratio
SE_REDUCTION_RATIO = 16

# Spatial attention convolution kernel
SPATIAL_KERNEL_SIZE = 7

# -------------------------
# Paths
# -------------------------
# IMPORTANT: checkpoints/results are stored directly in Google Drive.
# This protects them if the Colab runtime or GPU session is lost.

DRIVE_V35_DIR = "/content/drive/MyDrive/melanoma_checkpoints/V35"
CHECKPOINT_DIR = DRIVE_V35_DIR
MODEL_PATH = os.path.join(DRIVE_V35_DIR, "V35_EfficientNetV2S_ChannelSpatialAttention_final.keras")
RESULTS_PATH = os.path.join(DRIVE_V35_DIR, "V35_results.json")

# Backup directories for TensorFlow/Keras training-state recovery.
STAGE1_BACKUP_DIR = os.path.join(DRIVE_V35_DIR, "stage1_backup")
STAGE2_BACKUP_DIR = os.path.join(DRIVE_V35_DIR, "stage2_backup")

print("=" * 80)
print("V35 CONFIGURATION")
print("=" * 80)

print("Version                  :", VERSION)
print("Image size               :", IMG_SIZE)
print("Batch size               :", BATCH_SIZE)
print("Stage 1 epochs           :", STAGE1_EPOCHS)
print("Stage 2 epochs           :", STAGE2_EPOCHS)
print("Stage 1 learning rate    :", STAGE1_LR)
print("Stage 2 learning rate    :", STAGE2_LR)
print("Fine-tune fraction       :", FINE_TUNE_FRACTION)
print("Non-melanoma weight      :", NON_MELANOMA_WEIGHT)
print("Melanoma weight          :", MELANOMA_WEIGHT)
print("Focal gamma              :", FOCAL_GAMMA)
print("Minimum sensitivity      :", MIN_REQUIRED_SENSITIVITY)
print("SE reduction ratio       :", SE_REDUCTION_RATIO)
print("Spatial kernel size      :", SPATIAL_KERNEL_SIZE)

print("\nCONTROLLED CHANGE:")
print("V33 -> V35: Added Spatial Attention after the existing Channel Attention")


# =============================================================================
# CELL 4 - GOOGLE DRIVE
# =============================================================================

from google.colab import drive

drive.mount("/content/drive")

# Create the Drive folders only AFTER Drive has been mounted.
os.makedirs(DRIVE_V35_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("\nV35 Drive storage:")
print(CHECKPOINT_DIR)


# =============================================================================
# CELL 5 - COPY DATASET LOCALLY
# =============================================================================

print("=" * 80)
print("PREPARING DATASET")
print("=" * 80)

if not os.path.exists(DRIVE_DATASET):
    raise FileNotFoundError(
        f"Dataset not found at:\n{DRIVE_DATASET}"
    )

if os.path.exists(LOCAL_DATASET):
    print("Local dataset already exists:")
    print(LOCAL_DATASET)
else:
    print("Copying dataset from Google Drive...")
    shutil.copytree(DRIVE_DATASET, LOCAL_DATASET)
    print("Dataset copied successfully.")


# =============================================================================
# CELL 6 - VERIFY DATASET
# =============================================================================

CLASS_NAMES = ["non_melanoma", "melanoma"]

SPLITS = ["train", "valid", "test"]

print("=" * 80)
print("DATASET STRUCTURE")
print("=" * 80)

dataset_counts = {}

for split in SPLITS:

    dataset_counts[split] = {}

    for class_name in CLASS_NAMES:

        folder = os.path.join(
            LOCAL_DATASET,
            split,
            class_name
        )

        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Missing folder:\n{folder}"
            )

        count = len([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])

        dataset_counts[split][class_name] = count

        print(
            f"{split:>6} | "
            f"{class_name:>15} : {count}"
        )


# =============================================================================
# CELL 7 - DATASET GENERATORS
# =============================================================================

print("=" * 80)
print("CREATING DATASETS")
print("=" * 80)

# -----------------------------------------------------------------------------
# Training augmentation
# Same augmentation as V31
# -----------------------------------------------------------------------------

train_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),

        layers.RandomRotation(
            factor=0.08
        ),

        layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            factor=0.10
        ),

        layers.RandomBrightness(
            factor=0.08
        )
    ],
    name="train_augmentation"
)


# -----------------------------------------------------------------------------
# Load datasets
# -----------------------------------------------------------------------------

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(LOCAL_DATASET, "train"),
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(LOCAL_DATASET, "valid"),
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(LOCAL_DATASET, "test"),
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDatasets loaded.")


# =============================================================================
# CELL 8 - APPLY TRAINING AUGMENTATION
# =============================================================================

AUTOTUNE = tf.data.AUTOTUNE

def augment_batch(images, labels):
    images = train_augmentation(
        images,
        training=True
    )
    return images, labels


train_ds = train_ds_raw.map(
    augment_batch,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Augmentation pipeline ready.")


# =============================================================================
# CELL 9 - VERIFY CLASS LABELS
# =============================================================================

print("=" * 80)
print("CLASS MAPPING")
print("=" * 80)

print("0 =", CLASS_NAMES[0])
print("1 =", CLASS_NAMES[1])


# =============================================================================
# CELL 10 - BINARY FOCAL LOSS
# =============================================================================

def binary_focal_loss(
    gamma=1.0,
    positive_weight=1.5,
    negative_weight=1.0
):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        # Prevent log(0)
        epsilon = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        # Standard BCE
        bce = -(
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        # p_t
        p_t = (
            y_true * y_pred
            +
            (1.0 - y_true) * (1.0 - y_pred)
        )

        # Focal modulation
        focal_factor = tf.pow(
            1.0 - p_t,
            gamma
        )

        # Class-specific weighting
        class_weight = (
            y_true * positive_weight
            +
            (1.0 - y_true) * negative_weight
        )

        return tf.reduce_mean(
            focal_factor * class_weight * bce
        )

    return loss


loss_function = binary_focal_loss(
    gamma=FOCAL_GAMMA,
    positive_weight=MELANOMA_WEIGHT,
    negative_weight=NON_MELANOMA_WEIGHT
)

print("Focal loss configured.")
print("Gamma:", FOCAL_GAMMA)
print("Melanoma weight:", MELANOMA_WEIGHT)
print("Non-melanoma weight:", NON_MELANOMA_WEIGHT)


# =============================================================================
# CELL 11 - CHANNEL + SPATIAL ATTENTION MODULES
# =============================================================================
#
# V35 CONTROLLED CHANGE:
#     Add Spatial Attention after the existing V33 Channel Attention.
#
# V33:
#     EfficientNetV2-S -> Channel Attention -> GAP
#
# V35:
#     EfficientNetV2-S -> Channel Attention -> Spatial Attention -> GAP
#
# Channel Attention (SE):
#     Learns which feature channels are important.
#
# Spatial Attention:
#     Learns which spatial locations in the feature map are important.
#     It computes channel-wise average and maximum maps, concatenates them,
#     and applies a 7x7 convolution followed by sigmoid.
# =============================================================================

@keras.utils.register_keras_serializable()
class ChannelAttention(layers.Layer):

    def __init__(
        self,
        reduction_ratio=16,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.reduction_ratio = reduction_ratio

        self.global_pool = layers.GlobalAveragePooling2D(
            keepdims=False,
            name="channel_avg_pool"
        )

        self.reduce_dense = None
        self.expand_dense = None

    def build(self, input_shape):

        channels = int(input_shape[-1])

        reduced_channels = max(
            channels // self.reduction_ratio,
            1
        )

        self.reduce_dense = layers.Dense(
            reduced_channels,
            activation="relu",
            name="channel_reduce"
        )

        self.expand_dense = layers.Dense(
            channels,
            activation="sigmoid",
            name="channel_expand"
        )

        super().build(input_shape)

    def call(self, inputs):

        # Squeeze
        channel_descriptor = self.global_pool(inputs)

        # Excitation
        channel_attention = self.reduce_dense(
            channel_descriptor
        )

        channel_attention = self.expand_dense(
            channel_attention
        )

        # Reshape:
        # [B, C] -> [B, 1, 1, C]
        channel_attention = tf.expand_dims(
            channel_attention,
            axis=1
        )

        channel_attention = tf.expand_dims(
            channel_attention,
            axis=1
        )

        # Channel-wise recalibration
        return inputs * channel_attention

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "reduction_ratio": self.reduction_ratio
            }
        )

        return config


print("ChannelAttention layer defined.")


@keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):

    def __init__(
        self,
        kernel_size=7,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.kernel_size = kernel_size

        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="spatial_conv"
        )

    def call(self, inputs):

        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        spatial_descriptor = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        spatial_attention = self.spatial_conv(
            spatial_descriptor
        )

        return inputs * spatial_attention

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "kernel_size": self.kernel_size
            }
        )

        return config


print("SpatialAttention layer defined.")


# =============================================================================
# CELL 12 - BUILD V35 MODEL
# =============================================================================

print("=" * 80)
print("BUILDING V35 MODEL")
print("=" * 80)

# -----------------------------------------------------------------------------
# EfficientNetV2-S backbone
# Same as V33
# -----------------------------------------------------------------------------

base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

base_model.trainable = False


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="image"
)


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------

x = base_model(
    inputs,
    training=False
)


# -----------------------------------------------------------------------------
# V35 CONTROLLED CHANGE
# -----------------------------------------------------------------------------
#
# V33 already contains Channel Attention.
# V35 adds Spatial Attention immediately after it.
#
# V33:
#     backbone -> ChannelAttention -> GAP
#
# V35:
#     backbone -> ChannelAttention -> SpatialAttention -> GAP
# -----------------------------------------------------------------------------

x = ChannelAttention(
    reduction_ratio=SE_REDUCTION_RATIO,
    name="channel_attention"
)(x)

x = SpatialAttention(
    kernel_size=SPATIAL_KERNEL_SIZE,
    name="spatial_attention"
)(x)


# -----------------------------------------------------------------------------
# SAME CLASSIFIER HEAD AS V33
# -----------------------------------------------------------------------------

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.Dropout(
    0.45,
    name="dropout_1"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = layers.Dropout(
    0.35,
    name="dropout_2"
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="melanoma_probability"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs,
    name="V35_EfficientNetV2S_ChannelSpatialAttention"
)


# -----------------------------------------------------------------------------
# Compile Stage 1
# -----------------------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),
    loss=loss_function,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="auc"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)


print("\nModel created successfully.")

model.summary()


# =============================================================================
# CELL 13 - MODEL PARAMETER COUNTS
# =============================================================================

total_params = model.count_params()

trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.trainable_variables
])

non_trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.non_trainable_variables
])

print("=" * 80)
print("PARAMETER COUNTS")
print("=" * 80)

print("Total parameters     :", total_params)
print("Trainable parameters :", trainable_params)
print("Non-trainable        :", non_trainable_params)


# =============================================================================
# CELL 14 - CALLBACKS FOR STAGE 1
# =============================================================================

stage1_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "V35_stage1_best.keras"
)

print("Stage 1 best checkpoint:", stage1_checkpoint)

stage1_callbacks = [

    # Saves the BEST Stage 1 model directly to Google Drive.
    keras.callbacks.ModelCheckpoint(
        stage1_checkpoint,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Saves training state so an interrupted fit can be recovered.
    keras.callbacks.BackupAndRestore(
        backup_dir=STAGE1_BACKUP_DIR,
        delete_checkpoint=False
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# =============================================================================
# CELL 15 - STAGE 1 TRAINING
# =============================================================================

print("=" * 80)
print("V35 STAGE 1 TRAINING")
print("=" * 80)

print("Backbone: FROZEN")
print("Epochs:", STAGE1_EPOCHS)
print("Learning rate:", STAGE1_LR)

history_stage1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1
)


# =============================================================================
# CELL 16 - LOAD BEST STAGE 1 MODEL
# =============================================================================

print("=" * 80)
print("LOADING BEST STAGE 1 MODEL")
print("=" * 80)

# Recreate the exact V35 focal loss object
loss_function = binary_focal_loss(
    gamma=FOCAL_GAMMA,
    positive_weight=MELANOMA_WEIGHT,
    negative_weight=NON_MELANOMA_WEIGHT
)

model = keras.models.load_model(
    stage1_checkpoint,
    custom_objects={
        "loss": loss_function,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention
    },
    compile=False
)

# Recompile using the exact Stage 1 configuration
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),
    loss=loss_function,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="auc"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("Best Stage 1 model loaded successfully.")
print("Stage 1 model is ready for Stage 2 fine-tuning.")


# =============================================================================
# CELL 17 - STAGE 1 HISTORY
# =============================================================================

history1_df = pd.DataFrame(
    history_stage1.history
)

display(history1_df)

print("\nBest Stage 1 validation AUC:")

if "val_auc" in history1_df.columns:
    print(
        history1_df["val_auc"].max()
    )


# =============================================================================
# CELL 18 - PREPARE STAGE 2 FINE-TUNING
# =============================================================================

print("=" * 80)
print("V35 STAGE 2 FINE-TUNING PREPARATION")
print("=" * 80)

# Define redundantly here as protection against execution-order errors.
FINE_TUNE_FRACTION = 0.60
STAGE2_LR = 1e-5

base_model = model.get_layer(
    "efficientnetv2-s"
)

base_model.trainable = True

total_backbone_layers = len(
    base_model.layers
)

fine_tune_start = int(
    total_backbone_layers *
    (1.0 - FINE_TUNE_FRACTION)
)

print("Total backbone layers :", total_backbone_layers)
print("Fine-tune fraction    :", FINE_TUNE_FRACTION)
print("Fine-tune start index :", fine_tune_start)


# -----------------------------------------------------------------------------
# Freeze early backbone layers
# -----------------------------------------------------------------------------

for layer_index, layer in enumerate(
    base_model.layers
):

    if layer_index < fine_tune_start:

        layer.trainable = False

    else:

        # Keep BatchNorm frozen
        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

        else:
            layer.trainable = True


# -----------------------------------------------------------------------------
# Make sure Channel Attention and classifier are trainable
# -----------------------------------------------------------------------------

model.get_layer(
    "channel_attention"
).trainable = True

model.get_layer(
    "spatial_attention"
).trainable = True

model.get_layer(
    "global_average_pooling"
).trainable = True

model.get_layer(
    "dropout_1"
).trainable = True

model.get_layer(
    "dense_256"
).trainable = True

model.get_layer(
    "dropout_2"
).trainable = True

model.get_layer(
    "melanoma_probability"
).trainable = True


# -----------------------------------------------------------------------------
# Recompile with Stage 2 LR
# -----------------------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),
    loss=loss_function,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="auc"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)


trainable_params_stage2 = np.sum([
    np.prod(v.shape)
    for v in model.trainable_variables
])

print("\nStage 2 configuration:")
print("Fine-tune fraction    :", FINE_TUNE_FRACTION)
print("Stage 2 learning rate :", STAGE2_LR)
print("Trainable parameters :", trainable_params_stage2)


# =============================================================================
# CELL 19 - VERIFY TRAINABLE LAYERS
# =============================================================================

print("=" * 80)
print("TRAINABLE LAYER SUMMARY")
print("=" * 80)

trainable_count = 0
frozen_count = 0

for layer in model.layers:

    if layer.trainable:
        trainable_count += 1
    else:
        frozen_count += 1

print("Trainable layers:", trainable_count)
print("Frozen layers   :", frozen_count)

print("\nChannel Attention trainable:",
      model.get_layer(
          "channel_attention"
      ).trainable)

print("Spatial Attention trainable:",
      model.get_layer(
          "spatial_attention"
      ).trainable)


# =============================================================================
# CELL 20 - STAGE 2 CALLBACKS
# =============================================================================

stage2_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "V35_stage2_best.keras"
)

print("Stage 2 best checkpoint:", stage2_checkpoint)

stage2_callbacks = [

    # Saves the BEST Stage 2 model directly to Google Drive.
    keras.callbacks.ModelCheckpoint(
        stage2_checkpoint,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Saves optimizer/model training state for interruption recovery.
    keras.callbacks.BackupAndRestore(
        backup_dir=STAGE2_BACKUP_DIR,
        delete_checkpoint=False
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# =============================================================================
# CELL 21 - STAGE 2 TRAINING
# =============================================================================

print("=" * 80)
print("V35 STAGE 2 FINE-TUNING")
print("=" * 80)

print("Backbone fine-tuning :", FINE_TUNE_FRACTION)
print("Learning rate        :", STAGE2_LR)
print("Epochs               :", STAGE2_EPOCHS)
print("BatchNorm frozen     : YES")

history_stage2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=stage2_callbacks,
    verbose=1
)


# =============================================================================
# CELL 22 - LOAD BEST V35 MODEL
# =============================================================================

print("=" * 80)
print("LOADING BEST V35 MODEL")
print("=" * 80)

# Recreate focal loss
loss_function = binary_focal_loss(
    gamma=FOCAL_GAMMA,
    positive_weight=MELANOMA_WEIGHT,
    negative_weight=NON_MELANOMA_WEIGHT
)

model = keras.models.load_model(
    stage2_checkpoint,
    custom_objects={
        "loss": loss_function,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention
    },
    compile=False
)

# Recompile with Stage 2 configuration
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),
    loss=loss_function,
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.AUC(
            name="auc"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("Best V35 model loaded successfully.")


# =============================================================================
# CELL 23 - SAVE FINAL V35 MODEL
# =============================================================================

print("=" * 80)
print("SAVING V35 MODEL")
print("=" * 80)

model.save(
    MODEL_PATH
)

print("Model saved:")
print(MODEL_PATH)


# =============================================================================
# CELL 24 - VALIDATION PREDICTIONS
# =============================================================================

print("=" * 80)
print("VALIDATION PREDICTIONS")
print("=" * 80)

validation_probabilities = model.predict(
    valid_ds,
    verbose=1
).ravel()

validation_true = np.concatenate([
    y.numpy().ravel()
    for _, y in valid_ds
])

validation_true = validation_true.astype(int)

print("Validation samples:",
      len(validation_true))

print("Validation melanoma:",
      np.sum(validation_true == 1))

print("Validation non-melanoma:",
      np.sum(validation_true == 0))


# =============================================================================
# CELL 25 - TEST PREDICTIONS
# =============================================================================

print("=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)

test_probabilities = model.predict(
    test_ds,
    verbose=1
).ravel()

test_true = np.concatenate([
    y.numpy().ravel()
    for _, y in test_ds
])

test_true = test_true.astype(int)

print("Test samples:",
      len(test_true))

print("Test melanoma:",
      np.sum(test_true == 1))

print("Test non-melanoma:",
      np.sum(test_true == 0))


# =============================================================================
# CELL 26 - AUC RESULTS
# =============================================================================

validation_roc_auc = roc_auc_score(
    validation_true,
    validation_probabilities
)

validation_pr_auc = average_precision_score(
    validation_true,
    validation_probabilities
)

test_roc_auc = roc_auc_score(
    test_true,
    test_probabilities
)

test_pr_auc = average_precision_score(
    test_true,
    test_probabilities
)

print("=" * 80)
print("V35 AUC RESULTS")
print("=" * 80)

print(
    f"Validation ROC-AUC : {validation_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC  : {validation_pr_auc:.4f}"
)

print(
    f"Test ROC-AUC       : {test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC        : {test_pr_auc:.4f}"
)


# =============================================================================
# CELL 27 - METRIC FUNCTION
# =============================================================================

def calculate_metrics(
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

    balanced_accuracy = balanced_accuracy_score(
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
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# =============================================================================
# CELL 28 - VALIDATION THRESHOLD SELECTION
# =============================================================================
#
# IMPORTANT:
# Threshold is selected ONLY using validation data.
# Test data is NOT used to choose the deployment threshold.
# =============================================================================

thresholds = np.arange(
    0.01,
    1.0,
    0.005
)

validation_threshold_results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        validation_true,
        validation_probabilities,
        threshold
    )

    if (
        metrics["sensitivity"]
        >= MIN_REQUIRED_SENSITIVITY
    ):

        validation_threshold_results.append(
            metrics
        )


if len(validation_threshold_results) == 0:

    raise RuntimeError(
        "No validation threshold satisfies "
        "the minimum sensitivity requirement."
    )


validation_threshold_df = pd.DataFrame(
    validation_threshold_results
)

# Maximize F1 while satisfying sensitivity >= 70%
best_validation_row = (
    validation_threshold_df
    .sort_values(
        by=[
            "f1",
            "specificity"
        ],
        ascending=False
    )
    .iloc[0]
)

selected_threshold = float(
    best_validation_row["threshold"]
)

print("=" * 80)
print("V35 SELECTED THRESHOLD")
print("=" * 80)

print(
    f"Minimum required sensitivity: "
    f"{MIN_REQUIRED_SENSITIVITY:.2%}"
)

print(
    f"Selected threshold: "
    f"{selected_threshold:.3f}"
)

print(
    f"Validation F1: "
    f"{best_validation_row['f1']:.4f}"
)

print(
    f"Validation sensitivity: "
    f"{best_validation_row['sensitivity']:.4f}"
)

print(
    f"Validation specificity: "
    f"{best_validation_row['specificity']:.4f}"
)


# =============================================================================
# CELL 29 - V35 VALIDATION RESULTS
# =============================================================================

validation_selected_metrics = calculate_metrics(
    validation_true,
    validation_probabilities,
    selected_threshold
)

print("=" * 80)
print("V35 VALIDATION RESULTS")
print("=" * 80)

for key, value in validation_selected_metrics.items():

    if isinstance(value, float):

        print(
            f"{key:<22}: {value:.4f}"
        )

    else:

        print(
            f"{key:<22}: {value}"
        )


# =============================================================================
# CELL 30 - VALIDATION CLASSIFICATION REPORT
# =============================================================================

validation_predictions = (
    validation_probabilities
    >= selected_threshold
).astype(int)

print("=" * 80)
print("V35 VALIDATION CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        validation_true,
        validation_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =============================================================================
# CELL 31 - VALIDATION SENSITIVITY THRESHOLDS
# =============================================================================

print("=" * 80)
print("V35 VALIDATION SENSITIVITY THRESHOLDS")
print("=" * 80)

target_sensitivities = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

sensitivity_threshold_rows = []

for target_sensitivity in target_sensitivities:

    candidates = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            validation_true,
            validation_probabilities,
            threshold
        )

        if (
            metrics["sensitivity"]
            >= target_sensitivity
        ):

            candidates.append(metrics)

    if len(candidates) > 0:

        # Highest threshold satisfying target
        candidates = sorted(
            candidates,
            key=lambda x: x["threshold"],
            reverse=True
        )

        best = candidates[0]

        sensitivity_threshold_rows.append(
            {
                "target_sensitivity":
                    target_sensitivity,

                "threshold":
                    best["threshold"],

                "sensitivity":
                    best["sensitivity"],

                "specificity":
                    best["specificity"],

                "precision":
                    best["precision"],

                "f1":
                    best["f1"],

                "balanced_accuracy":
                    best["balanced_accuracy"]
            }
        )


validation_sensitivity_df = pd.DataFrame(
    sensitivity_threshold_rows
)

display(
    validation_sensitivity_df
)


# =============================================================================
# CELL 32 - FINAL TEST EVALUATION
# =============================================================================
#
# Uses the threshold selected from VALIDATION.
# =============================================================================

test_selected_metrics = calculate_metrics(
    test_true,
    test_probabilities,
    selected_threshold
)

print("=" * 80)
print("V35 FINAL TEST EVALUATION")
print("=" * 80)

for key, value in test_selected_metrics.items():

    if isinstance(value, float):

        print(
            f"{key:<22}: {value:.4f}"
        )

    else:

        print(
            f"{key:<22}: {value}"
        )


# =============================================================================
# CELL 33 - TEST CLASSIFICATION REPORT
# =============================================================================

test_predictions = (
    test_probabilities
    >= selected_threshold
).astype(int)

print("=" * 80)
print("V35 TEST CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        test_true,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =============================================================================
# CELL 34 - TEST CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    test_true,
    test_predictions,
    labels=[0, 1]
)

print("=" * 80)
print("V35 TEST CONFUSION MATRIX")
print("=" * 80)

print(cm)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("V35 Test Confusion Matrix")

plt.show()


# =============================================================================
# CELL 35 - TEST THRESHOLD COMPARISON
# =============================================================================

print("=" * 80)
print("V35 TEST THRESHOLD COMPARISON")
print("=" * 80)

test_threshold_rows = []

# Selected threshold
selected_test_metrics = calculate_metrics(
    test_true,
    test_probabilities,
    selected_threshold
)

test_threshold_rows.append(
    {
        "setting": "selected",
        **selected_test_metrics
    }
)


# Default 0.5
default_metrics = calculate_metrics(
    test_true,
    test_probabilities,
    0.5
)

test_threshold_rows.append(
    {
        "setting": "default_0.500",
        **default_metrics
    }
)


# Diagnostic best F1 on test
#
# IMPORTANT:
# This is NOT used for model selection.
# It is only diagnostic.
best_test_f1_row = None

for threshold in thresholds:

    metrics = calculate_metrics(
        test_true,
        test_probabilities,
        threshold
    )

    if (
        best_test_f1_row is None
        or metrics["f1"]
        > best_test_f1_row["f1"]
    ):

        best_test_f1_row = metrics

test_threshold_rows.append(
    {
        "setting": "best_f1",
        **best_test_f1_row
    }
)


# Diagnostic best balanced accuracy on test
best_test_balanced_row = None

for threshold in thresholds:

    metrics = calculate_metrics(
        test_true,
        test_probabilities,
        threshold
    )

    if (
        best_test_balanced_row is None
        or metrics["balanced_accuracy"]
        > best_test_balanced_row[
            "balanced_accuracy"
        ]
    ):

        best_test_balanced_row = metrics

test_threshold_rows.append(
    {
        "setting": "best_balanced",
        **best_test_balanced_row
    }
)


# Validation-selected sensitivity thresholds
for _, row in validation_sensitivity_df.iterrows():

    target = row["target_sensitivity"]

    threshold = row["threshold"]

    metrics = calculate_metrics(
        test_true,
        test_probabilities,
        threshold
    )

    test_threshold_rows.append(
        {
            "setting":
                f"sensitivity_{target:.2f}",

            **metrics
        }
    )


test_threshold_df = pd.DataFrame(
    test_threshold_rows
)

display(
    test_threshold_df
)


# =============================================================================
# CELL 36 - V35 PROBABILITY ANALYSIS
# =============================================================================

melanoma_probabilities = test_probabilities[
    test_true == 1
]

non_melanoma_probabilities = test_probabilities[
    test_true == 0
]

print("=" * 80)
print("V35 TEST PROBABILITY ANALYSIS")
print("=" * 80)

print("\nMelanoma probabilities")

print(
    "Mean   :",
    np.mean(melanoma_probabilities)
)

print(
    "Median :",
    np.median(melanoma_probabilities)
)

print(
    "Min    :",
    np.min(melanoma_probabilities)
)

print(
    "Max    :",
    np.max(melanoma_probabilities)
)


print("\nNon-melanoma probabilities")

print(
    "Mean   :",
    np.mean(non_melanoma_probabilities)
)

print(
    "Median :",
    np.median(non_melanoma_probabilities)
)

print(
    "Min    :",
    np.min(non_melanoma_probabilities)
)

print(
    "Max    :",
    np.max(non_melanoma_probabilities)
)


# =============================================================================
# CELL 37 - PROBABILITY DISTRIBUTION
# =============================================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    non_melanoma_probabilities,
    bins=40,
    alpha=0.6,
    label="Non-melanoma"
)

plt.hist(
    melanoma_probabilities,
    bins=40,
    alpha=0.6,
    label="Melanoma"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected threshold = {selected_threshold:.3f}"
)

plt.xlabel("Predicted melanoma probability")
plt.ylabel("Number of images")
plt.title("V35 Test Probability Distribution")
plt.legend()

plt.show()


# =============================================================================
# CELL 38 - ROC CURVE
# =============================================================================

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(
    test_true,
    test_probabilities
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"V35 ROC-AUC = {test_roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("V35 ROC Curve")
plt.legend()

plt.show()


# =============================================================================
# CELL 39 - PRECISION-RECALL CURVE
# =============================================================================

from sklearn.metrics import precision_recall_curve

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        test_true,
        test_probabilities
    )
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"V35 PR-AUC = {test_pr_auc:.4f}"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("V35 Precision-Recall Curve")
plt.legend()

plt.show()


# =============================================================================
# CELL 40 - ACCURACY TARGET ANALYSIS
# =============================================================================

TEST_SAMPLES = len(test_true)

CORRECT = int(
    np.sum(
        test_predictions == test_true
    )
)

ERRORS = TEST_SAMPLES - CORRECT

TARGET_ACCURACY = 0.96

CORRECT_REQUIRED = int(
    np.ceil(
        TEST_SAMPLES * TARGET_ACCURACY
    )
)

MAX_ERRORS_ALLOWED = (
    TEST_SAMPLES - CORRECT_REQUIRED
)

actual_accuracy = (
    CORRECT / TEST_SAMPLES
)

print("=" * 80)
print("V35 ACCURACY ANALYSIS")
print("=" * 80)

print(
    "Test samples:",
    TEST_SAMPLES
)

print(
    "Correct:",
    CORRECT
)

print(
    "Errors:",
    ERRORS
)

print(
    "Correct required for 96%:",
    CORRECT_REQUIRED
)

print(
    "Maximum errors allowed for 96%:",
    MAX_ERRORS_ALLOWED
)

print(
    f"Actual accuracy: "
    f"{actual_accuracy:.4f}"
)


# =============================================================================
# CELL 41 - SAVE V35 RESULTS
# =============================================================================

print("=" * 80)
print("SAVING V35 RESULTS")
print("=" * 80)


def make_json_serializable(obj):

    if isinstance(
        obj,
        dict
    ):

        return {
            str(k):
                make_json_serializable(v)

            for k, v in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            make_json_serializable(v)
            for v in obj
        ]

    if isinstance(
        obj,
        np.integer
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):

        return float(obj)

    if isinstance(
        obj,
        np.ndarray
    ):

        return obj.tolist()

    if pd.isna(obj):

        return None

    return obj


V35_results = {

    "version": VERSION,

    "controlled_change": {
        "description":
            "Added Spatial Attention after the existing Channel Attention",

        "baseline":
            "V33",

        "V33_stage2_learning_rate":
            1e-5,

        "V35_stage2_learning_rate":
            1e-5,

        "attention":
            "Squeeze-and-Excitation Channel Attention + Spatial Attention",

        "reduction_ratio":
            SE_REDUCTION_RATIO,

        "spatial_kernel_size":
            SPATIAL_KERNEL_SIZE
    },

    "configuration": {

        "image_size":
            IMG_SIZE,

        "batch_size":
            BATCH_SIZE,

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

        "melanoma_weight":
            MELANOMA_WEIGHT,

        "non_melanoma_weight":
            NON_MELANOMA_WEIGHT,

        "focal_gamma":
            FOCAL_GAMMA,

        "minimum_required_sensitivity":
            MIN_REQUIRED_SENSITIVITY
    },

    "validation": {

        "roc_auc":
            validation_roc_auc,

        "pr_auc":
            validation_pr_auc,

        "selected_threshold":
            selected_threshold,

        "metrics":
            validation_selected_metrics
    },

    "test": {

        "roc_auc":
            test_roc_auc,

        "pr_auc":
            test_pr_auc,

        "metrics":
            test_selected_metrics
    },

    "test_probability_analysis": {

        "melanoma_mean":
            float(
                np.mean(
                    melanoma_probabilities
                )
            ),

        "melanoma_median":
            float(
                np.median(
                    melanoma_probabilities
                )
            ),

        "melanoma_min":
            float(
                np.min(
                    melanoma_probabilities
                )
            ),

        "melanoma_max":
            float(
                np.max(
                    melanoma_probabilities
                )
            ),

        "non_melanoma_mean":
            float(
                np.mean(
                    non_melanoma_probabilities
                )
            ),

        "non_melanoma_median":
            float(
                np.median(
                    non_melanoma_probabilities
                )
            ),

        "non_melanoma_min":
            float(
                np.min(
                    non_melanoma_probabilities
                )
            ),

        "non_melanoma_max":
            float(
                np.max(
                    non_melanoma_probabilities
                )
            )
    },

    "accuracy_analysis": {

        "test_samples":
            TEST_SAMPLES,

        "correct":
            CORRECT,

        "errors":
            ERRORS,

        "correct_required_for_96_percent":
            CORRECT_REQUIRED,

        "maximum_errors_allowed_for_96_percent":
            MAX_ERRORS_ALLOWED
    }
}


V35_results = make_json_serializable(
    V35_results
)

with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        V35_results,
        f,
        indent=4
    )

print(
    "Results saved:",
    RESULTS_PATH
)


# =============================================================================
# CELL 42 - FINAL V35 SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("V35 FINAL SUMMARY")
print("=" * 80)

print("Controlled change:")
print(
    "V33 -> V35: Added Spatial Attention after existing Channel Attention"
)

print(
    "Attention reduction ratio:",
    SE_REDUCTION_RATIO
)

print(
    "Selected threshold       :",
    f"{selected_threshold:.3f}"
)

print(
    "Test ROC-AUC             :",
    f"{test_roc_auc:.4f}"
)

print(
    "Test PR-AUC              :",
    f"{test_pr_auc:.4f}"
)

print(
    "Test Accuracy            :",
    f"{test_selected_metrics['accuracy']:.4f}"
)

print(
    "Test Precision           :",
    f"{test_selected_metrics['precision']:.4f}"
)

print(
    "Test Sensitivity         :",
    f"{test_selected_metrics['sensitivity']:.4f}"
)

print(
    "Test Specificity         :",
    f"{test_selected_metrics['specificity']:.4f}"
)

print(
    "Test F1                  :",
    f"{test_selected_metrics['f1']:.4f}"
)

print(
    "Test Balanced Accuracy   :",
    f"{test_selected_metrics['balanced_accuracy']:.4f}"
)

print("\nConfusion Matrix:")

print(
    np.array([
        [
            test_selected_metrics["tn"],
            test_selected_metrics["fp"]
        ],
        [
            test_selected_metrics["fn"],
            test_selected_metrics["tp"]
        ]
    ])
)

print("=" * 80)
print("V35 COMPLETE")
print("=" * 80)