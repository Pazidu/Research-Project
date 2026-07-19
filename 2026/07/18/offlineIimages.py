# ==========================================================
# PART 1 - IMPORTS, SETTINGS, DATASET
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
    classification_report
)

# ==========================================================
# GPU MEMORY GROWTH
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================================================
# MIXED PRECISION
# ==========================================================

mixed_precision.set_global_policy("mixed_float16")

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==========================================================
# PATHS
# ==========================================================

DATASET = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata_backup"
CHECKPOINT = "/drive/MyDrive/checkpoints/best_model.keras"
MODEL_SAVE = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/final_model.keras"

# ==========================================================
# COPY DATASET TO COLAB
# ==========================================================

if os.path.exists(DATASET):
    shutil.rmtree(DATASET)

# copy dataset from Drive

shutil.copytree(
    IMG_SRC,
    DATASET
)  

# ==========================================================
# SETTINGS
# ==========================================================

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 10

AUTOTUNE = 2

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ==========================================================
# EDGE MAP
# ==========================================================

def add_edge(image, label):
    image = tf.cast(image, tf.float32)
    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge = tf.sqrt(
        tf.reduce_sum(
            tf.square(sobel),
            axis=-1
        )
    )
    edge = tf.clip_by_value(edge, 0.0, 1.0)
    return (image, edge), label

# ==========================================================
# DATASET LOADER
# ==========================================================

def load_dataset(path, shuffle):

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )

    ds = ds.map(
        add_edge,
        num_parallel_calls=AUTOTUNE
    )

    return ds.prefetch(1)


# ==========================================================
# LOAD DATASETS
# ==========================================================

train_ds = load_dataset(
    DATASET + "/train",
    True
)

val_ds = load_dataset(
    DATASET + "/valid",
    False
)

test_ds = load_dataset(
    DATASET + "/test",
    False
)

print(train_ds.class_names)
print("\nDatasets Loaded Successfully")

# ==========================================================
# LOSS FUNCTION
# ==========================================================

loss_fn = tf.keras.losses.CategoricalCrossentropy()

# =========================================================
# MODEL CREATION
# =========================================================

def create_model():

    rgb_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3),name="rgb")
    edge_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1),name="edge")

    # -------------------------
    # RGB BRANCH
    # -------------------------

    x = augmentation(rgb_input)
    x = preprocess_input(x)
    backbone = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Stage 1:
    # Freeze backbone to save memory

    backbone.trainable = False
    features = backbone(x)

    # -------------------------
    # EDGE BRANCH
    # -------------------------

    e = layers.Conv2D(32,3,padding="same",activation="relu")(edge_input)
    e = layers.BatchNormalization()(e)
    e = layers.MaxPooling2D()(e)
    e = layers.Conv2D(64,3,padding="same",activation="relu")(e)
    e = layers.BatchNormalization()(e)
    e = layers.MaxPooling2D()(e)
    e = layers.Conv2D(128,3,padding="same",activation="relu")(e)

    # Resize edge features to EfficientNet output

    e = layers.Resizing(features.shape[1],features.shape[2])(e)
    e = layers.Conv2D(features.shape[-1],1,padding="same")(e)

    # -------------------------
    # FUSION
    # -------------------------

    fused = layers.Concatenate()([features,e])
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dense(256,activation="relu")(fused)
    fused = layers.Dropout(0.5)(fused)
    output = layers.Dense(2,activation="softmax",dtype="float32")(fused)
    model = Model(
        inputs=[
            rgb_input,
            edge_input
        ],
        outputs=output
    )


    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn,

        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                name="auc"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )
    return model,backbone

model, backbone = create_model()
model.summary()

# =========================================================
# CALLBACKS
# =========================================================

checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT,
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=6,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stop,
    lr_reduce
]


# =========================================================
# STAGE 1 TRAINING
# Frozen EfficientNet
# =========================================================

print("\n==============================")
print("STAGE 1 TRAINING")
print("==============================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# =========================================================
# STAGE 2 FINE TUNING
# Unfreeze only last layers
# =========================================================

print("\n==============================")
print("STAGE 2 FINE TUNING")
print("==============================")

# Find EfficientNet backbone

backbone.trainable = True

# keep most pretrained layers frozen
for layer in backbone.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\nLoading best checkpoint...")
model.load_weights(CHECKPOINT)

# =========================================================
# TEST RESULTS
# =========================================================

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

results = model.evaluate(test_ds)
print("Loss, Accuracy, AUC, Precision, Recall:")
print(results)

# =========================================================
# CONFUSION MATRIX
# =========================================================

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images,verbose=0)
    y_true.extend(np.argmax(labels.numpy(),axis=1))
    y_pred.extend(np.argmax(predictions,axis=1))

cm = confusion_matrix(y_true,y_pred)
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(classification_report(
        y_true,
        y_pred,
        target_names=[
            "melanoma",
            "non_melanoma"
        ]
    )
)

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(MODEL_PATH)
print("\nMODEL SAVED SUCCESSFULLY")