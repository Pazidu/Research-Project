from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE            = "/content/newdata"
IMG_SRC         = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR  = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_dual_branch.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Precision & hyper-parameters ───────────────────────────────────────────────
mixed_precision.set_global_policy("float32")

BATCH_SIZE   = 16
IMAGE_SIZE   = 256
FUSION_LAYER = "block4c_add"
EPOCHS       = 30


# ── Class weights (fixes 8:1 imbalance) ───────────────────────────────────────
# Count samples per class from the training folder
train_classes = []
class_names = sorted(os.listdir(f"{BASE}/train"))   # alphabetical: melanoma=0, non_melanoma=1
for idx, cls in enumerate(class_names):
    n = len(os.listdir(f"{BASE}/train/{cls}"))
    train_classes.extend([idx] * n)
    print(f"  {cls}: {n} images (class {idx})")

train_labels = np.array(train_classes)
cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weight = dict(enumerate(cw))
print(f"\nClass weights: {class_weight}")
# Expected: melanoma ~4.0, non_melanoma ~0.5


# ── Augmentation (lighter — flip + rotation + zoom only) ──────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ── Dataset helpers ────────────────────────────────────────────────────────────
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    gray  = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge  = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge  = edge / (tf.reduce_max(edge) + 1e-6)
    return (image, edge), label


def prepare_dataset(path, shuffle, training=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
    )
    if training:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = prepare_dataset(f"{BASE}/train", shuffle=True,  training=True)
val_ds   = prepare_dataset(f"{BASE}/valid", shuffle=False, training=False)
test_ds  = prepare_dataset(f"{BASE}/test",  shuffle=False, training=False)


# ── Dual-branch model ──────────────────────────────────────────────────────────
def create_dual_model(num_train_batches):
    rgb_input  = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="rgb_input")
    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(FUSION_LAYER).output,
    )
    middle_feature = feature_extractor(rgb_input)

    # --- Edge branch ---
    edge_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="edge_input")
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(edge_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Resizing(middle_feature.shape[1], middle_feature.shape[2])(x)
    x = layers.Conv2D(middle_feature.shape[-1], 1, padding="same")(x)

    # --- Feature fusion ---
    fused = layers.Concatenate()([middle_feature, x])
    fused = layers.Conv2D(256, 3, activation="relu", padding="same")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(fused)
    fused = layers.Dropout(0.5)(fused)
    outputs = layers.Dense(2, activation="softmax")(fused)

    cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-4,
        decay_steps=EPOCHS * num_train_batches,
    )

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cosine_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),                        # <-- ADDED
            tf.keras.metrics.Recall(name="recall"),                   # <-- ADDED: catches melanoma
            tf.keras.metrics.Precision(name="precision"),             # <-- ADDED
        ],
    )
    return model


model = create_dual_model(num_train_batches=len(train_ds))
model.summary()


# ── Callbacks ──────────────────────────────────────────────────────────────────
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual_{FUSION_LAYER}.keras",
    monitor="val_auc",             # <-- CHANGED: AUC is a better monitor for imbalanced data
    save_best_only=True,
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_auc",             # <-- CHANGED: match checkpoint monitor
    patience=10,
    restore_best_weights=True,
    verbose=1,
)


# ── Training ───────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight,     # <-- KEY ADDITION: penalises missing melanoma 4x more
    callbacks=[checkpoint_best, early_stopping],
)


# ── Evaluation & save ──────────────────────────────────────────────────────────
results = model.evaluate(test_ds)
print("\n── Test results ──")
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

model.save(MODEL_SAVE_PATH)


# Best val AUC
# 85.43%
# epoch 21

# Best val accuracy
# 74.93%
# epoch 21

# Train AUC (ep 21)
# 96.30%
# gap: ~10.9%

# Val loss trend
# ↑ rising
# after ep 21