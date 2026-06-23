from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE            = "/content/newdata"
IMG_SRC         = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR  = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_dual_branch_run12.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Precision & hyper-parameters ───────────────────────────────────────────────
mixed_precision.set_global_policy("float32")

BATCH_SIZE       = 8               # <-- CHANGED: reduced from 16 because 456×456
                                   # images are 3x larger in memory
IMAGE_SIZE       = 456             # <-- CHANGED: matches paper's EfficientNet-B5
                                   # input size, captures finer dermoscopic detail
FUSION_LAYER     = "block4c_add"
EPOCHS           = 30             # matches the paper
STEPS_PER_EPOCH  = (7122 + 890) // BATCH_SIZE

CLASS_ORDER      = ["melanoma", "non_melanoma"]


# ── Augmentation ───────────────────────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ── Edge map helper ────────────────────────────────────────────────────────────
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    gray  = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge  = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge  = edge / (tf.reduce_max(edge) + 1e-6)
    return (image, edge), label


# ── Balanced training dataset (50/50 oversampling) ────────────────────────────
def make_class_stream(base_path, only_class):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE // 2,
        label_mode="categorical",
        shuffle=True,
        class_names=CLASS_ORDER,
    )
    ds = ds.filter(lambda x, y: tf.equal(tf.argmax(y[0]), only_class))
    ds = ds.repeat()
    ds = ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


melanoma_stream     = make_class_stream(f"{BASE}/train", only_class=0)
non_melanoma_stream = make_class_stream(f"{BASE}/train", only_class=1)

balanced_ds = tf.data.Dataset.sample_from_datasets(
    [melanoma_stream, non_melanoma_stream],
    weights=[0.5, 0.5],
)
balanced_ds = balanced_ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)


# ── Validation & test datasets ─────────────────────────────────────────────────
def prepare_eval_dataset(path):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=CLASS_ORDER,
    )
    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


val_ds  = prepare_eval_dataset(f"{BASE}/valid")
test_ds = prepare_eval_dataset(f"{BASE}/test")


# ── Dual-branch model ──────────────────────────────────────────────────────────
def create_dual_model(steps_per_epoch):
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
    fused   = layers.Dropout(0.5)(fused)
    outputs = layers.Dense(2, activation="softmax")(fused)

    cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-4,
        decay_steps=EPOCHS * steps_per_epoch,
    )

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cosine_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="sensitivity"),
            tf.keras.metrics.Recall(name="specificity", class_id=1),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


model = create_dual_model(steps_per_epoch=STEPS_PER_EPOCH)
model.summary()


# ── Callbacks ──────────────────────────────────────────────────────────────────
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual_{FUSION_LAYER}_run12.keras",
    monitor="val_auc",
    save_best_only=True,
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_auc",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)


# ── Training ───────────────────────────────────────────────────────────────────
history = model.fit(
    balanced_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    callbacks=[checkpoint_best, early_stopping],
)


# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n── Test results (run 12: 456×456 image size) ──")
results = model.evaluate(test_ds)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

model.save(MODEL_SAVE_PATH)