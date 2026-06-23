from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
from PIL import Image
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
BASE             = "/content/newdata"
BASE_BALANCED    = "/content/newdata_balanced"     # physically balanced copy
IMG_SRC          = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR   = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH  = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_dual_branch_attn.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Precision & hyper-parameters ───────────────────────────────────────────────
mixed_precision.set_global_policy("float32")

BATCH_SIZE       = 16
IMAGE_SIZE       = 256
FUSION_LAYER     = "block4c_add"
EPOCHS           = 30
CLASS_ORDER      = ["melanoma", "non_melanoma"]


# ── Step 1: Physical oversampling (paper-style) ───────────────────────────────
# The paper generates 7 augmented copies per melanoma image to reach near 1:1
# balance (1113 -> 8904). We do the same: bring melanoma up to match non_melanoma.

light_augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

def build_balanced_train_set(src_train_dir, dst_train_dir, image_size):
    """
    Copies non_melanoma as-is, and generates augmented melanoma copies
    until melanoma count matches non_melanoma count.
    """
    if os.path.exists(dst_train_dir):
        shutil.rmtree(dst_train_dir)
    os.makedirs(dst_train_dir)

    counts = {}
    for cls in CLASS_ORDER:
        src_cls_dir = f"{src_train_dir}/{cls}"
        dst_cls_dir = f"{dst_train_dir}/{cls}"
        os.makedirs(dst_cls_dir, exist_ok=True)
        files = os.listdir(src_cls_dir)
        counts[cls] = len(files)
        for f in files:
            shutil.copy(f"{src_cls_dir}/{f}", f"{dst_cls_dir}/{f}")

    target = max(counts.values())
    print(f"Class counts before balancing: {counts}")
    print(f"Target count per class: {target}")

    for cls in CLASS_ORDER:
        n_existing = counts[cls]
        n_needed = target - n_existing
        if n_needed <= 0:
            continue
        src_cls_dir = f"{src_train_dir}/{cls}"
        dst_cls_dir = f"{dst_train_dir}/{cls}"
        files = os.listdir(src_cls_dir)

        for i in range(n_needed):
            src_file = files[i % len(files)]
            img = Image.open(f"{src_cls_dir}/{src_file}").convert("RGB")
            img = img.resize((image_size, image_size))
            img_arr = np.expand_dims(np.array(img), 0).astype("float32")
            aug = light_augmenter(img_arr, training=True)[0].numpy().astype("uint8")
            Image.fromarray(aug).save(f"{dst_cls_dir}/aug_{i}_{src_file}")

        print(f"  {cls}: generated {n_needed} augmented copies "
              f"({n_existing} -> {target})")


build_balanced_train_set(f"{BASE}/train", f"{BASE_BALANCED}/train", IMAGE_SIZE)

# valid/test stay untouched — only training data is rebalanced
os.makedirs(f"{BASE_BALANCED}/valid", exist_ok=True)
os.makedirs(f"{BASE_BALANCED}/test", exist_ok=True)
if os.path.exists(f"{BASE_BALANCED}/valid"):
    shutil.rmtree(f"{BASE_BALANCED}/valid")
if os.path.exists(f"{BASE_BALANCED}/test"):
    shutil.rmtree(f"{BASE_BALANCED}/test")
shutil.copytree(f"{BASE}/valid", f"{BASE_BALANCED}/valid")
shutil.copytree(f"{BASE}/test",  f"{BASE_BALANCED}/test")


# ── Light augmentation for tf.data pipeline (in addition to physical copies) ──
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


# ── Datasets (now balanced at the file level, simple loading) ────────────────
def prepare_dataset(path, shuffle, training=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        class_names=CLASS_ORDER,
    )
    if training:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = prepare_dataset(f"{BASE_BALANCED}/train", shuffle=True,  training=True)
val_ds   = prepare_dataset(f"{BASE_BALANCED}/valid", shuffle=False, training=False)
test_ds  = prepare_dataset(f"{BASE_BALANCED}/test",  shuffle=False, training=False)


# ── Channel attention module (from the reference paper) ───────────────────────
def channel_attention_module(x, reduction_ratio=24, name="channel_attention"):
    """
    Squeeze-and-excitation style channel attention, as used in the paper.
    GAP -> FC(C/r) -> FC(C) -> Sigmoid -> multiply back into input feature map.
    """
    channels = x.shape[-1]
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gap = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(gap)
    fc1 = layers.Conv2D(max(channels // reduction_ratio, 1), 1,
                         activation="relu", name=f"{name}_fc1")(gap)
    fc2 = layers.Conv2D(channels, 1, activation="sigmoid",
                         name=f"{name}_fc2")(fc1)
    return layers.Multiply(name=f"{name}_scale")([x, fc2])


# ── Dual-branch model with channel attention ───────────────────────────────────
def create_dual_model(steps_per_epoch):
    # --- RGB branch ---
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

    # --- Feature fusion + channel attention (NEW) ---
    fused = layers.Concatenate()([middle_feature, x])
    fused = layers.Conv2D(256, 3, activation="relu", padding="same")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = channel_attention_module(fused, reduction_ratio=24)   # <-- ADDED
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
            tf.keras.metrics.Recall(name="sensitivity"),                # TP/(TP+FN) for melanoma (class 0)
            tf.keras.metrics.Recall(name="specificity", class_id=1),    # TN/(TN+FP), since class 1 = non_melanoma
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


model = create_dual_model(steps_per_epoch=len(train_ds))
model.summary()


# ── Callbacks ──────────────────────────────────────────────────────────────────
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual_{FUSION_LAYER}_attn.keras",
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
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_best, early_stopping],
)


# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n── Test results (with channel attention) ──")
results = model.evaluate(test_ds)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

model.save(MODEL_SAVE_PATH)