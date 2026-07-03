from google.colab import drive
drive.mount('/drive')

import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =========================================================
# PATHS
# =========================================================
BASE            = "/content/newdata"
IMG_SRC         = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR  = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = (
    "/drive/MyDrive/Colab Notebooks/Models/"
    "dermoscopy/efficientnetv2s_dual_branch_v2.keras"
)

if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
BATCH_SIZE    = 16
IMAGE_SIZE    = 256
FUSION_LAYER  = "block4c_add"
EPOCHS        = 40
WARMUP_EPOCHS = 3
LR_MAX        = 1e-4
LR_MIN        = 1e-6

# =========================================================
# AUGMENTATION
# =========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.15),
], name="augmentation")

# =========================================================
# MIXUP
# =========================================================
def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], minval=0.0, maxval=1.0)
    lam = tf.maximum(lam, 1.0 - lam)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

# =========================================================
# DATASET
# =========================================================
def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    gray  = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge  = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge  = edge / (tf.reduce_max(edge) + 1e-6)
    rgb   = preprocess_input(image)
    return (rgb, edge), label

def prepare_dataset(path, shuffle, apply_mixup=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
    )
    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    if apply_mixup:
        def mixup_map(inputs, labels):
            rgb, edge = inputs
            mixed_rgb, mixed_labels = mixup_batch(rgb, labels, alpha=0.2)
            return (mixed_rgb, edge), mixed_labels
        ds = ds.map(mixup_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = prepare_dataset(f"{BASE}/train", shuffle=True,  apply_mixup=True)
val_ds   = prepare_dataset(f"{BASE}/valid", shuffle=False, apply_mixup=False)
test_ds  = prepare_dataset(f"{BASE}/test",  shuffle=False, apply_mixup=False)

# =========================================================
# CLASS WEIGHTS
# =========================================================
def get_class_weights(train_path):
    class_dirs = sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ])
    counts  = [len(os.listdir(os.path.join(train_path, d))) for d in class_dirs]
    total   = sum(counts)
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("Class dirs   :", class_dirs)
    print("Class counts :", counts)
    print("Class weights:", weights)
    return weights

class_weights = get_class_weights(f"{BASE}/train")

# =========================================================
# LOSS
# =========================================================
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=0.25,
    label_smoothing=0.05,
)

# =========================================================
# LR SCHEDULE  (warmup → cosine decay, NO ReduceLROnPlateau)
#
# FIX: ReduceLROnPlateau is incompatible with a LearningRateSchedule
# optimizer — it tries to set .learning_rate on the optimizer which
# raises a TypeError.  We handle everything inside the schedule itself.
#
# Pacing fix: previous version decayed too fast → oscillating val_auc.
# Now warmup runs for WARMUP_EPOCHS, then cosine decays over the
# *remaining* epochs so the LR stays usefully high in the mid-run.
# =========================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, lr_max, lr_min):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        # --- warmup phase ---
        warmup_lr = self.lr_max * (step / self.warmup_steps)
        # --- cosine decay phase ---
        decay_steps = self.total_steps - self.warmup_steps
        cos_step    = tf.minimum(step - self.warmup_steps, decay_steps)
        cos_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + tf.cos(np.pi * cos_step / decay_steps)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return dict(
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
        )

steps_per_epoch = len(train_ds)
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = steps_per_epoch * WARMUP_EPOCHS

lr_schedule = WarmupCosineDecay(
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    lr_max=LR_MAX,
    lr_min=LR_MIN,
)

# =========================================================
# MODEL
# =========================================================
def create_dual_model():
    rgb_input  = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="rgb_input")
    edge_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="edge_input")

    # ── RGB BRANCH ──────────────────────────────────────────
    x_rgb = data_augmentation(rgb_input)
    x_rgb = preprocess_input(x_rgb)

    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    for layer in base_model.layers[:-160]:
        layer.trainable = False
    for layer in base_model.layers[-160:]:
        layer.trainable = True

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(FUSION_LAYER).output,
        name="rgb_backbone",
    )
    middle_feature = feature_extractor(x_rgb)   # (B, H', W', C')

    # ── EDGE BRANCH ─────────────────────────────────────────
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(edge_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Resizing(middle_feature.shape[1], middle_feature.shape[2])(x)
    x = layers.Conv2D(middle_feature.shape[-1], 1, padding="same")(x)

    # ── FUSION ──────────────────────────────────────────────
    fused = layers.Concatenate()([middle_feature, x])
    fused = layers.Conv2D(256, 3, activation="relu", padding="same",
                          kernel_regularizer=l2(1e-5))(fused)
    fused = layers.BatchNormalization()(fused)

    # ── CBAM-STYLE CHANNEL ATTENTION ────────────────────────
    gap = layers.GlobalAveragePooling2D()(fused)
    gmp = layers.GlobalMaxPooling2D()(fused)

    att_gap = layers.Dense(64,  activation="relu")(gap)
    att_gap = layers.Dense(256, activation="sigmoid")(att_gap)

    att_gmp = layers.Dense(64,  activation="relu")(gmp)
    att_gmp = layers.Dense(256, activation="sigmoid")(att_gmp)

    att = layers.Add()([att_gap, att_gmp])

    fused_gap      = layers.GlobalAveragePooling2D()(fused)
    fused_weighted = layers.Multiply()([fused_gap, att])

    # ── CLASSIFIER ──────────────────────────────────────────
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-5))(fused_weighted)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=[rgb_input, edge_input],
        outputs=outputs,
        name="dual_branch_v2",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

model = create_dual_model()
model.summary()

# =========================================================
# CALLBACKS  (no ReduceLROnPlateau — incompatible with schedule)
# =========================================================
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual_{FUSION_LAYER}_v2.keras",
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=12,          # generous: cosine LR causes occasional dips
    restore_best_weights=True,
    mode="max",
    verbose=1,
)

# =========================================================
# TRAINING
# =========================================================
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[checkpoint_best, early_stop],
)

# =========================================================
# TEST-TIME AUGMENTATION (TTA)
# =========================================================
def tta_predict(model, dataset, n_aug=8):
    all_preds  = []
    true_labels = None

    for _ in range(n_aug):
        preds  = []
        labels = []
        for (rgb, edge), lbl in dataset:
            rgb_aug = data_augmentation(rgb, training=True)
            pred    = model([rgb_aug, edge], training=False)
            preds.append(pred.numpy())
            labels.append(lbl.numpy())
        all_preds.append(np.concatenate(preds, axis=0))
        if true_labels is None:
            true_labels = np.concatenate(labels, axis=0)

    return np.mean(all_preds, axis=0), true_labels

print("\nRunning TTA on test set (8-pass)...")
tta_preds, true_labels = tta_predict(model, test_ds, n_aug=8)
tta_accuracy = np.mean(
    np.argmax(tta_preds, axis=1) == np.argmax(true_labels, axis=1)
)

# =========================================================
# STANDARD EVALUATION
# =========================================================
results = model.evaluate(test_ds, return_dict=True)

print("\n" + "=" * 40)
print("FINAL RESULTS")
print("=" * 40)
print(f"Fusion layer   : {FUSION_LAYER}")
print(f"Test accuracy  : {results['accuracy']:.4f}")
print(f"Test AUC       : {results['auc']:.4f}")
print(f"Test precision : {results['precision']:.4f}")
print(f"Test recall    : {results['recall']:.4f}")
print(f"TTA accuracy   : {tta_accuracy:.4f}  (8-pass average)")

# =========================================================
# SAVE
# =========================================================
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")