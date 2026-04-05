from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

print("TensorFlow:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())


# -------------------------------
# PATHS
# -------------------------------
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

STAGE1_MODEL_PATH = "/drive/MyDrive/Models/stage1_model.keras"
FINAL_MODEL_PATH  = "/drive/MyDrive/Models/stage2_final.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------------
# PARAMETERS
# -------------------------------
batch_size = 16
image_size = 300
FUSION_LAYER = "block4c_add"


# -------------------------------
# DATA AUGMENTATION
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
])


# -------------------------------
# FOCAL LOSS
# -------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss


# -------------------------------
# DATASET (FIXED)
# -------------------------------
def add_edge_map(image, label, training=False):
    image = tf.cast(image, tf.float32)

    # ✅ augmentation only for training
    if training:
        image = data_augmentation(image)

    image = preprocess_input(image)

    gray = tf.image.rgb_to_grayscale(image)

    # 🔥 multi-scale edge extraction
    e1 = tf.image.sobel_edges(gray)
    e2 = tf.image.sobel_edges(tf.image.resize(gray, [128,128]))

    e1 = tf.sqrt(tf.reduce_sum(tf.square(e1), axis=-1))
    e2 = tf.sqrt(tf.reduce_sum(tf.square(e2), axis=-1))

    e2 = tf.image.resize(e2, [image_size, image_size])

    edge = tf.concat([e1, e2], axis=-1)
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label


def prepare_dataset(path, training):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=training
    )

    ds = ds.map(lambda x,y: add_edge_map(x,y,training),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds   = prepare_dataset(f"{BASE}/valid", False)
test_ds  = prepare_dataset(f"{BASE}/test", False)


# -------------------------------
# CHANNEL ATTENTION
# -------------------------------
def channel_attention(x, ratio=8):
    c = x.shape[-1]

    gap = layers.GlobalAveragePooling2D()(x)
    d1 = layers.Dense(c // ratio, activation='relu')(gap)
    d2 = layers.Dense(c, activation='sigmoid')(d1)

    scale = layers.Reshape((1,1,c))(d2)
    return layers.Multiply()([x, scale])


# =========================================================
# 🔥 STAGE 1 MODEL
# =========================================================
def create_stage1_model():

    rgb_input = layers.Input(shape=(image_size, image_size, 3))
    edge_input = layers.Input(shape=(image_size, image_size, 2))  # 🔥 updated

    base = EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=rgb_input)
    rgb_feat = base.get_layer(FUSION_LAYER).output

    x = layers.Conv2D(32,3,padding='same',activation='relu')(edge_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Resizing(rgb_feat.shape[1], rgb_feat.shape[2])(x)

    rgb_feat = channel_attention(rgb_feat)
    x = channel_attention(x)

    fused = layers.Concatenate()([rgb_feat, x])

    fused = layers.Conv2D(256,3,padding='same',activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)

    features = layers.GlobalAveragePooling2D(name="feature_layer")(fused)

    x = layers.BatchNormalization()(features)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)


# -------------------------------
# CALLBACKS
# -------------------------------
lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-5   # 🔥 fixed
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# -------------------------------
# TRAIN STAGE 1
# -------------------------------
print("\n🔥 Training Stage 1")

stage1_model = create_stage1_model()

stage1_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),   # 🔥 improved
    metrics=["accuracy"]
)

checkpoint1 = ModelCheckpoint(
    f"{CHECKPOINT_DIR}/stage1_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

stage1_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint1, lr_schedule, early_stop]
)

stage1_model.save(STAGE1_MODEL_PATH)


# =========================================================
# 🔥 STAGE 2 (FULL FINE-TUNING)
# =========================================================
print("\n🔥 Training Stage 2")

stage2_model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/stage1_best.keras",
    custom_objects={'loss': focal_loss()}
)

# 🔥 unfreeze ALL layers
for layer in stage2_model.layers:
    layer.trainable = True

stage2_model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-6),  # 🔥 better LR
    loss=focal_loss(),
    metrics=["accuracy"]
)

checkpoint2 = ModelCheckpoint(
    f"{CHECKPOINT_DIR}/stage2_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

stage2_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint2, lr_schedule, early_stop]
)


# -------------------------------
# TEST-TIME AUGMENTATION
# -------------------------------
def tta_evaluate(model, dataset):
    preds = []
    for (img, edge), y in dataset:
        p1 = model.predict([img, edge], verbose=0)
        p2 = model.predict([tf.image.flip_left_right(img), edge], verbose=0)
        preds.append((p1 + p2)/2)

    preds = tf.concat(preds, axis=0)
    true  = tf.concat([y for _,y in dataset], axis=0)

    acc = tf.keras.metrics.categorical_accuracy(true, preds)
    print("Final Test Accuracy (TTA):", tf.reduce_mean(acc).numpy())


# -------------------------------
# EVALUATION
# -------------------------------
best_model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/stage2_best.keras",
    custom_objects={'loss': focal_loss()}
)

tta_evaluate(best_model, test_ds)

stage2_model.save(FINAL_MODEL_PATH)