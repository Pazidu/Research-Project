from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

# ===============================
# Paths
# ===============================
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_mid_fusion.keras"

# Clean Colab SSD and copy dataset
if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===============================
# Mixed precision
# ===============================
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ===============================
# Dataset + Edge Map
# ===============================
batch_size = 16
image_size = 256

def add_edge_map(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)

    return (image, edge), label

def prepare_dataset(path, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle
    )

    ds = ds.map(add_edge_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = prepare_dataset(f"{BASE}/train", True)
val_ds   = prepare_dataset(f"{BASE}/valid", False)
test_ds  = prepare_dataset(f"{BASE}/test", False)

# ===============================
# Mid-Level Fusion Model
# ===============================
def create_mid_fusion_model(image_size, fusion_layer_name="block4a_activation"):

    # RGB Input
    rgb_input = layers.Input(shape=(image_size, image_size, 3), name="rgb_input")

    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_tensor=rgb_input
    )

    # Freeze early layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Get middle feature map
    fusion_feature = base_model.get_layer(fusion_layer_name).output

    # Edge Input
    edge_input = layers.Input(shape=(image_size, image_size, 1), name="edge_input")

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(edge_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    # Resize edge feature to match EfficientNet feature size
    target_h = fusion_feature.shape[1]
    target_w = fusion_feature.shape[2]
    x = layers.Resizing(target_h, target_w)(x)

    # Match channels
    x = layers.Conv2D(fusion_feature.shape[-1], 1, activation='relu')(x)

    # Mid-level Fusion
    fused = layers.Concatenate()([fusion_feature, x])

    # Reduce channels back
    fused = layers.Conv2D(fusion_feature.shape[-1], 1, activation='relu')(fused)

    # Continue remaining EfficientNet layers
    remaining = fused
    fusion_index = base_model.layers.index(base_model.get_layer(fusion_layer_name))

    for layer in base_model.layers[fusion_index + 1:]:
        remaining = layer(remaining)

    # Classification head
    gap = layers.GlobalAveragePooling2D()(remaining)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Dropout(0.5)(gap)

    outputs = layers.Dense(2, activation='softmax', dtype='float32')(gap)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ===============================
# Create Model
# ===============================
model = create_mid_fusion_model(image_size, "block4a_activation")

model.summary()

# ===============================
# Training
# ===============================
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_mid_fusion.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
    callbacks=[checkpoint_best]
)

# ===============================
# Evaluation
# ===============================
loss, acc, auc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {acc:.4f}")
print(f"Final Test AUC: {auc:.4f}")

# ===============================
# Save Model
# ===============================
model.save(MODEL_SAVE_PATH)