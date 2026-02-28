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


# Paths
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_dual_branch.keras"

# Clean Colab SSD and copy dataset
if os.path.exists(BASE):
    shutil.rmtree(BASE)
shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Dataset + Edge Map
batch_size = 16
image_size = 256

def add_edge_map(image, label, image_size):
    # Preprocess for EfficientNetV2S
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    # Edge map
    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
    edge = edge / (tf.reduce_max(edge) + 1e-6)  # normalize
    edge = tf.ensure_shape(edge, (None, image_size, image_size, 1))
    return (image, edge), label

def prepare_dataset(path, batch_size, image_size, shuffle):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle
    )
    ds = ds.map(lambda x, y: add_edge_map(x, y, image_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = prepare_dataset(f"{BASE}/train", batch_size, image_size, True)
val_ds   = prepare_dataset(f"{BASE}/valid", batch_size, image_size, False)
test_ds  = prepare_dataset(f"{BASE}/test", batch_size, image_size, False)


# Dual-branch Model
def create_dual_model(image_size):

    # RGB branch
    rgb_input = layers.Input(shape=(image_size, image_size, 3), name="rgb_input")
    base = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=rgb_input)
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    rgb_features = layers.GlobalAveragePooling2D()(base.output)

    # Edge branch
    edge_input = layers.Input(shape=(image_size, image_size, 1), name="edge_input")
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(edge_input)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Feature fusion
    fused = layers.Concatenate()([rgb_features, x])
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.5)(fused)

    outputs = layers.Dense(2, activation='softmax', dtype='float32')(fused)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_dual_model(image_size)

# Training
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_dual.keras",
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

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {acc:.4f}")

# Save model
model.save(MODEL_SAVE_PATH)