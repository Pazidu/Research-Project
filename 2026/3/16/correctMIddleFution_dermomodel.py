from google.colab import drive
drive.mount('/drive')

import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

print("TensorFlow:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())


# Paths
BASE = "/content/newdata"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
MODEL_SAVE_PATH = "/drive/MyDrive/Models/efficientnetv2s_middlefusion.keras"

if os.path.exists(BASE):
    shutil.rmtree(BASE)

shutil.copytree(IMG_SRC, BASE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parameters
batch_size = 16
image_size = 256
FUSION_LAYER = "block4c_add"


# Edge Map Creation
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


# -------------------------------
# MODEL
# -------------------------------

def create_middle_fusion_model():

    # RGB branch
    rgb_input = layers.Input(shape=(image_size, image_size, 3))

    base = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_tensor=rgb_input
    )

    base.trainable = True

    # freeze early layers
    for layer in base.layers[:-50]:
        layer.trainable = False

    rgb_feature_map = base.get_layer(FUSION_LAYER).output


    # ---------------------------
    # Edge branch
    # ---------------------------

    edge_input = layers.Input(shape=(image_size, image_size, 1))

    x = layers.Conv2D(32,3,padding='same',activation='relu')(edge_input)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # x = layers.Conv2D(128,3,padding='same',activation='relu')(x)
    # x = layers.MaxPooling2D(2)(x)

    # x = layers.Conv2D(256,3,padding='same',activation='relu')(x)
    # x = layers.MaxPooling2D(2)(x)

    # resize edge map to match EfficientNet feature map
    x = layers.Resizing(
        rgb_feature_map.shape[1],
        rgb_feature_map.shape[2]
    )(x)


    # ---------------------------
    # Middle Fusion
    # ---------------------------
    fused = layers.Concatenate()([rgb_feature_map, x])

    # Continue CNN after fusion
    fused = layers.Conv2D(256, 3, padding='same', activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)

    fused = layers.Conv2D(256, 3, padding='same', activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)

    # Extra conv layer after fusion
    fused = layers.Conv2D(256, 3, padding='same', activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.3)(fused)  # Added dropout

    fused = layers.MaxPooling2D(2)(fused)


    # Classification
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.5)(fused)

    outputs = layers.Dense(
        2,
        activation='softmax',
        dtype='float32'
    )(fused)


    model = tf.keras.Model(
        inputs=[rgb_input, edge_input],
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = create_middle_fusion_model()

model.summary()


# -------------------------------
# Training
# -------------------------------

checkpoint = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_middlefusion.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


history = model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
    callbacks=[checkpoint]
)


# -------------------------------
# Evaluation
# -------------------------------

loss, acc = model.evaluate(test_ds)

print("Test Accuracy:", acc)


# Save model
model.save(MODEL_SAVE_PATH)