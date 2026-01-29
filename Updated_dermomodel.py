from google.colab import drive
drive.mount('/drive')

import os
import random as python_random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

print(tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

print('Found GPU:', device_name)

CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC  = "/drive/MyDrive/HAM10000/images"
BASE     = "/drive/MyDrive/Colab Notebooks/newdata"

df = pd.read_csv(CSV_PATH)

# Binary classification
df["label"] = df["dx"].apply(lambda x: "melanoma" if x == "mel" else "non_melanoma")

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

def make_dirs():
    for split in ["train", "valid", "test"]:
        for cls in ["melanoma", "non_melanoma"]:
            os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)

make_dirs()

def copy_images(df, split):
    for _, row in df.iterrows():
        img = row["image_id"] + ".jpg"
        src = os.path.join(IMG_SRC, img)
        dst = os.path.join(BASE, split, row["label"], img)
        if os.path.exists(src):
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(valid_df, "valid")
copy_images(test_df, "test")

def set_data(train, test, valid, batchSize, image_size):

    Image_size = (image_size, image_size)

    train_gen = ImageDataGenerator()
    test_gen  = ImageDataGenerator()
    val_gen   = ImageDataGenerator()

    train_set = train_gen.flow_from_directory(
        train, target_size=Image_size,
        batch_size=batchSize, class_mode='categorical'
    )

    test_set = test_gen.flow_from_directory(
        test, target_size=Image_size,
        batch_size=batchSize, class_mode='categorical'
    )

    val_set = val_gen.flow_from_directory(
        valid, target_size=Image_size,
        batch_size=batchSize, class_mode='categorical'
    )

    return train_set, test_set, val_set

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["train", "val"])
    plt.show()

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "val"])
    plt.show()

def unfreeze_model(model, num_layers):
    for layer in model.layers[num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

def create_model():
    inputs = layers.Input(shape=(256, 256, 3))  # reduced size (faster)

    base = EfficientNetB5(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet"
    )

    base.trainable = False
    base = unfreeze_model(base, -200)

    x = base.output

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1,1,2048))(se)
    se = layers.Dense(85, activation='swish', use_bias=False)(se)
    se = layers.Dense(2048, activation='sigmoid', use_bias=False)(se)
    x  = layers.Multiply()([se, x])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

BASE = "/drive/MyDrive/Colab Notebooks/newdata"

train_path = f"{BASE}/train"
valid_path = f"{BASE}/valid"
test_path  = f"{BASE}/test"

batchSize  = 8
image_size = 256
TOTAL_EPOCHS = 25

train_set, test_set, val_set = set_data(
    train_path,
    test_path,
    valid_path,
    batchSize,
    image_size
)

model = create_model()

CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Save BEST FULL MODEL
checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Save weights every epoch (for resume)
checkpoint_all = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/epoch_{{epoch:02d}}.weights.h5",
    save_weights_only=True,
    verbose=0
)

history = model.fit(
    train_set,
    epochs=5,
    validation_data=val_set,
    callbacks=[checkpoint_best, checkpoint_all]
)

# training part devided to two cells for loading weights and resuming training

model = create_model()
model.load_weights("/drive/MyDrive/checkpoints/epoch_05.weights.h5")

history = model.fit(
    train_set,
    initial_epoch=5,
    epochs=10,
    validation_data=val_set,
    callbacks=[checkpoint_best, checkpoint_all]
)

model = create_model()
model.load_weights("/drive/MyDrive/checkpoints/epoch_10.weights.h5")

history = model.fit(
    train_set,
    initial_epoch=10,
    epochs=15,
    validation_data=val_set,
    callbacks=[checkpoint_best, checkpoint_all]
)

model = create_model()
model.load_weights("/drive/MyDrive/checkpoints/epoch_15.weights.h5")

history = model.fit(
    train_set,
    initial_epoch=15,
    epochs=20,
    validation_data=val_set,
    callbacks=[checkpoint_best, checkpoint_all]
)

model = create_model()
model.load_weights("/drive/MyDrive/checkpoints/epoch_20.weights.h5")

history = model.fit(
    train_set,
    initial_epoch=20,
    epochs=25,
    validation_data=val_set,
    callbacks=[checkpoint_best, checkpoint_all]
)

model = tf.keras.models.load_model("/drive/MyDrive/checkpoints/best.keras")

model = tf.keras.models.load_model(
    f"{CHECKPOINT_DIR}/best.keras"
)

loss, acc = model.evaluate(test_set)
print("Final Test Accuracy:", acc)

# 126/126 ━━━━━━━━━━━━━━━━━━━━ 59s 284ms/step - accuracy: 0.9122 - loss: 0.4594
# Final Test Accuracy: 0.92514967918396

model.save(
    "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/big30data9615.keras"
)