from google.colab import drive
drive.mount('/drive')

import os
import shutil
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Enable mixed precision for speed
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC = "/drive/MyDrive/Colab Notebooks/newdata"
BASE = "/content/newdata"  # Local SSD path   

df = pd.read_csv(CSV_PATH)
df["label"] = df["dx"].apply(lambda x: "melanoma" if x == "mel" else "non_melanoma")

# Train/Val/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

for split in ["train", "valid", "test"]:
    for cls in ["melanoma", "non_melanoma"]:
        os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)

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

print("Images copied to local SSD")

def prepare_datasets(train_path, valid_path, test_path, batch_size, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

def create_model(image_size):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    base = EfficientNetV2S(
        include_top=False,  
        input_tensor=inputs,
        weights='imagenet',
    )

    base.trainable = False

    # Unfreeze last 50 layers (tune as needed)
    for layer in base.layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)  # float32 for output layer to avoid mixed precision issues

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

batch_size = 16
image_size = 256
TOTAL_EPOCHS = 25

train_path = f"{BASE}/train"
valid_path = f"{BASE}/valid"
test_path = f"{BASE}/test"

train_ds, val_ds, test_ds = prepare_datasets(train_path, valid_path, test_path, batch_size, image_size)

model = create_model(image_size)

CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_best]
)

loss, acc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {acc:.4f}")

model.save("/drive/MyDrive/Colab Notebooks/Models/dermoscopy/efficientnetv2s_optimized.keras")