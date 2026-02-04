from google.colab import drive
drive.mount('/drive')

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
CSV_PATH = "/drive/MyDrive/HAM10000/HAM10000_metadata.csv"
IMG_SRC  = "/drive/MyDrive/HAM10000/images"
BASE     = "/drive/MyDrive/Colab Notebooks/newdata"
CHECKPOINT_DIR = "/drive/MyDrive/checkpoints"

batchSize  = 32  # Increased for V2S speed
image_size = 256
TOTAL_EPOCHS = 25

# --- DATA PREPARATION ---
df = pd.read_csv(CSV_PATH)
df["label"] = df["dx"].apply(lambda x: "melanoma" if x == "mel" else "non_melanoma")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

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
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(valid_df, "valid")
copy_images(test_df, "test")

def set_data(train, test, valid, batchSize, image_size):
    # Added preprocess_input for EfficientNetV2 standards
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_set = datagen.flow_from_directory(
        train, target_size=(image_size, image_size),
        batch_size=batchSize, class_mode='categorical'
    )
    test_set = datagen.flow_from_directory(
        test, target_size=(image_size, image_size),
        batch_size=batchSize, class_mode='categorical', shuffle=False
    )
    val_set = datagen.flow_from_directory(
        valid, target_size=(image_size, image_size),
        batch_size=batchSize, class_mode='categorical'
    )
    return train_set, test_set, val_set

train_set, test_set, val_set = set_data(f"{BASE}/train", f"{BASE}/test", f"{BASE}/valid", batchSize, image_size)

# --- MODEL CREATION ---
def unfreeze_model(model, num_layers):
    for layer in model.layers[num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

def create_model():
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    base = EfficientNetV2S(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet"
    )
    
    base.trainable = False
    # Unfreezing last 100 layers is effective for V2S
    base = unfreeze_model(base, -100)
    
    x = base.output
    
    # Squeeze-and-Excitation (SE) Block
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // 4, activation='swish', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    x = layers.Multiply()([se, x])
    
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
    return model

model = create_model()

# --- CALLBACKS ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_best = ModelCheckpoint(
    filepath=f"{CHECKPOINT_DIR}/best_v2s.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# --- TRAINING ---
# This will run all 25 epochs. If it disconnects, 
# you can change initial_epoch to the last saved epoch number.
history = model.fit(
    train_set,
    epochs=TOTAL_EPOCHS,
    validation_data=val_set,
    callbacks=[checkpoint_best]
)

# --- EVALUATION ---
best_model = tf.keras.models.load_model(f"{CHECKPOINT_DIR}/best_v2s.keras")
loss, acc = best_model.evaluate(test_set)
print(f"Final Test Accuracy: {acc:.4f}")

best_model.save("/drive/MyDrive/Colab Notebooks/Models/dermoscopy/v2s_final_model.keras")