from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input

print("TensorFlow version:", tf.__version__)

IMG_SIZE = 188          # Image dimension (width & height)
BATCH_SIZE = 16         # Number of images per training batch
EPOCHS = 100            # Number of training epochs
LEARNING_RATE = 1e-4    # Learning rate for optimizer
R_RATIO = 24            # Reduction ratio for channel attention

SOURCE_DIR = "/content/drive/MyDrive/UMCG"              # Original dataset location
TARGET_DIR = "/content/drive/MyDrive/Colab Notebooks/kfolddata"  # Where to save k-fold data
CLASSES = ["melanoma", "non_melanoma"]                   # Class names
FOLDS = 5                                                # Number of K-Folds for cross-validation

images = []
labels = []

for idx, cls in enumerate(CLASSES):
    class_dir = os.path.join(SOURCE_DIR, cls)
    for img_name in os.listdir(class_dir):
        images.append(os.path.join(class_dir, img_name))
        labels.append(idx)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)  # Remove old fold data if exists

for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels), 1):
    for split, idxs in [("train", train_idx), ("test", test_idx)]:
        for i in idxs:
            src = images[i]
            cls = CLASSES[labels[i]]
            dst = os.path.join(TARGET_DIR, f"fold{fold}", split, cls)
            os.makedirs(dst, exist_ok=True)  # Create folder if not exists
            shutil.copy(src, dst)             # Copy images to respective fold/train/test folders

print("✅ K-Fold dataset created successfully")

def create_proposed_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Freeze all layers first
    base.trainable = False
    
    # Unfreeze last 20 layers except BatchNormalization layers
    for layer in base.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    x = base.output
    channels = x.shape[-1]

    # Channel Attention Module
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // R_RATIO, activation="swish", use_bias=False)(se)
    se = layers.Dense(channels, activation="sigmoid", use_bias=False)(se)
    x = layers.Multiply()([x, se])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

accuracies = []

BASE_PATH = TARGET_DIR
SAVE_PATH = "/content/drive/MyDrive/Models/clinical"
os.makedirs(SAVE_PATH, exist_ok=True)

for fold in range(1, FOLDS + 1):
    print(f"\n===== Fold {fold} =====")
    
    train_dir = os.path.join(BASE_PATH, f"fold{fold}", "train")
    test_dir = os.path.join(BASE_PATH, f"fold{fold}", "test")
    
    train_set = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )
    
    val_set = val_test_gen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    
    model = create_proposed_model()
    
    ckpt = ModelCheckpoint(
        filepath=os.path.join(SAVE_PATH, f"best_fold{fold}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )
    
    model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set,
        callbacks=[ckpt, reduce_lr]
    )
    
    model = tf.keras.models.load_model(os.path.join(SAVE_PATH, f"best_fold{fold}.keras"))
    loss, acc, auc = model.evaluate(val_set)
    accuracies.append(acc)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}, AUC: {auc:.4f}")

print(f"\nAverage Accuracy over {FOLDS} folds: {np.mean(accuracies):.4f}")
# Average Accuracy over 5 folds: 0.8294