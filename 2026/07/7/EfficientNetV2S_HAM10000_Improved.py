
# EfficientNetV2S HAM10000 Improved Training Pipeline
# Generated for Google Colab
# Features:
# - EfficientNetV2S backbone
# - 300x300 input
# - RGB + edge fusion
# - MixUp augmentation
# - Mixed precision
# - Reproducible seed
# - Two-stage fine tuning
# - Best model checkpoint

from google.colab import drive
drive.mount('/drive')

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from sklearn.metrics import confusion_matrix, classification_report


# ==========================
# SEED
# ==========================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==========================
# MIXED PRECISION
# ==========================

mixed_precision.set_global_policy("mixed_float16")

# ==========================
# PATHS
# ==========================

DATASET = "/content/newdata"
MODEL_PATH = "/drive/MyDrive/Colab Notebooks/Models/dermoscopy/best_efficientnetv2s_mixup.keras"
CHECKPOINT = "/drive/MyDrive/checkpoints/best_efficientnetv2s_mixup.keras"

# ==========================
# SETTINGS
# ==========================

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 30


# ==========================
# AUGMENTATION
# ==========================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# ==========================
# EDGE EXTRACTION
# ==========================

def add_edge(image, label):
    image = tf.cast(image, tf.float32)
    gray = tf.image.rgb_to_grayscale(image)
    sobel = tf.image.sobel_edges(gray)
    edge = tf.sqrt(
        tf.reduce_sum(tf.square(sobel), axis=-1)
    )
    edge = edge / (tf.reduce_max(edge)+1e-6)
    return (image, edge), label

# ==========================
# DATASET
# ==========================

def load_dataset(path, shuffle):

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )

    ds = ds.map(
        add_edge,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = load_dataset(
    DATASET+"/train",
    True
)

val_ds = load_dataset(
    DATASET+"/valid",
    False
)

test_ds = load_dataset(
    DATASET+"/test",
    False
)


# ==========================
# MIXUP
# ==========================

def mixup(batch1, batch2, alpha=0.2):

    (x1,e1), y1 = batch1
    (x2,e2), y2 = batch2

    lam = tf.random.uniform([],0,1)

    x = lam*x1 + (1-lam)*x2
    e = lam*e1 + (1-lam)*e2
    y = lam*y1 + (1-lam)*y2

    return (x,e),y


train_ds2 = tf.data.Dataset.zip(
    (train_ds, train_ds.shuffle(1000))
)

train_ds2 = train_ds2.map(
    lambda a,b: mixup(a,b),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds2 = train_ds2.prefetch(tf.data.AUTOTUNE)

# ==========================
# MODEL
# ==========================

def create_model():

    rgb = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    edge = layers.Input(shape=(IMG_SIZE,IMG_SIZE,1))
    x = augmentation(rgb)
    x = preprocess_input(x)

    backbone = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE,IMG_SIZE,3)
    )
    backbone.trainable = False
    features = backbone(x)
    e = layers.Conv2D(32,3,activation="relu",padding="same")(edge)
    e = layers.MaxPooling2D()(e)
    e = layers.Conv2D(64,3,activation="relu",padding="same")(e)
    e = layers.Resizing(features.shape[1],features.shape[2])(e)
    e = layers.Conv2D(features.shape[-1], 1)(e)
    fused = layers.Concatenate()([features,e])
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dense(256,activation="relu")(fused)
    fused = layers.Dropout(0.4)(fused)

    output = layers.Dense(
        2,
        activation="softmax",
        dtype="float32"
    )(fused)

    model = Model(
        [rgb,edge],
        output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

model = create_model()
model.summary()

# ==========================
# STAGE 1 TRAIN
# ==========================

callbacks=[

ModelCheckpoint(
    CHECKPOINT,
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1
),

EarlyStopping(
    monitor="val_auc",
    patience=8,
    restore_best_weights=True,
    mode="max"
),

ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=3
)]

history=model.fit(
    train_ds2,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# UNFREEZE LAST LAYERS
# ==========================

for layer in model.layers:
    layer.trainable=True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

history2=model.fit(
    train_ds2,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# ==========================
# TEST
# ==========================

model.load_weights(CHECKPOINT)
result=model.evaluate(test_ds)
print(result)

# Confusion matrix

y_true=[]
y_pred=[]

for x,y in test_ds:
    pred=model.predict(x,verbose=0)
    y_true.extend(np.argmax(y.numpy(),axis=1))
    y_pred.extend(np.argmax(pred,axis=1))


print(confusion_matrix(y_true,y_pred))

print(
classification_report(y_true,y_pred,target_names=["melanoma","non_melanoma"]))

model.save(MODEL_PATH)
print("DONE")
