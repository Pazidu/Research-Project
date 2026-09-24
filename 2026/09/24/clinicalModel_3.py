from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.losses import BinaryFocalCrossentropy  # NEW

print("TensorFlow version:", tf.__version__)

# ==============================
# PARAMETERS
# ==============================
IMG_SIZE = 188
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
FOLDS = 5

# Fusion layer (tune this)
FUSION_LAYER = "block6a_expand_activation"

# Thresholds to sweep for the sensitivity/specificity trade-off
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# --- NEW: focal loss + manual melanoma weight boost ---
FOCAL_GAMMA = 2.0            # NEW: focusing parameter, same as your HAM10000 experiments
MELANOMA_WEIGHT_MULTIPLIER = 1.5  # NEW: multiplies the "balanced" melanoma weight further

# ==============================
# PATHS
# ==============================
SOURCE_DIR = "/content/drive/MyDrive/UMCG"
TARGET_DIR = "/content/drive/MyDrive/Colab Notebooks/kfolddata"
CLASSES = ["melanoma", "non_melanoma"]

# ==============================
# K-FOLD DATASET PREPARATION
# ==============================
images = []
labels = []

for idx, cls in enumerate(CLASSES):
    class_dir = os.path.join(SOURCE_DIR, cls)
    for img_name in os.listdir(class_dir):
        images.append(os.path.join(class_dir, img_name))
        labels.append(idx)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels), 1):
    for split, idxs in [("train", train_idx), ("test", test_idx)]:
        for i in idxs:
            src = images[i]
            cls = CLASSES[labels[i]]
            dst = os.path.join(TARGET_DIR, f"fold{fold}", split, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

print("K-Fold dataset created successfully")

# ==============================
# EDGE MAP FUNCTION
# ==============================
def add_edge_map(image):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    image_exp = tf.expand_dims(image, axis=0)
    gray = tf.image.rgb_to_grayscale(image_exp)
    sobel = tf.image.sobel_edges(gray)
    edge = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))[0]
    edge = edge / (tf.reduce_max(edge) + 1e-6)
    return image, edge

# ==============================
# DUAL BRANCH MODEL
# ==============================
def create_dual_model(img_size):

    # RGB branch
    rgb_input = layers.Input(shape=(img_size, img_size, 3), name="rgb_input")
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=rgb_input
    )
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    fusion_layer = base_model.get_layer(FUSION_LAYER)
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=fusion_layer.output)
    middle_feature = feature_extractor(rgb_input)

    # Edge branch
    edge_input = layers.Input(shape=(img_size, img_size, 1), name="edge_input")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(edge_input)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    # Resize to match EfficientNet feature map
    x = layers.Resizing(middle_feature.shape[1], middle_feature.shape[2])(x)

    # Feature fusion
    fused = layers.Concatenate()([middle_feature, x])
    fused = layers.Conv2D(256, 3, activation="relu", padding="same")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.GlobalAveragePooling2D()(fused)
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.5)(fused)
    outputs = layers.Dense(1, activation="sigmoid")(fused)

    model = tf.keras.Model(inputs=[rgb_input, edge_input], outputs=outputs)

    # --- CHANGED: binary_crossentropy -> BinaryFocalCrossentropy ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, apply_class_balancing=False),  # CHANGED
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ==============================
# CUSTOM GENERATOR
# ==============================
class DualImageDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, generator, batch_size):
        self.generator = generator
        self.batch_size = batch_size

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        x_batch, y_batch = self.generator[idx]
        y_batch = y_batch.reshape(-1, 1)
        x_rgb = np.zeros_like(x_batch, dtype=np.float32)
        x_edge = np.zeros((x_batch.shape[0], x_batch.shape[1], x_batch.shape[2], 1), dtype=np.float32)
        for i in range(x_batch.shape[0]):
            rgb, edge = add_edge_map(x_batch[i])
            x_rgb[i] = rgb
            x_edge[i, :, :, 0] = edge[:, :, 0]
        return (x_rgb, x_edge), y_batch

# ==============================
# METRICS STORAGE
# ==============================
accuracies = []
aucs = []
sensitivities = []
specificities = []
precisions = []
f1_scores = []

# Per-threshold storage for the sweep (averaged across folds at the end)
threshold_results = {t: {"acc": [], "sens": [], "spec": [], "prec": [], "f1": []}
                      for t in THRESHOLDS}

BASE_PATH = TARGET_DIR
SAVE_PATH = "/content/drive/MyDrive/Models/clinical_dual"
os.makedirs(SAVE_PATH, exist_ok=True)

train_gen_base = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_test_gen_base = ImageDataGenerator()

for fold in range(1, FOLDS + 1):
    print(f"\n===== Fold {fold} =====")
    print("Fusion Layer:", FUSION_LAYER)

    train_dir = os.path.join(BASE_PATH, f"fold{fold}", "train")
    test_dir = os.path.join(BASE_PATH, f"fold{fold}", "test")

    train_set_base = train_gen_base.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    val_set_base = val_test_gen_base.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    train_set = DualImageDataGenerator(train_set_base, BATCH_SIZE)
    val_set = DualImageDataGenerator(val_set_base, BATCH_SIZE)

    # ==============================
    # CLASS WEIGHTS (boosted further on melanoma)
    # ==============================
    train_labels = train_set_base.classes
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}

    # --- NEW: push melanoma (class 0) weight higher than plain "balanced" ---
    class_weight_dict[0] = class_weight_dict[0] * MELANOMA_WEIGHT_MULTIPLIER  # NEW

    print(f"Fold {fold} class weights (melanoma boosted x{MELANOMA_WEIGHT_MULTIPLIER}): {class_weight_dict}")
    # class_weight_dict looks like {0: <boosted melanoma weight>, 1: <non_melanoma weight>}

    model = create_dual_model(IMG_SIZE)

    ckpt = ModelCheckpoint(
        filepath=os.path.join(SAVE_PATH, f"best_fold{fold}_{FUSION_LAYER}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set,
        class_weight=class_weight_dict,
        callbacks=[ckpt, reduce_lr, early_stop]
    )

    model = tf.keras.models.load_model(
        os.path.join(SAVE_PATH, f"best_fold{fold}_{FUSION_LAYER}.keras"),
        custom_objects={"BinaryFocalCrossentropy": BinaryFocalCrossentropy}  # NEW: needed to reload a model saved with focal loss
    )

    # ==============================
    # STANDARD EVALUATE (accuracy, auc) @ threshold 0.5
    # ==============================
    loss, acc, auc = model.evaluate(val_set)
    accuracies.append(acc)
    aucs.append(auc)

    # ==============================
    # PREDICTIONS FOR SENSITIVITY / SPECIFICITY / PRECISION / F1 @ threshold 0.5
    # ==============================
    y_true = val_set_base.classes  # 0 = melanoma, 1 = non_melanoma (alphabetical)
    y_prob = model.predict(val_set, steps=len(val_set)).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, pos_label=0)

    sensitivities.append(sensitivity)
    specificities.append(specificity)
    precisions.append(precision)
    f1_scores.append(f1)

    print(f"Fold {fold} Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, "
          f"Precision: {precision:.4f}, F1: {f1:.4f}")
    print(f"Fold {fold} Confusion Matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    # ==============================
    # THRESHOLD SWEEP (reuses y_true / y_prob from above)
    # ==============================
    for t in THRESHOLDS:
        y_pred_t = (y_prob >= t).astype(int)

        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_t, labels=[1, 0]).ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        f1_t   = f1_score(y_true, y_pred_t, pos_label=0)
        acc_t  = accuracy_score(y_true, y_pred_t)

        threshold_results[t]["acc"].append(acc_t)
        threshold_results[t]["sens"].append(sens_t)
        threshold_results[t]["spec"].append(spec_t)
        threshold_results[t]["prec"].append(prec_t)
        threshold_results[t]["f1"].append(f1_t)

# ==============================
# FINAL SUMMARY: DEFAULT THRESHOLD (0.5)
# ==============================
print("\n================================")
print(f"AVERAGE OVER {FOLDS} FOLDS (threshold = 0.50)")
print("================================")
print(f"Accuracy    : {np.mean(accuracies):.4f}")
print(f"AUC         : {np.mean(aucs):.4f}")
print(f"Sensitivity : {np.mean(sensitivities):.4f}")
print(f"Specificity : {np.mean(specificities):.4f}")
print(f"Precision   : {np.mean(precisions):.4f}")
print(f"F1          : {np.mean(f1_scores):.4f}")

# ==============================
# FINAL SUMMARY: THRESHOLD SWEEP
# ==============================
print("\n================================")
print("THRESHOLD SWEEP - AVERAGED OVER ALL FOLDS")
print("================================")
print(f"{'Threshold':<10}{'Accuracy':<10}{'Sensitivity':<12}{'Specificity':<12}{'Precision':<10}{'F1':<10}")
for t, vals in threshold_results.items():
    print(f"{t:<10}{np.mean(vals['acc']):<10.4f}{np.mean(vals['sens']):<12.4f}"
          f"{np.mean(vals['spec']):<12.4f}{np.mean(vals['prec']):<10.4f}{np.mean(vals['f1']):<10.4f}")