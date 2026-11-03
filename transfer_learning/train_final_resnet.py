import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# =====================================================
# PATHS
# =====================================================

TRAIN_DIR = "data/heart_mri/final_train"
TEST_DIR = "data/heart_mri/large_dataset/test"

MODEL_DIR = "transfer_learning/models"
RESULT_DIR = "transfer_learning/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# CONFIG
# =====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

NUM_CLASSES = 5


# =====================================================
# TRAIN DATA
# =====================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    shuffle=True,
    seed=42
)


# =====================================================
# TEST DATA
# =====================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    shuffle=False
)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =====================================================
# MRI AUGMENTATION
# =====================================================

augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.04),
    layers.RandomContrast(0.04),
], name="mri_augmentation")


# =====================================================
# RESNET50
# =====================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# =====================================================
# MODEL
# =====================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# =====================================================
# PHASE 1
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


best_path = os.path.join(
    MODEL_DIR,
    "acdc_resnet50_final_best.keras"
)


callbacks = [

    ModelCheckpoint(
        best_path,
        monitor="loss",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


print("\n==============================")
print("PHASE 1")
print("==============================")


model.fit(
    train_ds,
    epochs=15,
    callbacks=callbacks
)


# =====================================================
# PHASE 2 — FINE TUNING
# =====================================================

print("\n==============================")
print("PHASE 2 — FINE TUNING")
print("==============================")


base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:-60]:

    layer.trainable = False


# Keep BatchNorm frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-6
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_ds,
    epochs=20,
    callbacks=callbacks
)


# =====================================================
# LOAD BEST MODEL
# =====================================================

model = tf.keras.models.load_model(
    best_path
)


# =====================================================
# FINAL TEST
# =====================================================

print("\n==============================")
print("FINAL TEST")
print("==============================")


test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)


print(
    f"\nFINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)


# =====================================================
# PREDICTIONS
# =====================================================

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )

    y_pred.extend(
        np.argmax(
            predictions,
            axis=1
        )
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =====================================================
# REPORT
# =====================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    digits=4
)

print("\n")
print(report)


with open(
    os.path.join(
        RESULT_DIR,
        "acdc_resnet50_final_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


np.savetxt(
    os.path.join(
        RESULT_DIR,
        "acdc_resnet50_final_cm.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


# =====================================================
# SAVE MODEL
# =====================================================

final_path = os.path.join(
    MODEL_DIR,
    "acdc_resnet50_final.keras"
)

model.save(final_path)


print("\n================================")
print("DONE")
print("================================")

print(
    f"Model: {final_path}"
)

print(
    f"Accuracy: {test_accuracy * 100:.2f}%"
)