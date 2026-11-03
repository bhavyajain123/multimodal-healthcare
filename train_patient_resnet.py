import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

TRAIN_DIR = "data/heart_mri/large_dataset/train"
VAL_DIR = "data/heart_mri/patient_split/validation"
TEST_DIR = "data/heart_mri/large_dataset/test"

MODEL_DIR = "transfer_learning/models"
RESULT_DIR = "transfer_learning/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

NUM_CLASSES = 5

CLASS_NAMES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

SEED = 42


# =========================================================
# LOAD DATASETS
# =========================================================

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASS_NAMES,
    shuffle=False
)

print("\nDatasets loaded successfully.")


# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =========================================================
# DATA AUGMENTATION
# =========================================================

augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
], name="augmentation")


# =========================================================
# RESNET50
# =========================================================

print("\nLoading ResNet50...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(shape=(224, 224, 3))

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


# =========================================================
# PHASE 1
# =========================================================

print("\n==============================")
print("PHASE 1: FROZEN RESNET50")
print("==============================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# CLASS WEIGHTS
# =========================================================

# Training labels from directory structure
train_labels = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    train_labels.extend(
        [class_index] * count
    )


train_labels = np.array(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = {
    i: float(weights[i])
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")
print(class_weights)


# =========================================================
# CALLBACKS
# =========================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "acdc_patient_resnet50_best.keras"
)

callbacks = [

    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# =========================================================
# TRAIN PHASE 1
# =========================================================

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================================================
# PHASE 2
# FINE-TUNING
# =========================================================

print("\n==============================")
print("PHASE 2: FINE-TUNING")
print("==============================")


base_model.trainable = True


# Freeze most layers
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
        learning_rate=5e-6
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    best_model_path
)


# =========================================================
# TEST EVALUATION
# =========================================================

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


# =========================================================
# PREDICTIONS
# =========================================================

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        np.argmax(labels.numpy(), axis=1)
    )

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print("\nClassification Report:")
print(report)


with open(
    os.path.join(
        RESULT_DIR,
        "acdc_patient_classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "acdc_patient_confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


# =========================================================
# CONFUSION MATRIX IMAGE
# =========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "ACDC Patient-Level ResNet50"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "acdc_patient_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# =========================================================
# SAVE FINAL MODEL
# =========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "acdc_patient_resnet50_final.keras"
)

model.save(
    final_model_path
)

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print(
    f"Best model: {best_model_path}"
)

print(
    f"Final model: {final_model_path}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)