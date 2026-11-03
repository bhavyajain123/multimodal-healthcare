import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = "data/heart_mri/large_dataset/train"
TEST_DIR = "data/heart_mri/large_dataset/test"

MODEL_DIR = "transfer_learning/models"
RESULT_DIR = "transfer_learning/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 20

CLASS_NAMES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

NUM_CLASSES = len(CLASS_NAMES)

SEED = 42


# ============================================================
# LOAD TRAIN DATASET
# ============================================================

print("\n==============================================")
print("LOADING TRAIN DATASET")
print("==============================================")


train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    class_names=CLASS_NAMES,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    subset="training",

    seed=SEED,

    shuffle=True
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

val_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    class_names=CLASS_NAMES,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    subset="validation",

    seed=SEED,

    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="int",

    class_names=CLASS_NAMES,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


print("\n==============================================")
print("CLASS NAMES")
print("==============================================")

print(CLASS_NAMES)


# ============================================================
# DATASET INFORMATION
# ============================================================

train_count = tf.data.experimental.cardinality(
    train_dataset
).numpy()

val_count = tf.data.experimental.cardinality(
    val_dataset
).numpy()

test_count = tf.data.experimental.cardinality(
    test_dataset
).numpy()


print("\n==============================================")
print("DATASET INFORMATION")
print("==============================================")


print(
    "Training batches:",
    train_count
)

print(
    "Validation batches:",
    val_count
)

print(
    "Test batches:",
    test_count
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    # Mild rotation
    layers.RandomRotation(
        0.03
    ),

    # Mild zoom
    layers.RandomZoom(
        0.05
    ),

    # Mild contrast
    layers.RandomContrast(
        0.05
    )
])


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_dataset(
    images,
    labels,
    training=False
):

    images = tf.cast(
        images,
        tf.float32
    )

    if training:

        images = data_augmentation(
            images,
            training=True
        )

    images = preprocess_input(
        images
    )

    return images, labels


# ============================================================
# APPLY PREPROCESSING
# ============================================================

train_dataset = train_dataset.map(

    lambda x, y:
    preprocess_dataset(
        x,
        y,
        True
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


val_dataset = val_dataset.map(

    lambda x, y:
    preprocess_dataset(
        x,
        y,
        False
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.map(

    lambda x, y:
    preprocess_dataset(
        x,
        y,
        False
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n==============================================")
print("CALCULATING CLASS WEIGHTS")
print("==============================================")


# Read labels from directory structure
all_labels = []


for class_index, class_name in enumerate(CLASS_NAMES):

    class_path = os.path.join(
        TRAIN_DIR,
        class_name
    )

    if os.path.exists(class_path):

        images = [

            file

            for file in os.listdir(
                class_path
            )

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]

        all_labels.extend(
            [class_index] * len(images)
        )


all_labels = np.array(
    all_labels
)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(NUM_CLASSES),

    y=all_labels
)


class_weights = {

    i: float(
        class_weights_array[i]
    )

    for i in range(NUM_CLASSES)
}


print(
    "Class weights:",
    class_weights
)


# ============================================================
# BUILD RESNET50
# ============================================================

print("\n==============================================")
print("BUILDING RESNET50 MODEL")
print("==============================================")


base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )
)


# ============================================================
# PHASE 1
# FREEZE RESNET50
# ============================================================

base_model.trainable = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

inputs = layers.Input(

    shape=(
        224,
        224,
        3
    )
)


x = base_model(

    inputs,

    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.BatchNormalization()(x)


x = layers.Dense(

    256,

    activation="relu"

)(x)


x = layers.Dropout(
    0.35
)(x)


x = layers.Dense(

    128,

    activation="relu"

)(x)


x = layers.Dropout(
    0.25
)(x)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax"

)(x)


model = models.Model(

    inputs,

    outputs
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(

    MODEL_DIR,

    "acdc_resnet50_improved_best.keras"
)


# ============================================================
# PHASE 1 COMPILE
# ============================================================

print("\n==============================================")
print("PHASE 1 - TRANSFER LEARNING")
print("==============================================")


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# PHASE 1 CALLBACKS
# ============================================================

callbacks_phase1 = [

    ModelCheckpoint(

        checkpoint_path,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=4,

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


# ============================================================
# PHASE 1 TRAINING
# ============================================================

history1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=PHASE1_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase1
)


# ============================================================
# PHASE 2 - CONTROLLED FINE TUNING
# ============================================================

print("\n==============================================")
print("PHASE 2 - CONTROLLED FINE TUNING")
print("==============================================")


# Unfreeze ResNet50
base_model.trainable = True


# Freeze all except last 60 layers

for layer in base_model.layers[:-60]:

    layer.trainable = False


# Keep BatchNorm frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COMPILE FINE-TUNING MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=5e-6
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# PHASE 2 CALLBACKS
# ============================================================

callbacks_phase2 = [

    ModelCheckpoint(

        checkpoint_path,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=6,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-8,

        verbose=1
    )
]


# ============================================================
# PHASE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=PHASE2_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase2
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n==============================================")
print("LOADING BEST MODEL")
print("==============================================")


model = tf.keras.models.load_model(

    checkpoint_path
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST EVALUATION")
print("==============================================")


test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


print("\n==============================================")
print("FINAL TEST ACCURACY")
print("==============================================")


print(

    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

y_true = []

y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(

        images,

        verbose=0
    )

    predicted_classes = np.argmax(

        predictions,

        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4
)


print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")


print(report)


report_path = os.path.join(

    RESULT_DIR,

    "acdc_resnet50_improved_report.txt"
)


with open(

    report_path,

    "w"

) as f:

    f.write(
        "ACDC RESNET50 IMPROVED CLASSIFICATION REPORT\n\n"
    )

    f.write(
        report
    )

    f.write(

        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"

    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred
)


print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")


print(cm)


cm_path = os.path.join(

    RESULT_DIR,

    "acdc_resnet50_improved_confusion_matrix.csv"
)


pd.DataFrame(

    cm,

    index=CLASS_NAMES,

    columns=CLASS_NAMES

).to_csv(

    cm_path
)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(

    figsize=(
        8,
        7
    )
)


plt.imshow(cm)


plt.title(

    "ACDC MRI - Improved ResNet50"

)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
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


plt.colorbar()


plt.tight_layout()


cm_graph_path = os.path.join(

    RESULT_DIR,

    "acdc_resnet50_improved_confusion_matrix.png"
)


plt.savefig(

    cm_graph_path,

    dpi=200
)


plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

train_acc = (

    history1.history["accuracy"]

    +

    history2.history["accuracy"]

)


val_acc = (

    history1.history["val_accuracy"]

    +

    history2.history["val_accuracy"]

)


plt.figure(

    figsize=(
        9,
        6
    )
)


plt.plot(

    train_acc,

    label="Training Accuracy"
)


plt.plot(

    val_acc,

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(

    "ACDC MRI - Improved ResNet50 Training"

)


plt.legend()


plt.grid(True)


plt.tight_layout()


accuracy_graph_path = os.path.join(

    RESULT_DIR,

    "acdc_resnet50_improved_accuracy.png"
)


plt.savefig(

    accuracy_graph_path,

    dpi=200
)


plt.close()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "acdc_resnet50_improved_final.keras"
)


model.save(

    final_model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("TRANSFER LEARNING COMPLETED")
print("==============================================")


print(

    f"\nFINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"

)


print(
    "\nBest model:"
)

print(
    checkpoint_path
)


print(
    "\nFinal model:"
)

print(
    final_model_path
)


print(
    "\nClassification report:"
)

print(
    report_path
)


print(
    "\nConfusion matrix:"
)

print(
    cm_path
)


print(
    "\nConfusion matrix graph:"
)

print(
    cm_graph_path
)


print(
    "\nAccuracy graph:"
)

print(
    accuracy_graph_path
)


print("\nDONE!")