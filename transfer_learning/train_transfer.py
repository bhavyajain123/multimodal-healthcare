import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# =========================================================
# PATHS
# =========================================================

BASE_PATH = "data/processed/splits"

MODEL_PATH = "transfer_learning/models"
RESULT_PATH = "transfer_learning/results"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(dataset_name):

    X_train = pd.read_csv(
        f"{BASE_PATH}/{dataset_name}_X_train.csv"
    )

    X_test = pd.read_csv(
        f"{BASE_PATH}/{dataset_name}_X_test.csv"
    )

    y_train = pd.read_csv(
        f"{BASE_PATH}/{dataset_name}_y_train.csv"
    ).values.ravel()

    y_test = pd.read_csv(
        f"{BASE_PATH}/{dataset_name}_y_test.csv"
    ).values.ravel()

    return (
        X_train.values.astype("float32"),
        X_test.values.astype("float32"),
        y_train.astype("float32"),
        y_test.astype("float32")
    )


# =========================================================
# CREATE TRANSFER LEARNING MODEL
# =========================================================

def create_model(input_features):

    inputs = Input(shape=(input_features,), name="input")

    # Dataset-specific adapter
    adapter = Dense(
        32,
        activation="relu",
        name="adapter"
    )(inputs)

    adapter = Dropout(
        0.3,
        name="adapter_dropout"
    )(adapter)

    # Shared feature extractor
    feature_1 = Dense(
        32,
        activation="relu",
        name="feature_layer_1"
    )(adapter)

    feature_2 = Dense(
        16,
        activation="relu",
        name="feature_layer_2"
    )(feature_1)

    # Classification head
    classifier = Dense(
        16,
        activation="relu",
        name="classifier"
    )(feature_2)

    outputs = Dense(
        1,
        activation="sigmoid",
        name="output"
    )(classifier)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


# =========================================================
# METRICS
# =========================================================

def evaluate_model(model, X_test, y_test, dataset_name):

    probabilities = model.predict(
        X_test,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n------------------------------------------")
    print(f"{dataset_name} Transfer Learning Results")
    print("------------------------------------------")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Dataset": dataset_name,
        "Model": "Transfer Learning",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# =========================================================
# COMPILE MODEL
# =========================================================

def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


# =========================================================
# EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# =========================================================
# START
# =========================================================

print("\n==========================================")
print("TRANSFER LEARNING")
print("==========================================")


# =========================================================
# STEP 1: PRE-TRAIN ON HEART
# =========================================================

print("\n")
print("==========================================")
print("STEP 1: PRE-TRAINING ON HEART")
print("==========================================")


X_train_h, X_test_h, y_train_h, y_test_h = load_dataset(
    "heart"
)

print(f"Heart Train Shape: {X_train_h.shape}")
print(f"Heart Test Shape : {X_test_h.shape}")


heart_model = create_model(
    X_train_h.shape[1]
)

compile_model(
    heart_model
)


heart_model.fit(
    X_train_h,
    y_train_h,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# Evaluate Heart
heart_result = evaluate_model(
    heart_model,
    X_test_h,
    y_test_h,
    "Heart"
)


# =========================================================
# SAVE HEART MODEL
# =========================================================

heart_model.save(
    f"{MODEL_PATH}/heart_transfer_base.keras"
)

print(
    "\nHeart pre-trained model saved."
)


# =========================================================
# EXTRACT SHARED FEATURE WEIGHTS
# =========================================================

feature_weights_1 = (
    heart_model
    .get_layer("feature_layer_1")
    .get_weights()
)

feature_weights_2 = (
    heart_model
    .get_layer("feature_layer_2")
    .get_weights()
)


print(
    "\nShared feature extractor weights "
    "successfully extracted."
)


# =========================================================
# FUNCTION FOR TRANSFER LEARNING
# =========================================================

def transfer_to_dataset(
    dataset_name,
    feature_weights_1,
    feature_weights_2
):

    print("\n")
    print("==========================================")
    print(
        f"STEP: TRANSFER TO {dataset_name.upper()}"
    )
    print("==========================================")


    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name
    )

    print(
        f"{dataset_name} Train Shape: "
        f"{X_train.shape}"
    )

    print(
        f"{dataset_name} Test Shape : "
        f"{X_test.shape}"
    )


    # -----------------------------------------------------
    # Create new model for target dataset
    # -----------------------------------------------------

    model = create_model(
        X_train.shape[1]
    )


    # -----------------------------------------------------
    # Transfer pre-trained feature extractor
    # -----------------------------------------------------

    model.get_layer(
        "feature_layer_1"
    ).set_weights(
        feature_weights_1
    )

    model.get_layer(
        "feature_layer_2"
    ).set_weights(
        feature_weights_2
    )


    print(
        "\nPre-trained feature extractor "
        "transferred successfully."
    )


    # -----------------------------------------------------
    # Freeze transferred layers
    # -----------------------------------------------------

    model.get_layer(
        "feature_layer_1"
    ).trainable = False

    model.get_layer(
        "feature_layer_2"
    ).trainable = False


    print(
        "Transferred layers frozen."
    )


    # -----------------------------------------------------
    # Compile
    # -----------------------------------------------------

    compile_model(
        model
    )


    # -----------------------------------------------------
    # Phase 1: Train new adapter + classifier
    # -----------------------------------------------------

    print(
        "\nPhase 1: Training target-specific layers..."
    )


    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )


    # -----------------------------------------------------
    # Phase 2: Fine-tuning
    # -----------------------------------------------------

    print(
        "\nPhase 2: Fine-tuning transferred layers..."
    )


    model.get_layer(
        "feature_layer_1"
    ).trainable = True

    model.get_layer(
        "feature_layer_2"
    ).trainable = True


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )


    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    result = evaluate_model(
        model,
        X_test,
        y_test,
        dataset_name
    )


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model.save(
        f"{MODEL_PATH}/{dataset_name.lower()}_transfer.keras"
    )


    print(
        f"\n{dataset_name} transfer model saved."
    )


    return result


# =========================================================
# STEP 2: HEART → METABOLIC
# =========================================================

metabolic_result = transfer_to_dataset(
    "metabolic",
    feature_weights_1,
    feature_weights_2
)


# =========================================================
# STEP 3: HEART → DIABETES
# =========================================================

diabetes_result = transfer_to_dataset(
    "diabetes",
    feature_weights_1,
    feature_weights_2
)


# =========================================================
# SAVE ALL RESULTS
# =========================================================

results = pd.DataFrame([
    heart_result,
    metabolic_result,
    diabetes_result
])


results_path = (
    f"{RESULT_PATH}/transfer_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("==========================================")
print("TRANSFER LEARNING SUMMARY")
print("==========================================\n")

print(
    results.to_string(
        index=False
    )
)


print("\n==========================================")
print("TRANSFER LEARNING COMPLETED")
print("==========================================")

print(
    f"\nResults saved at:"
    f"\n{results_path}"
)

print(
    f"\nModels saved at:"
    f"\n{MODEL_PATH}"
)