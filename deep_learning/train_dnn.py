import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
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
# SETTINGS
# =========================================================

DATASETS = [
    "heart",
    "metabolic",
    "diabetes"
]

BASE_PATH = "data/processed/splits"

MODEL_DIR = "deep_learning/models"
RESULT_DIR = "deep_learning/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# FUNCTION TO TRAIN ONE DATASET
# =========================================================

def train_dnn(dataset):

    print("\n")
    print("=" * 60)
    print(f"TRAINING DNN FOR: {dataset.upper()}")
    print("=" * 60)

    # -----------------------------------------------------
    # File paths
    # -----------------------------------------------------

    X_train_path = f"{BASE_PATH}/{dataset}_X_train.csv"
    X_test_path = f"{BASE_PATH}/{dataset}_X_test.csv"

    y_train_path = f"{BASE_PATH}/{dataset}_y_train.csv"
    y_test_path = f"{BASE_PATH}/{dataset}_y_test.csv"


    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()


    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")


    # -----------------------------------------------------
    # Convert to float32
    # -----------------------------------------------------

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")


    # =====================================================
    # BUILD DNN
    # =====================================================

    model = Sequential([

        Input(shape=(X_train.shape[1],)),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])


    # =====================================================
    # COMPILE MODEL
    # =====================================================

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )


    # =====================================================
    # TRAIN
    # =====================================================

    print("\nTraining started...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save epoch-wise training history
    history_df = pd.DataFrame(history.history)

    history_path = f"{RESULT_DIR}/{dataset}_dnn_training_history.csv"

    history_df.to_csv(history_path, index=False)

    print(f"Training history saved at: {history_path}")

    



    # =====================================================
    # PREDICTION
    # =====================================================

    print("\nGenerating predictions...")

    y_probability = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (
        y_probability >= 0.5
    ).astype(int)

    # =====================================================
    # EVALUATION
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print(f"DNN RESULTS - {dataset.upper()}")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (
        f"{MODEL_DIR}/{dataset}_dnn.keras"
    )

    model.save(model_path)

    print(
        f"\nModel saved at: {model_path}"
    )


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {
        "Dataset": dataset.capitalize(),
        "Model": "DNN",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# =========================================================
# TRAIN ALL DATASETS
# =========================================================

all_results = []


for dataset in DATASETS:

    result = train_dnn(dataset)

    all_results.append(result)


# =========================================================
# SAVE ALL RESULTS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_path = (
    f"{RESULT_DIR}/dnn_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("ALL DNN MODELS TRAINED SUCCESSFULLY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nResults saved at:")
print(results_path)