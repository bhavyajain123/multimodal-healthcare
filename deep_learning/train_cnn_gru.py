
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, GRU
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "deep_learning" / "models"
RESULT_DIR = ROOT / "deep_learning" / "results"

DATASETS = ["heart", "metabolic", "diabetes"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_cnn(n_features):
    model = Sequential([
        Input(shape=(n_features, 1)),
        Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ], name="CNN1D")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru(n_features):
    model = Sequential([
        Input(shape=(n_features, 1)),
        GRU(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], name="GRU")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def evaluate_model(model, X_test, y_test, dataset, model_name):
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Dataset": dataset,
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Confusion_Matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }


def train_one(dataset, model_name):
    print(f"\n{'=' * 55}")
    print(f"Dataset: {dataset.upper()} | Model: {model_name}")
    print("=" * 55)

    X_train = pd.read_csv(DATA_DIR / f"{dataset}_X_train.csv")
    X_test = pd.read_csv(DATA_DIR / f"{dataset}_X_test.csv")
    y_train = pd.read_csv(DATA_DIR / f"{dataset}_y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / f"{dataset}_y_test.csv").values.ravel()

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(f"{dataset}: expected binary labels 0 and 1")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"{dataset}: train/test feature columns differ")

    # The existing split files are used without refitting a scaler.
    # Add a channel dimension for CNN/GRU.
    X_train = X_train.to_numpy(dtype=np.float32)[..., np.newaxis]
    X_test = X_test.to_numpy(dtype=np.float32)[..., np.newaxis]
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    tf.keras.backend.clear_session()

    model = (
        build_cnn(X_train.shape[1])
        if model_name == "CNN1D"
        else build_gru(X_train.shape[1])
    )

    model_path = MODEL_DIR / f"{dataset}_{model_name.lower()}.keras"

    # Do not overwrite a previously trained model accidentally.
    if model_path.exists():
        raise FileExistsError(
            f"{model_path} already exists. Rename or back it up first."
        )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=256 if len(X_train) > 5000 else 32,
        callbacks=[early_stop],
        verbose=1
    )

    result = evaluate_model(
        model, X_test, y_test, dataset, model_name
    )

    model.save(model_path)

    print("\nEvaluation:")
    for key, value in result.items():
        if key not in ("Dataset", "Model", "Confusion_Matrix"):
            print(f"{key}: {value:.4f}")
    print("Confusion matrix:", result["Confusion_Matrix"])
    print("Saved:", model_path)

    return result


def main():
    results = []

    for dataset in DATASETS:
        for model_name in ["CNN1D", "GRU"]:
            results.append(train_one(dataset, model_name))

    results_path = RESULT_DIR / "cnn_gru_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    print("\nAll six models trained successfully.")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nResults saved to:", results_path)


if __name__ == "__main__":
    main()
