
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, GRU
)
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "splits"
RESULT_DIR = ROOT / "deep_learning" / "results"
HISTORY_DIR = RESULT_DIR / "training_history"
GRAPH_DIR = RESULT_DIR / "graphs"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["heart", "metabolic", "diabetes"]
MODELS = ["DNN", "CNN1D", "GRU"]


def load_data(dataset):
    X_train = pd.read_csv(DATA_DIR / f"{dataset}_X_train.csv")
    y_train = pd.read_csv(DATA_DIR / f"{dataset}_y_train.csv").values.ravel()

    X_test = pd.read_csv(DATA_DIR / f"{dataset}_X_test.csv")
    y_test = pd.read_csv(DATA_DIR / f"{dataset}_y_test.csv").values.ravel()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"{dataset}: train/test columns do not match")

    return (
        X_train.to_numpy(dtype="float32"),
        y_train.astype("float32"),
        X_test.to_numpy(dtype="float32"),
        y_test.astype("float32"),
    )


def build_model(name, n_features):
    if name == "DNN":
        model = Sequential([
            Input(shape=(n_features,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])

    elif name == "CNN1D":
        model = Sequential([
            Input(shape=(n_features, 1)),
            Conv1D(32, 3, activation="relu", padding="same"),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])

    else:
        model = Sequential([
            Input(shape=(n_features, 1)),
            GRU(32),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, dataset, model_name):
    hist = pd.DataFrame(history.history)
    csv_path = HISTORY_DIR / f"{dataset}_{model_name.lower()}_history.csv"
    hist.to_csv(csv_path, index_label="epoch")

    epochs = range(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["accuracy"], label="Training Accuracy")
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{dataset.title()} - {model_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist["loss"], label="Training Loss")
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{dataset.title()} - {model_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    graph_path = GRAPH_DIR / f"{dataset}_{model_name.lower()}_training_curves.png"
    fig.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"History CSV: {csv_path}")
    print(f"Training graph: {graph_path}")


def main():
    for dataset in DATASETS:
        print(f"\n{'=' * 55}\nDataset: {dataset.upper()}\n{'=' * 55}")

        X_train, y_train, X_test, y_test = load_data(dataset)

        for model_name in MODELS:
            print(f"\nTraining {dataset} - {model_name}")

            X_fit = X_train
            X_eval = X_test

            if model_name in ("CNN1D", "GRU"):
                X_fit = X_fit[..., None]
                X_eval = X_eval[..., None]

            model = build_model(model_name, X_train.shape[1])

            patience = 10 if model_name == "DNN" else 7
            max_epochs = 100 if model_name == "DNN" else 40
            batch_size = 32 if dataset != "diabetes" else 256

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )

            history = model.fit(
                X_fit,
                y_train,
                validation_split=0.2,
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=1,
            )

            test_loss, test_accuracy = model.evaluate(
                X_eval, y_test, verbose=0
            )
            print(
                f"Test accuracy: {test_accuracy:.4f} | "
                f"Test loss: {test_loss:.4f}"
            )

            plot_history(history, dataset, model_name)

            # Do not save the model: existing trained models stay untouched.
            del model, history
            tf.keras.backend.clear_session()

    print("\nAll training-history graphs and CSV files generated.")


if __name__ == "__main__":
    main()
