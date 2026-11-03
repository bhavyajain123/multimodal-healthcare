
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "deep_learning" / "models"
OUT_DIR = ROOT / "deep_learning" / "results" / "fusion_runs"
FUSION_MODEL_DIR = MODEL_DIR / "fusion_runs"
GRAPH_DIR = OUT_DIR / "graphs"

for folder in (OUT_DIR, FUSION_MODEL_DIR, GRAPH_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATASETS = ["heart", "metabolic", "diabetes"]


def load_data(dataset):
    X_train = pd.read_csv(DATA_DIR / f"{dataset}_X_train.csv")
    X_test = pd.read_csv(DATA_DIR / f"{dataset}_X_test.csv")
    y_train = pd.read_csv(DATA_DIR / f"{dataset}_y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / f"{dataset}_y_test.csv").values.ravel()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"{dataset}: train/test columns do not match")

    if len(X_train.columns) != len(X_test.columns):
        raise ValueError(f"{dataset}: feature count mismatch")

    return (
        X_train.to_numpy(dtype=np.float32),
        y_train.astype(np.float32),
        X_test.to_numpy(dtype=np.float32),
        y_test.astype(np.int32),
        list(X_train.columns),
    )


def load_feature_extractors(dataset, n_features):
    cnn_path = MODEL_DIR / f"{dataset}_cnn1d.keras"
    gru_path = MODEL_DIR / f"{dataset}_gru.keras"

    if not cnn_path.exists() or not gru_path.exists():
        raise FileNotFoundError(
            f"Required CNN/GRU model missing for {dataset}"
        )

    cnn = tf.keras.models.load_model(cnn_path, compile=False)
    gru = tf.keras.models.load_model(gru_path, compile=False)

    expected_shape = (None, n_features, 1)
    if tuple(cnn.input_shape) != expected_shape:
        raise ValueError(
            f"CNN input mismatch for {dataset}: {cnn.input_shape}; "
            f"expected {expected_shape}"
        )
    if tuple(gru.input_shape) != expected_shape:
        raise ValueError(
            f"GRU input mismatch for {dataset}: {gru.input_shape}; "
            f"expected {expected_shape}"
        )

    # Existing CNN: Dense layer before Dropout and output.
    cnn_features = Model(
        inputs=cnn.inputs,
        outputs=cnn.get_layer("dense").output,
        name=f"{dataset}_cnn_feature_extractor",
    )

    # Existing GRU: output of its recurrent layer.
    gru_features = Model(
        inputs=gru.inputs,
        outputs=gru.get_layer("gru").output,
        name=f"{dataset}_gru_feature_extractor",
    )

    # Feature extractors are used only for inference.
    cnn.trainable = False
    gru.trainable = False

    return cnn_features, gru_features


def extract_features(cnn_features, gru_features, X, dataset, split):
    X_seq = X[..., np.newaxis]

    cnn_out = cnn_features.predict(
        X_seq, batch_size=512, verbose=0
    )
    gru_out = gru_features.predict(
        X_seq, batch_size=512, verbose=0
    )

    fused = np.concatenate([cnn_out, gru_out], axis=1)

    print(
        f"{dataset} {split}: CNN features {cnn_out.shape}, "
        f"GRU features {gru_out.shape}, fused {fused.shape}"
    )

    return fused.astype(np.float32)


def build_fusion_classifier(n_features):
    model = Sequential([
        Input(shape=(n_features,), name="fused_features"),
        Dense(32, activation="relu", name="fusion_dense"),
        Dropout(0.2, name="fusion_dropout"),
        Dense(16, activation="relu", name="fusion_dense_2"),
        Dense(1, activation="sigmoid", name="fusion_prediction"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history, dataset):
    hist = pd.DataFrame(history.history)
    hist.to_csv(
        OUT_DIR / f"{dataset}_fusion_history.csv",
        index_label="epoch",
    )

    epochs = range(1, len(hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["accuracy"], label="Training")
    axes[0].plot(epochs, hist["val_accuracy"], label="Validation")
    axes[0].set_title(f"{dataset.title()} Fusion Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist["loss"], label="Training")
    axes[1].plot(epochs, hist["val_loss"], label="Validation")
    axes[1].set_title(f"{dataset.title()} Fusion Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        GRAPH_DIR / f"{dataset}_fusion_training_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_evaluation_graphs(dataset, y_true, scores, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{dataset.title()} CNN-GRU Fusion Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / f"{dataset}_fusion_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"Fusion ROC-AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", label="Random baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset.title()} CNN-GRU Fusion ROC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            GRAPH_DIR / f"{dataset}_fusion_roc.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def train_dataset(dataset):
    print(f"\n{'=' * 60}\nDATASET: {dataset.upper()}\n{'=' * 60}")

    X_train, y_train, X_test, y_test, columns = load_data(dataset)

    cnn_features, gru_features = load_feature_extractors(
        dataset, X_train.shape[1]
    )

    # Training split is used to train the new fusion classifier.
    # Test split is reserved for final evaluation.
    F_train = extract_features(
        cnn_features, gru_features, X_train, dataset, "train"
    )
    F_test = extract_features(
        cnn_features, gru_features, X_test, dataset, "test"
    )

    fusion = build_fusion_classifier(F_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = fusion.fit(
        F_train,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=256 if dataset == "diabetes" else 32,
        callbacks=[early_stop],
        verbose=1,
    )

    scores = fusion.predict(F_test, batch_size=512, verbose=0).ravel()
    y_pred = (scores >= 0.5).astype(int)

    result = {
        "Dataset": dataset,
        "Model": "CNN-GRU Fusion",
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": (
            float(roc_auc_score(y_test, scores))
            if len(np.unique(y_test)) == 2 else None
        ),
        "Threshold": 0.5,
        "Train_samples": int(len(y_train)),
        "Test_samples": int(len(y_test)),
        "Feature_count": int(len(columns)),
    }

    print("\nTEST METRICS")
    print(json.dumps(result, indent=2))

    # All outputs go into new fusion-specific folders.
    model_path = FUSION_MODEL_DIR / f"{dataset}_cnn_gru_fusion.keras"
    fusion.save(model_path)

    save_training_curves(history, dataset)
    save_evaluation_graphs(dataset, y_test, scores, y_pred)

    with open(OUT_DIR / f"{dataset}_fusion_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    del fusion, history, cnn_features, gru_features
    tf.keras.backend.clear_session()

    return result


def main():
    results = []

    for dataset in DATASETS:
        try:
            results.append(train_dataset(dataset))
        except Exception as exc:
            print(f"\nERROR for {dataset}: {type(exc).__name__}: {exc}")
            raise

    results_df = pd.DataFrame(results)
    csv_path = OUT_DIR / "cnn_gru_fusion_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    print("\nFINAL CNN-GRU FUSION RESULTS")
    print(results_df.round(4).to_string(index=False))
    print(f"\nResults saved: {csv_path}")
    print(f"Graphs saved in: {GRAPH_DIR}")
    print(f"Fusion heads saved in: {FUSION_MODEL_DIR}")


if __name__ == "__main__":
    main()
