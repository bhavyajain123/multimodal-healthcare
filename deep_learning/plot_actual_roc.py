
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "deep_learning" / "models"
GRAPH_DIR = ROOT / "deep_learning" / "results" / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["heart", "metabolic", "diabetes"]
MODEL_TYPES = ["dnn", "cnn1d", "gru"]


def main():
    summary = []

    for dataset in DATASETS:
        print(f"\nProcessing {dataset.upper()}...")

        X_test = pd.read_csv(
            DATA_DIR / f"{dataset}_X_test.csv"
        ).to_numpy(dtype=np.float32)

        y_test = pd.read_csv(
            DATA_DIR / f"{dataset}_y_test.csv"
        ).values.ravel().astype(int)

        plt.figure(figsize=(8, 6))
        successful_models = 0

        for model_type in MODEL_TYPES:
            model_path = (
                MODEL_DIR / f"{dataset}_{model_type}.keras"
            )

            if not model_path.exists():
                print(f"Missing model: {model_path.name}")
                continue

            model = load_model(model_path, compile=False)

            # CNN1D and GRU require a channel dimension.
            if model_type in ("cnn1d", "gru"):
                X_input = X_test[..., np.newaxis]
            else:
                X_input = X_test

            probabilities = model.predict(
                X_input, verbose=0
            ).ravel()

            if len(probabilities) != len(y_test):
                raise ValueError(
                    f"Prediction count mismatch: "
                    f"{dataset} / {model_type}"
                )

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Invalid prediction values: "
                    f"{dataset} / {model_type}"
                )

            auc_value = roc_auc_score(
                y_test, probabilities
            )

            fpr, tpr, _ = roc_curve(
                y_test, probabilities
            )

            plt.plot(
                fpr,
                tpr,
                label=f"{model_type.upper()} (AUC={auc_value:.3f})"
            )

            summary.append({
                "Dataset": dataset,
                "Model": model_type.upper(),
                "ROC-AUC": auc_value
            })

            successful_models += 1
            print(
                f"{model_type.upper()} ROC-AUC: "
                f"{auc_value:.4f}"
            )

        if successful_models == 0:
            plt.close()
            continue

        plt.plot(
            [0, 1], [0, 1],
            linestyle="--",
            label="Random classifier"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset.title()} - Actual ROC Curves")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = (
            GRAPH_DIR / f"{dataset}_actual_roc_curves.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved graph: {output_path}")

    if summary:
        results_path = GRAPH_DIR / "actual_roc_auc.csv"
        pd.DataFrame(summary).to_csv(
            results_path, index=False
        )
        print(f"\nROC summary saved: {results_path}")

    print("\nROC curve generation finished.")


if __name__ == "__main__":
    main()
