
from pathlib import Path
import ast
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "deep_learning" / "results"
GRAPHS = RESULTS / "graphs"
ML_DIR = ROOT / "models" / "ml"

GRAPHS.mkdir(parents=True, exist_ok=True)

DATASETS = ["heart", "metabolic", "diabetes"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def load_test_data(dataset):
    split_dir = ROOT / "data" / "processed" / "splits"
    X = pd.read_csv(split_dir / f"{dataset}_X_test.csv")
    y = pd.read_csv(split_dir / f"{dataset}_y_test.csv").values.ravel()
    return X, y


def find_ml_model(dataset):
    candidates = [
        ML_DIR / f"{dataset}_XGBoost.pkl",
        ROOT / "models" / f"{dataset}_XGBoost.pkl",
        ROOT / "models" / "ml" / f"{dataset.lower()}_xgboost.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2:
            return p[:, 1]
        return p.ravel()

    if hasattr(model, "decision_function"):
        return model.decision_function(X).ravel()

    return np.asarray(model.predict(X)).ravel()


def evaluate_model(dataset, name, y_true, y_pred, scores):
    result = {
        "Dataset": dataset.title(),
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": np.nan,
    }

    try:
        if len(np.unique(y_true)) == 2:
            result["ROC-AUC"] = roc_auc_score(y_true, scores)
    except (ValueError, TypeError):
        pass

    return result


def main():
    all_results = []

    # Existing DNN/CNN/GRU models: use saved result CSV metrics.
    result_files = [
        RESULTS / "dnn_results.csv",
        RESULTS / "cnn_gru_results.csv",
    ]

    for file in result_files:
        if not file.exists():
            print(f"WARNING: Missing results file: {file}")
            continue

        df = pd.read_csv(file)
        for _, row in df.iterrows():
            dataset = str(row["Dataset"]).lower()
            name = str(row["Model"]).upper()

            if dataset not in DATASETS:
                continue

            all_results.append({
                "Dataset": dataset.title(),
                "Model": name,
                "Accuracy": row.get("Accuracy", np.nan),
                "Precision": row.get("Precision", np.nan),
                "Recall": row.get("Recall", np.nan),
                "F1": row.get("F1", np.nan),
                "ROC-AUC": row.get("ROC-AUC", np.nan),
            })

    # XGBoost: evaluate the saved pickle against the test split.
    for dataset in DATASETS:
        model_path = find_ml_model(dataset)
        if model_path is None:
            print(f"WARNING: XGBoost model not found for {dataset}")
            continue

        try:
            X_test, y_test = load_test_data(dataset)

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            scores = predict_scores(model, X_test)
            y_pred = (scores >= 0.5).astype(int)

            all_results.append(
                evaluate_model(dataset, "XGBOOST", y_test, y_pred, scores)
            )
            print(f"Evaluated {dataset} XGBoost: {model_path}")

        except Exception as exc:
            print(f"Could not evaluate {dataset} XGBoost: {exc}")

    if not all_results:
        raise RuntimeError("No model metrics found. Check your results CSV files.")

    df = pd.DataFrame(all_results)
    output_csv = RESULTS / "final_model_comparison.csv"
    df.to_csv(output_csv, index=False)

    print("\nFINAL MODEL COMPARISON")
    print(df.round(4).to_string(index=False))
    print(f"\nCSV saved: {output_csv}")

    # One metric-comparison chart per dataset.
    for dataset in DATASETS:
        subset = df[df["Dataset"].str.lower() == dataset].copy()
        if subset.empty:
            continue

        plot_df = subset.set_index("Model")[METRICS]
        ax = plot_df.plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"{dataset.title()} - Model Metric Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right")
        plt.tight_layout()

        path = GRAPHS / f"{dataset}_final_model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graph saved: {path}")

    # Best model per dataset by ROC-AUC (not accuracy).
    valid = df.dropna(subset=["ROC-AUC"])
    if not valid.empty:
        best = valid.loc[valid.groupby("Dataset")["ROC-AUC"].idxmax()]
        best_path = RESULTS / "best_model_by_roc_auc.csv"
        best[["Dataset", "Model", "ROC-AUC", "Accuracy", "Recall", "F1"]].to_csv(
            best_path, index=False
        )
        print("\nBEST MODEL PER DATASET BY ROC-AUC")
        print(best[["Dataset", "Model", "ROC-AUC", "Accuracy", "Recall", "F1"]]
              .round(4).to_string(index=False))
        print(f"\nBest-model table saved: {best_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
