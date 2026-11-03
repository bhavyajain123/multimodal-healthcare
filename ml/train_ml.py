import pandas as pd
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"

MODEL_DIR = BASE_DIR / "models" / "ml"

EVALUATION_DIR = BASE_DIR / "evaluation" / "ml_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name):

    X_train = pd.read_csv(
        DATA_DIR / f"{dataset_name}_X_train.csv"
    )

    X_test = pd.read_csv(
        DATA_DIR / f"{dataset_name}_X_test.csv"
    )

    y_train = pd.read_csv(
        DATA_DIR / f"{dataset_name}_y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        DATA_DIR / f"{dataset_name}_y_test.csv"
    ).squeeze()

    return X_train, X_test, y_train, y_test


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    dataset_name
):

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Get scores for ROC-AUC
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):

        probabilities = model.decision_function(X_test)

    else:

        probabilities = predictions

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"{dataset_name.upper()} - {model_name}"
    )

    print("-" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Dataset": dataset_name,

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc
    }


# ============================================================
# TRAIN DIABETES MODELS
# ============================================================

def train_diabetes_models():

    dataset_name = "diabetes"

    print("\n")
    print("=" * 70)
    print("DATASET: DIABETES")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name
    )

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Testing data :",
        X_test.shape
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        # ----------------------------------------------------
        # Fast Linear SVM
        # ----------------------------------------------------

        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    # ========================================================
    # TRAIN
    # ========================================================

    for model_name, model in models.items():

        print("\n")
        print(
            f"Training {model_name}..."
        )

        # Train model
        model.fit(
            X_train,
            y_train
        )

        # Evaluate model
        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name,
            dataset_name
        )

        results.append(result)

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR /
            f"{dataset_name}_{model_name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Model saved: {model_path}"
        )

    return results


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_previous_results():

    results_path = (
        EVALUATION_DIR /
        "ml_results.csv"
    )

    if results_path.exists():

        print("\n")
        print(
            "Previous ML results found."
        )

        previous_results = pd.read_csv(
            results_path
        )

        # Remove old diabetes rows
        # because we are recalculating diabetes
        previous_results = previous_results[
            previous_results["Dataset"] != "diabetes"
        ]

        return previous_results

    else:

        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 70)
print("DIABETES ML TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# Load Heart + Metabolic previous results
# ------------------------------------------------------------

previous_results = load_previous_results()

print("\nPrevious results preserved:")
print(
    previous_results[
        ["Dataset", "Model"]
    ].to_string(index=False)
    if not previous_results.empty
    else "No previous results found."
)


# ------------------------------------------------------------
# Train Diabetes models
# ------------------------------------------------------------

diabetes_results = train_diabetes_models()

diabetes_results_df = pd.DataFrame(
    diabetes_results
)


# ------------------------------------------------------------
# Combine results
# ------------------------------------------------------------

if not previous_results.empty:

    final_results = pd.concat(
        [
            previous_results,
            diabetes_results_df
        ],
        ignore_index=True
    )

else:

    final_results = diabetes_results_df


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_path = (
    EVALUATION_DIR /
    "ml_results.csv"
)

final_results.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ML RESULTS")
print("=" * 70)

print(
    final_results.to_string(
        index=False
    )
)

print("\n")
print(
    "Results saved at:"
)

print(
    results_path
)

print("\n")
print(
    "ML TRAINING COMPLETED SUCCESSFULLY."
)