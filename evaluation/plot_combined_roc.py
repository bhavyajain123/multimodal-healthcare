import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import roc_curve, auc


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits"
)

ML_DIR = os.path.join(
    BASE_DIR,
    "models",
    "ml"
)

DNN_DIR = os.path.join(
    BASE_DIR,
    "deep_learning",
    "models"
)

TRANSFER_DIR = os.path.join(
    BASE_DIR,
    "transfer_learning",
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
    "roc_curves"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {

    "Heart": {

        "prefix": "heart",

        "ml_model": os.path.join(
            ML_DIR,
            "heart_RandomForest.pkl"
        ),

        "dnn_model": os.path.join(
            DNN_DIR,
            "heart_dnn.keras"
        ),

        # ACTUAL HEART TRANSFER MODEL
        "transfer_model": os.path.join(
            TRANSFER_DIR,
            "heart_transfer_base.keras"
        ),
    },


    "Metabolic": {

        "prefix": "metabolic",

        "ml_model": os.path.join(
            ML_DIR,
            "metabolic_XGBoost.pkl"
        ),

        "dnn_model": os.path.join(
            DNN_DIR,
            "metabolic_dnn.keras"
        ),

        "transfer_model": os.path.join(
            TRANSFER_DIR,
            "metabolic_transfer.keras"
        ),
    },


    "Diabetes": {

        "prefix": "diabetes",

        "ml_model": os.path.join(
            ML_DIR,
            "diabetes_XGBoost.pkl"
        ),

        "dnn_model": os.path.join(
            DNN_DIR,
            "diabetes_dnn.keras"
        ),

        "transfer_model": os.path.join(
            TRANSFER_DIR,
            "diabetes_transfer.keras"
        ),
    }
}


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(prefix):

    X_path = os.path.join(
        SPLIT_DIR,
        f"{prefix}_X_test.csv"
    )

    y_path = os.path.join(
        SPLIT_DIR,
        f"{prefix}_y_test.csv"
    )

    print(
        f"Loading X_test: {os.path.basename(X_path)}"
    )

    print(
        f"Loading y_test: {os.path.basename(y_path)}"
    )

    if not os.path.exists(X_path):

        raise FileNotFoundError(
            f"\nX_test not found:\n{X_path}"
        )

    if not os.path.exists(y_path):

        raise FileNotFoundError(
            f"\ny_test not found:\n{y_path}"
        )

    X_test = pd.read_csv(
        X_path
    )

    y_test = pd.read_csv(
        y_path
    ).values.ravel()

    return X_test, y_test


# ============================================================
# ML PREDICTION
# ============================================================

def get_ml_predictions(
    model_path,
    X_test,
    y_test
):

    print(
        f"Loading ML model: "
        f"{os.path.basename(model_path)}"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nML model not found:\n{model_path}"
        )

    model = joblib.load(
        model_path
    )

    # Probability score
    if hasattr(
        model,
        "predict_proba"
    ):

        scores = model.predict_proba(
            X_test
        )[:, 1]

    # Decision score
    elif hasattr(
        model,
        "decision_function"
    ):

        scores = model.decision_function(
            X_test
        )

    else:

        scores = model.predict(
            X_test
        )

    scores = np.asarray(
        scores
    ).reshape(-1)

    return y_test, scores


# ============================================================
# DNN PREDICTION
# ============================================================

def get_dnn_predictions(
    model_path,
    X_test,
    y_test
):

    print(
        f"Loading DNN model: "
        f"{os.path.basename(model_path)}"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nDNN model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    X_values = X_test.values.astype(
        np.float32
    )

    scores = model.predict(
        X_values,
        verbose=0
    )

    scores = np.asarray(
        scores
    ).reshape(-1)

    return y_test, scores


# ============================================================
# TRANSFER LEARNING PREDICTION
# ============================================================

def get_transfer_predictions(
    model_path,
    X_test,
    y_test
):

    print(
        f"Loading Transfer Learning model: "
        f"{os.path.basename(model_path)}"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nTransfer learning model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    X_values = X_test.values.astype(
        np.float32
    )

    scores = model.predict(
        X_values,
        verbose=0
    )

    scores = np.asarray(
        scores
    ).reshape(-1)

    return y_test, scores


# ============================================================
# PLOT ROC
# ============================================================

def plot_roc_curve(
    y_true,
    scores,
    label,
    ax
):

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{label} (AUC = {roc_auc:.3f})"
    )

    return roc_auc


# ============================================================
# GENERATE ROC FOR DATASET
# ============================================================

def generate_dataset_roc(
    dataset_name,
    config
):

    print()
    print("=" * 60)
    print(
        f"Generating ROC curve for {dataset_name}..."
    )
    print("=" * 60)

    prefix = config["prefix"]

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    X_test, y_test = load_test_data(
        prefix
    )

    print(
        f"Test data shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    y_ml, score_ml = get_ml_predictions(
        config["ml_model"],
        X_test,
        y_test
    )

    if dataset_name == "Heart":

        ml_name = "ML - Random Forest"

    else:

        ml_name = "ML - XGBoost"

    ml_auc = plot_roc_curve(
        y_ml,
        score_ml,
        ml_name,
        ax
    )

    print(
        f"ML ROC-AUC: {ml_auc:.4f}"
    )


    # ========================================================
    # DNN
    # ========================================================

    y_dnn, score_dnn = get_dnn_predictions(
        config["dnn_model"],
        X_test,
        y_test
    )

    dnn_auc = plot_roc_curve(
        y_dnn,
        score_dnn,
        "DNN",
        ax
    )

    print(
        f"DNN ROC-AUC: {dnn_auc:.4f}"
    )


    # ========================================================
    # TRANSFER LEARNING
    # ========================================================

    y_transfer, score_transfer = get_transfer_predictions(
        config["transfer_model"],
        X_test,
        y_test
    )

    transfer_auc = plot_roc_curve(
        y_transfer,
        score_transfer,
        "Transfer Learning",
        ax
    )

    print(
        f"Transfer Learning ROC-AUC: "
        f"{transfer_auc:.4f}"
    )


    # ========================================================
    # RANDOM CLASSIFIER
    # ========================================================

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier"
    )


    # ========================================================
    # GRAPH SETTINGS
    # ========================================================

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=12
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=12
    )

    ax.set_title(
        f"{dataset_name} Dataset\n"
        f"ROC Curve: ML vs DNN vs Transfer Learning",
        fontsize=14
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.05
    )

    plt.tight_layout()


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_combined_roc.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"ROC curve saved successfully:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "COMBINED ROC CURVE GENERATION"
    )

    print(
        "=" * 60
    )

    for dataset_name, config in DATASETS.items():

        generate_dataset_roc(
            dataset_name,
            config
        )

    print()
    print(
        "=" * 60
    )

    print(
        "ALL ROC CURVES GENERATED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print()
    print(
        f"Output folder:\n{OUTPUT_DIR}"
    )