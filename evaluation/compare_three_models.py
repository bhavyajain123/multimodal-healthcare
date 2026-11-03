import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# =========================================================
# PATHS
# =========================================================

ML_PATH = "evaluation/ml_results/ml_results.csv"
DNN_PATH = "deep_learning/results/dnn_results.csv"
TL_PATH = "transfer_learning/results/transfer_results.csv"

OUTPUT_DIR = "evaluation/three_model_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD RESULTS
# =========================================================

ml_df = pd.read_csv(ML_PATH)
dnn_df = pd.read_csv(DNN_PATH)
tl_df = pd.read_csv(TL_PATH)


# =========================================================
# STANDARDIZE DATASET NAMES
# =========================================================

def standardize_dataset(name):

    name = str(name).strip().lower()

    if name == "heart":
        return "Heart"

    if name == "metabolic":
        return "Metabolic"

    if name == "diabetes":
        return "Diabetes"

    return name.title()


ml_df["Dataset"] = ml_df["Dataset"].apply(
    standardize_dataset
)

dnn_df["Dataset"] = dnn_df["Dataset"].apply(
    standardize_dataset
)

tl_df["Dataset"] = tl_df["Dataset"].apply(
    standardize_dataset
)


# =========================================================
# FIND BEST ML MODEL
# =========================================================

best_ml = (
    ml_df
    .sort_values(
        ["Dataset", "ROC-AUC"],
        ascending=[True, False]
    )
    .groupby("Dataset")
    .first()
    .reset_index()
)


print("\n==========================================")
print("BEST ML MODEL FOR EACH DATASET")
print("==========================================\n")

print(
    best_ml[
        [
            "Dataset",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)


# =========================================================
# PREPARE ML
# =========================================================

ml_compare = best_ml[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
].copy()

ml_compare["Approach"] = "ML"


# =========================================================
# PREPARE DNN
# =========================================================

dnn_compare = dnn_df[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
].copy()

dnn_compare["Approach"] = "DNN"


# =========================================================
# PREPARE TRANSFER LEARNING
# =========================================================

tl_compare = tl_df[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
].copy()

tl_compare["Approach"] = "Transfer Learning"


# =========================================================
# COMBINE
# =========================================================

combined_df = pd.concat(
    [
        ml_compare,
        dnn_compare,
        tl_compare
    ],
    ignore_index=True
)


# =========================================================
# COLUMN ORDER
# =========================================================

combined_df = combined_df[
    [
        "Dataset",
        "Approach",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
]


# =========================================================
# SAVE COMBINED CSV
# =========================================================

combined_csv = (
    f"{OUTPUT_DIR}/three_model_comparison.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


# =========================================================
# PRINT COMPARISON
# =========================================================

print("\n==========================================")
print("ML vs DNN vs TRANSFER LEARNING")
print("==========================================\n")

print(
    combined_df.to_string(index=False)
)


# =========================================================
# METRICS
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]


# =========================================================
# DATASET-WISE GRAPHS
# =========================================================

datasets = [
    "Heart",
    "Metabolic",
    "Diabetes"
]


for dataset in datasets:

    data = combined_df[
        combined_df["Dataset"] == dataset
    ].copy()


    # -----------------------------------------------------
    # Force exact approach order
    # -----------------------------------------------------

    approach_order = [
        "ML",
        "DNN",
        "Transfer Learning"
    ]


    data["Approach"] = pd.Categorical(
        data["Approach"],
        categories=approach_order,
        ordered=True
    )


    data = data.sort_values(
        "Approach"
    )


    # -----------------------------------------------------
    # Check
    # -----------------------------------------------------

    if len(data) != 3:

        print(
            f"\nWARNING: {dataset} does not have "
            f"exactly 3 approaches."
        )

        continue


    # -----------------------------------------------------
    # Values
    # -----------------------------------------------------

    ml_values = data[
        data["Approach"] == "ML"
    ][metrics].iloc[0].values


    dnn_values = data[
        data["Approach"] == "DNN"
    ][metrics].iloc[0].values


    tl_values = data[
        data["Approach"] == "Transfer Learning"
    ][metrics].iloc[0].values


    # -----------------------------------------------------
    # X positions
    # -----------------------------------------------------

    x = np.arange(
        len(metrics)
    )

    width = 0.25


    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    plt.figure(
        figsize=(13, 7)
    )


    # -----------------------------------------------------
    # Bars
    # -----------------------------------------------------

    bars_ml = plt.bar(
        x - width,
        ml_values,
        width,
        label="ML"
    )


    bars_dnn = plt.bar(
        x,
        dnn_values,
        width,
        label="DNN"
    )


    bars_tl = plt.bar(
        x + width,
        tl_values,
        width,
        label="Transfer Learning"
    )


    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    plt.xticks(
        x,
        metrics,
        fontsize=11
    )

    plt.ylabel(
        "Score",
        fontsize=12
    )

    plt.xlabel(
        "Evaluation Metrics",
        fontsize=12
    )

    plt.title(
        f"{dataset}: ML vs DNN vs Transfer Learning",
        fontsize=16,
        fontweight="bold"
    )

    plt.ylim(
        0,
        1
    )


    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    plt.legend(
        fontsize=11
    )


    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------

    plt.grid(
        axis="y",
        alpha=0.3
    )


    # -----------------------------------------------------
    # Value labels
    # -----------------------------------------------------

    for bars in [
        bars_ml,
        bars_dnn,
        bars_tl
    ]:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )


    plt.tight_layout()


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    filename = (
        f"{dataset.lower()}_"
        f"ml_dnn_transfer_comparison.png"
    )


    plt.savefig(
        f"{OUTPUT_DIR}/{filename}",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"\n{dataset} graph saved."
    )


# =========================================================
# OVERALL AVERAGE
# =========================================================

average_df = (
    combined_df
    .groupby("Approach", as_index=False)[metrics]
    .mean()
)


# Force order

approach_order = [
    "ML",
    "DNN",
    "Transfer Learning"
]


average_df["Approach"] = pd.Categorical(
    average_df["Approach"],
    categories=approach_order,
    ordered=True
)


average_df = average_df.sort_values(
    "Approach"
)


# =========================================================
# PRINT OVERALL AVERAGE
# =========================================================

print("\n==========================================")
print("OVERALL AVERAGE PERFORMANCE")
print("==========================================\n")

print(
    average_df.to_string(index=False)
)


# =========================================================
# SAVE OVERALL CSV
# =========================================================

average_csv = (
    f"{OUTPUT_DIR}/overall_average_comparison.csv"
)

average_df.to_csv(
    average_csv,
    index=False
)


# =========================================================
# OVERALL GRAPH
# =========================================================

x = np.arange(
    len(metrics)
)

width = 0.25


plt.figure(
    figsize=(13, 7)
)


bars_ml = plt.bar(
    x - width,
    average_df.iloc[0][metrics].astype(float).values,
    width,
    label="ML"
)


bars_dnn = plt.bar(
    x,
    average_df.iloc[1][metrics].astype(float).values,
    width,
    label="DNN"
)


bars_tl = plt.bar(
    x + width,
    average_df.iloc[2][metrics].astype(float).values,
    width,
    label="Transfer Learning"
)


plt.xticks(
    x,
    metrics,
    fontsize=11
)

plt.ylabel(
    "Average Score",
    fontsize=12
)

plt.xlabel(
    "Evaluation Metrics",
    fontsize=12
)

plt.title(
    "Overall Comparison: ML vs DNN vs Transfer Learning",
    fontsize=16,
    fontweight="bold"
)

plt.ylim(
    0,
    1
)

plt.legend(
    fontsize=11
)

plt.grid(
    axis="y",
    alpha=0.3
)


# =========================================================
# OVERALL VALUE LABELS
# =========================================================

for bars in [
    bars_ml,
    bars_dnn,
    bars_tl
]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


plt.tight_layout()


plt.savefig(
    f"{OUTPUT_DIR}/overall_ml_dnn_transfer_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# DONE
# =========================================================

print("\n==========================================")
print("THREE-MODEL COMPARISON COMPLETED")
print("==========================================")

print(
    f"\nCombined CSV:"
    f"\n{combined_csv}"
)

print(
    f"\nOverall CSV:"
    f"\n{average_csv}"
)

print(
    f"\nGraphs:"
    f"\n{OUTPUT_DIR}"
)