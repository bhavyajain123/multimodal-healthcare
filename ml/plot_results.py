import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
file_path = "evaluation/ml_results/ml_results.csv"

df = pd.read_csv(file_path)

# Create output directory
output_dir = "evaluation/ml_results/plots"
os.makedirs(output_dir, exist_ok=True)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

# ==============================
# 1. Overall Model Comparison
# ==============================

for metric in metrics:

    plt.figure(figsize=(10, 6))

    for dataset in df["Dataset"].unique():

        data = df[df["Dataset"] == dataset]

        plt.plot(
            data["Model"],
            data[metric],
            marker="o",
            label=dataset
        )

    plt.title(f"{metric} Comparison")
    plt.xlabel("Machine Learning Model")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xticks(rotation=20)

    plt.tight_layout()

    filename = f"{metric.lower().replace('-', '_')}_comparison.png"

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300
    )

    plt.show()


# =================================
# 2. Dataset-wise Model Comparison
# =================================

for dataset in df["Dataset"].unique():

    data = df[df["Dataset"] == dataset]

    ax = data.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(f"{dataset} Dataset - Model Performance")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    plt.xticks(rotation=20)

    plt.legend(
        title="Metrics",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"{dataset.lower()}_model_comparison.png"
    )

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300
    )

    plt.show()

print("All graphs generated successfully!")
print(f"Graphs saved in: {output_dir}")