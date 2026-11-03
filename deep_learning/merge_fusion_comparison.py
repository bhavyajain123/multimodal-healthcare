
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "deep_learning" / "results"
FUSION_DIR = RESULTS / "fusion_runs"
GRAPHS = RESULTS / "graphs"

base_path = RESULTS / "final_model_comparison.csv"
fusion_path = FUSION_DIR / "cnn_gru_fusion_comparison.csv"

if not base_path.exists():
    raise FileNotFoundError(f"Missing file: {base_path}")

if not fusion_path.exists():
    raise FileNotFoundError(f"Missing file: {fusion_path}")

base = pd.read_csv(base_path)
fusion = pd.read_csv(fusion_path)

# Match columns without changing the original CSV.
fusion = fusion.rename(columns={
    "Dataset": "Dataset",
    "Model": "Model",
    "F1": "F1",
    "ROC-AUC": "ROC-AUC",
})

columns = ["Dataset", "Model", "Accuracy", "Precision",
           "Recall", "F1", "ROC-AUC"]

base = base[columns]
fusion = fusion[columns]

combined = pd.concat([base, fusion], ignore_index=True)

# Save to a new file so the original comparison remains unchanged.
out_path = RESULTS / "all_models_with_fusion_comparison.csv"
combined.to_csv(out_path, index=False)

print("\nALL MODELS INCLUDING CNN-GRU FUSION")
print(combined.round(4).to_string(index=False))
print(f"\nSaved: {out_path}")

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

for dataset in ["Heart", "Metabolic", "Diabetes"]:
    subset = combined[
        combined["Dataset"].str.lower() == dataset.lower()
    ].copy()

    if subset.empty:
        continue

    ax = subset.set_index("Model")[metrics].plot(
        kind="bar", figsize=(12, 6)
    )
    ax.set_title(f"{dataset} - All Models Including CNN-GRU Fusion")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output = GRAPHS / f"{dataset.lower()}_all_models_with_fusion.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Graph saved: {output}")
