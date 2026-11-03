from pathlib import Path
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "deep_learning" / "results"
OUT_DIR = RESULT_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cnn_gru_path = RESULT_DIR / "cnn_gru_results.csv"
dnn_path = RESULT_DIR / "dnn_results.csv"

if not cnn_gru_path.exists():
    raise FileNotFoundError(f"Missing results file: {cnn_gru_path}")

cnn_gru = pd.read_csv(cnn_gru_path)
cnn_gru["Dataset"] = cnn_gru["Dataset"].astype(str).str.strip().str.lower()
cnn_gru["Model"] = cnn_gru["Model"].astype(str).str.strip()

frames = [cnn_gru]
if dnn_path.exists():
    dnn = pd.read_csv(dnn_path)
    dnn["Dataset"] = dnn["Dataset"].astype(str).str.strip().str.lower()
    dnn["Model"] = dnn["Model"].astype(str).str.strip()
    frames.append(dnn)

results = pd.concat(frames, ignore_index=True, sort=False)
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
for col in metrics:
    if col in results.columns:
        results[col] = pd.to_numeric(results[col], errors="coerce")

# 1. Metric comparison charts, one per dataset
for dataset in sorted(results["Dataset"].dropna().unique()):
    subset = results[results["Dataset"] == dataset].copy()
    available = [m for m in metrics if m in subset.columns and subset[m].notna().any()]
    if subset.empty or not available:
        continue

    x = np.arange(len(available))
    width = 0.8 / max(len(subset), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, row) in enumerate(subset.iterrows()):
        vals = [row[m] if pd.notna(row[m]) else 0 for m in available]
        positions = x - 0.4 + width / 2 + i * width
        bars = ax.bar(positions, vals, width, label=row["Model"])
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title(f"{dataset.title()} Dataset — Model Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset}_metric_comparison.png", dpi=180)
    plt.close(fig)

# 2. Accuracy and ROC-AUC comparison across datasets/models
for metric in ["Accuracy", "ROC-AUC"]:
    if metric not in results.columns:
        continue
    pivot = results.pivot_table(index="Dataset", columns="Model", values=metric, aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(10, 5.5), ylim=(0, 1.05), rot=0)
    ax.set_title(f"{metric} Comparison Across Datasets")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"all_datasets_{metric.lower()}_comparison.png", dpi=180)
    plt.close()

# 3. Confusion matrices for CNN1D and GRU (if matrix values exist in CSV)
if "Confusion_Matrix" in cnn_gru.columns:
    for _, row in cnn_gru.iterrows():
        raw = row["Confusion_Matrix"]
        try:
            cm = np.array(ast.literal_eval(raw) if isinstance(raw, str) else raw)
            if cm.shape != (2, 2):
                continue
        except (ValueError, SyntaxError, TypeError):
            continue

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f'{row["Dataset"].title()} — {row["Model"]} Confusion Matrix')
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["0", "1"])
        ax.set_yticks([0, 1], labels=["0", "1"])
        threshold = cm.max() / 2 if cm.size else 0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > threshold else "black", fontsize=12)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe_dataset = str(row["Dataset"]).lower().replace(" ", "_")
        safe_model = str(row["Model"]).lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{safe_dataset}_{safe_model}_confusion_matrix.png", dpi=180)
        plt.close(fig)

# Save a combined normalized results table for inspection
results.to_csv(OUT_DIR / "combined_model_metrics.csv", index=False)

print(f"Graphs and combined metrics saved to: {OUT_DIR}")
print("Files created:")
for path in sorted(OUT_DIR.iterdir()):
    print(" -", path.name)
print("\nNote: these are evaluation graphs from saved CSV metrics.")
print("They are not epoch-wise training accuracy/loss curves.")
