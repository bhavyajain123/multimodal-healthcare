import os
import pandas as pd

results = [
    # ================= HEART =================
    {
        "Dataset": "Heart",
        "Model": "Logistic Regression",
        "Accuracy": 0.8689,
        "Precision": 0.8125,
        "Recall": 0.9286,
        "F1": 0.8667,
        "ROC-AUC": 0.9502
    },
    {
        "Dataset": "Heart",
        "Model": "SVM",
        "Accuracy": 0.8689,
        "Precision": 0.8125,
        "Recall": 0.9286,
        "F1": 0.8667,
        "ROC-AUC": 0.9437
    },
    {
        "Dataset": "Heart",
        "Model": "Random Forest",
        "Accuracy": 0.9180,
        "Precision": 0.8710,
        "Recall": 0.9643,
        "F1": 0.9153,
        "ROC-AUC": 0.9567
    },
    {
        "Dataset": "Heart",
        "Model": "XGBoost",
        "Accuracy": 0.9016,
        "Precision": 0.8438,
        "Recall": 0.9643,
        "F1": 0.9000,
        "ROC-AUC": 0.9470
    },

    # ================= METABOLIC =================
    {
        "Dataset": "Metabolic",
        "Model": "Logistic Regression",
        "Accuracy": 0.8337,
        "Precision": 0.7297,
        "Recall": 0.8182,
        "F1": 0.7714,
        "ROC-AUC": 0.9097
    },
    {
        "Dataset": "Metabolic",
        "Model": "SVM",
        "Accuracy": 0.8233,
        "Precision": 0.7247,
        "Recall": 0.7818,
        "F1": 0.7522,
        "ROC-AUC": 0.9111
    },
    {
        "Dataset": "Metabolic",
        "Model": "Random Forest",
        "Accuracy": 0.8877,
        "Precision": 0.8581,
        "Recall": 0.8061,
        "F1": 0.8313,
        "ROC-AUC": 0.9602
    },
    {
        "Dataset": "Metabolic",
        "Model": "XGBoost",
        "Accuracy": 0.9002,
        "Precision": 0.8634,
        "Recall": 0.8424,
        "F1": 0.8528,
        "ROC-AUC": 0.9683
    },

    # ================= DIABETES =================
    {
        "Dataset": "Diabetes",
        "Model": "Logistic Regression",
        "Accuracy": 0.7143,
        "Precision": 0.3183,
        "Recall": 0.7601,
        "F1": 0.4487,
        "ROC-AUC": 0.8106
    },
    {
        "Dataset": "Diabetes",
        "Model": "SVM",
        "Accuracy": 0.710361,
        "Precision": 0.315536,
        "Recall": 0.764496,
        "F1": 0.446701,
        "ROC-AUC": 0.810333
    },
    {
        "Dataset": "Diabetes",
        "Model": "Random Forest",
        "Accuracy": 0.843948,
        "Precision": 0.467894,
        "Recall": 0.148454,
        "F1": 0.225395,
        "ROC-AUC": 0.781658
    },
    {
        "Dataset": "Diabetes",
        "Model": "XGBoost",
        "Accuracy": 0.855692,
        "Precision": 0.597826,
        "Recall": 0.172389,
        "F1": 0.267610,
        "ROC-AUC": 0.819937
    }
]

df = pd.DataFrame(results)

output_dir = "evaluation/ml_results"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "ml_results.csv")

df.to_csv(output_file, index=False)

print("ML results compiled successfully!")
print(f"Saved at: {output_file}")
print()
print(df)