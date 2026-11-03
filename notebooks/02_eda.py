import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "evaluation" / "eda_plots"

# Create output folder automatically
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON FUNCTION - SAVE PLOT
# ============================================================

def save_plot(filename):
    """
    Saves the current matplotlib figure inside:
    evaluation/eda_plots/
    """

    path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved: {path}")


# ============================================================
# 1. LOAD HEART DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING HEART DATASET")
print("=" * 70)

heart_path = DATA_DIR / "heart" / "processed.cleveland.data"

heart_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

heart = pd.read_csv(
    heart_path,
    header=None,
    names=heart_columns,
    na_values="?"
)

print("Heart dataset loaded.")
print("Shape:", heart.shape)


# ============================================================
# 2. LOAD METABOLIC DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING METABOLIC DATASET")
print("=" * 70)

metabolic_folder = DATA_DIR / "metabolism"

metabolic_files = list(metabolic_folder.glob("*.csv"))

if len(metabolic_files) == 0:
    raise FileNotFoundError(
        "No CSV file found inside data/metabolism/"
    )

metabolic_path = metabolic_files[0]

metabolic = pd.read_csv(metabolic_path)

print("Metabolic dataset loaded.")
print("File:", metabolic_path.name)
print("Shape:", metabolic.shape)


# ============================================================
# 3. LOAD DIABETES DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DIABETES DATASET")
print("=" * 70)

diabetes_path = DATA_DIR / "diabetes" / "diabetes.csv"

diabetes = pd.read_csv(diabetes_path)

print("Diabetes dataset loaded.")
print("Shape:", diabetes.shape)


# ============================================================
# 4. BASIC INFORMATION
# ============================================================

print("\n\n")
print("=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)


def basic_information(df, name):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


basic_information(heart, "HEART")
basic_information(metabolic, "METABOLIC")
basic_information(diabetes, "DIABETES")


# ============================================================
# 5. HEART EDA
# ============================================================

print("\n\n")
print("=" * 70)
print("HEART DISEASE EDA")
print("=" * 70)


# ------------------------------------------------------------
# Heart Target Distribution
# ------------------------------------------------------------

print("\nHeart Target Distribution:")
print(heart["target"].value_counts().sort_index())


plt.figure(figsize=(7, 5))

heart["target"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)

save_plot("heart_target_distribution.png")


# ------------------------------------------------------------
# Heart Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    heart["age"].dropna(),
    bins=15
)

plt.title("Heart Dataset - Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

save_plot("heart_age_distribution.png")


# ------------------------------------------------------------
# Heart Cholesterol Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    heart["chol"].dropna(),
    bins=20
)

plt.title("Heart Dataset - Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Number of Patients")

save_plot("heart_cholesterol_distribution.png")


# ------------------------------------------------------------
# Heart Blood Pressure Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    heart["trestbps"].dropna(),
    bins=20
)

plt.title("Heart Dataset - Resting Blood Pressure")
plt.xlabel("Resting Blood Pressure")
plt.ylabel("Number of Patients")

save_plot("heart_blood_pressure_distribution.png")


# ------------------------------------------------------------
# Heart Maximum Heart Rate
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    heart["thalach"].dropna(),
    bins=20
)

plt.title("Heart Dataset - Maximum Heart Rate")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Number of Patients")

save_plot("heart_max_heart_rate.png")


# ============================================================
# 6. HEART CORRELATION HEATMAP
# ============================================================

print("\nGenerating Heart correlation matrix...")

heart_numeric = heart.select_dtypes(include=np.number)

heart_corr = heart_numeric.corr()

plt.figure(figsize=(12, 9))

plt.imshow(
    heart_corr,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(heart_corr.columns)),
    heart_corr.columns,
    rotation=90
)

plt.yticks(
    range(len(heart_corr.columns)),
    heart_corr.columns
)

plt.title("Heart Disease Feature Correlation")

save_plot("heart_correlation_heatmap.png")


# ============================================================
# 7. METABOLIC EDA
# ============================================================

print("\n\n")
print("=" * 70)
print("METABOLIC SYNDROME EDA")
print("=" * 70)


print("\nMetabolic Columns:")
print(metabolic.columns.tolist())


# ------------------------------------------------------------
# Find Metabolic Target
# ------------------------------------------------------------

metabolic_target = None

possible_targets = [
    "MetabolicSyndrome",
    "Metabolic_Syndrome",
    "metabolic_syndrome",
    "target",
    "Target"
]

for column in possible_targets:

    if column in metabolic.columns:
        metabolic_target = column
        break


if metabolic_target is None:

    print(
        "\nWARNING: Metabolic target column was not automatically found."
    )

else:

    print("\nMetabolic Target:", metabolic_target)

    print("\nTarget Distribution:")

    print(
        metabolic[metabolic_target]
        .value_counts(dropna=False)
    )

    # Target graph
    plt.figure(figsize=(7, 5))

    metabolic[metabolic_target].value_counts().plot(
        kind="bar"
    )

    plt.title("Metabolic Syndrome Target Distribution")
    plt.xlabel("Metabolic Syndrome")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)

    save_plot("metabolic_target_distribution.png")


# ------------------------------------------------------------
# Metabolic Numerical Distributions
# ------------------------------------------------------------

metabolic_numeric = metabolic.select_dtypes(
    include=np.number
)

print("\nMetabolic Numerical Columns:")

print(
    metabolic_numeric.columns.tolist()
)


# Plot selected common clinical variables
metabolic_features = [
    "Age",
    "age",
    "BMI",
    "WaistCirc",
    "WaistCircumference",
    "BloodGlucose",
    "Glucose",
    "HDL",
    "Triglycerides",
    "UricAcid"
]


for feature in metabolic_features:

    if feature in metabolic.columns:

        # Make sure values are numeric
        values = pd.to_numeric(
            metabolic[feature],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        plt.figure(figsize=(8, 5))

        plt.hist(
            values,
            bins=20
        )

        plt.title(
            f"Metabolic Dataset - {feature} Distribution"
        )

        plt.xlabel(feature)
        plt.ylabel("Number of Patients")

        save_plot(
            f"metabolic_{feature.lower()}_distribution.png"
        )


# ============================================================
# 8. METABOLIC CORRELATION
# ============================================================

if len(metabolic_numeric.columns) > 1:

    metabolic_corr = metabolic_numeric.corr()

    plt.figure(figsize=(12, 9))

    plt.imshow(
        metabolic_corr,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(metabolic_corr.columns)),
        metabolic_corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(metabolic_corr.columns)),
        metabolic_corr.columns
    )

    plt.title("Metabolic Syndrome Feature Correlation")

    save_plot("metabolic_correlation_heatmap.png")


# ============================================================
# 9. DIABETES EDA
# ============================================================

print("\n\n")
print("=" * 70)
print("DIABETES EDA")
print("=" * 70)


print("\nDiabetes Columns:")
print(diabetes.columns.tolist())


# ------------------------------------------------------------
# Diabetes Target
# ------------------------------------------------------------

diabetes_target = "Diabetes_binary"

if diabetes_target not in diabetes.columns:

    print(
        "\nWARNING: Diabetes_binary column was not found."
    )

else:

    print("\nDiabetes Target Distribution:")

    print(
        diabetes[diabetes_target]
        .value_counts(dropna=False)
    )


    plt.figure(figsize=(7, 5))

    diabetes[diabetes_target].value_counts().plot(
        kind="bar"
    )

    plt.title("Diabetes Target Distribution")
    plt.xlabel("Diabetes")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)

    save_plot("diabetes_target_distribution.png")


# ------------------------------------------------------------
# Diabetes BMI
# ------------------------------------------------------------

if "BMI" in diabetes.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        diabetes["BMI"].dropna(),
        bins=25
    )

    plt.title("Diabetes Dataset - BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Number of Patients")

    save_plot("diabetes_bmi_distribution.png")


# ------------------------------------------------------------
# Diabetes Age
# ------------------------------------------------------------

if "Age" in diabetes.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        diabetes["Age"].dropna(),
        bins=13
    )

    plt.title("Diabetes Dataset - Age Distribution")
    plt.xlabel("Age Category")
    plt.ylabel("Number of Patients")

    save_plot("diabetes_age_distribution.png")


# ------------------------------------------------------------
# Diabetes General Health
# ------------------------------------------------------------

if "GenHlth" in diabetes.columns:

    plt.figure(figsize=(8, 5))

    diabetes["GenHlth"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Diabetes Dataset - General Health")
    plt.xlabel("General Health")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)

    save_plot("diabetes_general_health.png")


# ------------------------------------------------------------
# Diabetes High Blood Pressure
# ------------------------------------------------------------

if "HighBP" in diabetes.columns:

    plt.figure(figsize=(7, 5))

    diabetes["HighBP"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Diabetes Dataset - High Blood Pressure")
    plt.xlabel("HighBP")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)

    save_plot("diabetes_high_bp.png")


# ------------------------------------------------------------
# Diabetes High Cholesterol
# ------------------------------------------------------------

if "HighChol" in diabetes.columns:

    plt.figure(figsize=(7, 5))

    diabetes["HighChol"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Diabetes Dataset - High Cholesterol")
    plt.xlabel("HighChol")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)

    save_plot("diabetes_high_chol.png")


# ============================================================
# 10. DIABETES CORRELATION
# ============================================================

diabetes_numeric = diabetes.select_dtypes(
    include=np.number
)

diabetes_corr = diabetes_numeric.corr()

plt.figure(figsize=(15, 12))

plt.imshow(
    diabetes_corr,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(diabetes_corr.columns)),
    diabetes_corr.columns,
    rotation=90
)

plt.yticks(
    range(len(diabetes_corr.columns)),
    diabetes_corr.columns
)

plt.title("Diabetes Feature Correlation")

save_plot("diabetes_correlation_heatmap.png")


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPlots have been saved in:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):

    print(" -", file.name)

print("\nNext Step:")
print("Data Cleaning + Preprocessing")