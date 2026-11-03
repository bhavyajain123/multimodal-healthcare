import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_dataset(df, filename):
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# HEART DISEASE PREPROCESSING
# ============================================================

def preprocess_heart():

    print("\n" + "=" * 70)
    print("HEART DISEASE PREPROCESSING")
    print("=" * 70)

    heart_path = (
        DATA_DIR
        / "heart"
        / "processed.cleveland.data"
    )

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

    print("Original shape:", heart.shape)

    # --------------------------------------------------------
    # Convert all columns to numeric
    # --------------------------------------------------------

    for column in heart.columns:
        heart[column] = pd.to_numeric(
            heart[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values before cleaning:")
    print(heart.isnull().sum())

    # Numerical columns
    feature_columns = [
        column
        for column in heart.columns
        if column != "target"
    ]

    for column in feature_columns:

        heart[column] = heart[column].fillna(
            heart[column].median()
        )

    # --------------------------------------------------------
    # Convert target into binary classification
    # --------------------------------------------------------

    heart["target"] = (
        heart["target"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    heart = heart.drop_duplicates()

    print("\nMissing values after cleaning:")
    print(heart.isnull().sum())

    print("\nHeart target distribution:")
    print(heart["target"].value_counts())

    # --------------------------------------------------------
    # Save cleaned complete dataset
    # --------------------------------------------------------

    save_dataset(
        heart,
        "heart_processed.csv"
    )

    return heart


# ============================================================
# METABOLIC SYNDROME PREPROCESSING
# ============================================================

def preprocess_metabolic():

    print("\n" + "=" * 70)
    print("METABOLIC SYNDROME PREPROCESSING")
    print("=" * 70)

    metabolic_folder = DATA_DIR / "metabolism"

    csv_files = list(
        metabolic_folder.glob("*.csv")
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No metabolic CSV found!"
        )

    metabolic_path = csv_files[0]

    metabolic = pd.read_csv(
        metabolic_path
    )

    print("Original shape:", metabolic.shape)

    # --------------------------------------------------------
    # Remove patient ID
    # --------------------------------------------------------

    if "seqn" in metabolic.columns:

        metabolic = metabolic.drop(
            columns=["seqn"]
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    numerical_columns = metabolic.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = metabolic.select_dtypes(
        include=["object", "category"]
    ).columns

    # Numerical → median
    for column in numerical_columns:

        if column != "MetabolicSyndrome":

            metabolic[column] = metabolic[column].fillna(
                metabolic[column].median()
            )

    # Categorical → mode
    for column in categorical_columns:

        if len(metabolic[column].mode()) > 0:

            metabolic[column] = metabolic[column].fillna(
                metabolic[column].mode()[0]
            )

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    categorical_columns = metabolic.select_dtypes(
        include=["object", "category"]
    ).columns

    metabolic = pd.get_dummies(
        metabolic,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    metabolic = metabolic.drop_duplicates()

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nMetabolic target distribution:")
    print(
        metabolic["MetabolicSyndrome"].value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        metabolic,
        "metabolic_processed.csv"
    )

    return metabolic


# ============================================================
# DIABETES PREPROCESSING
# ============================================================

def preprocess_diabetes():

    print("\n" + "=" * 70)
    print("DIABETES PREPROCESSING")
    print("=" * 70)

    diabetes_path = (
        DATA_DIR
        / "diabetes"
        / "diabetes.csv"
    )

    diabetes = pd.read_csv(
        diabetes_path
    )

    print("Original shape:", diabetes.shape)

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicates = diabetes.duplicated().sum()

    print(
        f"\nDuplicate rows found: {duplicates}"
    )

    diabetes = diabetes.drop_duplicates()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for column in diabetes.columns:

        diabetes[column] = pd.to_numeric(
            diabetes[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in diabetes.columns:

        if column != "Diabetes_binary":

            diabetes[column] = diabetes[column].fillna(
                diabetes[column].median()
            )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    diabetes["Diabetes_binary"] = (
        diabetes["Diabetes_binary"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nDiabetes target distribution:")
    print(
        diabetes["Diabetes_binary"].value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        diabetes,
        "diabetes_processed.csv"
    )

    return diabetes


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("      MULTIMODAL HEALTHCARE PREPROCESSING")
    print("=" * 70)

    # Process all three datasets

    heart = preprocess_heart()

    metabolic = preprocess_metabolic()

    diabetes = preprocess_diabetes()

    print("\n")
    print("=" * 70)
    print("       ALL DATASETS PREPROCESSED")
    print("=" * 70)

    print("\nProcessed datasets:")

    print(
        PROCESSED_DIR / "heart_processed.csv"
    )

    print(
        PROCESSED_DIR / "metabolic_processed.csv"
    )

    print(
        PROCESSED_DIR / "diabetes_processed.csv"
    )