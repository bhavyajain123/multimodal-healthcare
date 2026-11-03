import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = PROCESSED_DIR / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FUNCTION
# =========================

def process_dataset(file_name, target_column, dataset_name):

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    file_path = PROCESSED_DIR / file_name

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)

    # -------------------------
    # Separate X and y
    # -------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Features:", X.shape)
    print("Target:", y.shape)

    # -------------------------
    # Train Test Split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    # -------------------------
    # Scaling
    # -------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    # -------------------------
    # Save files
    # -------------------------

    X_train_scaled.to_csv(
        OUTPUT_DIR / f"{dataset_name}_X_train.csv",
        index=False
    )

    X_test_scaled.to_csv(
        OUTPUT_DIR / f"{dataset_name}_X_test.csv",
        index=False
    )

    y_train.to_csv(
        OUTPUT_DIR / f"{dataset_name}_y_train.csv",
        index=False
    )

    y_test.to_csv(
        OUTPUT_DIR / f"{dataset_name}_y_test.csv",
        index=False
    )

    # Save scaler
    joblib.dump(
        scaler,
        OUTPUT_DIR / f"{dataset_name}_scaler.pkl"
    )

    print("\nSaved successfully.")


# =========================
# DATASETS
# =========================

process_dataset(
    "heart_processed.csv",
    "target",
    "heart"
)

process_dataset(
    "metabolic_processed.csv",
    "MetabolicSyndrome",
    "metabolic"
)

process_dataset(
    "diabetes_processed.csv",
    "Diabetes_binary",
    "diabetes"
)


print("\n" + "=" * 60)
print("ALL DATASETS PROCESSED SUCCESSFULLY")
print("=" * 60)