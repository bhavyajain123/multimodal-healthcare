import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"



def inspect_dataset(df, name):

    print("\n")
    print("=" * 70)
    print(f"                 {name} DATASET")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nBasic Statistics:")
    print(df.describe(include="all"))




diabetes_path = DATA_DIR / "diabetes" / "diabetes.csv"

print("\nLoading Diabetes Dataset...")

diabetes = pd.read_csv(diabetes_path)

inspect_dataset(
    diabetes,
    "DIABETES"
)



metabolic_folder = DATA_DIR / "metabolism"

csv_files = list(metabolic_folder.glob("*.csv"))

if len(csv_files) == 0:

    print("\n❌ Metabolic CSV not found!")

else:

    metabolic_path = csv_files[0]

    print("\nLoading Metabolic Dataset...")

    metabolic = pd.read_csv(metabolic_path)

    inspect_dataset(
        metabolic,
        "METABOLIC SYNDROME"
    )



heart_path = DATA_DIR / "heart" / "processed.cleveland.data"

print("\nLoading Heart Dataset...")


# UCI Cleveland dataset normally has no header.
# Therefore we provide the official feature names.

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

inspect_dataset(
    heart,
    "HEART DISEASE"
)



print("\n")
print("=" * 70)
print("             DATASET INSPECTION COMPLETED")
print("=" * 70)