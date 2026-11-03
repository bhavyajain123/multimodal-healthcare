import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/heart_mri/labels.csv"

TRAIN_FILE = "data/heart_mri/train_labels.csv"
VAL_FILE = "data/heart_mri/val_labels.csv"

# Load labels
df = pd.read_csv(INPUT_FILE)

print("Total patients:", len(df))
print("\nOriginal distribution:")
print(df["label"].value_counts())

# Patient-level stratified split
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Save
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)

print("\n====================================")
print("MRI DATASET SPLIT COMPLETED")
print("====================================")

print("\nTraining patients:", len(train_df))
print("Validation patients:", len(val_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

print("\nSaved:")
print(TRAIN_FILE)
print(VAL_FILE)