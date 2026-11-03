import os
import shutil
import pandas as pd

# =========================
# PATHS
# =========================
PROCESSED_DIR = "data/heart_mri/processed"
VAL_CSV = "data/heart_mri/val_labels.csv"
VAL_DIR = "data/heart_mri/patient_split/validation"

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]

# =========================
# READ VALIDATION LABELS
# =========================
df = pd.read_csv(VAL_CSV)

print("\nValidation patients:")
print(df.groupby("label").size())

# =========================
# CREATE DIRECTORIES
# =========================
for label in CLASSES:
    os.makedirs(
        os.path.join(VAL_DIR, label),
        exist_ok=True
    )

# =========================
# COPY PATIENT IMAGES
# =========================
total = 0

for _, row in df.iterrows():

    patient = row["patient"]
    label = row["label"]

    patient_dir = os.path.join(
        PROCESSED_DIR,
        patient
    )

    destination_dir = os.path.join(
        VAL_DIR,
        label
    )

    if not os.path.exists(patient_dir):
        print(f"WARNING: Missing patient folder: {patient_dir}")
        continue

    copied = 0

    for filename in os.listdir(patient_dir):

        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            source = os.path.join(
                patient_dir,
                filename
            )

            # Patient ID filename mein preserve rahega
            new_filename = f"{patient}_{filename}"

            destination = os.path.join(
                destination_dir,
                new_filename
            )

            shutil.copy2(
                source,
                destination
            )

            copied += 1
            total += 1

    print(
        f"{patient} ({label}) -> {copied} images"
    )

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 50)
print("VALIDATION DATASET CREATED")
print("=" * 50)

for label in CLASSES:

    folder = os.path.join(
        VAL_DIR,
        label
    )

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    print(f"{label}: {count} images")

print(f"\nTOTAL VALIDATION IMAGES: {total}")

print(
    f"\nValidation dataset location:\n{VAL_DIR}"
)