import os
import random
import shutil
import pandas as pd

# ==========================================
# PATHS
# ==========================================

LABEL_FILE = "data/heart_mri/train_labels.csv"

IMAGE_DIR = "data/heart_mri/processed"

OUTPUT_DIR = "data/heart_mri/small_dataset"

# 30 PATIENTS PER CLASS
PATIENTS_PER_CLASS = 30

CLASSES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

random.seed(42)


# ==========================================
# READ LABELS
# ==========================================

df = pd.read_csv(LABEL_FILE)

print("\nOriginal dataset:")
print(df["label"].value_counts())


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for label in CLASSES:

    os.makedirs(
        os.path.join(OUTPUT_DIR, label),
        exist_ok=True
    )


# ==========================================
# SELECT PATIENTS
# ==========================================

for label in CLASSES:

    class_patients = df[
        df["label"] == label
    ]["patient"].tolist()

    print(
        f"\n{label}: "
        f"{len(class_patients)} patients available"
    )

    if len(class_patients) < PATIENTS_PER_CLASS:

        raise ValueError(
            f"Not enough patients for {label}"
        )

    random.shuffle(class_patients)

    selected_patients = class_patients[
        :PATIENTS_PER_CLASS
    ]


    # ======================================
    # TAKE ONE IMAGE FROM EACH PATIENT
    # ======================================

    copied = 0

    for patient in selected_patients:

        patient_folder = os.path.join(
            IMAGE_DIR,
            patient
        )

        if not os.path.exists(patient_folder):

            print(
                "Folder not found:",
                patient_folder
            )

            continue


        images = [
            f for f in os.listdir(patient_folder)
            if f.lower().endswith(".png")
        ]


        if len(images) == 0:

            print(
                "No PNG image found:",
                patient
            )

            continue


        # Randomly select ONE image
        image = random.choice(images)

        source = os.path.join(
            patient_folder,
            image
        )


        # Give unique filename
        new_name = (
            f"{patient}_{image}"
        )

        destination = os.path.join(
            OUTPUT_DIR,
            label,
            new_name
        )


        shutil.copy2(
            source,
            destination
        )

        copied += 1


    print(
        f"{label}: {copied} images copied"
    )


# ==========================================
# FINAL COUNT
# ==========================================

print("\n==============================")
print("SMALL MRI DATASET CREATED")
print("==============================")

total = 0

for label in CLASSES:

    folder = os.path.join(
        OUTPUT_DIR,
        label
    )

    count = len([
        f for f in os.listdir(folder)
        if f.endswith(".png")
    ])

    print(
        f"{label}: {count}"
    )

    total += count


print("------------------------------")
print("TOTAL IMAGES:", total)
print("------------------------------")