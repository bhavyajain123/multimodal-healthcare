import os
import shutil
import random
import pandas as pd

random.seed(42)

# ==============================
# PATHS
# ==============================

PROCESSED_DIR = "data/heart_mri/processed"
LABEL_FILE = "data/heart_mri/train_labels.csv"

OUTPUT_DIR = "data/heart_mri/large_dataset"

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]


# ==============================
# REMOVE OLD DATASET
# ==============================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)


# ==============================
# LOAD LABELS
# ==============================

df = pd.read_csv(LABEL_FILE)

print("\n================================")
print("CREATING LARGER MRI DATASET")
print("================================")


# ==============================
# CREATE FOLDERS
# ==============================

for split in ["train", "test"]:

    for cls in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                cls
            ),
            exist_ok=True
        )


# ==============================
# PROCESS EACH CLASS
# ==============================

total_train = 0
total_test = 0


for cls in CLASSES:

    patients = df[
        df["label"] == cls
    ]["patient"].tolist()

    random.shuffle(patients)

    # 12 patients for training
    train_patients = patients[:12]

    # 4 patients for testing
    test_patients = patients[12:16]

    print(f"\n{cls}")

    print(
        "Train patients:",
        len(train_patients)
    )

    print(
        "Test patients:",
        len(test_patients)
    )


    # ==========================
    # TRAIN IMAGES
    # ==========================

    train_images = []

    for patient in train_patients:

        patient_dir = os.path.join(
            PROCESSED_DIR,
            patient
        )

        if not os.path.exists(patient_dir):
            continue

        images = [
            f for f in os.listdir(patient_dir)
            if f.lower().endswith(".png")
        ]

        for image in images:

            train_images.append(
                (
                    patient,
                    os.path.join(
                        patient_dir,
                        image
                    )
                )
            )


    # Shuffle slices
    random.shuffle(train_images)


    # Use maximum 100 images/class
    train_images = train_images[:100]


    for index, (patient, image_path) in enumerate(
        train_images
    ):

        destination = os.path.join(
            OUTPUT_DIR,
            "train",
            cls,
            f"{patient}_{index}.png"
        )

        shutil.copy2(
            image_path,
            destination
        )


    # ==========================
    # TEST IMAGES
    # ==========================

    test_images = []

    for patient in test_patients:

        patient_dir = os.path.join(
            PROCESSED_DIR,
            patient
        )

        if not os.path.exists(patient_dir):
            continue

        images = [
            f for f in os.listdir(patient_dir)
            if f.lower().endswith(".png")
        ]

        for image in images:

            test_images.append(
                (
                    patient,
                    os.path.join(
                        patient_dir,
                        image
                    )
                )
            )


    random.shuffle(test_images)

    # 20 test images/class
    test_images = test_images[:20]


    for index, (patient, image_path) in enumerate(
        test_images
    ):

        destination = os.path.join(
            OUTPUT_DIR,
            "test",
            cls,
            f"{patient}_{index}.png"
        )

        shutil.copy2(
            image_path,
            destination
        )


    print(
        "Train images:",
        len(train_images)
    )

    print(
        "Test images:",
        len(test_images)
    )

    total_train += len(train_images)
    total_test += len(test_images)


# ==============================
# FINAL
# ==============================

print("\n================================")
print("DATASET CREATED")
print("================================")

print(
    "TOTAL TRAIN:",
    total_train
)

print(
    "TOTAL TEST:",
    total_test
)

print(
    "\nDataset:",
    OUTPUT_DIR
)