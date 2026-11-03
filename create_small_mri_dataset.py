import os
import random
import shutil
import pandas as pd

# =====================================================
# SETTINGS
# =====================================================

LABEL_FILE = "data/heart_mri/train_labels.csv"

IMAGE_DIR = "data/heart_mri/processed"

OUTPUT_DIR = "data/heart_mri/small_dataset"

CLASSES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

TRAIN_PATIENTS = 12
TEST_PATIENTS = 4

TRAIN_IMAGES_PER_CLASS = 24
TEST_IMAGES_PER_CLASS = 6

random.seed(42)


# =====================================================
# REMOVE OLD SMALL DATASET
# =====================================================

if os.path.exists(OUTPUT_DIR):
    print("Removing old small dataset...")
    shutil.rmtree(OUTPUT_DIR)


# =====================================================
# CREATE TRAIN / TEST FOLDERS
# =====================================================

for split in ["train", "test"]:

    for label in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                label
            ),
            exist_ok=True
        )


# =====================================================
# READ LABELS
# =====================================================

df = pd.read_csv(LABEL_FILE)

print("\nOriginal dataset:")
print(df["label"].value_counts())


# =====================================================
# PROCESS EACH CLASS
# =====================================================

for label in CLASSES:

    print("\n================================")
    print("Processing:", label)
    print("================================")

    patients = df[
        df["label"] == label
    ]["patient"].tolist()

    print("Patients available:", len(patients))

    if len(patients) < 16:

        raise ValueError(
            f"{label} has less than 16 patients."
        )

    # Randomly shuffle patients
    random.shuffle(patients)

    # ---------------------------------------------
    # PATIENT LEVEL SPLIT
    # ---------------------------------------------

    train_patients = patients[:TRAIN_PATIENTS]

    test_patients = patients[
        TRAIN_PATIENTS:
        TRAIN_PATIENTS + TEST_PATIENTS
    ]

    print("Train patients:", len(train_patients))
    print("Test patients :", len(test_patients))


    # =================================================
    # GET TRAIN IMAGES
    # =================================================

    train_images = []

    for patient in train_patients:

        patient_folder = os.path.join(
            IMAGE_DIR,
            patient
        )

        if not os.path.exists(patient_folder):
            continue

        images = [
            f for f in os.listdir(patient_folder)
            if f.lower().endswith(".png")
        ]

        for image in images:

            train_images.append(
                (
                    patient,
                    image
                )
            )


    random.shuffle(train_images)


    # =================================================
    # GET TEST IMAGES
    # =================================================

    test_images = []

    for patient in test_patients:

        patient_folder = os.path.join(
            IMAGE_DIR,
            patient
        )

        if not os.path.exists(patient_folder):
            continue

        images = [
            f for f in os.listdir(patient_folder)
            if f.lower().endswith(".png")
        ]

        for image in images:

            test_images.append(
                (
                    patient,
                    image
                )
            )


    random.shuffle(test_images)


    # =================================================
    # CHECK ENOUGH IMAGES
    # =================================================

    if len(train_images) < TRAIN_IMAGES_PER_CLASS:

        raise ValueError(
            f"{label}: only "
            f"{len(train_images)} train images found."
        )

    if len(test_images) < TEST_IMAGES_PER_CLASS:

        raise ValueError(
            f"{label}: only "
            f"{len(test_images)} test images found."
        )


    # =================================================
    # SELECT IMAGES
    # =================================================

    selected_train = train_images[
        :TRAIN_IMAGES_PER_CLASS
    ]

    selected_test = test_images[
        :TEST_IMAGES_PER_CLASS
    ]


    # =================================================
    # COPY TRAIN IMAGES
    # =================================================

    for patient, image in selected_train:

        source = os.path.join(
            IMAGE_DIR,
            patient,
            image
        )

        new_name = (
            f"{patient}_{image}"
        )

        destination = os.path.join(
            OUTPUT_DIR,
            "train",
            label,
            new_name
        )

        shutil.copy2(
            source,
            destination
        )


    # =================================================
    # COPY TEST IMAGES
    # =================================================

    for patient, image in selected_test:

        source = os.path.join(
            IMAGE_DIR,
            patient,
            image
        )

        new_name = (
            f"{patient}_{image}"
        )

        destination = os.path.join(
            OUTPUT_DIR,
            "test",
            label,
            new_name
        )

        shutil.copy2(
            source,
            destination
        )


    print(
        f"{label}: "
        f"{len(selected_train)} train images, "
        f"{len(selected_test)} test images"
    )


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")
print("========================================")
print("SMALL MRI DATASET CREATED SUCCESSFULLY")
print("========================================")

total_train = 0
total_test = 0

for label in CLASSES:

    train_folder = os.path.join(
        OUTPUT_DIR,
        "train",
        label
    )

    test_folder = os.path.join(
        OUTPUT_DIR,
        "test",
        label
    )

    train_count = len([
        f for f in os.listdir(train_folder)
        if f.endswith(".png")
    ])

    test_count = len([
        f for f in os.listdir(test_folder)
        if f.endswith(".png")
    ])

    print(
        f"{label}: "
        f"Train={train_count}, "
        f"Test={test_count}"
    )

    total_train += train_count
    total_test += test_count


print("----------------------------------------")
print("TOTAL TRAIN:", total_train)
print("TOTAL TEST :", total_test)
print("TOTAL IMAGES:", total_train + total_test)
print("----------------------------------------")

print("\nDataset location:")
print(OUTPUT_DIR)