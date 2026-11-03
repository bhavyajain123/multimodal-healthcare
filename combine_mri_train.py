import os
import shutil

TRAIN_DIR = "data/heart_mri/large_dataset/train"
VAL_DIR = "data/heart_mri/patient_split/validation"

FINAL_TRAIN_DIR = "data/heart_mri/final_train"

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]

os.makedirs(FINAL_TRAIN_DIR, exist_ok=True)

for cls in CLASSES:

    destination = os.path.join(
        FINAL_TRAIN_DIR,
        cls
    )

    os.makedirs(destination, exist_ok=True)

    # -------------------------
    # Original training images
    # -------------------------
    source_train = os.path.join(
        TRAIN_DIR,
        cls
    )

    for file in os.listdir(source_train):

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            shutil.copy2(
                os.path.join(source_train, file),
                os.path.join(destination, "train_" + file)
            )

    # -------------------------
    # Validation patients
    # -------------------------
    source_val = os.path.join(
        VAL_DIR,
        cls
    )

    for file in os.listdir(source_val):

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            shutil.copy2(
                os.path.join(source_val, file),
                os.path.join(destination, "val_" + file)
            )

print("\n================================")
print("FINAL TRAIN DATASET")
print("================================")

total = 0

for cls in CLASSES:

    folder = os.path.join(
        FINAL_TRAIN_DIR,
        cls
    )

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    total += count

    print(f"{cls}: {count}")

print("--------------------------------")
print(f"TOTAL: {total}")
print("--------------------------------")