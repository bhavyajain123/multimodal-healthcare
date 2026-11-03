import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

TEST_DIR = "data/heart_mri/large_dataset/test"

MODEL_PATH = (
    "transfer_learning/models/"
    "acdc_resnet50_improved_best.keras"
)

RESULT_DIR = "transfer_learning/results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "DCM",
    "HCM",
    "MINF",
    "NOR",
    "RV"
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("LOADING RESNET50 MODEL")
print("==============================================")


model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully.")


# ============================================================
# FIND TEST IMAGES
# ============================================================

print("\n==============================================")
print("FINDING TEST IMAGES")
print("==============================================")


image_extensions = (
    ".png",
    ".jpg",
    ".jpeg"
)


patient_images = {}


for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.exists(class_dir):

        continue


    files = os.listdir(
        class_dir
    )


    for file_name in files:

        if not file_name.lower().endswith(
            image_extensions
        ):

            continue


        # Example:
        # patient076_0.png
        #
        # Patient ID:
        # patient076

        name_without_extension = os.path.splitext(
            file_name
        )[0]


        parts = name_without_extension.split(
            "_"
        )


        patient_id = parts[0]


        image_path = os.path.join(
            class_dir,
            file_name
        )


        if patient_id not in patient_images:

            patient_images[patient_id] = {
                "true_class": class_name,
                "images": []
            }


        patient_images[patient_id]["images"].append(
            image_path
        )


# ============================================================
# PRINT PATIENT INFORMATION
# ============================================================

print(
    "\nTotal patients found:",
    len(patient_images)
)


for patient_id, data in patient_images.items():

    print(
        patient_id,
        "->",
        data["true_class"],
        "->",
        len(data["images"]),
        "images"
    )


# ============================================================
# PATIENT LEVEL PREDICTION
# ============================================================

print("\n==============================================")
print("PATIENT LEVEL PREDICTION")
print("==============================================")


patient_results = []


for patient_id, data in patient_images.items():

    true_class = data["true_class"]

    image_paths = data["images"]


    all_probabilities = []


    # --------------------------------------------------------
    # Predict every slice
    # --------------------------------------------------------

    for image_path in image_paths:

        image = tf.keras.utils.load_img(

            image_path,

            target_size=IMG_SIZE
        )


        image_array = tf.keras.utils.img_to_array(
            image
        )


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        image_array = preprocess_input(
            image_array
        )


        probabilities = model.predict(

            image_array,

            verbose=0
        )[0]


        all_probabilities.append(
            probabilities
        )


    # --------------------------------------------------------
    # Average slice probabilities
    # --------------------------------------------------------

    all_probabilities = np.array(
        all_probabilities
    )


    average_probability = np.mean(
        all_probabilities,
        axis=0
    )


    predicted_index = np.argmax(
        average_probability
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = (
        average_probability[
            predicted_index
        ]
        * 100
    )


    patient_results.append({

        "patient": patient_id,

        "true_class": true_class,

        "predicted_class": predicted_class,

        "confidence": confidence,

        "number_of_slices": len(
            image_paths
        )

    })


    print(

        f"{patient_id}: "

        f"Actual={true_class}, "

        f"Predicted={predicted_class}, "

        f"Confidence={confidence:.2f}%"

    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    patient_results
)


# ============================================================
# PATIENT LEVEL ACCURACY
# ============================================================

y_true = results_df[
    "true_class"
].values


y_pred = results_df[
    "predicted_class"
].values


patient_accuracy = accuracy_score(

    y_true,

    y_pred
)


print("\n==============================================")
print("PATIENT LEVEL ACCURACY")
print("==============================================")


print(

    f"Patient-Level Accuracy: "
    f"{patient_accuracy * 100:.2f}%"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    labels=CLASS_NAMES,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)


print("\n==============================================")
print("PATIENT LEVEL CLASSIFICATION REPORT")
print("==============================================")


print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=CLASS_NAMES
)


print("\n==============================================")
print("PATIENT LEVEL CONFUSION MATRIX")
print("==============================================")


print(cm)


# ============================================================
# SAVE PATIENT RESULTS
# ============================================================

results_path = os.path.join(

    RESULT_DIR,

    "patient_level_results.csv"
)


results_df.to_csv(

    results_path,

    index=False
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(

    RESULT_DIR,

    "patient_level_classification_report.txt"
)


with open(

    report_path,

    "w"

) as f:

    f.write(

        "ACDC PATIENT LEVEL CLASSIFICATION\n\n"

    )

    f.write(

        f"Patient-Level Accuracy: "
        f"{patient_accuracy * 100:.2f}%\n\n"

    )

    f.write(
        report
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(

    RESULT_DIR,

    "patient_level_confusion_matrix.csv"
)


pd.DataFrame(

    cm,

    index=CLASS_NAMES,

    columns=CLASS_NAMES

).to_csv(

    cm_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("PATIENT LEVEL EVALUATION COMPLETED")
print("==============================================")


print(

    f"\nPatient-Level Accuracy: "
    f"{patient_accuracy * 100:.2f}%"

)


print(
    "\nPatient results:"
)

print(
    results_path
)


print(
    "\nClassification report:"
)

print(
    report_path
)


print(
    "\nConfusion matrix:"
)

print(
    cm_path
)


print("\nDONE!")