import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "heart_mri",
    "raw"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "heart_mri",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

# Number of patients to process
# None = all patients
MAX_PATIENTS = None


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum - minimum == 0:
        return np.zeros_like(image)

    image = (
        (image - minimum)
        / (maximum - minimum)
    )

    image = (
        image * 255
    ).astype(np.uint8)

    return image


# ============================================================
# PROCESS ONE MRI VOLUME
# ============================================================

def process_patient(patient_dir):

    patient_name = os.path.basename(
        patient_dir
    )

    print(
        f"\nProcessing {patient_name}..."
    )

    # Look for 4D MRI
    mri_files = glob.glob(
        os.path.join(
            patient_dir,
            "*_4d.nii.gz"
        )
    )

    # If 4D file doesn't exist,
    # look for individual frame files
    if len(mri_files) == 0:

        mri_files = glob.glob(
            os.path.join(
                patient_dir,
                "*_frame*.nii.gz"
            )
        )

        # Remove ground-truth files
        mri_files = [
            f for f in mri_files
            if "_gt" not in f
        ]

    if len(mri_files) == 0:

        print(
            f"No MRI file found for {patient_name}"
        )

        return 0


    saved_count = 0


    # ========================================================
    # PROCESS MRI FILES
    # ========================================================

    for mri_path in mri_files:

        print(
            f"Reading: "
            f"{os.path.basename(mri_path)}"
        )

        nii = nib.load(
            mri_path
        )

        image_data = nii.get_fdata()

        print(
            f"Shape: {image_data.shape}"
        )


        # ----------------------------------------------------
        # Handle 3D image
        # ----------------------------------------------------

        if image_data.ndim == 3:

            volumes = [
                image_data
            ]


        # ----------------------------------------------------
        # Handle 4D image
        # ----------------------------------------------------

        elif image_data.ndim == 4:

            volumes = [
                image_data[:, :, :, i]
                for i in range(
                    image_data.shape[3]
                )
            ]

        else:

            print(
                "Unsupported image dimension."
            )

            continue


        # ====================================================
        # EXTRACT 2D SLICES
        # ====================================================

        for volume_index, volume in enumerate(
            volumes
        ):

            # Take middle slices around
            # the cardiac region
            number_of_slices = volume.shape[2]

            start = max(
                0,
                number_of_slices // 2 - 5
            )

            end = min(
                number_of_slices,
                number_of_slices // 2 + 5
            )


            for slice_index in range(
                start,
                end
            ):

                slice_image = volume[
                    :, :,
                    slice_index
                ]


                # Normalize
                slice_image = normalize_image(
                    slice_image
                )


                # Convert to PIL
                pil_image = Image.fromarray(
                    slice_image
                )


                # Resize
                pil_image = pil_image.resize(
                    IMAGE_SIZE
                )


                # Convert grayscale → RGB
                pil_image = pil_image.convert(
                    "RGB"
                )


                # Output patient folder
                patient_output = os.path.join(
                    OUTPUT_DIR,
                    patient_name
                )

                os.makedirs(
                    patient_output,
                    exist_ok=True
                )


                output_filename = (
                    f"{patient_name}_"
                    f"volume{volume_index}_"
                    f"slice{slice_index}.png"
                )


                output_path = os.path.join(
                    patient_output,
                    output_filename
                )


                pil_image.save(
                    output_path
                )

                saved_count += 1


    print(
        f"Saved {saved_count} images "
        f"for {patient_name}"
    )

    return saved_count


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=============================================="
    )

    print(
        "ACDC MRI PREPROCESSING"
    )

    print(
        "=============================================="
    )

    if not os.path.exists(RAW_DIR):

        raise FileNotFoundError(
            f"Raw directory not found:\n{RAW_DIR}"
        )


    # Find patient folders
    patient_dirs = sorted(
        [
            os.path.join(
                RAW_DIR,
                folder
            )

            for folder in os.listdir(
                RAW_DIR
            )

            if os.path.isdir(
                os.path.join(
                    RAW_DIR,
                    folder
                )
            )
        ]
    )


    if MAX_PATIENTS is not None:

        patient_dirs = patient_dirs[
            :MAX_PATIENTS
        ]


    print(
        f"Patients found: "
        f"{len(patient_dirs)}"
    )


    total_images = 0


    for patient_dir in patient_dirs:

        total_images += process_patient(
            patient_dir
        )


    print()
    print(
        "=============================================="
    )

    print(
        "PREPROCESSING COMPLETED"
    )

    print(
        f"Total images created: {total_images}"
    )

    print(
        f"Saved inside:\n{OUTPUT_DIR}"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":

    main()