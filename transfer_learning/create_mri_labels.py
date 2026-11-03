import os
import csv

RAW_DIR = "data/heart_mri/raw"
OUTPUT_FILE = "data/heart_mri/labels.csv"

rows = []

# Only ACDC training patients: patient001 - patient100
for i in range(1, 101):

    patient = f"patient{i:03d}"
    patient_dir = os.path.join(RAW_DIR, patient)
    info_file = os.path.join(patient_dir, "Info.cfg")

    if not os.path.exists(info_file):
        print(f"Missing Info.cfg: {patient}")
        continue

    group = None

    with open(info_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Group:"):
                group = line.split(":", 1)[1].strip()

    if group:
        rows.append({
            "patient": patient,
            "label": group
        })
        print(f"{patient} -> {group}")
    else:
        print(f"Label not found: {patient}")


# Save CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["patient", "label"]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n====================================")
print("MRI LABEL EXTRACTION COMPLETED")
print("====================================")
print(f"Patients with labels: {len(rows)}")
print(f"Saved to: {OUTPUT_FILE}")