from pathlib import Path
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/workspace/dataset")

RESULTS_DIR = PROJECT_ROOT / "csv"
RESULTS_DIR.mkdir(exist_ok=True)

rows = []

def add_row(source, image_path, person_id=None, height=None, weight=None, bmi=None, gender=None):
    rows.append({
        "source": source,
        "image_path": str(image_path),
        "person_id": person_id,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "gender": gender,
    })

# ------------------------
# 2DImage2BMI
# ------------------------
for split in ["Image_train", "Image_val", "Image_test"]:
    folder = DATASET_ROOT / "2DImage2BMI" / split
    for img in folder.rglob("*"):
        if img.suffix.lower() not in [".jpg", ".png"]:
            continue

        name = img.stem.split("_")

        # Example: 0_F_20_160020_6713168
        try:
            gender = 0 if name[1] == "F" else 1
            # dataset-specific parsing (adjust if needed)
        except:
            gender = None

        add_row(
            source="2DImage2BMI",
            image_path=img,
            person_id=img.stem,
            gender=gender
        )

# ------------------------
# Celeb-FBI
# ------------------------
folder = DATASET_ROOT / "Celeb-FBI"
for img in folder.rglob("*"):
    if img.suffix.lower() not in [".jpg", ".png"]:
        continue

    add_row(
        source="Celeb-FBI",
        image_path=img,
        person_id=img.stem
    )

# ------------------------
# visual_body_to_BMI
# ------------------------
folder = DATASET_ROOT / "visual_body_to_BMI"
for img in folder.rglob("*"):
    if img.suffix.lower() not in [".jpg", ".png"]:
        continue

    add_row(
        source="visual_body_to_BMI",
        image_path=img,
        person_id=img.stem
    )

df = pd.DataFrame(rows)
out_path = RESULTS_DIR / "dataset_metadata.csv"
df.to_csv(out_path, index=False)

print(f"Saved metadata: {out_path}")
print(f"Total images: {len(df)}")