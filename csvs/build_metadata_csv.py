from pathlib import Path
import pandas as pd
import re
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/workspace/dataset")

RESULTS_DIR = PROJECT_ROOT / "csvs"
RESULTS_DIR.mkdir(exist_ok=True)

rows = []

def add_row(source, image_path, height=None, weight=None, bmi=None, sex=None):
    rows.append({
        "source": source,
        "image_path": str(image_path),
        "sex": sex,
        "height": height,
        "weight": weight,
        "bmi": bmi,

    })

# ------------------------
# 2DImage2BMI
# ------------------------
for split in ["Image_train", "Image_val", "Image_test"]:
    folder = DATASET_ROOT / "2DImage2BMI" / split
    for img in folder.rglob("*"):
        if img.suffix.lower() not in [".jpg", ".png"]:
            continue

        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img.name)
        sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
        height = float(ret.group(3)) / 100000
        weight = float(ret.group(4)) / 100000
        bmi = round(float(weight) / (float(height) ** 2), 4)


        add_row(
            source="2DImage2BMI",
            image_path=img,
            sex=sex,
            height=height,
            weight=weight,
            bmi = bmi,
        )

# ------------------------
# Celeb-FBI
# ------------------------
folder = DATASET_ROOT / "Celeb-FBI Dataset"
for img in folder.rglob("*"):
    if img.suffix.lower() not in [".jpg", ".png"]:
        continue
    name = img.name.replace(" ", "")
    name = name.replace("a_.", "a.")
    name = img.stem.replace(" ", "")


    parts = name.split("_")

    # Expected rough format:
    # id_heighth_weightw_sex_agea
    # but sex / age may be messy or missing

    if len(parts) < 3:
        print(f"❌ Failed: {img.name}")
        continue

    # -------------------------
    # Extract raw values
    # -------------------------

    # height
    height_part = parts[1]   # e.g. 5.11h or 6h
    if not height_part.endswith("h"):
        print(f"❌ Bad height: {img.name}")
        continue

    height_str = height_part[:-1]
    if "." in height_str:
        feet_str, inches_str = height_str.split(".", 1)
        feet = int(feet_str)
        inches = int(inches_str) if inches_str else 0
    else:
        feet = int(height_str)
        inches = 0

    # weight
    weight_part = parts[2]   # e.g. 76w or 54.5w
    if not weight_part.endswith("w"):
        print(f"❌ Bad weight: {img.name}")
        continue

    weight_kg = float(weight_part[:-1])   # keep your variable name

    # sex
    sex = parts[3].lower().strip() if len(parts) > 3 else ""

    # Convert height to meters
    height_m = (feet * 12 + inches) * 0.0254

    if height_m == 0 or weight_kg == 0:
        continue

    # Encode gender (0 = female, 1 = male)
    if sex.startswith("f"):
        sex = 0
    elif sex.startswith("m"):
        sex = 1
    else:
        sex = None

    # If the filename weight is actually pounds, convert first:
    # weight_kg = weight_kg * 0.453592

    bmi = round(float(weight_kg) / (float(height_m) ** 2), 4)

    add_row(
        source="Celeb-FBI",
        image_path=img,
        sex=sex,
        height=height_m,
        weight=weight_kg,
        bmi = bmi,
    )

# ------------------------
# visual_body_to_BMI
# ------------------------
folder = DATASET_ROOT / "visual_body_to_BMI"
for img in folder.rglob("*"):
    if img.suffix.lower() not in [".jpg", ".png"]:
        continue
    
folder = DATASET_ROOT / "visual_body_to_BMI"
for img in folder.rglob("*"):
    if img.suffix.lower() not in [".jpg", ".png"]:
        continue

    name = img.stem
    print(name)

    # split and remove empty pieces caused by double underscores
    parts = [p for p in name.split("_") if p != ""]

    # expected relevant tail:
    # ... , weight_lb, height_in, true/false, ...
    if len(parts) < 5:
        print(f"❌ Failed: {img.name}")
        continue

    try:
        weight_lb = float(parts[2])
        height_in = float(parts[3])
        gender_str = parts[4].lower()
    except ValueError:
        print(f"❌ Failed numeric parse: {img.name}")
        continue

    # Convert
    weight_kg = weight_lb * 0.4536
    height_m = height_in * 0.0254

    if weight_kg <= 0 or height_m <= 0:
        print(f"❌ Skipped (invalid height/weight): {img.name}")
        continue

    bmi = round(weight_kg / (height_m ** 2), 4)

    # Sex encoding: 0 if female
    sex = 0 if gender_str == "true" else 1 if gender_str == "false" else None

    add_row(
        source="visual_body_to_BMI",
        image_path=img,
        sex=sex,
        height=height_m,
        weight=weight_kg,
        bmi=bmi,
    )

df = pd.DataFrame(rows)
out_path = RESULTS_DIR / "dataset_metadata.csv"
df.to_csv(out_path, index=False)

print(f"Saved metadata: {out_path}")
print(f"Total images: {len(df)}")