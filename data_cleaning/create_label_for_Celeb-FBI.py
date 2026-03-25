import os
import re
import csv

# ====== CONFIG ======
IMAGE_FOLDER = "Celeb-FBI Dataset"
OUTPUT_CSV = os.path.join(IMAGE_FOLDER, "Celeb-FBILabel.csv")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def feet_inches_token_to_meters(height_token: str) -> float:
    """
    Convert '5.5h' → 5 feet 5 inches → meters
    Also supports '5h'
    """
    token = height_token[:-1]

    if "." in token:
        feet, inches = map(int, token.split("."))
    else:
        feet = int(token)
        inches = 0

    total_inches = feet * 12 + inches
    return total_inches * 0.0254


def parse_filename(filename: str):
    name, ext = os.path.splitext(filename)

    if ext.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = name.split("_")
    if len(parts) < 5:
        return None

    height_token = parts[1]
    weight_token = parts[2]

    if not re.fullmatch(r"\d+(?:\.\d+)?h", height_token):
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?w", weight_token):
        return None

    try:
        height_m = feet_inches_token_to_meters(height_token)
        weight_kg = float(weight_token[:-1])
    except:
        return None

    if height_m <= 0 or weight_kg <= 0:
        return None

    bmi = weight_kg / (height_m ** 2)

    return {
        "serial_no": f"C{name}",  
        "image_name": filename,
        "height_feet_inches": height_token[:-1],
        "height_meters": round(height_m, 4),
        "weight_kg": round(weight_kg, 2),
        "bmi": round(bmi, 2),
    }


def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folder not found: {IMAGE_FOLDER}")
        return

    rows = []
    skipped = []

    for filename in os.listdir(IMAGE_FOLDER):
        parsed = parse_filename(filename)
        if parsed:
            rows.append(parsed)
        else:
            skipped.append(filename)


    rows.sort(key=lambda x: x["serial_no"])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "serial_no",
            "image_name",
            "height_feet_inches",
            "height_meters",
            "weight_kg",
            "bmi"
        ])

        for row in rows:
            writer.writerow([
                row["serial_no"],
                row["image_name"],
                row["height_feet_inches"],
                row["height_meters"],
                row["weight_kg"],
                row["bmi"]
            ])

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Valid rows: {len(rows)}")
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()