import os
import csv
import re

# ====== CONFIG ======
BASE_FOLDER = r"2DImage2BMI"
TRAIN_CSV = os.path.join(BASE_FOLDER, "Image_train.csv")
TEST_CSV = os.path.join(BASE_FOLDER, "Image_test.csv")
OUTPUT_CSV = os.path.join(BASE_FOLDER, "2DImage2BMILabel.csv")


def parse_number_from_filename(filename: str):
    """
    Example filenames:
    0_F_18_157480_4535924.png
    000891_F_25_167640_11022295.jpg

    Format:
    serial_gender_age_height_weight.ext

    height is stored as meters * 100000
    weight is stored as kg * 100000
    """
    name, _ = os.path.splitext(os.path.basename(filename))
    parts = name.split("_")

    if len(parts) < 5:
        return None

    raw_serial = parts[0]
    raw_height = parts[3]
    raw_weight = parts[4]

    if not raw_height.isdigit() or not raw_weight.isdigit():
        return None

    height_m = float(raw_height) / 100000.0
    weight_kg = float(raw_weight) / 100000.0

    return raw_serial, height_m, weight_kg


def process_csv_file(csv_path, rows, seen_serials, next_serial):
    """
    Reads a CSV file and appends cleaned rows into rows list.
    Returns updated next_serial.
    """
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return next_serial

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            img_name = row[0].strip()

            parsed = parse_number_from_filename(img_name)
            if parsed is None:
                print(f"Skipped malformed filename at {csv_path}:{line_num} -> {img_name}")
                continue

            raw_serial, height_m, weight_kg = parsed

            # Remove zero / invalid values
            if height_m <= 0 or weight_kg <= 0:
                continue

            bmi = weight_kg / (height_m ** 2)

            # Use existing serial if present and numeric, else create one
            if raw_serial.isdigit():
                serial_num = str(int(raw_serial))  # removes leading zeros like 000891 -> 891
            else:
                serial_num = str(next_serial)
                next_serial += 1

            new_serial = f"B{serial_num}"

            # Avoid duplicates if same serial appears twice
            if new_serial in seen_serials:
                while f"B{next_serial}" in seen_serials:
                    next_serial += 1
                new_serial = f"B{next_serial}"
                next_serial += 1

            seen_serials.add(new_serial)

            rows.append({
                "serial_no": new_serial,
                "image_name": img_name,
                "height_meters": round(height_m, 5),
                "weight_kg": round(weight_kg, 5),
                "bmi": round(bmi, 5),
            })

    return next_serial


def main():
    rows = []
    seen_serials = set()
    next_serial = 1

    next_serial = process_csv_file(TRAIN_CSV, rows, seen_serials, next_serial)
    next_serial = process_csv_file(TEST_CSV, rows, seen_serials, next_serial)

    # Sort by numeric part of B serial
    rows.sort(key=lambda x: int(x["serial_no"][1:]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "serial_no",
            "image_name",
            "height_meters",
            "weight_kg",
            "bmi"
        ])

        for row in rows:
            writer.writerow([
                row["serial_no"],
                row["image_name"],
                row["height_meters"],
                row["weight_kg"],
                row["bmi"]
            ])

    print(f"Saved CSV to: {OUTPUT_CSV}")
    print(f"Total valid records written: {len(rows)}")


if __name__ == "__main__":
    main()