import csv
import os

input_csv = r"csvs/filtered_images.csv"
output_csv = r"csvs/filtered_images_cleaned.csv"

with open(input_csv, mode="r", newline="", encoding="utf-8") as infile, \
     open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    
    fieldnames = ["source", "image_path", "bmi"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()

    for row in reader:
        source = row["source"]
        image_path = row["image_path"]
        bmi = row["bmi"]

        # Extract filename only
        filename = os.path.basename(image_path)

        writer.writerow({
            "source": source,
            "image_path": filename,
            "bmi": bmi
        })

print("Done. Cleaned CSV saved to:", output_csv)