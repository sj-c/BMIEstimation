from pathlib import Path
import pandas as pd
import cv2

FILTERED_CSV = Path("/workspace/BMIEstimation/csvs/filtered_images.csv")
OUTPUT_DIR = Path("/workspace/filtered_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_and_save(df):
    success = 0
    failed = 0

    for _, row in df.iterrows():
        src = Path(row["image_path"])

        if not src.exists():
            print(f"Missing: {src}")
            failed += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"Failed to read: {src}")
            failed += 1
            continue

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"Empty crop for: {src}")
            failed += 1
            continue

        # Preserve folder structure: source/filename.jpg
        out_path = OUTPUT_DIR / row["source"] / src.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), cropped)
        success += 1

    print(f"\nDone. Saved: {success} | Failed: {failed}")


def main():
    df = pd.read_csv(FILTERED_CSV)
    print(f"Loaded {len(df)} images from filtered CSV")
    crop_and_save(df)


if __name__ == "__main__":
    main()