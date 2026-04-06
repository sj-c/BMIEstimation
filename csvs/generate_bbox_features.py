from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import cv2

from filter_images.models.bounding_box_detection import BoundingBoxModule


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csvs"

METADATA_CSV = CSV_DIR / "dataset_metadata.csv"
OUTPUT_CSV = CSV_DIR / "dataset_with_bbox.csv"


def main():
    metadata_df = pd.read_csv(METADATA_CSV)

    detector = BoundingBoxModule(batch_size=4)

    samples = []
    for _, row in metadata_df.iterrows():
        samples.append(
            SimpleNamespace(
                original_path=Path(row["image_path"]),
                bounding_box=None,
                metadata=row.to_dict(),
            )
        )

    print(f"Loaded {len(samples)} images from metadata CSV")

    detector.run(samples)

    rows = []

    for s in samples:
        row = s.metadata
        img_path = s.original_path

        if not img_path.exists():
            print(f"❌ Missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        image_height, image_width = img.shape[:2]
        image_area = image_width * image_height

        # No human detected
        if s.bounding_box is None:
            rows.append({
                **row,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "confidence": 0.0,
                "image_width": image_width,
                "image_height": image_height,
                "image_area": image_area,
                "bbox_width": 0,
                "bbox_height": 0,
                "bbox_area": 0,
                "bbox_ratio": 0.0,
                "has_human": False,
            })
            continue

        x1 = s.bounding_box.x1
        y1 = s.bounding_box.y1
        x2 = s.bounding_box.x2
        y2 = s.bounding_box.y2
        confidence = s.bounding_box.confidence

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        bbox_ratio = bbox_area / image_area if image_area > 0 else 0.0

        rows.append({
            **row,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "image_width": image_width,
            "image_height": image_height,
            "image_area": image_area,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "bbox_area": bbox_area,
            "bbox_ratio": bbox_ratio,
            "has_human": True,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Saved: {OUTPUT_CSV}")
    print(f"Total rows: {len(out_df)}")
    print(f"Humans detected: {out_df['has_human'].sum()}")
    print(f"No human detected: {(~out_df['has_human']).sum()}")


if __name__ == "__main__":
    main()