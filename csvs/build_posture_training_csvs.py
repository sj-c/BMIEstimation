from types import SimpleNamespace
import pandas as pd

from filter_images.models.bounding_box_detection import BoundingBoxModule
from filter_images.models.keypoint_detection import KeyPointDetectionModule

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/workspace/dataset")

RESULTS_DIR = PROJECT_ROOT / "csvs"

METADATA_CSV = PROJECT_ROOT / "csvs/dataset_metadata.csv"

def keypoints_to_wide_row(sample):
    row = {
        "source": sample.source,
        "image_path": sample.image_path,
    }
    for kp in sample.keypoints:
        row[f"{kp.label}-x"] = float(kp.x)
        row[f"{kp.label}-y"] = float(kp.y)
    return row

def main():
    df = pd.read_csv(METADATA_CSV)

    samples = []
    for _, row in df.iterrows():
        samples.append(
            SimpleNamespace(
                original_path=Path(row["image_path"]),
                image_path=row["image_path"],
                source=row["source"],

                bounding_box=None,
                keypoints=None,
            )
        )

    print(f"Loaded {len(samples)} images from metadata")

    bbox_detector = BoundingBoxModule(batch_size=4)
    bbox_detector.run(samples)

    keypoint_detector = KeyPointDetectionModule(batch_size=4)
    keypoint_detector.run(samples)

    bbox_rows = []
    kp_rows = []

    for s in samples:
        if s.bounding_box is not None:
            bbox_rows.append({
                "source": s.source,
                "image_path": s.image_path,
                "x1": s.bounding_box.x1,
                "y1": s.bounding_box.y1,
                "x2": s.bounding_box.x2,
                "y2": s.bounding_box.y2,
                "confidence": s.bounding_box.confidence,
            })

        if s.keypoints is not None:
            kp_rows.append(keypoints_to_wide_row(s))

    pd.DataFrame(bbox_rows).to_csv(RESULTS_DIR / "bounding_boxes.csv", index=False)
    pd.DataFrame(kp_rows).to_csv(RESULTS_DIR / "keypoints_wide.csv", index=False)

    print("Saved bbox + keypoints CSVs")

if __name__ == "__main__":
    main()