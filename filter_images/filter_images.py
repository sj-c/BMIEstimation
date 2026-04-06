from pathlib import Path
import pandas as pd
import numpy as np

KEYPOINTS_CSV = Path("csvs/keypoints_wide.csv")
BBOX_CSV = Path("csvs/bounding_boxes.csv")
OUTPUT_CSV = Path("csvs/filtered_images.csv")


def load_and_merge(keypoints_csv, bbox_csv):
    kp_df = pd.read_csv(keypoints_csv)
    bbox_df = pd.read_csv(bbox_csv).copy()
    metadata_df = pd.read_csv("csvs/dataset_metadata.csv")  # has sex, height, weight, bmi

    bbox_df["width_bbox"] = bbox_df["x2"] - bbox_df["x1"]
    bbox_df["height_bbox"] = bbox_df["y2"] - bbox_df["y1"]  # renamed to avoid clash with height column

    df = kp_df.merge(bbox_df, on=["source", "image_path"], how="inner")
    df = df.merge(metadata_df[["source", "image_path", "sex", "height", "weight", "bmi"]], 
                  on=["source", "image_path"], how="inner")

    print(f"Total images after merge: {len(df)}")
    return df


def filter_zero_keypoints(df):
    """Remove images where any keypoint is exactly (0, 0) — means undetected"""
    x_cols = [c for c in df.columns if c.endswith("-x")]
    y_cols = [c for c in df.columns if c.endswith("-y")]

    # A keypoint is zero if both x and y are 0
    has_zero_kp = pd.Series(False, index=df.index)
    for x_col, y_col in zip(x_cols, y_cols):
        is_zero = (df[x_col] == 0) & (df[y_col] == 0)
        has_zero_kp |= is_zero

    df = df[~has_zero_kp].reset_index(drop=True)
    print(f"After removing zero keypoints: {len(df)}")
    return df


def filter_missing_keypoints(df):
    """Remove images with NaN keypoints"""
    x_cols = [c for c in df.columns if c.endswith("-x")]
    y_cols = [c for c in df.columns if c.endswith("-y")]

    before = len(df)
    df = df.dropna(subset=x_cols + y_cols).reset_index(drop=True)
    print(f"After removing NaN keypoints: {len(df)} (dropped {before - len(df)})")
    return df


def normalize_keypoints(df):
    """Normalize keypoints relative to bounding box"""
    x_cols = [c for c in df.columns if c.endswith("-x")]
    y_cols = [c for c in df.columns if c.endswith("-y")]

    df = df.copy()
    df[x_cols] = df[x_cols].subtract(df["x1"], axis=0).div(df["width_bbox"], axis=0)
    df[y_cols] = df[y_cols].subtract(df["y1"], axis=0).div(df["height_bbox"], axis=0)
    return df


def filter_front_facing(df, shoulder_width_min=0.15, symmetry_max=0.25):
    """Keep only roughly front-facing images based on shoulder width and symmetry"""
    df = df.copy()

    df["shoulder_width"] = abs(df["left_shoulder-x"] - df["right_shoulder-x"])

    # How centered the shoulders are (0 = perfectly centered)
    df["shoulder_symmetry"] = abs(
        (df["left_shoulder-x"] + df["right_shoulder-x"]) / 2 - 0.5
    )

    before = len(df)
    df = df[
        (df["shoulder_width"] > shoulder_width_min) &
        (df["shoulder_symmetry"] < symmetry_max)
    ].reset_index(drop=True)
    print(f"After front-facing filter: {len(df)} (dropped {before - len(df)})")
    return df


def filter_upright(df):
    """Keep only upright poses — shoulders above hips (y increases downward)"""
    df = df.copy()

    df["shoulder_y"] = (df["left_shoulder-y"] + df["right_shoulder-y"]) / 2
    df["hip_y"] = (df["left_hip-y"] + df["right_hip-y"]) / 2

    before = len(df)
    df = df[df["shoulder_y"] < df["hip_y"]].reset_index(drop=True)
    print(f"After upright filter: {len(df)} (dropped {before - len(df)})")
    return df


def filter_full_body(df, min_body_height=0.4):
    """Keep images where the full body is visible — ankles near bottom of bbox"""
    df = df.copy()

    # Average ankle y position (should be near 1.0 = bottom of bbox)
    df["ankle_y"] = (df["left_ankle-y"] + df["right_ankle-y"]) / 2

    # Torso height — shoulders to hips should span reasonable portion
    df["shoulder_y"] = (df["left_shoulder-y"] + df["right_shoulder-y"]) / 2
    df["hip_y"] = (df["left_hip-y"] + df["right_hip-y"]) / 2
    df["body_height"] = df["hip_y"] - df["shoulder_y"]

    before = len(df)
    df = df[
        (df["ankle_y"] > 0.7) &               # ankles near bottom
        (df["body_height"] > min_body_height)  # torso is visible
    ].reset_index(drop=True)
    print(f"After full-body filter: {len(df)} (dropped {before - len(df)})")
    return df


def filter_single_person(df, bbox_csv):
    """Keep images where bounding box confidence is high (likely clean single person shot)"""
    df = df.copy()
    before = len(df)
    df = df[df["confidence"] > 0.9].reset_index(drop=True)
    print(f"After confidence filter: {len(df)} (dropped {before - len(df)})")
    return df


def main():
    df = load_and_merge(KEYPOINTS_CSV, BBOX_CSV)

    # Step 1: Remove zero/missing keypoints
    df = filter_zero_keypoints(df)
    df = filter_missing_keypoints(df)

    # Step 2: Normalize keypoints to bounding box
    df = normalize_keypoints(df)

    # Step 3: Front-facing
    df = filter_front_facing(df)

    # Step 4: Upright pose
    df = filter_upright(df)

    # Step 5: Full body visible
    df = filter_full_body(df)

    # Step 6: High confidence detection
    df = filter_single_person(df,BBOX_CSV)

    # Save only the columns needed downstream
    out = df.copy()
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFinal filtered dataset: {len(out)} images")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()