from pathlib import Path
import pandas as pd
import numpy as np

from train_posture_clustering import (
    prepare_keypoint_data_from_csv,
    train_clustering,
    save_trained_models,
    save_elbow_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "filter_images/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "filter_images/trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

KEYPOINTS_CSV = PROJECT_ROOT / "csv/keypoints_wide.csv"
BBOX_CSV = Path("csv/bounding_boxes.csv")


def save_cluster_results(results, normalized_df, out_csv):
    df = normalized_df.copy()
    df["cluster"] = results["clusters"]

    distances = results["kmeans"].transform(results["X_pca"])
    df["cluster_confidence"] = 1 / (1 + distances.min(axis=1))

    keep_cols = [
        c for c in
        ["source", "person_id", "image_path", "cluster", "cluster_confidence"]
        if c in df.columns
    ]
    df[keep_cols].to_csv(out_csv, index=False)
    print(f"Saved cluster assignments to: {out_csv}")


def save_cluster_summary(results, normalized_df, out_csv):
    temp = normalized_df.copy()
    temp["cluster"] = results["clusters"]

    summary = (
        temp.groupby(["cluster", "source"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "source"])
    )
    summary["percentage_within_all"] = summary["count"] / summary["count"].sum()
    summary.to_csv(out_csv, index=False)
    print(f"Saved cluster summary to: {out_csv}")


def main():
    features, normalized_df = prepare_keypoint_data_from_csv(KEYPOINTS_CSV, BBOX_CSV)

    save_elbow_curve(features, RESULTS_DIR / "elbow_curve.png", max_k=10)

    results = train_clustering(features=features, n_clusters=4)

    save_trained_models(results, model_dir=MODELS_DIR)

    save_cluster_results(results, normalized_df, RESULTS_DIR / "cluster_assignments.csv")
    save_cluster_summary(results, normalized_df, RESULTS_DIR / "cluster_summary.csv")

    print("Training complete.")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()