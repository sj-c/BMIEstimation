from pathlib import Path
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/workspace/dataset")

RESULTS_DIR = PROJECT_ROOT / "filter_images/results"
ASSIGNMENTS_CSV = RESULTS_DIR / "cluster_assignments.csv"
OUT_DIR = RESULTS_DIR / "cluster_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLUSTER = 20


def main():
    df = pd.read_csv(ASSIGNMENTS_CSV)

    for cluster_id, group in df.groupby("cluster"):
        cluster_dir = OUT_DIR / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        sample_group = group.head(SAMPLES_PER_CLUSTER)

        for _, row in sample_group.iterrows():
            src = Path(row["image_path"])
            if src.exists():
                dst_name = f"{row['source']}__{row['person_id']}__{src.name}"
                dst = cluster_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)

        print(f"Saved samples for cluster {cluster_id} to {cluster_dir}")


if __name__ == "__main__":
    main()