import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def prepare_keypoint_data_from_csv(keypoints_csv, bounding_boxes_csv):
    keypoints_df = pd.read_csv(keypoints_csv)
    bounding_boxes_df = pd.read_csv(bounding_boxes_csv).copy()

    merge_cols = ["source", "image_path"]

    bounding_boxes_df["width"] = bounding_boxes_df["x2"] - bounding_boxes_df["x1"]
    bounding_boxes_df["height"] = bounding_boxes_df["y2"] - bounding_boxes_df["y1"]

    joined_df = keypoints_df.merge(
        bounding_boxes_df,
        on=merge_cols,
        how="inner",
        suffixes=("", "_bbox"),
    )

    x_columns = [col for col in joined_df.columns if col.endswith("-x")]
    y_columns = [col for col in joined_df.columns if col.endswith("-y")]

    normalized_df = joined_df.copy()
    normalized_df[x_columns] = (
        normalized_df[x_columns]
        .subtract(joined_df["x1"], axis=0)
        .div(joined_df["width"], axis=0)
    )
    normalized_df[y_columns] = (
        normalized_df[y_columns]
        .subtract(joined_df["y1"], axis=0)
        .div(joined_df["height"], axis=0)
    )

    features = normalized_df[x_columns + y_columns]

    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask].reset_index(drop=True)
    normalized_df = normalized_df.loc[valid_mask].reset_index(drop=True)

    print(f"Loaded {len(features)} valid samples after merge/dropna")
    return features, normalized_df


def train_clustering(features, n_clusters=4, pca_variance=0.95, random_state=42):
    print(f"Training clustering models...")
    print(f"Features: {features.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state)
    clusters = kmeans.fit_predict(X_pca)

    results = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "features": features,
        "clusters": clusters,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
    }

    cluster_counts = np.bincount(clusters)
    print("\nCluster distribution:")
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count} samples ({count/len(clusters)*100:.1f}%)")

    return results


def save_trained_models(results, model_dir):
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / "scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)
    with open(model_path / "pca.pkl", "wb") as f:
        pickle.dump(results["pca"], f)
    with open(model_path / "kmeans.pkl", "wb") as f:
        pickle.dump(results["kmeans"], f)

    print(f"Models saved to: {model_path}")


def save_elbow_curve(features, save_path, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    wcss = []
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), wcss, "bo-")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved elbow curve to: {save_path}")