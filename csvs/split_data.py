import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
input_csv = Path("csvs/filtered_images_cleaned.csv")
output_dir = Path("csvs")

# Load data
df = pd.read_csv(input_csv)

train_list = []
val_list = []
test_list = []

# Split per source
for source in df["source"].unique():
    df_source = df[df["source"] == source]

    # First split: train (80%) + temp (20%)
    train, temp = train_test_split(
        df_source,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Second split: val (10%) + test (10%)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_list.append(train)
    val_list.append(val)
    test_list.append(test)

# Combine all sources
train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Save
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("Done.")
print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")

print(train_df["source"].value_counts())
print(val_df["source"].value_counts())
print(test_df["source"].value_counts())