import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class BMIDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        source = row["source"]
        filename = row["image_path"]
        bmi = float(row["bmi"])

        img_path = os.path.join(self.root_dir, source, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bmi, dtype=torch.float32)