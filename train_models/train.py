import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BMIDataset
from model import DinoBMIModel
from utils import get_train_transform, get_val_transform, plot_loss

import pandas as pd
import matplotlib.pyplot as plt

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Datasets ---
# --- Datasets ---
train_dataset = BMIDataset(
    "../csvs/train.csv",
    "../../filtered_and_cropped_images",
    get_train_transform()
)

val_dataset = BMIDataset(
    "../csvs/val.csv",
    "../../filtered_and_cropped_images",
    get_val_transform()
)

test_dataset = BMIDataset(
    "../csvs/test.csv",
    "../../filtered_and_cropped_images",
    get_val_transform()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# --- Model ---
model = DinoBMIModel().to(device)

# --- Loss + optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-3)

# --- Early stopping ---
best_val_loss = float("inf")
patience = 5
counter = 0

train_losses = []
val_losses = []

# --- Training loop ---
for epoch in range(50):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")

    # --- Early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# --- Plot loss ---
plot_loss(train_losses, val_losses)

# --- Test evaluation ---
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

true_vals = []
pred_vals = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        preds = model(images)

        true_vals.extend(labels.numpy())
        pred_vals.extend(preds.cpu().numpy())

# --- Scatter plot ---
plt.scatter(true_vals, pred_vals, alpha=0.5)
plt.xlabel("True BMI")
plt.ylabel("Predicted BMI")

min_val = min(true_vals)
max_val = max(true_vals)
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.savefig("pred_vs_true_test.png")
plt.close()