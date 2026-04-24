import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataset import BMIDataset
from model import DinoBMIModel
from utils import get_val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

test_dataset = BMIDataset(
    "../csvs/test.csv",
    "../../filtered_and_cropped_images",
    get_val_transform()
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = DinoBMIModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()



mse_fn = nn.MSELoss()
mae_fn = nn.L1Loss()

all_true = []
all_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()

        preds = model(images)

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

all_true = np.array(all_true)
all_pred = np.array(all_pred)

mse = np.mean((all_true - all_pred) ** 2)
mae = np.mean(np.abs(all_true - all_pred))
rmse = np.sqrt(mse)
epsilon = 1e-8  # to avoid division by zero
mape = np.mean(np.abs((all_true - all_pred) / (all_true + epsilon))) * 100

print(f"Test MAPE: {mape:.2f}%")
print(f"Test MSE:  {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")

# Save predictions
results_df = pd.DataFrame({
    "true_bmi": all_true,
    "pred_bmi": all_pred,
    "error": all_pred - all_true,
    "abs_error": np.abs(all_pred - all_true),
    "percentage_error": np.abs((all_true - all_pred) / (all_true + epsilon)) * 100
})
results_df.to_csv("test_predictions.csv", index=False)

# Plot 1: Predicted vs Actual
plt.figure()
plt.scatter(all_true, all_pred, alpha=0.5)
plt.xlabel("True BMI")
plt.ylabel("Predicted BMI")
plt.title("Predicted vs True BMI - Test Set")

min_val = min(all_true.min(), all_pred.min())
max_val = max(all_true.max(), all_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.savefig("pred_vs_true_test.png", dpi=200)
plt.close()

# Plot 2: Error distribution
errors = all_pred - all_true

plt.figure()
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error (Predicted - True BMI)")
plt.ylabel("Count")
plt.title("Error Distribution - Test Set")
plt.savefig("error_distribution_test.png", dpi=200)
plt.close()

# Plot 3: Absolute error vs true BMI
plt.figure()
plt.scatter(all_true, np.abs(errors), alpha=0.5)
plt.xlabel("True BMI")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs True BMI - Test Set")
plt.savefig("abs_error_vs_true_bmi_test.png", dpi=200)
plt.close()