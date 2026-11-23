import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_PATH, TRAIN_SPLIT_DAY, SOLAR_CAPACITY_KW, RANDOM_SEED


class SolarLSTM(nn.Module):
    """1-layer LSTM with 32 hidden units for 24-hour solar generation forecasting."""

    def __init__(self, input_size=3, hidden_size=32, output_size=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length=48, input_size=3)
        lstm_out, _ = self.lstm(x)
        # Use final timestep output for 24-hour forecast
        out = self.fc(lstm_out[:, -1, :])
        return out


def train_solar_lstm():
    """Trains the LSTM forecasting model on solar data and validates performance."""
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # Compute cyclical hour features
    hours = df["hour"].values
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)

    # Normalize solar generation to [0, 1] using capacity
    solar_norm = df["solar_kw"].values / SOLAR_CAPACITY_KW

    # Feature matrix: shape (N, 3)
    features = np.column_stack([solar_norm, hour_sin, hour_cos]).astype(np.float32)
    targets = solar_norm.astype(np.float32)

    total_hours = len(df)
    train_split_hour = TRAIN_SPLIT_DAY * 24

    # 2. Build training dataset (days 0 to TRAIN_SPLIT_DAY-1)
    # Using 48 hours history to predict next 24 hours
    x_train_list, y_train_list = [], []
    for i in range(train_split_hour - 48 - 24 + 1):
        x_train_list.append(features[i : i + 48])
        y_train_list.append(targets[i + 48 : i + 48 + 24])

    X_train = torch.tensor(np.array(x_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)

    # 3. Build validation dataset (remaining days)
    x_val_list, y_val_list = [], []
    for idx in range(train_split_hour, total_hours - 24 + 1):
        if idx - 48 >= 0:
            x_val_list.append(features[idx - 48 : idx])
            y_val_list.append(targets[idx : idx + 24])

    X_val = torch.tensor(np.array(x_val_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )

    # 4. Initialize model, loss, and optimizer
    model = SolarLSTM(input_size=3, hidden_size=32, output_size=24)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    print(f"Starting LSTM training for {epochs} epochs on real Kolkata solar data...")

    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_preds_kw = val_preds * SOLAR_CAPACITY_KW
            val_targets_kw = y_val * SOLAR_CAPACITY_KW
            val_mae = torch.mean(torch.abs(val_preds_kw - val_targets_kw)).item()

        print(f"Epoch {epoch + 1:2d}/{epochs} - Validation MAE: {val_mae:.4f} kW")

    # 6. Save model to models/lstm_solar.pt
    os.makedirs("models", exist_ok=True)
    model_path = "models/lstm_solar.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # 7. Generate 3-day validation forecast (first 72 hours of validation set)
    os.makedirs("results", exist_ok=True)
    model.eval()
    day_preds = []

    with torch.no_grad():
        for d in range(3):
            start_idx = train_split_hour + (d * 24)
            hist = torch.tensor(
                features[start_idx - 48 : start_idx], dtype=torch.float32
            ).unsqueeze(0)
            pred_24 = model(hist).squeeze(0).numpy() * SOLAR_CAPACITY_KW
            day_preds.extend(np.clip(pred_24, 0.0, SOLAR_CAPACITY_KW))

    actual_3days = df["solar_kw"].iloc[train_split_hour : train_split_hour + 72].values

    # Plot actual vs predicted for the first 3 validation days
    plt.figure(figsize=(12, 5))
    hours_axis = list(range(72))
    plt.plot(hours_axis, actual_3days, label="Actual Solar (kW)", color="black", linewidth=1.5)
    plt.plot(
        hours_axis,
        day_preds,
        label="Predicted Solar (kW)",
        color="orange",
        linestyle="--",
        linewidth=1.5,
    )
    plt.xlabel("Hour (First 3 Validation Days)")
    plt.ylabel("Solar Generation (kW)")
    plt.title("Solar Generation Forecast vs Actual - Kolkata (First 3 Validation Days)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    chart_path = "results/forecast_check.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Validation forecast chart saved to: {chart_path}")

    # 8. Precompute forecasts with NO LEAKAGE (using only prior 48 hours for each hour)
    print("Precomputing 3-hour and 6-hour forecasts across all 8760 hours (no leakage)...")
    pred_3h_norm = []
    pred_6h_norm = []

    with torch.no_grad():
        for t in range(total_hours):
            if t >= 48:
                hist_window = features[t - 48 : t]
            else:
                # Pad beginning of dataset with earliest available history
                pad_len = 48 - t
                pad = np.repeat(features[0:1], pad_len, axis=0)
                hist_window = np.vstack([pad, features[:t]])

            hist_tensor = torch.tensor(hist_window, dtype=torch.float32).unsqueeze(0)
            pred_24 = model(hist_tensor).squeeze(0).numpy()
            pred_24 = np.clip(pred_24, 0.0, 1.0)

            # Average predicted solar for the next 3 hours and 6 hours
            p3 = float(np.mean(pred_24[:3]))
            p6 = float(np.mean(pred_24[:6]))

            pred_3h_norm.append(round(p3, 4))
            pred_6h_norm.append(round(p6, 4))

    # Save enriched dataset with forecast features
    df["solar_pred_3h_norm"] = pred_3h_norm
    df["solar_pred_6h_norm"] = pred_6h_norm
    df["solar_pred_3h_kw"] = np.round(np.array(pred_3h_norm) * SOLAR_CAPACITY_KW, 2)
    df["solar_pred_6h_kw"] = np.round(np.array(pred_6h_norm) * SOLAR_CAPACITY_KW, 2)

    forecast_data_path = "data/grid_data_forecast.csv"
    df.to_csv(forecast_data_path, index=False)
    print(f"Enriched dataset with forecast features saved to: {forecast_data_path}")

    return model, df


if __name__ == "__main__":
    train_solar_lstm()
