import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_PATH


def plot_sample_week():
    """Loads grid data and plots a 7-day sample week (days 100-106) with real weather data."""
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Filter for days 100 to 106 (7 days total)
    sample_df = df[(df["day"] >= 100) & (df["day"] <= 106)].copy()

    # Reset index to represent continuous hours from 0 to 167
    sample_df = sample_df.reset_index(drop=True)

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(sample_df.index, sample_df["solar_kw"], label="Real Solar Generation (kW)", color="orange")
    plt.plot(sample_df.index, sample_df["wind_kw"], label="Real Wind Generation (kW)", color="cyan")
    plt.plot(sample_df.index, sample_df["demand_kw"], label="Campus Demand (kW)", color="red")

    # Add labels, title, and legend
    plt.xlabel("Hour of the Week (0 to 167)")
    plt.ylabel("Power (kW)")
    plt.title("Real Weather Power Profile - Kolkata, India (Days 100 to 106)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Ensure results directory exists
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_week.png")

    # Save chart and display it
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_sample_week()
