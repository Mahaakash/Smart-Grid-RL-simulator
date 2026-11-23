import os
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd


def load_monitor_data(monitor_dir):
    """Loads and concatenates SB3 monitor CSV files from a directory."""
    files = glob.glob(os.path.join(monitor_dir, "*.monitor.csv"))
    if not files:
        files = glob.glob(os.path.join(monitor_dir, "*.csv"))
    if not files:
        return None

    dfs = []
    for f in sorted(files):
        try:
            # SB3 monitor files have a JSON comment line on line 1, headers on line 2
            df = pd.read_csv(f, skiprows=1)
            if "r" in df.columns:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_single_curve(df, agent_name, output_path, window=20):
    """Plots the episode reward and rolling average for an agent."""
    plt.figure(figsize=(10, 5))

    # Calculate rolling average
    rolling_series = df["r"].rolling(window=window, min_periods=1).mean()

    # Plot raw episode rewards and rolling average
    plt.plot(df.index, df["r"], alpha=0.35, color="#1f77b4" if agent_name == "DQN" else "#ff7f0e", label="Episode Reward")
    plt.plot(df.index, rolling_series, color="#1f77b4" if agent_name == "DQN" else "#ff7f0e", linewidth=2.0, label=f"Rolling Mean (window={window})")

    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title(f"{agent_name} Training Reward Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {agent_name} reward curve to: {output_path}")


def plot_training_curves():
    """Generates training reward curves with rolling averages for DQN and PPO."""
    os.makedirs("results", exist_ok=True)

    dqn_df = load_monitor_data("results/monitor_dqn")
    ppo_df = load_monitor_data("results/monitor_ppo")

    # 1. Plot individual DQN reward curve
    if dqn_df is not None and len(dqn_df) > 0:
        plot_single_curve(dqn_df, "DQN", "results/dqn_reward_curve.png")
    else:
        print("No DQN monitor logs found to plot.")

    # 2. Plot individual PPO reward curve
    if ppo_df is not None and len(ppo_df) > 0:
        plot_single_curve(ppo_df, "PPO", "results/ppo_reward_curve.png")
    else:
        print("No PPO monitor logs found to plot.")

    # 3. Plot combined comparison curve if both exist
    if dqn_df is not None and ppo_df is not None and len(dqn_df) > 0 and len(ppo_df) > 0:
        plt.figure(figsize=(12, 6))

        dqn_roll = dqn_df["r"].rolling(window=20, min_periods=1).mean()
        ppo_roll = ppo_df["r"].rolling(window=20, min_periods=1).mean()

        plt.plot(dqn_df.index, dqn_roll, label="DQN (Rolling Mean)", color="#1f77b4", linewidth=2.0)
        plt.plot(ppo_df.index, ppo_roll, label="PPO (Rolling Mean)", color="#ff7f0e", linewidth=2.0)

        plt.xlabel("Episode")
        plt.ylabel("Total Episode Reward")
        plt.title("Training Reward Curve Comparison (DQN vs PPO)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        combined_path = "results/training_reward_curves.png"
        plt.savefig(combined_path)
        plt.close()
        print(f"Saved combined training reward curves to: {combined_path}")


if __name__ == "__main__":
    plot_training_curves()

