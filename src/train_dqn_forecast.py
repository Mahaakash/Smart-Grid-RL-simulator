import os
import sys
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_SEED
from src.grid_env_forecast import SmartGridForecastEnv


class TimestepPrintCallback(BaseCallback):
    """Callback to print the current timestep count every check_freq steps."""

    def __init__(self, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            print(f"DQN+Forecast Timestep: {self.num_timesteps}")
        return True


def train_dqn_forecast(timesteps=200000):
    """Trains a DQN agent on SmartGridForecastEnv with solar forecast observations."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/tensorboard", exist_ok=True)
    monitor_dir = "results/monitor_dqn_forecast"
    os.makedirs(monitor_dir, exist_ok=True)

    # Initialize environment with forecast observations and Monitor
    raw_env = SmartGridForecastEnv(mode="train")
    env = Monitor(raw_env, filename=os.path.join(monitor_dir, "dqn_forecast"))

    # Initialize DQN model with specified hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=2000,
        exploration_fraction=0.3,
        seed=RANDOM_SEED,
        tensorboard_log="results/tensorboard/",
        verbose=0,
    )

    check_freq = 5000 if timesteps <= 20000 else 10000
    callback = TimestepPrintCallback(check_freq=check_freq)

    print(f"Starting DQN + Forecast training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save final model to models/dqn_forecast.zip
    save_path = "models/dqn_forecast.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN + Forecast agent on SmartGrid microgrid environment")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total timesteps to train")
    args = parser.parse_args()

    train_dqn_forecast(timesteps=args.timesteps)

