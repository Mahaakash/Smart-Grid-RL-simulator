import os
import sys
import numpy as np
import gymnasium
from gymnasium import spaces

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    EPISODE_DAYS,
    TRAIN_SPLIT_DAY,
    SOLAR_CAPACITY_KW,
    WIND_CAPACITY_KW,
    PEAK_PRICE,
    PEAK_HOURS,
    COST_WEIGHT,
    CO2_WEIGHT,
    UNMET_WEIGHT,
    INVALID_ACTION_PENALTY,
)
from src.grid_simulator import MicrogridSimulator


class SmartGridForecastEnv(gymnasium.Env):
    """Gymnasium environment wrapping MicrogridSimulator with LSTM solar forecast observations."""

    def __init__(
        self,
        mode="train",
        episode_days=EPISODE_DAYS,
        fixed_start_day=None,
        data_path="data/grid_data_forecast.csv",
    ):
        super().__init__()
        self.mode = mode
        self.episode_days = episode_days
        self.fixed_start_day = TRAIN_SPLIT_DAY if fixed_start_day is None else fixed_start_day

        # Simulator using forecast-enriched dataset
        self.sim = MicrogridSimulator(data_path=data_path)

        # Action space: 0=Charge, 1=Discharge, 2=Buy, 3=Sell
        self.action_space = spaces.Discrete(4)

        # Observation space: 11 continuous normalized values [0, 1]
        # (9 baseline grid features + 2 future solar forecast features)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self.current_step = 0

    def _get_obs(self, state):
        """Constructs the 11-dimensional normalized observation vector."""
        row = self.sim.data.iloc[self.sim.current_index]
        hour = state["hour"]
        is_peak = 1.0 if (PEAK_HOURS[0] <= hour < PEAK_HOURS[1]) else 0.0

        p3_norm = float(row.get("solar_pred_3h_norm", 0.0))
        p6_norm = float(row.get("solar_pred_6h_norm", 0.0))

        obs = np.array([
            state["solar_kw"] / SOLAR_CAPACITY_KW,
            state["wind_kw"] / WIND_CAPACITY_KW,
            state["demand_kw"] / 100.0,
            state["battery_level"],
            state["price"] / PEAK_PRICE,
            (np.sin(2.0 * np.pi * hour / 24.0) + 1.0) / 2.0,
            (np.cos(2.0 * np.pi * hour / 24.0) + 1.0) / 2.0,
            state["day_of_week"] / 6.0,
            is_peak,
            p3_norm,
            p6_norm,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0

        # Determine starting day based on mode
        if self.mode == "train":
            max_start = TRAIN_SPLIT_DAY - self.episode_days
            start_day = int(self.np_random.integers(0, max_start + 1))
        else:
            start_day = int(self.fixed_start_day)

        start_index = start_day * 24
        state = self.sim.reset(start_index)

        obs = self._get_obs(state)
        info = {}
        return obs, info

    def step(self, action):
        """Steps the environment by executing one hour simulation."""
        result = self.sim.step(action)
        self.current_step += 1

        obs = self._get_obs(result["state"])

        # Calculate reward
        reward = -(
            COST_WEIGHT * result["cost_inr"]
            + CO2_WEIGHT * result["co2_kg"]
            + UNMET_WEIGHT * result["unmet_demand_kwh"]
        )
        if result["invalid_action"]:
            reward -= INVALID_ACTION_PENALTY

        # Check termination conditions
        truncated = bool(self.current_step >= self.episode_days * 24)
        terminated = bool(result["done"])
        info = result

        return obs, float(reward), terminated, truncated, info

