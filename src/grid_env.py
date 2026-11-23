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


class SmartGridEnv(gymnasium.Env):
    """Gymnasium environment wrapping the MicrogridSimulator."""

    def __init__(self, mode="train", episode_days=EPISODE_DAYS, fixed_start_day=None):
        super().__init__()
        self.mode = mode
        self.episode_days = episode_days
        self.fixed_start_day = TRAIN_SPLIT_DAY if fixed_start_day is None else fixed_start_day

        self.sim = MicrogridSimulator()

        # Action space: 0=Charge, 1=Discharge, 2=Buy, 3=Sell
        self.action_space = spaces.Discrete(4)

        # Observation space: 9 continuous normalized values [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0

        # Determine starting day based on mode
        if self.mode == "train":
            # Random starting day within training split
            max_start = TRAIN_SPLIT_DAY - self.episode_days
            start_day = int(self.np_random.integers(0, max_start + 1))
        else:
            # Fixed starting day for evaluation
            start_day = int(self.fixed_start_day)

        start_index = start_day * 24
        state = self.sim.reset(start_index)

        # Construct normalized observation vector
        hour = state["hour"]
        is_peak = 1.0 if (PEAK_HOURS[0] <= hour < PEAK_HOURS[1]) else 0.0

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
        ], dtype=np.float32)

        obs = np.clip(obs, 0.0, 1.0)
        info = {}
        return obs, info

    def step(self, action):
        """Steps the environment by executing one hour simulation."""
        result = self.sim.step(action)
        self.current_step += 1

        state = result["state"]
        hour = state["hour"]
        is_peak = 1.0 if (PEAK_HOURS[0] <= hour < PEAK_HOURS[1]) else 0.0

        # Construct normalized observation vector for the new state
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
        ], dtype=np.float32)

        obs = np.clip(obs, 0.0, 1.0)

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

