import os
import sys
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid_env import SmartGridEnv
from src.config import RANDOM_SEED


def test_env_checker():
    """1. gymnasium.utils.env_checker.check_env passes for SmartGridEnv."""
    env = SmartGridEnv()
    check_env(env)


def test_full_random_episode():
    """2. A full random-action episode finishes without errors, and every obs value is in [0, 1]."""
    env = SmartGridEnv()
    obs, info = env.reset(seed=RANDOM_SEED)

    # Check initial observation bounds
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    done = False
    while not done:
        # Sample random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check every observation value stays between 0 and 1
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        done = terminated or truncated

