import os
import sys
import numpy as np
import pytest

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid_simulator import MicrogridSimulator
from src.config import RANDOM_SEED


def test_battery_level_never_exceeds_max():
    """1. Battery level never goes above 1.0 after 200 steps of CHARGE."""
    sim = MicrogridSimulator()
    sim.reset(0)
    for _ in range(200):
        result = sim.step(0)  # action 0 = CHARGE
        assert result["battery_level"] <= 1.0


def test_battery_level_never_below_zero():
    """2. Battery level never goes below 0.0 after 200 steps of DISCHARGE."""
    sim = MicrogridSimulator()
    sim.reset(0)
    for _ in range(200):
        result = sim.step(1)  # action 1 = DISCHARGE
        assert result["battery_level"] >= 0.0


def test_unmet_demand_always_zero():
    """3. unmet_demand_kwh is always 0 across 200 steps of random actions (fixed seed)."""
    np.random.seed(RANDOM_SEED)
    sim = MicrogridSimulator()
    sim.reset(0)
    for _ in range(200):
        action = int(np.random.choice([0, 1, 2, 3]))
        result = sim.step(action)
        assert result["unmet_demand_kwh"] == 0.0


def test_cost_never_negative_on_buy():
    """4. cost_inr is never negative when action is BUY."""
    sim = MicrogridSimulator()
    sim.reset(0)
    for _ in range(200):
        result = sim.step(2)  # action 2 = BUY
        assert result["cost_inr"] >= 0.0


def test_invalid_action_when_discharging_empty_battery():
    """5. invalid_action is True when discharging with an empty battery."""
    sim = MicrogridSimulator()
    sim.reset(0)
    sim.battery_level = 0.0
    result = sim.step(1)  # action 1 = DISCHARGE
    assert result["invalid_action"] is True

