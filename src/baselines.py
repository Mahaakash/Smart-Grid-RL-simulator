import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PEAK_HOURS


class GridOnly:
    """Baseline strategy that always buys all required power from the grid."""

    def act(self, state: dict) -> int:
        """Always returns action 2 (BUY)."""
        return 2


class RuleBased:
    """Rule-based heuristic strategy for battery and grid management."""

    def act(self, state: dict) -> int:
        """Chooses an action based on prioritized operational rules."""
        renewable = state["solar_kw"] + state["wind_kw"]
        demand = state["demand_kw"]
        battery = state["battery_level"]
        hour = state["hour"]

        # Check if current hour falls in peak price window
        is_peak = PEAK_HOURS[0] <= hour < PEAK_HOURS[1]

        # Rule 1: Use surplus renewable generation to charge battery if not full
        if renewable > demand and battery < 1.0:
            return 0  # CHARGE

        # Rule 2: Sell surplus renewable energy to the grid if battery is full
        if renewable > demand and battery >= 1.0:
            return 3  # SELL

        # Rule 3: Discharge battery during peak price hours if it has sufficient charge
        if is_peak and battery > 0.1:
            return 1  # DISCHARGE

        # Rule 4: Charge battery during cheap off-peak hours if battery is low
        if (not is_peak) and battery < 0.3:
            return 0  # CHARGE

        # Rule 5: Discharge battery to cover energy deficit if battery has charge
        if battery > 0.1 and renewable < demand:
            return 1  # DISCHARGE

        # Rule 6: Otherwise buy energy from the grid
        return 2  # BUY

