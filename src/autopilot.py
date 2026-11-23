"""
Auto-Pilot autonomous microgrid dispatch controller.
Coordinates RL agents, heuristics, safety filtering, diagnostics fallbacks, and manual overrides.
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    SOLAR_CAPACITY_KW,
    WIND_CAPACITY_KW,
    PEAK_PRICE,
    PEAK_HOURS,
    BATTERY_CAPACITY_KWH,
    MAX_RATE_KW,
)
from src.grid_simulator import MicrogridSimulator
from src.baselines import RuleBased
from src.diagnostics import scan

ACTION_NAMES = {
    0: "CHARGE BATTERY",
    1: "DISCHARGE BATTERY",
    2: "BUY FROM GRID",
    3: "SELL TO GRID",
}


class AutoPilot:
    """
    Autonomous intelligent controller for microgrid energy management.
    Applies RL policies or heuristic rules, filtered by CommandController safety rules,
    with automatic diagnostics fallback and temporary manual overrides.
    """

    def __init__(
        self,
        simulator: Optional[MicrogridSimulator] = None,
        strategy: str = "DQN",
        override_hours: int = 2,
    ):
        self.simulator = simulator if simulator is not None else MicrogridSimulator()
        self.strategy = strategy
        self.enabled = True
        self.default_override_hours = int(override_hours)

        # Active manual override state: None or dict(mode, hours_left, action)
        self.active_override = None

        # Diagnostic and model notices
        self.model_notice = None
        self.fallback_notice = None

        # Rule-based fallback instance
        self.rule_based_policy = RuleBased()

        # Decision and event logs in memory
        self.decision_log = []
        self.event_log = []

        # Cached RL model
        self.model = None
        self._load_strategy_model()

    @property
    def energy_system(self):
        """Accesses the underlying physical/simulated device collection."""
        return self.simulator.energy_system

    def is_override_active(self) -> bool:
        """Returns True if a manual battery override is currently in effect."""
        return bool(self.active_override is not None and self.active_override.get("hours_left", 0) > 0)

    def set_manual_override(self, mode: str, hours: Optional[int] = None):
        """
        Overrides Auto-Pilot for N hours (default 2) with a manual battery directive.
        Modes: 'CHARGE' (0), 'DISCHARGE' (1), 'HOLD' (2).
        """
        mode_upper = str(mode).strip().upper()
        action_map = {"CHARGE": 0, "DISCHARGE": 1, "HOLD": 2}
        if mode_upper not in action_map:
            raise ValueError(f"Unknown override mode '{mode}'. Choose from: CHARGE, DISCHARGE, HOLD")

        duration = int(hours) if hours is not None else self.default_override_hours
        self.active_override = {
            "mode": mode_upper,
            "action": action_map[mode_upper],
            "hours_left": duration,
            "total_hours": duration,
        }
        self.log_event(
            "OVERRIDE_ACTIVATED",
            f"Manual battery override '{mode_upper}' activated for {duration} hours.",
        )

    def clear_override(self):
        """Clears any active manual override and resumes automatic control."""
        if self.active_override:
            mode = self.active_override.get("mode")
            self.active_override = None
            self.log_event("OVERRIDE_CLEARED", f"Manual override ({mode}) cleared. Auto-Pilot resumed.")

    def log_event(self, event_type: str, details: str):
        """Records an event in memory."""
        self.event_log.append({
            "hour_index": self.simulator.current_index,
            "event_type": str(event_type),
            "details": str(details),
        })

    def _load_strategy_model(self):
        """Attempts to load the chosen RL model, or falls back to Rule-Based if missing."""
        self.model_notice = None
        if self.strategy == "Rule-Based":
            self.model = None
            return

        model_paths = {
            "DQN": "models/dqn_grid.zip",
            "PPO": "models/ppo_grid.zip",
            "DQN+Forecast": "models/dqn_forecast.zip",
            "DQN + Forecast": "models/dqn_forecast.zip",
        }

        path = model_paths.get(self.strategy)
        if not path or not os.path.exists(path):
            self.model = None
            self.model_notice = f"Notice: Model file '{path}' not found. Falling back to Rule-Based strategy."
            self.log_event("MODEL_FALLBACK", self.model_notice)
            return

        try:
            if "PPO" in self.strategy:
                from stable_baselines3 import PPO
                self.model = PPO.load(path)
            else:
                from stable_baselines3 import DQN
                self.model = DQN.load(path)
        except Exception as err:
            self.model = None
            self.model_notice = f"Notice: Failed to load '{path}' ({err}). Falling back to Rule-Based strategy."
            self.log_event("MODEL_LOAD_ERROR", self.model_notice)

    def build_observation(self, state: dict) -> np.ndarray:
        """
        Constructs the exact normalized observation vector expected by the RL agent.
        Adapts cleanly when devices are OFF or in FAULT (e.g. seeing 0 kW for solar if offline).
        """
        solar_kw = float(state.get("solar_kw", 0.0))
        wind_kw = float(state.get("wind_kw", 0.0))
        demand_kw = float(state.get("demand_kw", 0.0))
        battery_level = float(state.get("battery_level", 0.5))
        price = float(state.get("price", 5.0))
        hour = int(state.get("hour", 0))
        dow = int(state.get("day_of_week", 0))

        is_peak = 1.0 if (PEAK_HOURS[0] <= hour < PEAK_HOURS[1]) else 0.0

        obs_9 = [
            solar_kw / SOLAR_CAPACITY_KW,
            wind_kw / WIND_CAPACITY_KW,
            demand_kw / 100.0,
            battery_level,
            price / PEAK_PRICE,
            (np.sin(2.0 * np.pi * hour / 24.0) + 1.0) / 2.0,
            (np.cos(2.0 * np.pi * hour / 24.0) + 1.0) / 2.0,
            dow / 6.0,
            is_peak,
        ]

        if "forecast" in self.strategy.lower():
            # Extract 3h and 6h predictions from dataset row if available
            row = self.simulator.data.iloc[self.simulator.current_index]
            p3 = float(row.get("solar_pred_3h_norm", 0.0))
            p6 = float(row.get("solar_pred_6h_norm", 0.0))
            obs_11 = obs_9 + [p3, p6]
            return np.clip(np.array(obs_11, dtype=np.float32), 0.0, 1.0)

        return np.clip(np.array(obs_9, dtype=np.float32), 0.0, 1.0)

    def filter_action(self, candidate_action: int, state: dict) -> Tuple[int, Optional[str]]:
        """
        Safety layer from CommandController:
        - Never discharge below 10%
        - Never charge above 100%
        - Never act on FAULT or disabled devices
        """
        battery = self.energy_system.battery
        grid = self.energy_system.grid
        soc = float(state.get("battery_level", self.simulator.battery_level))
        renew = float(state.get("solar_kw", 0.0)) + float(state.get("wind_kw", 0.0))
        demand = float(state.get("demand_kw", 0.0))
        has_surplus = renew > demand

        # 1. Device FAULT: Battery
        if battery.status == "FAULT" and candidate_action in [0, 1]:
            if battery.fault_type != "battery degradation":
                fallback = 3 if (has_surplus and grid.status == "ON") else 2
                return fallback, f"Safety intervention: Battery in FAULT state ({battery.fault_type}). Blocked battery action; routed to {ACTION_NAMES[fallback]}."

        # 2. Device OFF: Battery
        if battery.status == "OFF" and candidate_action in [0, 1]:
            fallback = 3 if (has_surplus and grid.status == "ON") else 2
            return fallback, f"Safety intervention: Battery is switched OFF. Switched to {ACTION_NAMES[fallback]}."

        # 3. Minimum SOC Safety limit (<= 10%)
        if candidate_action == 1 and soc <= 0.10:
            fallback = 2  # BUY
            return fallback, f"Safety intervention: Battery at {soc * 100:.1f}% (<= 10% safety limit). Discharge blocked to prevent cell degradation; switched to BUY."

        # 4. Maximum Capacity limit (>= 100%)
        if candidate_action == 0 and soc >= (1.0 - 1e-4):
            fallback = 3 if (has_surplus and grid.status == "ON") else 2
            return fallback, f"Safety intervention: Battery at 100% capacity. Charge blocked; switched to {ACTION_NAMES[fallback]}."

        # 5. Grid Disconnection / Islanding
        if grid.status != "ON" and candidate_action in [2, 3]:
            # In island mode, cannot import or export
            if battery.status == "ON" and soc > 0.10 and not has_surplus:
                return 1, "Safety intervention: Grid disconnected (island mode). Switched from BUY to DISCHARGE."
            elif battery.status == "ON" and soc < 1.0 and has_surplus:
                return 0, "Safety intervention: Grid disconnected (island mode). Switched from SELL to CHARGE."
            else:
                return 2, "Safety intervention: Grid disconnected (island mode). Operating on available generation."

        return candidate_action, None

    def generate_reason(
        self,
        action: int,
        state: dict,
        strategy: str,
        safety_reason: Optional[str] = None,
    ) -> str:
        """Constructs a one-line plain-English explanation for the decision."""
        if safety_reason:
            return safety_reason

        solar = float(state.get("solar_kw", 0.0))
        wind = float(state.get("wind_kw", 0.0))
        renew = solar + wind
        demand = float(state.get("demand_kw", 0.0))
        soc = float(state.get("battery_level", 0.0)) * 100
        price = float(state.get("price", 5.0))
        hour = int(state.get("hour", 12))
        is_peak = (PEAK_HOURS[0] <= hour < PEAK_HOURS[1])

        if action == 0:  # CHARGE
            if renew > demand:
                return f"Surplus clean energy ({renew - demand:.1f} kW), battery at {soc:.0f}%: charging battery."
            else:
                return f"Off-peak tariff (₹{price:.0f}/kWh), battery at {soc:.0f}%: pre-charging from grid."
        elif action == 1:  # DISCHARGE
            if is_peak:
                return f"Peak tariff window (₹{price:.0f}/kWh), battery at {soc:.0f}%: discharging to avoid grid costs."
            else:
                return f"Energy deficit ({demand - renew:.1f} kW), battery at {soc:.0f}%: discharging clean storage."
        elif action == 2:  # BUY
            return f"Deficit ({demand - renew:.1f} kW), battery conserved at {soc:.0f}%: importing from main grid."
        elif action == 3:  # SELL
            return f"Surplus clean energy ({renew - demand:.1f} kW), battery at {soc:.0f}%: exporting to grid for revenue."

        return f"Action {ACTION_NAMES.get(action, str(action))} commanded by {strategy}."

    def step(self) -> Dict[str, Any]:
        """
        Executes one hourly simulation step under Auto-Pilot:
        1. Checks manual override.
        2. Checks diagnostics for critical faults (falls back to Rule-Based if critical).
        3. Generates candidate action using strategy.
        4. Filters through CommandController safety layer.
        5. Executes step on simulator and logs result.
        """
        state = self.simulator.get_state()
        actual_strategy = self.strategy
        safety_intervention = None

        # 1. Manual override check
        if self.is_override_active():
            candidate_action = self.active_override["action"]
            mode = self.active_override["mode"]
            hours_left = self.active_override["hours_left"]

            reason = f"Manual override active ({mode}, {hours_left}h remaining)."
            self.active_override["hours_left"] -= 1
            if self.active_override["hours_left"] <= 0:
                self.clear_override()

            # Filter override through safety layer as well
            final_action, safety_intervention = self.filter_action(candidate_action, state)
            if safety_intervention:
                reason = safety_intervention

        else:
            # 2. Check diagnostic scan for critical alerts
            diag_report = scan(simulator=self.simulator, energy_system=self.energy_system)
            if diag_report.critical_issues:
                first_crit = diag_report.critical_issues[0]
                actual_strategy = "Conservative Rule-Based Fallback"
                candidate_action = self.rule_based_policy.act(state)
                reason = f"Emergency Fallback: Critical scan alert ({first_crit['description']}). Switched to Rule-Based."
                final_action, safety_intervention = self.filter_action(candidate_action, state)
                if safety_intervention:
                    reason = safety_intervention
            else:
                # 3. Model or Rule-Based decision
                if self.model is not None:
                    obs = self.build_observation(state)
                    pred, _ = self.model.predict(obs, deterministic=True)
                    candidate_action = int(pred)
                else:
                    actual_strategy = "Rule-Based"
                    candidate_action = self.rule_based_policy.act(state)

                # 4. Filter candidate action through safety layer
                final_action, safety_intervention = self.filter_action(candidate_action, state)
                reason = self.generate_reason(final_action, state, actual_strategy, safety_intervention)

        # 5. Execute action in simulator
        result = self.simulator.step(final_action)

        # 6. Record decision in log
        decision_record = {
            "index": state["index"],
            "hour": state["hour"],
            "strategy": actual_strategy,
            "action": final_action,
            "action_name": ACTION_NAMES[final_action],
            "reason": reason,
            "safety_intervention": safety_intervention,
            "override_active": self.is_override_active(),
            "cost_inr": result["cost_inr"],
            "battery_level": result["battery_level"],
        }
        self.decision_log.append(decision_record)

        return {
            "action": final_action,
            "action_name": ACTION_NAMES[final_action],
            "reason": reason,
            "strategy_used": actual_strategy,
            "override_active": self.is_override_active(),
            "safety_intervention": safety_intervention,
            "state": state,
            "result": result,
        }

    def run(self, steps: int = 24) -> list:
        """Runs Auto-Pilot for a specified number of consecutive hours."""
        outputs = []
        for _ in range(steps):
            out = self.step()
            outputs.append(out)
            if out["result"]["done"]:
                break
        return outputs
