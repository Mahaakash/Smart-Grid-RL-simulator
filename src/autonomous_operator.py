"""
AutonomousOperator: Fully autonomous supervisory microgrid control and resilience management.
Runs observation, diagnostic scans, RL/heuristic decision dispatch, safety filtering,
supervisor policy fault response, and structured audit/incident logging without human input.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RANDOM_SEED,
    BATTERY_CAPACITY_KWH,
    PEAK_PRICE,
    PEAK_HOURS,
)
from src.grid_simulator import MicrogridSimulator
from src.devices import EnergySystem
from src.command_controller import CommandController
from src.autopilot import AutoPilot, ACTION_NAMES
from src.diagnostics import scan, DiagnosticReport
from src.scenarios import ScenarioEngine, ScenarioEvent
from src.baselines import RuleBased


class AutonomousOperator:
    """
    Autonomous supervisory microgrid manager.
    Coordinates observation, diagnostics, decision-making, safety filtering,
    incident response, load shedding, and recovery tracking.
    """

    def __init__(
        self,
        simulator: Optional[MicrogridSimulator] = None,
        strategy: str = "DQN",
        scenario_engine: Optional[ScenarioEngine] = None,
        enable_load_shedding: bool = True,
    ):
        self.simulator = simulator if simulator is not None else MicrogridSimulator()
        self.strategy = strategy
        self.scenario_engine = scenario_engine
        self.enable_load_shedding = enable_load_shedding

        # Subsystems
        self.autopilot = AutoPilot(simulator=self.simulator, strategy=self.strategy)
        self.controller = CommandController(autopilot=self.autopilot)
        self.rule_based_fallback = RuleBased()
        self.model_notice = self.autopilot.model_notice

        # Operational tracking states
        self.current_hour = 0
        self.last_scan_hour = -99
        self.latest_scan_report: Optional[DiagnosticReport] = None
        self.conservative_fallback_active = False

        # Supervisor fault tracking
        self.solar_fault_hours = 0
        self.solar_marked_for_maintenance = False
        self.load_shedding_active = False

        # Incident log: list of dicts with detected -> responded -> resolved tracking
        self.incident_log: List[Dict[str, Any]] = []
        self._active_incidents: Dict[str, Dict[str, Any]] = {}
        self._incident_counter = 0

        # Structured decision log and step records
        self.decision_log: List[Dict[str, Any]] = []
        self.step_history: List[Dict[str, Any]] = []

    @property
    def energy_system(self) -> EnergySystem:
        return self.simulator.energy_system

    def reset(self, start_index: int = 0):
        """Resets the operator, simulator, and logs to the beginning."""
        self.current_hour = start_index
        self.last_scan_hour = -99
        self.latest_scan_report = None
        self.conservative_fallback_active = False
        self.solar_fault_hours = 0
        self.solar_marked_for_maintenance = False
        self.load_shedding_active = False
        self.incident_log.clear()
        self._active_incidents.clear()
        self._incident_counter = 0
        self.decision_log.clear()
        self.step_history.clear()
        self.simulator.reset(start_index)
        self.energy_system.clear_faults()
        self.autopilot.clear_override()

    def run_step(self) -> Dict[str, Any]:
        """
        Executes one full autonomous supervisory control cycle for the current simulated hour.
        Steps:
        1. OBSERVE: read state and device statuses.
        2. DIAGNOSE: scan every 6h or immediately on suspected anomaly.
        3. DECIDE: get action from strategy (or fallback).
        4. SAFETY: pass action through safety layer.
        5. ACT: apply action via simulator/energy system.
        6. RESPOND: supervisor policy (solar restart, load shed, price spike, fault handling).
        7. LOG: structured decision record and incident updates.
        """
        hour = self.current_hour
        es = self.energy_system

        # Apply scenario timeline events if an engine is attached
        if self.scenario_engine is not None:
            self.scenario_engine.apply_hour(hour, es, self.simulator)

        # -------------------------------------------------------------
        # 1. OBSERVE
        # -------------------------------------------------------------
        state = self.simulator.get_state()
        device_statuses = es.get_status()
        raw_solar = float(self.simulator.data.iloc[self.simulator.current_index]["solar_kw"])
        raw_wind = float(self.simulator.data.iloc[self.simulator.current_index]["wind_kw"])
        effective_solar = float(state["solar_kw"])
        effective_wind = float(state["wind_kw"])
        demand_kw = float(state["demand_kw"])
        price = float(state["price"])
        battery_level = float(state["battery_level"])

        # Check for active device faults
        active_fault_names = [k for k, v in device_statuses.items() if v["status"] == "FAULT"]

        # Check for sensor stuck or flatline
        sensor_flatline_suspected = any(
            getattr(dev, "sensor_stuck", False) or getattr(dev, "fault_type", None) == "sensor stuck"
            for dev in es.devices.values()
        )
        if not sensor_flatline_suspected and len(self.step_history) >= 6:
            for col in ["solar_kw", "wind_kw", "demand_kw"]:
                vals = [r.get(col, 0.0) for r in self.step_history[-6:]]
                if np.all(np.array(vals) > 0.0) and np.std(vals) < 1e-6:
                    sensor_flatline_suspected = True
                    break

        # -------------------------------------------------------------
        # 2. DIAGNOSE
        # -------------------------------------------------------------
        suspect_anomaly = (
            (raw_solar > 20.0 and effective_solar < 5.0)  # Solar output far below expectation
            or (raw_wind > 5.0 and effective_wind == 0.0)  # Wind output far below expectation
            or bool(active_fault_names)                     # Active device fault
            or sensor_flatline_suspected                   # Sensor flatline
            or (self.step_history and self.step_history[-1].get("unmet_demand_kwh", 0.0) > 0.0)  # Recent unmet demand
        )

        should_scan = ((hour - self.last_scan_hour) >= 6) or suspect_anomaly
        if should_scan:
            self.latest_scan_report = scan(simulator=self.simulator, energy_system=es, history=self.step_history)
            self.last_scan_hour = hour
            self._update_incidents_from_scan(self.latest_scan_report, hour)

        # Check if critical scan alert warrants conservative fallback
        critical_alerts = self.latest_scan_report.critical_issues if self.latest_scan_report else []
        if critical_alerts and self.strategy != "Rule-Based":
            if not self.conservative_fallback_active:
                self.conservative_fallback_active = True
                self.controller.log_event("FALLBACK_ENGAGED", f"Hour {hour}: Switched from {self.strategy} to Conservative Rule-Based due to critical alert.")
        elif not critical_alerts and self.conservative_fallback_active:
            self.conservative_fallback_active = False
            self.controller.log_event("FALLBACK_DISENGAGED", f"Hour {hour}: Critical issues cleared. Resumed {self.strategy} strategy.")

        # -------------------------------------------------------------
        # 3. DECIDE & 6. RESPOND (Supervisor Policy)
        # -------------------------------------------------------------
        supervisor_reason = None
        forced_action = None

        # Supervisor Rule A: Solar fault handling & lost generation tracking
        if es.solar.status == "FAULT":
            lost_gen = max(0.0, raw_solar - effective_solar)
            self.solar_fault_hours += 1
            if lost_gen > 0:
                self.controller.log_event("SOLAR_LOST_GEN", f"Hour {hour}: Solar in FAULT. Lost generation: {lost_gen:.1f} kWh.")
            if self.solar_fault_hours == 3:
                # Attempt clear fault / restart after a few hours
                es.solar.enable()
                self.controller.log_event("SOLAR_RESTART", f"Hour {hour}: Attempted automatic inverter restart after 3 hours.")
            elif self.solar_fault_hours >= 6 and not self.solar_marked_for_maintenance:
                self.solar_marked_for_maintenance = True
                self.controller.log_event("SOLAR_MAINTENANCE", f"Hour {hour}: Solar inverter fault persists after restart. Marked for maintenance.")
        else:
            self.solar_fault_hours = 0
            self.solar_marked_for_maintenance = False

        # Supervisor Rule B: Grid outage -> Island mode, prioritize battery, shed non-critical load if battery low
        load_shed_kwh = 0.0
        original_demand_mult = getattr(es, "demand_multiplier", 1.0)
        if es.grid.status != "ON":
            if self.enable_load_shedding and battery_level <= 0.15:
                # Battery nearly empty in island mode: shed 30% of demand
                if not self.load_shedding_active:
                    self.load_shedding_active = True
                    self.controller.log_event("LOAD_SHEDDING_ACTIVE", f"Hour {hour}: Battery nearly empty ({battery_level*100:.1f}%) in island mode. Shed 30% non-critical load.")
                load_shed_kwh = demand_kw * 0.30
                demand_kw = demand_kw * 0.70
                # Temporarily update demand multiplier for this step
                es.demand_multiplier = original_demand_mult * 0.70
            else:
                if self.load_shedding_active:
                    self.load_shedding_active = False
                    self.controller.log_event("LOAD_SHEDDING_OFF", f"Hour {hour}: Battery level restored or island mode ended. Resumed full load.")

            # Prioritize battery discharge for demand in island mode
            if battery_level > 0.10 and (effective_solar + effective_wind) < demand_kw and es.battery.status == "ON":
                forced_action = 1  # DISCHARGE
                supervisor_reason = "Supervisor Policy: Grid outage island mode. Discharging battery to protect critical campus load."

        # Supervisor Rule C: Battery fault -> Stop using battery, buy from grid
        if es.battery.status == "FAULT":
            forced_action = 2  # BUY
            fault_desc = es.battery.fault_type or "internal fault"
            supervisor_reason = f"Supervisor Policy: Battery in FAULT ({fault_desc}). Bypassed battery to buy from grid."

        # Supervisor Rule D: Price spike -> Prefer discharging battery if it has charge
        if forced_action is None and (price > PEAK_PRICE or es.price_multiplier > 1.0 or price >= 15.0):
            if battery_level > 0.15 and es.battery.status == "ON":
                forced_action = 1  # DISCHARGE
                supervisor_reason = f"Supervisor Policy: Electricity price spike (₹{price:.1f}/kWh). Discharging battery storage to avoid extreme tariff."

        # Obtain candidate action if not supervisor-forced
        strategy_used = self.strategy
        if forced_action is not None:
            candidate_action = forced_action
            strategy_used = "Supervisor Policy"
        elif self.conservative_fallback_active or self.autopilot.model is None or self.strategy == "Rule-Based":
            candidate_action = self.rule_based_fallback.act(state)
            strategy_used = "Rule-Based" if not self.conservative_fallback_active else "Conservative Rule-Based Fallback"
        else:
            obs = self.autopilot.build_observation(state)
            pred, _ = self.autopilot.model.predict(obs, deterministic=True)
            candidate_action = int(pred)

        # -------------------------------------------------------------
        # 4. SAFETY
        # -------------------------------------------------------------
        final_action, safety_intervention = self.autopilot.filter_action(candidate_action, state)

        # Construct decision reason
        if supervisor_reason:
            reason = supervisor_reason
        elif safety_intervention:
            reason = safety_intervention
        else:
            reason = self.autopilot.generate_reason(final_action, state, strategy_used)

        # -------------------------------------------------------------
        # 5. ACT
        # -------------------------------------------------------------
        step_result = self.simulator.step(final_action)

        # Reset demand multiplier if load shedding was active for this step
        if self.load_shedding_active:
            es.demand_multiplier = original_demand_mult
            if es.grid.status == "ON" or battery_level > 0.15:
                self.load_shedding_active = False

        # -------------------------------------------------------------
        # 7. LOG (Structured decision record and incident log update)
        # -------------------------------------------------------------
        obs_summary = f"Solar:{effective_solar:.1f}kW, Wind:{effective_wind:.1f}kW, Demand:{demand_kw:.1f}kW, SoC:{battery_level*100:.1f}%, Price:₹{price:.1f}"

        decision_record = {
            "hour": hour,
            "day": hour // 24,
            "obs_summary": obs_summary,
            "solar_kw": round(effective_solar, 2),
            "wind_kw": round(effective_wind, 2),
            "demand_kw": round(demand_kw, 2),
            "price": round(price, 2),
            "action": final_action,
            "action_name": ACTION_NAMES[final_action],
            "strategy": strategy_used,
            "reason": reason,
            "active_faults": list(active_fault_names),
            "safety_intervention": safety_intervention,
            "cost_inr": step_result["cost_inr"],
            "co2_kg": step_result["co2_kg"],
            "battery_level": step_result["battery_level"],
            "bought_kwh": step_result["bought_kwh"],
            "sold_kwh": step_result["sold_kwh"],
            "unmet_demand_kwh": step_result["unmet_demand_kwh"],
            "load_shed_kwh": load_shed_kwh,
            "done": step_result["done"],
        }

        self.decision_log.append(decision_record)
        self.step_history.append(decision_record)

        # Check for incident resolutions
        self._check_incident_resolutions(hour)

        # Advance hour index
        self.current_hour += 1

        return decision_record

    def _update_incidents_from_scan(self, report: DiagnosticReport, hour: int):
        """Creates or updates incident log entries from diagnostic scan results."""
        for issue in report:
            dev = issue["device"]
            key = f"{dev}:{issue['description'][:25]}"
            if key not in self._active_incidents:
                self._incident_counter += 1
                detection_delay = 0
                if self.scenario_engine is not None:
                    for ev in self.scenario_engine.active_events:
                        if (ev.event_type.lower() in issue["description"].lower()) or (dev.lower() in ev.event_type.lower()) or (dev == "grid" and "outage" in ev.event_type.lower()):
                            detection_delay = max(0, hour - ev.start_hour)
                            break

                incident = {
                    "incident_id": self._incident_counter,
                    "device": dev,
                    "severity": issue["severity"],
                    "description": issue["description"],
                    "suggested_fix": issue["suggested_fix"],
                    "detected_hour": hour,
                    "responded_hour": hour,  # Supervisor policy responds immediately
                    "resolved_hour": None,
                    "detection_delay": detection_delay,
                    "time_to_recover": None,
                    "status": "ACTIVE",
                }
                self._active_incidents[key] = incident
                self.incident_log.append(incident)

    def _check_incident_resolutions(self, hour: int):
        """Marks resolved incidents when devices return to healthy status."""
        for key in list(self._active_incidents.keys()):
            incident = self._active_incidents[key]
            dev_name = incident["device"]
            dev = self.energy_system.devices.get(dev_name)

            is_resolved = False
            if dev is not None:
                eff_ok = getattr(dev, "efficiency", 1.0) >= 0.90
                stuck_ok = not getattr(dev, "sensor_stuck", False)
                if dev.status == "ON" and dev.derate >= 0.95 and stuck_ok and eff_ok:
                    is_resolved = True
            elif dev_name == "grid" and self.energy_system.grid.status == "ON":
                is_resolved = True
            elif dev_name == "demand" and self.step_history and self.step_history[-1].get("unmet_demand_kwh", 0.0) == 0.0:
                is_resolved = True

            if is_resolved:
                incident["resolved_hour"] = hour
                incident["time_to_recover"] = hour - incident["detected_hour"]
                incident["status"] = "RESOLVED"
                del self._active_incidents[key]

    def run(
        self,
        hours: int = 168,
        scenario: Any = None,
        strategy: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes an end-to-end autonomous simulation run.

        Args:
            hours: Total simulation steps (e.g. 168 for 1 week).
            scenario: Preset name string ('Stress test', 'Grid outage drill', 'Normal week', 'Random'),
                      a ScenarioEngine instance, or None.
            strategy: Optional strategy override ('DQN', 'PPO', 'DQN+Forecast', 'Rule-Based').

        Returns:
            history_df: Full DataFrame containing hourly telemetry and decisions.
            incident_log: List of all detected incidents with timeline tracking.
            kpis: Summary dictionary of key performance indicators.
        """
        if strategy is not None:
            self.strategy = strategy
            self.autopilot.strategy = strategy
            self.autopilot._load_strategy_model()
            self.model_notice = self.autopilot.model_notice

        if isinstance(scenario, ScenarioEngine):
            self.scenario_engine = scenario
        elif isinstance(scenario, str):
            self.scenario_engine = ScenarioEngine(seed=RANDOM_SEED, total_hours=hours)
            self.scenario_engine.generate_timeline(preset=scenario, total_hours=hours)
        else:
            self.scenario_engine = None

        self.reset(start_index=0)

        for _ in range(int(hours)):
            rec = self.run_step()
            if rec["done"]:
                break

        # Assemble History DataFrame
        df = pd.DataFrame(self.step_history)
        if not df.empty:
            df["cum_cost"] = df["cost_inr"].cumsum()
            df["cum_co2"] = df["co2_kg"].cumsum()

        # Compute Summary KPIs
        total_cost = float(df["cost_inr"].sum()) if not df.empty else 0.0
        total_co2 = float(df["co2_kg"].sum()) if not df.empty else 0.0
        unmet_demand = float(df["unmet_demand_kwh"].sum()) if not df.empty else 0.0
        load_shed_total = float(df["load_shed_kwh"].sum()) if not df.empty else 0.0

        detection_delays = [i.get("detection_delay", 0) for i in self.incident_log]
        recovery_times = [i["time_to_recover"] for i in self.incident_log if i.get("time_to_recover") is not None]

        avg_detect = float(np.mean(detection_delays)) if detection_delays else 0.0
        avg_recovery = float(np.mean(recovery_times)) if recovery_times else 0.0

        kpis = {
            "total_cost": round(total_cost, 2),
            "total_cost_inr": round(total_cost, 2),
            "co2": round(total_co2, 2),
            "total_co2_kg": round(total_co2, 2),
            "unmet_demand": round(unmet_demand, 2),
            "unmet_demand_kwh": round(unmet_demand, 2),
            "load_shed_total_kwh": round(load_shed_total, 2),
            "number_of_incidents": len(self.incident_log),
            "average_detection_delay": round(avg_detect, 2),
            "average_detection_delay_hours": round(avg_detect, 2),
            "average_time_to_recover": round(avg_recovery, 2),
            "average_time_to_recover_hours": round(avg_recovery, 2),
        }

        return df, list(self.incident_log), kpis
