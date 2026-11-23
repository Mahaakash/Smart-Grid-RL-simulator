"""
Diagnostics subsystem for microgrid health monitoring, anomaly detection, and fault root-cause analysis.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.devices import EnergySystem


class DiagnosticReport(list):
    """A list of diagnostic issue dicts with convenience filtering properties."""

    @property
    def is_clean(self) -> bool:
        """Returns True if no issues were detected."""
        return len(self) == 0

    @property
    def critical_issues(self) -> List[Dict[str, Any]]:
        """Returns only critical severity issues."""
        return [i for i in self if i.get("severity") == "critical"]

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        """Returns only warning severity issues."""
        return [i for i in self if i.get("severity") == "warning"]

    @property
    def info_alerts(self) -> List[Dict[str, Any]]:
        """Returns only info severity issues."""
        return [i for i in self if i.get("severity") == "info"]

    def summary(self) -> str:
        """Returns a formatted human-readable summary of all issues."""
        if self.is_clean:
            return "System healthy: All diagnostic checks passed with zero anomalies."
        lines = [f"Found {len(self)} diagnostic issue(s):"]
        for idx, item in enumerate(self, 1):
            lines.append(
                f"{idx}. [{item['severity'].upper()}] {item['device'].upper()}: "
                f"{item['description']} -> Fix: {item['suggested_fix']}"
            )
        return "\n".join(lines)


def scan(
    simulator: Any = None,
    energy_system: Optional[EnergySystem] = None,
    history: Any = None,
    window_hours: int = 24,
) -> DiagnosticReport:
    """
    Inspects recent simulation history and current device states to detect faults and anomalies.

    Args:
        simulator: Optional MicrogridSimulator instance.
        energy_system: Optional EnergySystem instance. Defaults to simulator.energy_system or a new EnergySystem.
        history: Optional history list of dicts or pd.DataFrame. Defaults to simulator.history.
        window_hours: Number of recent hourly steps to inspect (default: 24).

    Returns:
        DiagnosticReport: A list of issue dicts, each containing:
            - 'severity': 'info', 'warning', or 'critical'
            - 'device': component name ('solar', 'wind', 'battery', 'grid', 'demand')
            - 'description': plain-English description of the anomaly
            - 'suggested_fix': actionable recommendation to resolve the issue
    """
    report = DiagnosticReport()

    # 1. Resolve EnergySystem
    if energy_system is None:
        if simulator is not None and hasattr(simulator, "energy_system"):
            energy_system = simulator.energy_system
        else:
            energy_system = EnergySystem()

    # 2. Resolve recent history records
    history_records = []
    if history is not None:
        if isinstance(history, pd.DataFrame):
            history_records = history.tail(window_hours).to_dict(orient="records")
        elif isinstance(history, list):
            history_records = history[-window_hours:]
    elif simulator is not None and hasattr(simulator, "history") and simulator.history:
        history_records = simulator.history[-window_hours:]

    solar = energy_system.solar
    wind = energy_system.wind
    battery = energy_system.battery
    grid = energy_system.grid

    # =========================================================================
    # CHECK 1: Solar Underperformance & Inverter Faults
    # =========================================================================
    if solar.fault_type == "inverter failure" or (solar.status == "FAULT" and solar.derate == 0.0):
        report.append({
            "severity": "critical",
            "device": "solar",
            "description": "Solar inverter failure: Solar array output is at 0 kW despite available sunlight.",
            "suggested_fix": "Inspect inverter AC/DC contactor breakers, check DC isolator switch, and reboot inverter controller.",
        })
    elif solar.fault_type == "solar soiling" or (solar.status == "FAULT" and 0.0 < solar.derate < 0.85):
        loss_pct = round((1.0 - solar.derate) * 100)
        report.append({
            "severity": "warning",
            "device": "solar",
            "description": f"Solar underperformance detected: Output reduced by {loss_pct}% below expected radiation due to dust/dirt soiling or partial shading.",
            "suggested_fix": "Schedule physical PV module cleaning and inspect for physical shading obstructions.",
        })
    elif solar.status == "OFF":
        report.append({
            "severity": "info",
            "device": "solar",
            "description": "Solar PV array is switched OFF.",
            "suggested_fix": "Enable solar array via command controller or dashboard when daylight is available.",
        })

    # History-based solar check (actual vs expected)
    if history_records:
        daylight_steps = [
            r for r in history_records
            if (6 <= r.get("hour", 0) <= 17) and r.get("expected_solar_kw", 0.0) >= 20.0
        ]
        if daylight_steps and not any(i["device"] == "solar" and i["severity"] in ["warning", "critical"] for i in report):
            avg_actual = np.mean([r.get("solar_kw", 0.0) for r in daylight_steps])
            avg_expected = np.mean([r.get("expected_solar_kw", 0.0) for r in daylight_steps])
            if avg_expected > 0.0 and (avg_actual / avg_expected) < 0.70:
                report.append({
                    "severity": "warning",
                    "device": "solar",
                    "description": f"Solar generation is underperforming solar radiation data (averaging {avg_actual:.1f} kW vs {avg_expected:.1f} kW expected).",
                    "suggested_fix": "Check for panel dirt accumulation, micro-cracks, or string inverter clipping.",
                })

    # =========================================================================
    # CHECK 2: Battery Problems (Degradation, stuck SOC, repeatedly hitting limits)
    # =========================================================================
    bat_eff = getattr(battery, "efficiency", 1.0)
    if battery.fault_type == "battery degradation" or bat_eff < 0.85:
        eff_pct = round(bat_eff * 100, 1)
        report.append({
            "severity": "warning",
            "device": "battery",
            "description": f"Abnormal battery degradation: Round-trip efficiency has dropped to {eff_pct}% (normal is ~95%).",
            "suggested_fix": "Execute BMS cell balancing calibration, inspect thermal coolant loops, and review cell internal impedance.",
        })
    elif battery.status == "FAULT":
        report.append({
            "severity": "critical",
            "device": "battery",
            "description": f"Battery storage in FAULT state ({battery.fault_type or 'internal fault'}). Charge and discharge disabled.",
            "suggested_fix": "Inspect BMS alarm register, check safety disconnect fuses, and clear fault.",
        })
    elif battery.status == "OFF":
        report.append({
            "severity": "info",
            "device": "battery",
            "description": "Battery BESS is switched OFF.",
            "suggested_fix": "Enable battery to allow peak shaving and renewable storage buffering.",
        })

    # History-based battery checks
    if history_records and len(history_records) >= 6:
        soc_series = [r.get("battery_level", 0.5) for r in history_records]
        actions = [r.get("action", 0) for r in history_records]
        # Check if SOC is stuck while active charge/discharge is requested
        if np.std(soc_series) < 1e-5 and any(a in [0, 1] for a in actions):
            report.append({
                "severity": "warning",
                "device": "battery",
                "description": f"Battery SOC stuck at {soc_series[-1]*100:.1f}% across {len(history_records)} consecutive active hours.",
                "suggested_fix": "Recalibrate BMS State of Charge Coulomb counter and inspect power converter gate drivers.",
            })

        # Repeatedly hitting capacity limits (0% or 100%)
        limit_hits = sum(1 for s in soc_series if s <= 0.05 or s >= 0.95)
        if limit_hits >= (0.75 * len(soc_series)) and len(soc_series) >= 12:
            report.append({
                "severity": "info",
                "device": "battery",
                "description": f"Battery frequently hitting capacity limits (0% or 100%) during {limit_hits}/{len(soc_series)} hours.",
                "suggested_fix": "Review battery capacity sizing or adjust charge/discharge scheduling heuristics.",
            })

    # =========================================================================
    # CHECK 3: Sensor Flatline (Reading that never changes for many hours)
    # =========================================================================
    # Check device-level sensor stuck fault
    for dev_name, dev in energy_system.devices.items():
        if getattr(dev, "sensor_stuck", False) or dev.fault_type == "sensor stuck":
            report.append({
                "severity": "warning",
                "device": dev_name,
                "description": f"Sensor flatline on {dev_name}: Telemetry signal remains frozen with zero variance.",
                "suggested_fix": f"Check {dev_name} sensor wiring harness, verify ADC bus communications, and reboot telemetry gateway.",
            })

    # History-based flatline detection
    if history_records and len(history_records) >= 6:
        for col, dev_name in [("solar_kw", "solar"), ("wind_kw", "wind"), ("demand_kw", "demand")]:
            vals = [r.get(col, 0.0) for r in history_records]
            # If all values are non-zero and identical for >= 6 consecutive hours
            if len(vals) >= 6 and np.all(np.array(vals) > 0.0) and np.std(vals) < 1e-6:
                if not any(i["device"] == dev_name and "flatline" in i["description"] for i in report):
                    report.append({
                        "severity": "warning",
                        "device": dev_name,
                        "description": f"Sensor flatline detected: {col} signal has remained completely unchanged at {vals[0]:.2f} kW for {len(vals)} hours.",
                        "suggested_fix": f"Inspect {dev_name} transducer and check if sensor telemetry buffer is frozen.",
                    })

    # =========================================================================
    # CHECK 4: Wind Turbine Producing Nothing During Good Wind
    # =========================================================================
    if wind.status == "FAULT":
        report.append({
            "severity": "critical",
            "device": "wind",
            "description": f"Wind turbine in FAULT state ({wind.fault_type or 'rotor lock'}). No power generated.",
            "suggested_fix": "Inspect yaw drive mechanism, release mechanical brake, and clear turbine alarm.",
        })
    elif wind.status == "OFF":
        report.append({
            "severity": "info",
            "device": "wind",
            "description": "Wind turbine is switched OFF.",
            "suggested_fix": "Enable wind turbine when generation is desired.",
        })

    # History check: good wind conditions but zero output
    if history_records:
        zero_wind_good_conditions = sum(
            1 for r in history_records
            if r.get("wind_kw", 0.0) == 0.0 and (r.get("expected_wind_kw", 0.0) >= 5.0 or r.get("wind_speed", 0.0) >= 4.0)
        )
        if zero_wind_good_conditions >= 3 and wind.status == "ON" and not any(i["device"] == "wind" for i in report):
            report.append({
                "severity": "critical",
                "device": "wind",
                "description": f"Wind turbine produced 0 kW across {zero_wind_good_conditions} hours of favorable wind conditions.",
                "suggested_fix": "Check blade pitch actuator, verify anemometer sensor reading, and inspect generator inverter.",
            })

    # =========================================================================
    # CHECK 5: Grid Outage or Disconnection
    # =========================================================================
    if grid.fault_type == "grid outage" or (grid.status == "FAULT" and grid.derate == 0.0):
        report.append({
            "severity": "critical",
            "device": "grid",
            "description": "Grid outage detected: Utility grid connection lost or tripped by substation protection relay.",
            "suggested_fix": "Inspect substation PCC circuit breakers, check utility transformer status, and coordinate with grid operator.",
        })
    elif grid.status == "OFF":
        report.append({
            "severity": "warning",
            "device": "grid",
            "description": "Main grid is disconnected (microgrid operating in island mode).",
            "suggested_fix": "Reconnect to the utility grid once mains voltage and frequency are stable.",
        })

    # =========================================================================
    # CHECK 6: Demand Spikes Well Above Normal for That Hour
    # =========================================================================
    if history_records:
        demand_vals = [r.get("demand_kw", 0.0) for r in history_records]
        max_demand = max(demand_vals) if demand_vals else 0.0
        # Normal campus business peak is ~50-60 kW, spikes > 85 kW are abnormal
        if max_demand > 85.0:
            report.append({
                "severity": "warning",
                "device": "demand",
                "description": f"Campus demand spike detected: Peak reached {max_demand:.1f} kW, significantly higher than typical 50-60 kW load.",
                "suggested_fix": "Investigate heavy loads (HVAC chillers, lab machinery, EV charging clusters) and apply demand response curtailment.",
            })

    # =========================================================================
    # CHECK 7: Unmet Demand Events
    # =========================================================================
    if history_records:
        unmet_events = [r for r in history_records if r.get("unmet_demand_kwh", 0.0) > 0.0]
        if unmet_events:
            total_unmet = sum(r.get("unmet_demand_kwh", 0.0) for r in unmet_events)
            report.append({
                "severity": "critical",
                "device": "demand",
                "description": f"Unmet demand event recorded: Total of {total_unmet:.1f} kWh unserved demand during islanded operation.",
                "suggested_fix": "Reconnect to the main grid immediately, start standby generator, or activate priority load shedding.",
            })

    return report
