"""
Unit tests for AutonomousOperator: autonomous supervisory control,
diagnostics integration, supervisor policies, safety filtering, and incident recovery.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.config import BATTERY_CAPACITY_KWH
from src.grid_simulator import MicrogridSimulator
from src.devices import EnergySystem
from src.scenarios import ScenarioEngine, ScenarioEvent
from src.autonomous_operator import AutonomousOperator


@pytest.fixture
def clean_operator():
    """Returns a freshly initialized AutonomousOperator with Rule-Based strategy."""
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="Rule-Based")
    return op


def test_operator_initialization_and_run_step(clean_operator):
    """Verifies that run_step() executes the 7-phase control cycle and returns structured record."""
    op = clean_operator
    rec = op.run_step()

    assert "hour" in rec
    assert rec["hour"] == 0
    assert "obs_summary" in rec
    assert "action" in rec
    assert "action_name" in rec
    assert "reason" in rec
    assert "cost_inr" in rec
    assert "battery_level" in rec
    assert "load_shed_kwh" in rec
    assert len(op.decision_log) == 1
    assert len(op.step_history) == 1
    assert op.current_hour == 1


def test_periodic_and_immediate_diagnostics(clean_operator):
    """Verifies diagnostics.scan() runs every 6 hours and immediately when an anomaly is suspected."""
    op = clean_operator

    # Step 0: runs initial scan (hour 0 - (-99) >= 6)
    op.run_step()
    assert op.last_scan_hour == 0

    # Step 1-5: should not scan under normal operation
    for _ in range(5):
        op.run_step()
    assert op.last_scan_hour == 0

    # Step 6: 6-hour periodic scan triggers
    op.run_step()
    assert op.last_scan_hour == 6

    # Step 7: Inject anomaly (inverter failure) -> immediate scan triggered!
    op.energy_system.inject_fault("inverter failure")
    op.run_step()
    assert op.last_scan_hour == 7
    assert op.latest_scan_report is not None
    assert len(op.latest_scan_report.critical_issues) >= 1


def test_supervisor_solar_fault_restart_and_maintenance():
    """
    Verifies Supervisor Policy for solar fault:
    - continues operation and logs lost generation
    - attempts automatic clear/restart after 3 hours
    - marks for maintenance if fault persists for >= 6 hours
    """
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="Rule-Based")
    op.reset(start_index=10)  # midday hour with sunlight

    # Inject inverter failure
    op.energy_system.inject_fault("inverter failure")
    assert op.energy_system.solar.status == "FAULT"

    # Step 1 hour: fault tracked
    rec1 = op.run_step()
    assert op.solar_fault_hours == 1
    assert any("SOLAR_LOST_GEN" in e["event_type"] for e in op.controller.event_log)

    # Step 2nd hour: fault tracked
    op.run_step()
    assert op.solar_fault_hours == 2

    # Step 3rd hour: automatic restart attempted!
    op.run_step()
    assert any("SOLAR_RESTART" in e["event_type"] for e in op.controller.event_log)

    # Re-inject fault to simulate persistent hardware defect
    op.energy_system.inject_fault("inverter failure")
    for _ in range(3):
        op.run_step()

    # After 6 total hours of fault, marked for maintenance
    assert op.solar_fault_hours >= 6
    assert op.solar_marked_for_maintenance is True
    assert any("SOLAR_MAINTENANCE" in e["event_type"] for e in op.controller.event_log)


def test_supervisor_grid_outage_island_and_load_shedding():
    """
    Verifies Supervisor Policy for grid outage:
    - switches to island mode
    - prioritizes battery for demand
    - sheds non-critical load (30%) if battery is nearly empty (SoC <= 15%)
    """
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="Rule-Based", enable_load_shedding=True)
    op.reset(start_index=20)  # evening hour (no solar, high demand)

    # Inject grid outage
    op.energy_system.inject_fault("grid outage")
    assert op.energy_system.grid.status == "FAULT"

    # Battery starts at 50% (> 10%): should prioritize battery discharge (action 1)
    rec = op.run_step()
    assert rec["action"] == 1  # DISCHARGE
    assert "Supervisor Policy: Grid outage island mode" in rec["reason"]

    # Now force battery nearly empty (<= 15%)
    op.simulator.battery_level = 0.12
    rec_shed = op.run_step()
    assert rec_shed["load_shed_kwh"] > 0.0
    assert any("LOAD_SHEDDING_ACTIVE" in e["event_type"] for e in op.controller.event_log)


def test_supervisor_battery_fault_grid_fallback():
    """Verifies that when battery is in FAULT, operator bypasses battery and buys from grid."""
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="Rule-Based")
    op.reset(start_index=12)

    op.energy_system.battery.status = "FAULT"
    op.energy_system.battery.fault_type = "BMS overvoltage trip"

    rec = op.run_step()
    assert rec["action"] == 2  # BUY
    assert "Supervisor Policy: Battery in FAULT" in rec["reason"]


def test_supervisor_price_spike_discharge():
    """Verifies that during electricity price spike, battery discharge is prioritized if SoC > 15%."""
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="Rule-Based")
    op.reset(start_index=19)  # evening
    op.simulator.battery_level = 0.50

    op.energy_system.price_multiplier = 2.0  # 2x peak price = ₹20/kWh

    rec = op.run_step()
    assert rec["action"] == 1  # DISCHARGE
    assert "price spike" in rec["reason"].lower()


def test_critical_scan_fallback_and_recovery():
    """Verifies switching from RL strategy to conservative Rule-Based fallback on critical scan alert."""
    sim = MicrogridSimulator()
    # Use DQN strategy
    op = AutonomousOperator(simulator=sim, strategy="DQN")
    op.reset(start_index=0)

    # Inject inverter failure -> critical issue
    op.energy_system.inject_fault("inverter failure")
    rec = op.run_step()

    # Conservative fallback engaged
    assert op.conservative_fallback_active is True
    assert "Conservative" in rec["strategy"] or "Fallback" in rec["strategy"] or "Supervisor" in rec["strategy"]
    assert any("FALLBACK_ENGAGED" in e["event_type"] for e in op.controller.event_log)

    # Clear fault and advance until scan clears
    op.energy_system.clear_faults()
    # Advance past next scan interval (6 hours)
    for _ in range(6):
        op.run_step()

    # Conservative fallback disengaged
    assert op.conservative_fallback_active is False
    assert any("FALLBACK_DISENGAGED" in e["event_type"] for e in op.controller.event_log)


def test_run_grid_outage_drill_scenario():
    """
    Executes a multi-day simulation with 'Grid outage drill' scenario preset.
    Confirms history DataFrame, incident logging, and summary KPIs.
    """
    op = AutonomousOperator(strategy="Rule-Based")
    df, incidents, kpis = op.run(hours=72, scenario="Grid outage drill")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 72
    assert "cum_cost" in df.columns
    assert "cum_co2" in df.columns
    assert isinstance(incidents, list)
    assert len(incidents) >= 1  # Grid outage was detected and logged

    # Verify incident structure
    inc = incidents[0]
    assert "incident_id" in inc
    assert "device" in inc
    assert "detected_hour" in inc
    assert "responded_hour" in inc
    assert "status" in inc

    # Verify KPIs dictionary
    assert "total_cost" in kpis or "total_cost_inr" in kpis
    assert "co2" in kpis or "total_co2_kg" in kpis
    assert "unmet_demand" in kpis or "unmet_demand_kwh" in kpis
    assert "number_of_incidents" in kpis
    assert "average_detection_delay" in kpis or "average_detection_delay_hours" in kpis
    assert "average_time_to_recover" in kpis or "average_time_to_recover_hours" in kpis
    assert kpis["number_of_incidents"] == len(incidents)


def test_run_stress_test_scenario():
    """
    Executes an autonomous week-long simulation with 'Stress test' preset.
    Confirms multiple overlapping fault handling, load shedding, and recovery tracking.
    """
    op = AutonomousOperator(strategy="Rule-Based")
    df, incidents, kpis = op.run(hours=168, scenario="Stress test")

    assert len(df) == 168
    assert len(incidents) >= 2  # Compounding events logged
    assert kpis["number_of_incidents"] >= 2
    # Check that resolved incidents have time_to_recover computed
    resolved = [i for i in incidents if i["status"] == "RESOLVED"]
    assert len(resolved) >= 1
    for r in resolved:
        assert r["time_to_recover"] is not None
        assert r["time_to_recover"] >= 0


def test_missing_model_fallback_notice():
    """Verifies that missing model files fall back to Rule-Based with a notice."""
    sim = MicrogridSimulator()
    op = AutonomousOperator(simulator=sim, strategy="NonExistentModel")
    assert op.autopilot.model is None
    assert op.model_notice is not None
    assert "not found" in op.model_notice.lower() or "falling back" in op.model_notice.lower()

    rec = op.run_step()
    assert rec["strategy"] == "Rule-Based"
