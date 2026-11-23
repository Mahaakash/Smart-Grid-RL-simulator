import os
import sys
import pytest

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid_simulator import MicrogridSimulator
from src.autopilot import AutoPilot, ACTION_NAMES
from src.command_controller import CommandController


def test_autopilot_step_decision_logging():
    """Auto-Pilot steps simulation, decides actions automatically, and logs a one-line reason."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based")

    out = ap.step()
    assert "action" in out
    assert out["action"] in [0, 1, 2, 3]
    assert out["action_name"] in ACTION_NAMES.values()
    assert isinstance(out["reason"], str) and len(out["reason"]) > 10
    assert len(ap.decision_log) == 1
    assert ap.decision_log[0]["action"] == out["action"]


def test_autopilot_observation_device_adaptation():
    """When a device is OFF or in FAULT, observation adapts cleanly (e.g. solar off -> 0)."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based")

    # Disable solar
    ap.energy_system.set_device_state("solar", "OFF")
    state = sim.get_state()
    assert state["solar_kw"] == 0.0

    obs = ap.build_observation(state)
    assert obs[0] == 0.0  # Normalized solar observation is 0.0

    # AutoPilot executes cleanly
    out = ap.step()
    assert out["action"] in [0, 1, 2, 3]


def test_autopilot_safety_layer_filters_discharge_below_10():
    """Safety layer filters every automatic action: never discharges below 10%."""
    sim = MicrogridSimulator()
    sim.reset(0)
    sim.battery_level = 0.08  # 8% (below 10% safety limit)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based")

    state = sim.get_state()
    # Force a candidate action of 1 (DISCHARGE)
    filtered_action, safety_reason = ap.filter_action(1, state)

    # Must be blocked and switched to BUY (2)
    assert filtered_action == 2
    assert safety_reason is not None
    assert "10% safety limit" in safety_reason


def test_autopilot_safety_layer_fault_device():
    """Safety layer never acts on FAULT devices (battery fault blocks charge/discharge)."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based")

    # Inject internal fault on battery
    ap.energy_system.battery.status = "FAULT"
    ap.energy_system.battery.fault_type = "thermal_runaway"

    state = sim.get_state()
    action, safety_reason = ap.filter_action(0, state)  # Candidate CHARGE
    assert action in [2, 3]
    assert "FAULT state" in safety_reason

    action2, safety_reason2 = ap.filter_action(1, state)  # Candidate DISCHARGE
    assert action2 in [2, 3]
    assert "FAULT state" in safety_reason2


def test_autopilot_manual_override_and_expiration():
    """A manual battery command overrides Auto-Pilot for N hours (default 2) and then resumes."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based", override_hours=2)

    # Set manual override to CHARGE for 2 hours
    ap.set_manual_override("CHARGE", hours=2)
    assert ap.is_override_active() is True
    assert ap.active_override["hours_left"] == 2

    # Step 1: override is active
    out1 = ap.step()
    assert out1["override_active"] is True
    assert out1["action"] == 0  # CHARGE
    assert "Manual override active" in out1["reason"]
    assert ap.active_override["hours_left"] == 1

    # Step 2: override is active on last hour
    out2 = ap.step()
    assert out2["action"] == 0  # CHARGE

    # Step 3: override has expired, Auto-Pilot resumes
    assert ap.is_override_active() is False
    out3 = ap.step()
    assert out3["override_active"] is False


def test_autopilot_critical_scan_fallback():
    """If a scan finds a critical issue, Auto-Pilot switches to conservative rule-based fallback and logs reason."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based")

    # Inject critical fault: grid outage
    ap.energy_system.inject_fault("grid outage")

    out = ap.step()
    assert "Emergency Fallback" in out["reason"] or "Conservative Rule-Based" in out["strategy_used"]
    assert len(ap.decision_log) >= 1
    assert "Emergency Fallback" in ap.decision_log[-1]["reason"]


def test_autopilot_missing_model_fallback():
    """Auto-Pilot falls back to Rule-Based with a notice if model files are missing."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="NonExistentModel")

    assert ap.model is None
    assert ap.model_notice is not None
    assert "Falling back to Rule-Based" in ap.model_notice

    # Step executes smoothly via Rule-Based fallback
    out = ap.step()
    assert out["action"] in [0, 1, 2, 3]
    assert out["strategy_used"] == "Rule-Based"


def test_command_controller_triggers_autopilot_override():
    """CommandController triggers manual override on AutoPilot when a manual battery command is received."""
    sim = MicrogridSimulator()
    sim.reset(0)
    ap = AutoPilot(simulator=sim, strategy="Rule-Based", override_hours=2)
    controller = CommandController(autopilot=ap)

    assert ap.is_override_active() is False

    # Issue manual battery charge command
    res = controller.execute("battery charge")
    assert res.success is True
    assert "Auto-Pilot overridden for 2 hours" in res.message
    assert ap.is_override_active() is True
    assert ap.active_override["mode"] == "CHARGE"
    assert ap.active_override["hours_left"] == 2
