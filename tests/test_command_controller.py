import os
import sys
import pytest

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.devices import EnergySystem
from src.command_controller import CommandController, CommandResult


def test_command_result_properties():
    res = CommandResult(True, "Success", {"solar": "ON"}, warning="Minor warning")
    assert res.success is True
    assert res.message == "Success"
    assert res["solar"] == "ON"
    assert res.warning == "Minor warning"


def test_parse_solar_commands():
    controller = CommandController()
    # Test forgiving natural language input
    res_off = controller.execute("please switch the solar panel off")
    assert res_off.success is True
    assert res_off.changed.get("solar") == "OFF"
    assert controller.energy_system.solar.status == "OFF"

    res_on = controller.execute("turn solar on")
    assert res_on.success is True
    assert res_on.changed.get("solar") == "ON"
    assert controller.energy_system.solar.status == "ON"


def test_parse_wind_commands():
    controller = CommandController()
    res_off = controller.execute("turn off the wind turbine")
    assert res_off.success is True
    assert res_off.changed.get("wind") == "OFF"
    assert controller.energy_system.wind.status == "OFF"

    res_on = controller.execute("wind on")
    assert res_on.success is True
    assert res_on.changed.get("wind") == "ON"
    assert controller.energy_system.wind.status == "ON"


def test_parse_battery_modes():
    controller = CommandController()
    controller.set_battery_level(0.5)

    res_charge = controller.execute("battery charge")
    assert res_charge.success is True
    assert controller.battery_mode == "CHARGE"

    res_discharge = controller.execute("please discharge the battery now")
    assert res_discharge.success is True
    assert controller.battery_mode == "DISCHARGE"

    res_hold = controller.execute("hold battery")
    assert res_hold.success is True
    assert controller.battery_mode == "HOLD"


def test_parse_grid_connect_disconnect():
    controller = CommandController()
    res_disc = controller.execute("disconnect the main grid")
    assert res_disc.success is True
    assert controller.energy_system.grid.status == "OFF"

    res_conn = controller.execute("grid connect")
    assert res_conn.success is True
    assert controller.energy_system.grid.status == "ON"


def test_parse_auto_on_off():
    controller = CommandController()
    res_off = controller.execute("auto off")
    assert res_off.success is True
    assert controller.auto_mode is False

    res_on = controller.execute("turn auto on")
    assert res_on.success is True
    assert controller.auto_mode is True


def test_parse_scan_status_help():
    controller = CommandController()
    res_scan = controller.execute("scan system health")
    assert res_scan.success is True
    assert "System scan complete" in res_scan.message

    res_status = controller.execute("status")
    assert res_status.success is True
    assert "Solar:" in res_status.message

    res_help = controller.execute("help")
    assert res_help.success is True
    assert "Supported commands:" in res_help.message


def test_safety_conflicting_commands_blocked():
    controller = CommandController()
    # Conflicting charge and discharge
    res_conflict1 = controller.execute("charge and discharge the battery at once")
    assert res_conflict1.success is False
    assert "Conflicting command" in res_conflict1.message

    # Conflicting on and off
    res_conflict2 = controller.execute("turn solar on and off")
    assert res_conflict2.success is False
    assert "Conflicting command" in res_conflict2.message


def test_safety_battery_limits():
    controller = CommandController()

    # Low battery: refuse discharge below 10%
    controller.set_battery_level(0.08)
    res_dis = controller.execute("discharge battery")
    assert res_dis.success is False
    assert "10% minimum safety threshold" in res_dis.message

    # Full battery: refuse charge at 100%
    controller.set_battery_level(1.0)
    res_chg = controller.execute("charge battery")
    assert res_chg.success is False
    assert "100% capacity" in res_chg.message


def test_safety_refuse_on_when_device_faulted():
    controller = CommandController()
    # Introduce fault on solar
    controller.energy_system.apply_fault("solar", "inverter_overheat", 0.5)
    assert controller.energy_system.solar.status == "FAULT"

    # Attempt to turn solar ON
    res_fail = controller.execute("solar on")
    assert res_fail.success is False
    assert "FAULT state" in res_fail.message
    assert "Clear the fault before switching ON" in res_fail.message

    # Clear the fault
    res_clear = controller.execute("clear fault solar")
    assert res_clear.success is True
    assert controller.energy_system.solar.status == "ON"

    # Now turning solar ON works
    res_ok = controller.execute("solar on")
    assert res_ok.success is True


def test_safety_grid_disconnect_warning():
    controller = CommandController()
    # High demand context without sufficient renewables
    context = {"demand_kw": 80.0, "solar_kw": 10.0, "wind_kw": 5.0, "battery_available_kw": 10.0}
    res = controller.execute("disconnect grid", context=context)
    assert res.success is True
    assert res.warning is not None
    assert "risks unmet demand" in res.warning


def test_audit_logs():
    controller = CommandController()
    controller.execute("solar off")
    controller.execute("charge and discharge the battery")

    assert len(controller.command_log) == 2
    assert controller.command_log[0]["command"] == "solar off"
    assert controller.command_log[0]["success"] is True
    assert controller.command_log[1]["success"] is False

    # Event log has entries
    assert len(controller.event_log) >= 2
    event_types = [e["event_type"] for e in controller.event_log]
    assert "SAFETY_BLOCK" in event_types
