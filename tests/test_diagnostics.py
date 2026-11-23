import os
import sys
import pytest

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.devices import EnergySystem
from src.grid_simulator import MicrogridSimulator
from src.diagnostics import scan, DiagnosticReport


def test_clean_report_on_healthy_system():
    """Confirms that a healthy system gives a clean report with zero issues."""
    es = EnergySystem()
    report = scan(energy_system=es)
    assert isinstance(report, list)
    assert report.is_clean is True
    assert len(report) == 0
    assert "healthy" in report.summary().lower()


def test_detect_solar_soiling_fault():
    """Confirms that injecting 'solar soiling' is detected as solar underperformance."""
    es = EnergySystem()
    es.inject_fault("solar soiling")
    assert es.solar.status == "FAULT"
    assert round(es.solar.derate, 2) == 0.70

    report = scan(energy_system=es)
    assert report.is_clean is False
    solar_issues = [i for i in report if i["device"] == "solar"]
    assert len(solar_issues) >= 1
    issue = solar_issues[0]
    assert issue["severity"] == "warning"
    assert "soiling" in issue["description"].lower() or "underperformance" in issue["description"].lower()
    assert "cleaning" in issue["suggested_fix"].lower()


def test_detect_inverter_failure_fault():
    """Confirms that injecting 'inverter failure' is detected as a critical solar issue."""
    es = EnergySystem()
    es.inject_fault("inverter failure")
    assert es.solar.status == "FAULT"
    assert es.solar.derate == 0.0

    report = scan(energy_system=es)
    assert report.is_clean is False
    solar_issues = [i for i in report if i["device"] == "solar"]
    assert len(solar_issues) >= 1
    issue = solar_issues[0]
    assert issue["severity"] == "critical"
    assert "inverter failure" in issue["description"].lower()


def test_detect_battery_degradation_fault():
    """Confirms that injecting 'battery degradation' is detected as an abnormal battery issue."""
    es = EnergySystem()
    es.inject_fault("battery degradation")
    assert es.battery.status == "FAULT"
    assert es.battery.efficiency < 0.85

    report = scan(energy_system=es)
    assert report.is_clean is False
    bat_issues = [i for i in report if i["device"] == "battery"]
    assert len(bat_issues) >= 1
    issue = bat_issues[0]
    assert issue["severity"] == "warning"
    assert "degradation" in issue["description"].lower() or "efficiency" in issue["description"].lower()


def test_detect_sensor_stuck_fault():
    """Confirms that injecting 'sensor stuck' is detected as sensor flatline."""
    es = EnergySystem()
    es.inject_fault("sensor stuck", target="solar")

    report = scan(energy_system=es)
    assert report.is_clean is False
    flatline_issues = [i for i in report if "flatline" in i["description"].lower() or "sensor" in i["description"].lower()]
    assert len(flatline_issues) >= 1
    assert flatline_issues[0]["severity"] == "warning"


def test_detect_grid_outage_fault():
    """Confirms that injecting 'grid outage' is detected as a critical grid fault."""
    es = EnergySystem()
    es.inject_fault("grid outage")
    assert es.grid.status == "FAULT"

    report = scan(energy_system=es)
    assert report.is_clean is False
    grid_issues = [i for i in report if i["device"] == "grid"]
    assert len(grid_issues) >= 1
    assert grid_issues[0]["severity"] == "critical"
    assert "grid outage" in grid_issues[0]["description"].lower()


def test_clear_faults_restores_clean_report():
    """Confirms that clear_faults() clears all active faults and returns a clean report."""
    es = EnergySystem()
    es.inject_fault("solar soiling")
    es.inject_fault("grid outage")
    assert len(scan(energy_system=es)) >= 2

    es.clear_faults()
    report = scan(energy_system=es)
    assert report.is_clean is True


def test_detect_wind_producing_nothing_during_good_wind():
    """Confirms scan detects wind turbine producing 0 kW despite favorable wind speed."""
    es = EnergySystem()
    history = [
        {"hour": 10, "wind_speed": 6.5, "expected_wind_kw": 12.0, "wind_kw": 0.0, "solar_kw": 50.0, "demand_kw": 40.0},
        {"hour": 11, "wind_speed": 7.0, "expected_wind_kw": 14.0, "wind_kw": 0.0, "solar_kw": 60.0, "demand_kw": 45.0},
        {"hour": 12, "wind_speed": 6.8, "expected_wind_kw": 13.0, "wind_kw": 0.0, "solar_kw": 65.0, "demand_kw": 45.0},
    ]
    report = scan(energy_system=es, history=history)
    wind_issues = [i for i in report if i["device"] == "wind"]
    assert len(wind_issues) >= 1
    assert wind_issues[0]["severity"] == "critical"
    assert "0 kw" in wind_issues[0]["description"].lower()


def test_detect_demand_spike():
    """Confirms scan detects demand spikes well above normal campus baseline."""
    es = EnergySystem()
    history = [
        {"hour": 14, "demand_kw": 55.0, "solar_kw": 40.0, "wind_kw": 5.0},
        {"hour": 15, "demand_kw": 110.0, "solar_kw": 35.0, "wind_kw": 5.0},  # Abnormal spike!
    ]
    report = scan(energy_system=es, history=history)
    demand_issues = [i for i in report if i["device"] == "demand" and "spike" in i["description"].lower()]
    assert len(demand_issues) >= 1
    assert demand_issues[0]["severity"] == "warning"
    assert "110.0 kw" in demand_issues[0]["description"].lower()


def test_detect_unmet_demand_event():
    """Confirms scan detects unmet demand during islanding or capacity deficit."""
    es = EnergySystem()
    history = [
        {"hour": 19, "demand_kw": 50.0, "unmet_demand_kwh": 15.5},
    ]
    report = scan(energy_system=es, history=history)
    unmet_issues = [i for i in report if i["device"] == "demand" and "unmet" in i["description"].lower()]
    assert len(unmet_issues) >= 1
    assert unmet_issues[0]["severity"] == "critical"
    assert "15.5 kwh" in unmet_issues[0]["description"].lower()


def test_detect_stuck_battery_soc():
    """Confirms scan detects battery SOC stuck over consecutive active hours."""
    es = EnergySystem()
    # 8 consecutive hours where action is CHARGE but SOC does not move
    history = [
        {"hour": h, "battery_level": 0.45, "action": 0, "demand_kw": 30.0, "solar_kw": 50.0}
        for h in range(8)
    ]
    report = scan(energy_system=es, history=history)
    bat_stuck = [i for i in report if i["device"] == "battery" and "stuck" in i["description"].lower()]
    assert len(bat_stuck) >= 1
    assert bat_stuck[0]["severity"] == "warning"


def test_scan_with_simulator_instance():
    """Confirms scan works seamlessly when passed a MicrogridSimulator instance."""
    sim = MicrogridSimulator()
    sim.reset(0)
    for _ in range(5):
        sim.step(0)
    report = scan(simulator=sim)
    assert isinstance(report, DiagnosticReport)
    assert report.is_clean is True
