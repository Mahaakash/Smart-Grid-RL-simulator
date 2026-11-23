import os
import sys
import pytest

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.devices import EnergySystem
from src.grid_simulator import MicrogridSimulator
from src.scenarios import ScenarioEngine, ScenarioEvent


def test_seed_reproducibility():
    """Confirms that the same random seed produces the exact same event timeline."""
    engine1 = ScenarioEngine(seed=42)
    timeline1 = engine1.generate_timeline(preset="Random", seed=42)

    engine2 = ScenarioEngine(seed=42)
    timeline2 = engine2.generate_timeline(preset="Random", seed=42)

    assert len(timeline1) == len(timeline2)
    assert len(timeline1) >= 3

    for e1, e2 in zip(timeline1, timeline2):
        assert e1.event_type == e2.event_type
        assert e1.start_hour == e2.start_hour
        assert e1.duration == e2.duration
        assert e1.severity == e2.severity
        assert e1.description == e2.description

    # Different seed produces different timeline
    engine3 = ScenarioEngine(seed=999)
    timeline3 = engine3.generate_timeline(preset="Random", seed=999)
    assert any(
        e1.event_type != e3.event_type or e1.start_hour != e3.start_hour
        for e1, e3 in zip(timeline1, timeline3)
    )


def test_presets_structure():
    """Validates the 4 built-in scenario presets."""
    engine = ScenarioEngine(seed=42)

    # 1. Normal week
    norm_tl = engine.generate_timeline("Normal week")
    assert len(norm_tl) == 2
    types = [e.event_type for e in norm_tl]
    assert "cloudy spell" in types
    assert "price spike" in types

    # 2. Stress test
    stress_tl = engine.generate_timeline("Stress test")
    assert len(stress_tl) == 6
    stress_types = [e.event_type for e in stress_tl]
    assert "heatwave" in stress_types
    assert "grid outage" in stress_types
    assert "cloudy spell" in stress_types
    assert "battery degradation" in stress_types

    # 3. Grid outage drill
    drill_tl = engine.generate_timeline("Grid outage drill")
    assert len(drill_tl) == 3
    drill_types = [e.event_type for e in drill_tl]
    assert "grid outage" in drill_types
    assert "heatwave" in drill_types
    outage_event = [e for e in drill_tl if e.event_type == "grid outage"][0]
    assert outage_event.duration == 8

    # 4. Random
    rand_tl = engine.generate_timeline("Random", seed=123)
    assert len(rand_tl) >= 3


def test_event_injection_and_removal_cycle():
    """Confirms that apply_hour correctly injects events at start hour and restores devices when they end."""
    es = EnergySystem()
    sim = MicrogridSimulator(energy_system=es)
    sim.reset(0)

    engine = ScenarioEngine(seed=42)
    # Manually configure a simple 2-event timeline
    engine.timeline = [
        ScenarioEvent("heatwave", start_hour=5, duration=3, severity=1.25),
        ScenarioEvent("grid outage", start_hour=6, duration=2, severity=0.0),
    ]

    # Hour 4: Normal baseline
    engine.apply_hour(4, es, sim)
    assert es.demand_multiplier == 1.0
    assert es.grid.status == "ON"

    # Hour 5: Heatwave begins (demand +25%)
    logs5 = engine.apply_hour(5, es, sim)
    assert any("heatwave" in l for l in logs5)
    assert es.demand_multiplier == 1.25
    state5 = sim.get_state()
    raw_demand = sim.data.iloc[sim.current_index]["demand_kw"]
    assert round(state5["demand_kw"], 2) == round(raw_demand * 1.25, 2)

    # Hour 6: Grid outage begins (island mode)
    logs6 = engine.apply_hour(6, es, sim)
    assert any("grid outage" in l for l in logs6)
    assert es.grid.status == "FAULT"

    # Hour 8: Grid outage ends (start 6 + dur 2 = 8), heatwave ends (start 5 + dur 3 = 8)
    logs8 = engine.apply_hour(8, es, sim)
    assert any("grid outage" in l for l in logs8)
    assert es.grid.status == "ON"
    assert es.demand_multiplier == 1.0


def test_cloudy_spell_and_price_spike():
    """Confirms cloudy spell reduces solar output by 0.3x and price spike doubles tariff."""
    es = EnergySystem()
    sim = MicrogridSimulator(energy_system=es)
    sim.reset(0)

    engine = ScenarioEngine(seed=42)
    engine.timeline = [
        ScenarioEvent("cloudy spell", start_hour=10, duration=4, severity=0.30),
        ScenarioEvent("price spike", start_hour=10, duration=2, severity=2.00),
    ]

    # Inject at hour 10
    engine.apply_hour(10, es, sim)
    assert round(es.solar.derate, 2) == 0.30
    assert es.price_multiplier == 2.00

    state10 = sim.get_state()
    raw_price = sim.data.iloc[sim.current_index]["price_per_kwh"]
    assert round(state10["price"], 2) == round(raw_price * 2.0, 2)

    # Hour 12: price spike ends (10 + 2 = 12), cloudy spell still active
    engine.apply_hour(12, es, sim)
    assert es.price_multiplier == 1.00
    assert round(es.solar.derate, 2) == 0.30

    # Hour 14: cloudy spell ends (10 + 4 = 14)
    engine.apply_hour(14, es, sim)
    assert es.solar.status == "ON"
    assert round(es.solar.derate, 2) == 1.00
