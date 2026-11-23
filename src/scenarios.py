"""
ScenarioEngine for microgrid stress testing, operational drills, and reproducible event timelines.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_SEED
from src.devices import EnergySystem
from src.grid_simulator import MicrogridSimulator


class ScenarioEvent:
    """Represents a scheduled environmental disruption or hardware fault event."""

    def __init__(
        self,
        event_type: str,
        start_hour: int,
        duration: int,
        severity: float = 1.0,
        description: str = "",
    ):
        self.event_type = str(event_type)
        self.start_hour = int(start_hour)
        self.duration = int(duration)
        self.severity = float(severity)
        self.description = description or f"{self.event_type} (start={self.start_hour}h, dur={self.duration}h, sev={self.severity})"

    @property
    def end_hour(self) -> int:
        """The hour at which the event concludes and normal operation resumes."""
        return self.start_hour + self.duration

    def is_active(self, hour: int) -> bool:
        """Returns True if the event is active at the given hour."""
        return self.start_hour <= hour < self.end_hour

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary representation of the event."""
        return {
            "event_type": self.event_type,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "duration": self.duration,
            "severity": self.severity,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"<ScenarioEvent: {self.event_type} ({self.start_hour}h-{self.end_hour}h, sev={self.severity})>"


class ScenarioEngine:
    """
    Generates and executes reproducible event timelines for microgrid testing.
    Supports 4 presets: 'Normal week', 'Stress test', 'Grid outage drill', and seeded 'Random'.
    """

    EVENT_CATALOG = [
        "cloudy spell",
        "heatwave",
        "solar soiling",
        "inverter failure",
        "battery degradation",
        "sensor stuck",
        "wind turbine trip",
        "grid outage",
        "price spike",
    ]

    def __init__(self, seed: int = RANDOM_SEED, total_hours: int = 168):
        self.seed = int(seed)
        self.total_hours = int(total_hours)
        self.preset_name = "Normal week"
        self.timeline: List[ScenarioEvent] = []
        self.active_events: List[ScenarioEvent] = []
        self.event_history: List[str] = []

        # Automatically generate default timeline
        self.generate_timeline(preset="Normal week", seed=self.seed, total_hours=self.total_hours)

    def generate_timeline(
        self,
        preset: str = "Normal week",
        seed: Optional[int] = None,
        total_hours: Optional[int] = None,
    ) -> List[ScenarioEvent]:
        """
        Generates a chronological list of ScenarioEvents based on preset configuration and seed.
        """
        if seed is not None:
            self.seed = int(seed)
        if total_hours is not None:
            self.total_hours = int(total_hours)

        self.preset_name = str(preset)
        self.active_events.clear()
        events = []

        if preset == "Normal week":
            # Few minor non-overlapping events over 168 hours
            events.append(
                ScenarioEvent(
                    event_type="cloudy spell",
                    start_hour=32,
                    duration=8,
                    severity=0.30,
                    description="Cloudy spell: Solar output reduced to 30% for 8 hours",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="price spike",
                    start_hour=90,
                    duration=4,
                    severity=2.00,
                    description="Evening price spike: Peak tariff multiplied by 2x for 4 hours",
                )
            )

        elif preset == "Stress test":
            # Compounding severe events with overlapping failures
            events.append(
                ScenarioEvent(
                    event_type="heatwave",
                    start_hour=24,
                    duration=48,
                    severity=1.25,
                    description="Severe heatwave: Campus demand increased by +25% for 48 hours",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="cloudy spell",
                    start_hour=36,
                    duration=18,
                    severity=0.30,
                    description="Overcast monsoon spell: Solar output reduced to 30% during heatwave",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="grid outage",
                    start_hour=42,
                    duration=6,
                    severity=0.00,
                    description="Substation breaker trip: 6-hour utility grid outage during peak heatwave",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="wind turbine trip",
                    start_hour=84,
                    duration=12,
                    severity=0.00,
                    description="Wind turbine mechanical trip: Zero wind generation for 12 hours",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="battery degradation",
                    start_hour=100,
                    duration=36,
                    severity=0.70,
                    description="Thermal BESS degradation: Round-trip efficiency drops to 70% for 36 hours",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="price spike",
                    start_hour=110,
                    duration=4,
                    severity=2.00,
                    description="Grid tariff spike: Electricity price doubled for 4 hours",
                )
            )

        elif preset == "Grid outage drill":
            # Coordinated resilience drill: 8-hour outage during high heatwave demand
            events.append(
                ScenarioEvent(
                    event_type="heatwave",
                    start_hour=18,
                    duration=42,
                    severity=1.25,
                    description="Summer heatwave: Demand elevated +25%",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="grid outage",
                    start_hour=24,
                    duration=8,
                    severity=0.00,
                    description="Grid outage drill: 8-hour disconnection (island mode) during heatwave",
                )
            )
            events.append(
                ScenarioEvent(
                    event_type="sensor stuck",
                    start_hour=28,
                    duration=14,
                    severity=1.00,
                    description="Telemetry sensor flatline: Solar irradiance sensor frozen",
                )
            )

        elif preset == "Random":
            # Deterministically seeded random timeline
            rng = np.random.RandomState(self.seed)
            num_events = rng.randint(3, 7)
            chosen_types = rng.choice(self.EVENT_CATALOG, size=num_events, replace=False)

            for etype in chosen_types:
                dur = int(self._get_random_duration(etype, rng))
                max_start = max(1, self.total_hours - dur - 4)
                start_h = int(rng.randint(4, max_start))
                sev = float(self._get_event_severity(etype))
                events.append(
                    ScenarioEvent(
                        event_type=etype,
                        start_hour=start_h,
                        duration=dur,
                        severity=sev,
                        description=f"{etype.title()} event (duration={dur}h, severity={sev})",
                    )
                )

        else:
            raise ValueError(f"Unknown preset '{preset}'. Available: 'Normal week', 'Stress test', 'Grid outage drill', 'Random'")

        # Sort timeline chronologically by start hour
        events.sort(key=lambda e: e.start_hour)
        self.timeline = events
        return self.timeline

    def _get_random_duration(self, event_type: str, rng: np.random.RandomState) -> int:
        """Returns typical duration range for a specific event type."""
        ranges = {
            "cloudy spell": (6, 24),
            "heatwave": (12, 48),
            "solar soiling": (24, 72),
            "inverter failure": (6, 18),
            "battery degradation": (12, 48),
            "sensor stuck": (8, 24),
            "wind turbine trip": (6, 18),
            "grid outage": (2, 8),
            "price spike": (2, 6),
        }
        low, high = ranges.get(event_type, (4, 12))
        return int(rng.randint(low, high + 1))

    def _get_event_severity(self, event_type: str) -> float:
        """Returns standard severity factor for an event type."""
        severities = {
            "cloudy spell": 0.30,
            "heatwave": 1.25,
            "solar soiling": 0.70,
            "inverter failure": 0.00,
            "battery degradation": 0.70,
            "sensor stuck": 1.00,
            "wind turbine trip": 0.00,
            "grid outage": 0.00,
            "price spike": 2.00,
        }
        return severities.get(event_type, 1.0)

    def apply_hour(
        self,
        hour: int,
        energy_system: EnergySystem,
        simulator: Optional[MicrogridSimulator] = None,
    ) -> List[str]:
        """
        Injects events that start at the given hour, and clears events that end.
        Returns a list of human-readable log messages of actions taken.
        """
        logs = []

        # 1. Check for events ending at this hour -> REMOVE
        for event in list(self.active_events):
            if hour >= event.end_hour:
                self.active_events.remove(event)
                msg = f"Hour {hour}: Event ended -> '{event.event_type}'. Restoring normal device operation."
                logs.append(msg)
                self.event_history.append(msg)
                self._remove_event(event, energy_system)

        # 2. Check for events starting at this hour -> INJECT
        for event in self.timeline:
            if event.start_hour == hour and event not in self.active_events:
                self.active_events.append(event)
                msg = f"Hour {hour}: Event injected -> '{event.event_type}' ({event.description})."
                logs.append(msg)
                self.event_history.append(msg)
                self._inject_event(event, energy_system)

        return logs

    def _inject_event(self, event: ScenarioEvent, energy_system: EnergySystem):
        """Applies physical modification or fault injection to the EnergySystem."""
        etype = event.event_type.lower()

        if etype == "cloudy spell":
            energy_system.solar.status = "FAULT"
            energy_system.solar.fault_type = "cloudy spell"
            energy_system.solar.derate = float(event.severity)

        elif etype == "heatwave":
            energy_system.demand_multiplier = float(event.severity)

        elif etype == "solar soiling":
            energy_system.inject_fault("solar soiling")

        elif etype == "inverter failure":
            energy_system.inject_fault("inverter failure")

        elif etype == "battery degradation":
            energy_system.inject_fault("battery degradation")

        elif etype == "sensor stuck":
            energy_system.inject_fault("sensor stuck", target="solar")

        elif etype == "wind turbine trip":
            energy_system.wind.status = "FAULT"
            energy_system.wind.fault_type = "wind turbine trip"
            energy_system.wind.derate = 0.0

        elif etype == "grid outage":
            energy_system.inject_fault("grid outage")

        elif etype == "price spike":
            energy_system.price_multiplier = float(event.severity)

    def _remove_event(self, event: ScenarioEvent, energy_system: EnergySystem):
        """Restores normal device parameters when an event expires."""
        etype = event.event_type.lower()

        if etype in ["cloudy spell", "solar soiling", "inverter failure"]:
            # If no other solar fault is active, restore solar
            other_solar = any(
                e.event_type in ["cloudy spell", "solar soiling", "inverter failure"]
                for e in self.active_events
            )
            if not other_solar:
                energy_system.solar.enable()

        elif etype == "heatwave":
            energy_system.demand_multiplier = 1.0

        elif etype == "battery degradation":
            other_bat = any(e.event_type == "battery degradation" for e in self.active_events)
            if not other_bat:
                energy_system.battery.enable()

        elif etype == "sensor stuck":
            energy_system.solar.sensor_stuck = False
            energy_system.solar.enable()

        elif etype == "wind turbine trip":
            energy_system.wind.enable()

        elif etype == "grid outage":
            energy_system.grid.enable()

        elif etype == "price spike":
            energy_system.price_multiplier = 1.0

    def get_active_events(self, hour: int) -> List[ScenarioEvent]:
        """Returns all events active at a given simulated hour."""
        return [e for e in self.timeline if e.is_active(hour)]

    def get_timeline(self) -> List[ScenarioEvent]:
        """Returns the full list of scheduled events."""
        return list(self.timeline)
