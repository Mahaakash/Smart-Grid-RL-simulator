"""
Device abstraction layer and backend interfaces for the microgrid simulator.
"""

from abc import ABC, abstractmethod


class Device:
    """Base class for all microgrid physical and simulated devices."""

    def __init__(self, name, status="ON", derate=1.0):
        self.name = str(name)
        self.status = str(status).upper()  # "ON", "OFF", or "FAULT"
        self.derate = max(0.0, min(1.0, float(derate)))  # 1.0 = healthy, lower = degraded
        self.fault_type = None

    def enable(self):
        """Enables the device (sets status to ON)."""
        self.status = "ON"
        self.fault_type = None
        self.derate = 1.0
        if hasattr(self, "sensor_stuck"):
            self.sensor_stuck = False
        if hasattr(self, "efficiency"):
            from src.config import BATTERY_EFFICIENCY
            self.efficiency = float(BATTERY_EFFICIENCY)

    def disable(self):
        """Disables the device (sets status to OFF)."""
        self.status = "OFF"

    def get_health(self):
        """Returns health and status information for the device."""
        return {
            "name": self.name,
            "status": self.status,
            "derate": float(self.derate),
            "fault_type": self.fault_type,
            "is_healthy": (self.status == "ON" and self.derate >= 1.0),
        }


class SolarArray(Device):
    """Solar photovoltaic array device."""

    def __init__(self, name="solar", status="ON", derate=1.0):
        super().__init__(name=name, status=status, derate=derate)


class WindTurbine(Device):
    """Wind turbine generation device."""

    def __init__(self, name="wind", status="ON", derate=1.0):
        super().__init__(name=name, status=status, derate=derate)


class Battery(Device):
    """Battery energy storage system (BESS) device."""

    def __init__(self, name="battery", status="ON", derate=1.0, efficiency=None):
        super().__init__(name=name, status=status, derate=derate)
        from src.config import BATTERY_EFFICIENCY
        self.efficiency = float(efficiency) if efficiency is not None else float(BATTERY_EFFICIENCY)


class GridConnection(Device):
    """Main grid connection point of common coupling (PCC)."""

    def __init__(self, name="grid", status="ON", derate=1.0):
        super().__init__(name=name, status=status, derate=derate)


class EnergySystem:
    """Holds all microgrid devices and exposes a unified control and telemetry API."""

    def __init__(self, solar=None, wind=None, battery=None, grid=None):
        self.solar = solar if solar is not None else SolarArray("solar")
        self.wind = wind if wind is not None else WindTurbine("wind")
        self.battery = battery if battery is not None else Battery("battery")
        self.grid = grid if grid is not None else GridConnection("grid")

        self.devices = {
            "solar": self.solar,
            "wind": self.wind,
            "battery": self.battery,
            "grid": self.grid,
        }

        # Environmental & Market Modifiers (for scenarios / drills)
        self.demand_multiplier = 1.0
        self.price_multiplier = 1.0

    def _resolve_device(self, name):
        """Resolves device by key, name, or common alias."""
        norm = str(name).strip().lower().replace("_", "").replace(" ", "")
        for key, dev in self.devices.items():
            if norm == key.lower():
                return dev
            if norm == dev.name.lower():
                return dev
            if norm == dev.__class__.__name__.lower():
                return dev

        aliases = {
            "pv": "solar",
            "solararray": "solar",
            "solarpv": "solar",
            "turbine": "wind",
            "windturbine": "wind",
            "bess": "battery",
            "storage": "battery",
            "pcc": "grid",
            "gridconnection": "grid",
            "maingrid": "grid",
        }
        if norm in aliases and aliases[norm] in self.devices:
            return self.devices[aliases[norm]]

        raise KeyError(f"Unknown device: '{name}'. Available: {list(self.devices.keys())}")

    def set_device_state(self, name, on_off):
        """Sets device state to ON or OFF."""
        device = self._resolve_device(name)
        if isinstance(on_off, str):
            is_on = on_off.strip().upper() in ["ON", "TRUE", "ENABLE", "ENABLED", "1"]
        else:
            is_on = bool(on_off)

        if is_on:
            device.enable()
        else:
            device.disable()

    def get_status(self):
        """Returns health and operating status for all devices."""
        return {key: dev.get_health() for key, dev in self.devices.items()}

    def apply_fault(self, name, fault_type, severity=0.5):
        """Applies a fault to a device and derates its capability."""
        device = self._resolve_device(name)
        device.status = "FAULT"
        device.fault_type = str(fault_type)
        clamped_severity = max(0.0, min(1.0, float(severity)))
        device.derate = max(0.0, min(1.0, 1.0 - clamped_severity))

    def inject_fault(self, fault_name, target=None, severity=None):
        """Injects a simulated fault for demonstration, testing, and diagnostics."""
        fname = str(fault_name).strip().lower().replace("_", " ")

        if "soiling" in fname:
            # Solar soiling (output reduced 30%)
            self.solar.status = "FAULT"
            self.solar.fault_type = "solar soiling"
            self.solar.derate = 0.70

        elif "inverter" in fname:
            # Inverter failure (solar to 0)
            self.solar.status = "FAULT"
            self.solar.fault_type = "inverter failure"
            self.solar.derate = 0.0

        elif "battery degradation" in fname or "degradation" in fname:
            # Battery degradation (efficiency drops)
            self.battery.status = "FAULT"
            self.battery.fault_type = "battery degradation"
            self.battery.efficiency = 0.70
            self.battery.derate = 0.70

        elif "sensor stuck" in fname or "flatline" in fname:
            # Sensor stuck
            dev = self._resolve_device(target) if target else self.solar
            dev.status = "FAULT"
            dev.fault_type = "sensor stuck"
            dev.sensor_stuck = True

        elif "grid outage" in fname or "outage" in fname:
            # Grid outage
            self.grid.status = "FAULT"
            self.grid.fault_type = "grid outage"
            self.grid.derate = 0.0

        else:
            dev_name = target or "solar"
            sev = severity if severity is not None else 0.5
            self.apply_fault(dev_name, fault_name, severity=sev)

    def clear_faults(self):
        """Clears all faults across all devices and restores 100% capacity."""
        from src.config import BATTERY_EFFICIENCY
        for dev in self.devices.values():
            dev.enable()
            dev.fault_type = None
            dev.derate = 1.0
            if hasattr(dev, "sensor_stuck"):
                dev.sensor_stuck = False
            if hasattr(dev, "efficiency"):
                dev.efficiency = float(BATTERY_EFFICIENCY)
        self.demand_multiplier = 1.0
        self.price_multiplier = 1.0


class EnergyBackend(ABC):
    """Abstract interface for energy backends (simulated or real hardware)."""

    @abstractmethod
    def reset(self, start_index=0):
        """Resets the backend and returns the initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Simulates or commands one step given an action and returns the transition result."""
        pass

    @abstractmethod
    def get_state(self):
        """Returns the current state dictionary."""
        pass


class SimulatedBackend(EnergyBackend):
    """Simulation backend using MicrogridSimulator and EnergySystem."""

    def __init__(self, simulator=None, energy_system=None, data_path=None):
        from src.grid_simulator import MicrogridSimulator

        if simulator is not None:
            self.simulator = simulator
            if energy_system is not None:
                self.simulator.energy_system = energy_system
        else:
            kwargs = {}
            if energy_system is not None:
                kwargs["energy_system"] = energy_system
            if data_path is not None:
                kwargs["data_path"] = data_path
            self.simulator = MicrogridSimulator(**kwargs)

    @property
    def energy_system(self):
        return self.simulator.energy_system

    def reset(self, start_index=0):
        return self.simulator.reset(start_index=start_index)

    def step(self, action):
        return self.simulator.step(action)

    def get_state(self):
        return self.simulator.get_state()

    def set_device_state(self, name, on_off):
        self.energy_system.set_device_state(name, on_off)

    def get_status(self):
        return self.energy_system.get_status()

    def apply_fault(self, name, fault_type, severity=0.5):
        self.energy_system.apply_fault(name, fault_type, severity)


class HardwareBackend(EnergyBackend):
    """Placeholder hardware backend for physical microgrid devices (Modbus/MQTT)."""

    def __init__(self, host="localhost", port=502):
        self.host = host
        self.port = port

    def reset(self, start_index=0):
        raise NotImplementedError(
            "HardwareBackend is not implemented yet. Connect real microgrid hardware (Modbus/MQTT) to enable."
        )

    def step(self, action):
        raise NotImplementedError(
            "HardwareBackend is not implemented yet. Connect real microgrid hardware (Modbus/MQTT) to enable."
        )

    def get_state(self):
        raise NotImplementedError(
            "HardwareBackend is not implemented yet. Connect real microgrid hardware (Modbus/MQTT) to enable."
        )
