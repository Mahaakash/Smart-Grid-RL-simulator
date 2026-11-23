import os
import sys
import re
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    BATTERY_CAPACITY_KWH,
    MAX_RATE_KW,
    INITIAL_BATTERY_LEVEL,
)
from src.devices import EnergySystem

# Safety limits for battery state of charge
MIN_BATTERY_SOC = 0.10  # 10% lower limit to prevent cell degradation
MAX_BATTERY_SOC = 1.00  # 100% upper limit


class CommandResult:
    """Result object returned by CommandController.execute()."""

    def __init__(self, success: bool, message: str, changed: dict = None, warning: str = None):
        self.success = bool(success)
        self.message = str(message)
        self.changed = changed if changed is not None else {}
        self.warning = warning

    def __getitem__(self, key):
        """Allows dictionary-style access to attributes or changed items."""
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.changed, dict) and key in self.changed:
            return self.changed[key]
        raise KeyError(key)

    def __repr__(self):
        return f"CommandResult(success={self.success}, message='{self.message}', changed={self.changed})"


class CommandController:
    """Text-based command interface and safety controller for the EnergySystem."""

    def __init__(self, energy_system=None, simulator=None, autopilot=None, override_hours=2):
        if autopilot is not None:
            self.autopilot = autopilot
            self.simulator = autopilot.simulator
            self.energy_system = autopilot.energy_system
        elif simulator is not None:
            self.simulator = simulator
            self.energy_system = simulator.energy_system
            self.autopilot = getattr(simulator, "autopilot", None)
        elif energy_system is not None:
            self.energy_system = energy_system
            self.simulator = None
            self.autopilot = None
        else:
            self.energy_system = EnergySystem()
            self.simulator = None
            self.autopilot = None

        self.auto_mode = True
        self.battery_mode = "HOLD"
        self._battery_level = INITIAL_BATTERY_LEVEL
        self.override_hours = int(override_hours)

        # In-memory timestamped audit logs
        self.command_log = []
        self.event_log = []

    def get_battery_level(self) -> float:
        """Returns the current battery state of charge fraction (0.0 to 1.0)."""
        if self.simulator is not None:
            return float(self.simulator.battery_level)
        return float(self._battery_level)

    def set_battery_level(self, level: float):
        """Sets the battery level for testing or external state synchronization."""
        self._battery_level = max(0.0, min(1.0, float(level)))
        if self.simulator is not None:
            self.simulator.battery_level = self._battery_level

    def log_event(self, event_type: str, details: str):
        """Records an event with a timestamp into memory."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": str(event_type),
            "details": str(details),
        }
        self.event_log.append(record)

    def log_command(self, raw_command: str, result: CommandResult):
        """Records an executed or refused command with its result into memory."""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": str(raw_command),
            "success": result.success,
            "message": result.message,
            "changed": result.changed,
            "warning": result.warning,
        }
        self.command_log.append(record)

    def execute(self, text: str, context: dict = None) -> CommandResult:
        """Parses a text command, applies safety rules, and controls the EnergySystem."""
        raw_text = str(text).strip()
        cmd = raw_text.lower()

        # 1. Safety check: detect conflicting commands
        conflict_result = self._check_conflicting_commands(cmd)
        if conflict_result is not None:
            self.log_event("SAFETY_BLOCK", conflict_result.message)
            self.log_command(raw_text, conflict_result)
            return conflict_result

        # 2. Parse command intent
        result = self._parse_and_execute(cmd, raw_text, context)

        # Log and return
        self.log_command(raw_text, result)
        if not result.success:
            self.log_event("SAFETY_BLOCK", result.message)
        elif result.warning:
            self.log_event("WARNING", result.warning)
        else:
            self.log_event("COMMAND_SUCCESS", result.message)

        return result

    def _check_conflicting_commands(self, cmd: str) -> CommandResult:
        """Blocks logically impossible or contradictory instructions."""
        has_charge = bool(re.search(r"\bcharge\b", cmd))
        has_discharge = bool(re.search(r"\bdischarge\b", cmd))
        if has_charge and has_discharge:
            return CommandResult(
                success=False,
                message="Refused: Conflicting command. Cannot charge and discharge the battery at the same time.",
                changed={},
            )

        # Conflicting on/off for the same device
        has_on = bool(re.search(r"\bon\b", cmd))
        has_off = bool(re.search(r"\boff\b", cmd))
        if has_on and has_off:
            return CommandResult(
                success=False,
                message="Refused: Conflicting command. Cannot enable and disable in the same command.",
                changed={},
            )

        has_connect = bool(re.search(r"\bconnect\b", cmd))
        has_disconnect = bool(re.search(r"\bdisconnect\b", cmd))
        if has_connect and has_disconnect:
            return CommandResult(
                success=False,
                message="Refused: Conflicting command. Cannot connect and disconnect the grid simultaneously.",
                changed={},
            )

        return None

    def _parse_and_execute(self, cmd: str, raw_text: str, context: dict = None) -> CommandResult:
        """Dispatches recognized keywords to their handler."""
        # HELP
        if re.search(r"\bhelp\b", cmd):
            return self._handle_help()

        # STATUS
        if re.search(r"\b(status|telemetry|metrics)\b", cmd):
            return self._handle_status()

        # SCAN
        if re.search(r"\b(scan|diagnose|diagnostic|health)\b", cmd):
            return self._handle_scan()

        # CLEAR FAULT
        if re.search(r"\b(clear|reset|fix)\b", cmd) and re.search(r"\b(fault|faults|alarm|alarms|error|errors)\b", cmd):
            return self._handle_clear_fault(cmd)

        # AUTO ON / OFF
        if re.search(r"\bauto\b", cmd):
            if re.search(r"\b(on|enable|start|activate)\b", cmd):
                self.auto_mode = True
                if self.autopilot is not None:
                    self.autopilot.enabled = True
                return CommandResult(
                    success=True,
                    message="Autonomous RL control enabled.",
                    changed={"auto_mode": True},
                )
            if re.search(r"\b(off|disable|stop|deactivate)\b", cmd):
                self.auto_mode = False
                if self.autopilot is not None:
                    self.autopilot.enabled = False
                return CommandResult(
                    success=True,
                    message="Autonomous RL control disabled (manual override active).",
                    changed={"auto_mode": False},
                )

        # SOLAR ON / OFF
        if re.search(r"\b(solar|pv)\b", cmd):
            return self._handle_solar(cmd)

        # WIND ON / OFF
        if re.search(r"\b(wind|turbine)\b", cmd):
            return self._handle_wind(cmd)

        # BATTERY CHARGE / DISCHARGE / HOLD / ON / OFF
        if re.search(r"\b(battery|bess|storage)\b", cmd) or re.search(r"\b(charge|discharge|hold)\b", cmd):
            return self._handle_battery(cmd)

        # GRID CONNECT / DISCONNECT
        if re.search(r"\b(grid|utility|mains)\b", cmd) or re.search(r"\b(connect|disconnect)\b", cmd):
            return self._handle_grid(cmd, context)

        # Unrecognized command
        return CommandResult(
            success=False,
            message="Unrecognized command. Type 'help' to view the list of supported instructions.",
            changed={},
        )

    def _handle_solar(self, cmd: str) -> CommandResult:
        """Handles solar commands."""
        solar = self.energy_system.solar

        if re.search(r"\b(off|disable|stop|disconnect|shutdown)\b", cmd):
            self.energy_system.set_device_state("solar", "OFF")
            return CommandResult(
                success=True,
                message="Solar array switched OFF.",
                changed={"solar": "OFF"},
            )

        if re.search(r"\b(on|enable|start|connect)\b", cmd):
            # Safety Rule: Refuse to switch ON a device in FAULT
            if solar.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: SolarArray is in FAULT state ({solar.fault_type}). Clear the fault before switching ON.",
                    changed={},
                )
            self.energy_system.set_device_state("solar", "ON")
            return CommandResult(
                success=True,
                message="Solar array switched ON.",
                changed={"solar": "ON"},
            )

        return CommandResult(
            success=False,
            message="Please specify whether to switch solar ON or OFF.",
            changed={},
        )

    def _handle_wind(self, cmd: str) -> CommandResult:
        """Handles wind turbine commands."""
        wind = self.energy_system.wind

        if re.search(r"\b(off|disable|stop|disconnect|shutdown)\b", cmd):
            self.energy_system.set_device_state("wind", "OFF")
            return CommandResult(
                success=True,
                message="Wind turbine switched OFF.",
                changed={"wind": "OFF"},
            )

        if re.search(r"\b(on|enable|start|connect)\b", cmd):
            # Safety Rule: Refuse to switch ON a device in FAULT
            if wind.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: WindTurbine is in FAULT state ({wind.fault_type}). Clear the fault before switching ON.",
                    changed={},
                )
            self.energy_system.set_device_state("wind", "ON")
            return CommandResult(
                success=True,
                message="Wind turbine switched ON.",
                changed={"wind": "ON"},
            )

        return CommandResult(
            success=False,
            message="Please specify whether to switch wind ON or OFF.",
            changed={},
        )

    def _handle_battery(self, cmd: str) -> CommandResult:
        """Handles battery charge, discharge, hold, on, and off commands with safety limits."""
        battery = self.energy_system.battery
        soc = self.get_battery_level()

        # 1. DISCHARGE
        if re.search(r"\bdischarge\b", cmd):
            if battery.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: Battery is in FAULT state ({battery.fault_type}). Clear the fault before operation.",
                    changed={},
                )
            if soc <= MIN_BATTERY_SOC:
                return CommandResult(
                    success=False,
                    message=f"Refused: Battery level is {soc * 100:.1f}%, which is at or below the 10% minimum safety threshold. Discharging blocked to prevent degradation.",
                    changed={},
                )
            self.battery_mode = "DISCHARGE"
            override_msg = ""
            if self.autopilot is not None:
                self.autopilot.set_manual_override("DISCHARGE", hours=self.override_hours)
                override_msg = f" (Auto-Pilot overridden for {self.override_hours} hours)"
            return CommandResult(
                success=True,
                message=f"Battery set to DISCHARGE mode (current level: {soc * 100:.1f}%).{override_msg}",
                changed={"battery_mode": "DISCHARGE"},
            )

        # 2. CHARGE
        if re.search(r"\bcharge\b", cmd):
            if battery.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: Battery is in FAULT state ({battery.fault_type}). Clear the fault before operation.",
                    changed={},
                )
            if soc >= MAX_BATTERY_SOC:
                return CommandResult(
                    success=False,
                    message="Refused: Battery is already at 100% capacity. Charging blocked to prevent overcharging.",
                    changed={},
                )
            self.battery_mode = "CHARGE"
            override_msg = ""
            if self.autopilot is not None:
                self.autopilot.set_manual_override("CHARGE", hours=self.override_hours)
                override_msg = f" (Auto-Pilot overridden for {self.override_hours} hours)"
            return CommandResult(
                success=True,
                message=f"Battery set to CHARGE mode (current level: {soc * 100:.1f}%).{override_msg}",
                changed={"battery_mode": "CHARGE"},
            )

        # 3. HOLD
        if re.search(r"\b(hold|idle|pause)\b", cmd):
            self.battery_mode = "HOLD"
            override_msg = ""
            if self.autopilot is not None:
                self.autopilot.set_manual_override("HOLD", hours=self.override_hours)
                override_msg = f" (Auto-Pilot overridden for {self.override_hours} hours)"
            return CommandResult(
                success=True,
                message=f"Battery set to HOLD mode (idle/standby).{override_msg}",
                changed={"battery_mode": "HOLD"},
            )

        # 4. ON / OFF
        if re.search(r"\b(off|disable)\b", cmd):
            self.energy_system.set_device_state("battery", "OFF")
            return CommandResult(
                success=True,
                message="Battery disabled (switched OFF).",
                changed={"battery": "OFF"},
            )

        if re.search(r"\b(on|enable)\b", cmd):
            if battery.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: Battery is in FAULT state ({battery.fault_type}). Clear the fault before switching ON.",
                    changed={},
                )
            self.energy_system.set_device_state("battery", "ON")
            return CommandResult(
                success=True,
                message="Battery enabled (switched ON).",
                changed={"battery": "ON"},
            )

        return CommandResult(
            success=False,
            message="Please specify battery action: charge, discharge, hold, on, or off.",
            changed={},
        )

    def _handle_grid(self, cmd: str, context: dict = None) -> CommandResult:
        """Handles grid connect and disconnect commands with safety warnings."""
        grid = self.energy_system.grid

        if re.search(r"\b(disconnect|off|isolate|island)\b", cmd):
            # Check safety: warn if demand cannot be met without the grid
            demand_risk = self._check_grid_demand_risk(context)
            self.energy_system.set_device_state("grid", "OFF")

            warning = None
            if demand_risk:
                warning = "Warning: Current demand may exceed available on-site generation and storage. Operating in island mode risks unmet demand."
                msg = f"Main grid disconnected (island mode active). {warning}"
            else:
                msg = "Main grid disconnected (island mode active)."

            return CommandResult(
                success=True,
                message=msg,
                changed={"grid": "OFF"},
                warning=warning,
            )

        if re.search(r"\b(connect|reconnect|on|enable)\b", cmd):
            if grid.status == "FAULT":
                return CommandResult(
                    success=False,
                    message=f"Refused: Grid connection is in FAULT state ({grid.fault_type}). Clear the fault before connecting.",
                    changed={},
                )
            self.energy_system.set_device_state("grid", "ON")
            return CommandResult(
                success=True,
                message="Main grid connected (grid-tied mode active).",
                changed={"grid": "ON"},
            )

        return CommandResult(
            success=False,
            message="Please specify whether to connect or disconnect the grid.",
            changed={},
        )

    def _check_grid_demand_risk(self, context: dict = None) -> bool:
        """Returns True if there is a risk that campus demand cannot be satisfied without grid import."""
        if context is not None and "demand_kw" in context:
            demand = float(context.get("demand_kw", 0.0))
            solar = float(context.get("solar_kw", 0.0))
            wind = float(context.get("wind_kw", 0.0))
            battery_avail = float(context.get("battery_available_kw", 0.0))
            return (solar + wind + battery_avail) < demand

        if self.simulator is not None:
            try:
                state = self.simulator.get_state()
                demand = float(state.get("demand_kw", 0.0))
                solar = float(state.get("solar_kw", 0.0))
                wind = float(state.get("wind_kw", 0.0))
                battery_energy = float(self.simulator.battery_level) * BATTERY_CAPACITY_KWH
                battery_avail = min(float(MAX_RATE_KW), battery_energy)
                return (solar + wind + battery_avail) < demand
            except Exception:
                pass

        # If solar or wind are currently disabled or faulted, demand cannot be guaranteed
        solar_off = (self.energy_system.solar.status != "ON")
        wind_off = (self.energy_system.wind.status != "ON")
        battery_low = (self.get_battery_level() <= MIN_BATTERY_SOC)
        return solar_off or wind_off or battery_low

    def _handle_scan(self) -> CommandResult:
        """Executes a diagnostic scan across all devices."""
        statuses = self.energy_system.get_status()
        faulted = [name for name, s in statuses.items() if s["status"] == "FAULT"]
        disabled = [name for name, s in statuses.items() if s["status"] == "OFF"]
        derated = [name for name, s in statuses.items() if s["derate"] < 1.0 and s["status"] != "FAULT"]

        findings = []
        if faulted:
            findings.append(f"Faulted devices: {', '.join(faulted)}")
        if disabled:
            findings.append(f"Disabled devices: {', '.join(disabled)}")
        if derated:
            findings.append(f"Derated devices: {', '.join(derated)}")

        if not findings:
            msg = "System scan complete: All devices healthy, active (ON), and at 100% capacity."
        else:
            msg = f"System scan complete with alerts — {'; '.join(findings)}."

        return CommandResult(
            success=True,
            message=msg,
            changed={"scan_results": statuses},
        )

    def _handle_status(self) -> CommandResult:
        """Returns the current operational telemetry."""
        statuses = self.energy_system.get_status()
        soc = self.get_battery_level() * 100
        summary_parts = [
            f"Solar: {statuses['solar']['status']}",
            f"Wind: {statuses['wind']['status']}",
            f"Battery: {statuses['battery']['status']} (SoC: {soc:.1f}%, Mode: {self.battery_mode})",
            f"Grid: {statuses['grid']['status']}",
            f"Auto-Mode: {'ON' if self.auto_mode else 'OFF'}",
        ]
        return CommandResult(
            success=True,
            message=" | ".join(summary_parts),
            changed={"status": statuses, "auto_mode": self.auto_mode, "battery_mode": self.battery_mode},
        )

    def _handle_clear_fault(self, cmd: str) -> CommandResult:
        """Clears faults on a specified device or all devices."""
        cleared = []
        for name, dev in self.energy_system.devices.items():
            # If a specific device was named or 'all'/unspecified
            if name in cmd or "all" in cmd or not any(d in cmd for d in self.energy_system.devices):
                if dev.status == "FAULT":
                    dev.enable()
                    cleared.append(name)

        if cleared:
            return CommandResult(
                success=True,
                message=f"Fault cleared successfully on: {', '.join(cleared)}. Normal operation restored.",
                changed={name: "ON" for name in cleared},
            )

        return CommandResult(
            success=True,
            message="No active device faults were found to clear.",
            changed={},
        )

    def _handle_help(self) -> CommandResult:
        """Returns instructions for all supported commands."""
        help_text = (
            "Supported commands:\n"
            "- solar on / off: Enable or disable the solar array\n"
            "- wind on / off: Enable or disable the wind turbine\n"
            "- battery charge / discharge / hold: Set battery mode (safety-governed)\n"
            "- grid connect / disconnect: Switch between grid-tied and island mode\n"
            "- auto on / off: Toggle autonomous RL dispatch control\n"
            "- scan: Run diagnostic telemetry scan on all devices\n"
            "- status: View current health and operational summary\n"
            "- clear fault: Clear faults on a specific device or all devices\n"
            "- help: Display this command reference"
        )
        return CommandResult(
            success=True,
            message=help_text,
            changed={},
        )
