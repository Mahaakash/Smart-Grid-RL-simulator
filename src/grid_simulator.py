import os
import sys
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DATA_PATH,
    BATTERY_CAPACITY_KWH,
    MAX_RATE_KW,
    BATTERY_EFFICIENCY,
    INITIAL_BATTERY_LEVEL,
    SELL_PRICE,
    GRID_CO2_KG_PER_KWH,
)
from src.devices import EnergySystem


class MicrogridSimulator:
    """Microgrid physics simulator for a campus energy system."""

    def __init__(self, data_path=DATA_PATH, energy_system=None):
        # Load the hourly grid dataset
        self.data = pd.read_csv(data_path)
        self.current_index = 0
        self.battery_level = INITIAL_BATTERY_LEVEL
        self.energy_system = energy_system if energy_system is not None else EnergySystem()
        self.history = []

    def reset(self, start_index=0):
        """Sets the current hour index and resets the battery level."""
        self.current_index = start_index
        self.battery_level = float(INITIAL_BATTERY_LEVEL)
        self.history = []
        return self.get_state()

    def get_state(self):
        """Returns the state dictionary for the current hour."""
        row = self.data.iloc[self.current_index]
        solar_dev = self.energy_system.solar
        wind_dev = self.energy_system.wind

        raw_solar = float(row["solar_kw"])
        raw_wind = float(row["wind_kw"])

        effective_solar = 0.0 if solar_dev.status == "OFF" else max(0.0, raw_solar * solar_dev.derate)
        effective_wind = 0.0 if wind_dev.status == "OFF" else max(0.0, raw_wind * wind_dev.derate)

        demand_mult = getattr(self.energy_system, "demand_multiplier", 1.0)
        price_mult = getattr(self.energy_system, "price_multiplier", 1.0)

        return {
            "index": int(self.current_index),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "solar_kw": float(effective_solar),
            "wind_kw": float(effective_wind),
            "demand_kw": float(row["demand_kw"] * demand_mult),
            "price": float(row["price_per_kwh"] * price_mult),
            "battery_level": float(self.battery_level),
        }

    def step(self, action):
        """Simulates exactly 1 hour given an action (0=Charge, 1=Discharge, 2=Buy, 3=Sell)."""
        row = self.data.iloc[self.current_index]
        raw_solar = float(row["solar_kw"])
        raw_wind = float(row["wind_kw"])

        demand_mult = getattr(self.energy_system, "demand_multiplier", 1.0)
        price_mult = getattr(self.energy_system, "price_multiplier", 1.0)

        demand_kw = float(row["demand_kw"] * demand_mult)
        price = float(row["price_per_kwh"] * price_mult)

        # Device states
        solar_dev = self.energy_system.solar
        wind_dev = self.energy_system.wind
        battery_dev = self.energy_system.battery
        grid_dev = self.energy_system.grid

        # 1. Renewable generation respecting device state (OFF = 0, FAULT/ON = derated)
        if solar_dev.status == "OFF":
            solar_kw = 0.0
        else:
            solar_kw = max(0.0, raw_solar * solar_dev.derate)

        if wind_dev.status == "OFF":
            wind_kw = 0.0
        else:
            wind_kw = max(0.0, raw_wind * wind_dev.derate)

        renewable = solar_kw + wind_kw
        net = renewable - demand_kw
        surplus = max(0.0, net)
        deficit = max(0.0, -net)

        # 2. Battery energy in kWh
        battery_energy = self.battery_level * BATTERY_CAPACITY_KWH

        # Battery availability (OFF or FAULT means it cannot charge or discharge, unless degraded)
        battery_active = (battery_dev.status == "ON" or battery_dev.fault_type == "battery degradation")
        # Grid connection (OFF or FAULT means disconnected / island mode)
        grid_connected = (grid_dev.status == "ON")

        # Effective round-trip battery efficiency
        eff = getattr(battery_dev, "efficiency", BATTERY_EFFICIENCY)

        # Check for invalid action (charge when full/disabled, or discharge when empty/disabled)
        is_full = self.battery_level >= (1.0 - 1e-6)
        is_empty = self.battery_level <= 1e-6
        invalid_action = False
        if action == 0 and (is_full or not battery_active):
            invalid_action = True
        elif action == 1 and (is_empty or not battery_active):
            invalid_action = True

        bought_kwh = 0.0
        sold_kwh = 0.0
        unmet_demand_kwh = 0.0

        # 3. Actions logic
        if action == 0:
            # CHARGE
            if battery_active:
                remaining_capacity = BATTERY_CAPACITY_KWH - battery_energy
                max_energy_in = remaining_capacity / eff
                max_rate = float(MAX_RATE_KW * battery_dev.derate)
                target_energy_in = min(max_rate, max(0.0, max_energy_in))

                energy_from_surplus = min(surplus, target_energy_in)
                charge_needed_from_grid = target_energy_in - energy_from_surplus

                if grid_connected:
                    charge_bought = charge_needed_from_grid
                    energy_in = target_energy_in
                    bought_kwh = deficit + charge_bought
                    unmet_demand_kwh = 0.0
                else:
                    # Island mode: no grid buying, only surplus can charge
                    energy_in = energy_from_surplus
                    bought_kwh = 0.0
                    unmet_demand_kwh = deficit

                stored_energy = energy_in * eff
                battery_energy = min(float(BATTERY_CAPACITY_KWH), battery_energy + stored_energy)
            else:
                # Battery OFF/FAULT: cannot charge
                if grid_connected:
                    bought_kwh = deficit
                    unmet_demand_kwh = 0.0
                else:
                    bought_kwh = 0.0
                    unmet_demand_kwh = deficit
            sold_kwh = 0.0

        elif action == 1:
            # DISCHARGE
            if battery_active:
                needed_draw = deficit / eff
                max_draw = min(float(MAX_RATE_KW * battery_dev.derate), battery_energy)
                energy_drawn = min(max_draw, needed_draw)

                energy_delivered = energy_drawn * eff
                battery_energy = max(0.0, battery_energy - energy_drawn)
                remaining_deficit = max(0.0, deficit - energy_delivered)

                if grid_connected:
                    bought_kwh = remaining_deficit
                    unmet_demand_kwh = 0.0
                else:
                    # Island mode: remaining deficit cannot be bought from grid
                    bought_kwh = 0.0
                    unmet_demand_kwh = remaining_deficit
            else:
                # Battery OFF/FAULT: cannot discharge
                if grid_connected:
                    bought_kwh = deficit
                    unmet_demand_kwh = 0.0
                else:
                    bought_kwh = 0.0
                    unmet_demand_kwh = deficit
            sold_kwh = 0.0

        elif action == 2:
            # BUY
            if grid_connected:
                bought_kwh = deficit
                unmet_demand_kwh = 0.0
            else:
                # Island mode: cannot buy from grid
                bought_kwh = 0.0
                unmet_demand_kwh = deficit
            sold_kwh = 0.0

        elif action == 3:
            # SELL
            if grid_connected:
                bought_kwh = deficit
                sold_kwh = surplus * grid_dev.derate
                unmet_demand_kwh = 0.0
            else:
                # Island mode: cannot buy or sell
                bought_kwh = 0.0
                sold_kwh = 0.0
                unmet_demand_kwh = deficit

        # Update battery level fraction (0 to 1)
        self.battery_level = max(0.0, min(1.0, battery_energy / BATTERY_CAPACITY_KWH))

        # 4 & 5. Cost calculation (buying uses dynamic price, selling uses fixed SELL_PRICE)
        cost_inr = (bought_kwh * price) - (sold_kwh * SELL_PRICE)

        # 6. CO2 emissions calculation
        co2_kg = bought_kwh * GRID_CO2_KG_PER_KWH

        # Record recent step history for diagnostics
        self.history.append({
            "index": int(self.current_index),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "solar_kw": float(solar_kw),
            "expected_solar_kw": float(raw_solar),
            "wind_kw": float(wind_kw),
            "expected_wind_kw": float(raw_wind),
            "demand_kw": float(demand_kw),
            "price": float(price),
            "battery_level": float(self.battery_level),
            "action": int(action),
            "bought_kwh": float(bought_kwh),
            "sold_kwh": float(sold_kwh),
            "unmet_demand_kwh": float(unmet_demand_kwh),
            "cost_inr": float(cost_inr),
            "co2_kg": float(co2_kg),
        })
        if len(self.history) > 168:
            self.history.pop(0)

        # Move to next hour
        self.current_index += 1

        # 9. Check if simulation is done (reaches or exceeds the last row)
        done = bool(self.current_index >= len(self.data) - 1)
        if self.current_index >= len(self.data):
            self.current_index = len(self.data) - 1
            done = True

        new_state = self.get_state()

        return {
            "bought_kwh": bought_kwh,
            "sold_kwh": sold_kwh,
            "unmet_demand_kwh": unmet_demand_kwh,
            "cost_inr": cost_inr,
            "co2_kg": co2_kg,
            "battery_level": self.battery_level,
            "invalid_action": invalid_action,
            "done": done,
            "state": new_state,
        }
