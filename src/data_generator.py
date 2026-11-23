import json
import os
import shutil
import sys
import urllib.request
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DATA_PATH,
    RANDOM_SEED,
    SOLAR_CAPACITY_KW,
    WIND_CAPACITY_KW,
    PEAK_PRICE,
    OFFPEAK_PRICE,
    PEAK_HOURS,
)

# Open-Meteo Historical Weather API URL for Kolkata, India (lat: 22.57, lon: 88.36)
OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=22.57&longitude=88.36&"
    "start_date=2023-01-01&end_date=2023-12-31&"
    "hourly=temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m&"
    "wind_speed_unit=ms&timezone=Asia/Kolkata"
)


def calculate_wind_power(wind_speed_ms, capacity_kw=WIND_CAPACITY_KW):
    """Calculates turbine power output using standard cut-in, rated, and cut-out speeds."""
    cut_in = 3.0   # Cut-in wind speed (m/s)
    rated = 12.0   # Rated wind speed (m/s)
    cut_out = 25.0 # Cut-out wind speed (m/s)

    if wind_speed_ms < cut_in or wind_speed_ms >= cut_out:
        return 0.0
    elif wind_speed_ms >= rated:
        return float(capacity_kw)
    else:
        # Linear power ramp between cut-in and rated speeds
        return float(capacity_kw) * ((wind_speed_ms - cut_in) / (rated - cut_in))


def download_weather_data():
    """Downloads 1 full year (2023, 8760 hours) of historical weather data for Kolkata."""
    headers = {"User-Agent": "SmartGridRLSimulator/1.0"}
    req = urllib.request.Request(OPEN_METEO_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["hourly"]


def generate_real_grid_data():
    """Generates real-data hourly dataset for Kolkata and saves it to DATA_PATH."""
    np.random.seed(RANDOM_SEED)

    # Back up existing file if it exists
    if os.path.exists(DATA_PATH):
        backup_path = "data/grid_data_synthetic_backup.csv"
        shutil.copy(DATA_PATH, backup_path)
        print(f"Backed up old dataset to: {backup_path}")

    print("Fetching historical weather data for Kolkata (lat 22.57, lon 88.36) from Open-Meteo...")
    weather = download_weather_data()

    times = weather["time"]
    temperatures = np.array(weather["temperature_2m"], dtype=np.float32)
    humidities = np.array(weather["relative_humidity_2m"], dtype=np.float32)
    cloud_covers = np.array(weather["cloud_cover"], dtype=np.float32)
    radiations = np.array(weather["shortwave_radiation"], dtype=np.float32)
    wind_speeds = np.array(weather["wind_speed_10m"], dtype=np.float32)

    total_hours = len(times)
    assert total_hours == 8760, f"Expected 8760 hours, got {total_hours}"

    # Parse timestamps and time features
    timestamps = pd.to_datetime(times)
    days = [h // 24 for h in range(total_hours)]
    hours = [ts.hour for ts in timestamps]
    days_of_week = [ts.day_of_week for ts in timestamps]

    solar_values = []
    wind_values = []
    demand_values = []
    price_values = []

    for h in range(total_hours):
        hour = hours[h]
        dow = days_of_week[h]
        is_weekend = (dow >= 5)

        # 1. Solar generation (kW): standard test irradiance is 1000 W/m²
        rad = max(0.0, float(radiations[h]))
        if 6 <= hour <= 18:
            solar_kw = (rad / 1000.0) * SOLAR_CAPACITY_KW
            solar_kw = min(float(SOLAR_CAPACITY_KW), max(0.0, solar_kw))
        else:
            solar_kw = 0.0
        solar_values.append(round(solar_kw, 2))

        # 2. Wind generation (kW): turbine power curve
        ws = max(0.0, float(wind_speeds[h]))
        wind_kw = calculate_wind_power(ws, capacity_kw=WIND_CAPACITY_KW)
        wind_values.append(round(wind_kw, 2))

        # 3. Campus demand (kW) with temperature/cooling sensitivity
        if 9 <= hour <= 17:
            base_demand = 55.0
        elif 18 <= hour <= 22:
            base_demand = 30.0
        else:
            base_demand = 17.5

        if is_weekend:
            base_demand *= 0.6

        # Temperature-dependent air conditioning (rises above comfortable 24°C)
        temp_c = float(temperatures[h])
        temp_excess = max(0.0, temp_c - 24.0)
        if 9 <= hour <= 17 and not is_weekend:
            cooling_demand = temp_excess * 1.0
        elif 18 <= hour <= 22:
            cooling_demand = temp_excess * 0.5
        else:
            cooling_demand = temp_excess * 0.3

        noise = np.random.uniform(-1.5, 1.5)
        demand_kw = max(0.0, round(base_demand + cooling_demand + noise, 2))
        demand_values.append(demand_kw)

        # 4. Time-of-use price per kWh
        if PEAK_HOURS[0] <= hour < PEAK_HOURS[1]:
            price = PEAK_PRICE
        else:
            price = OFFPEAK_PRICE
        price_values.append(price)

    # Assemble DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "day": days,
        "hour": hours,
        "day_of_week": days_of_week,
        "solar_kw": solar_values,
        "wind_kw": wind_values,
        "demand_kw": demand_values,
        "price_per_kwh": price_values,
        "temperature_c": np.round(temperatures, 1),
        "cloud_cover": np.round(cloud_covers, 1),
        "relative_humidity": np.round(humidities, 1),
    })

    # Forward fill any potential missing data, ensure no NaNs
    df = df.ffill().bfill()
    assert len(df) == 8760, f"Expected 8760 rows, got {len(df)}"
    assert df.isna().sum().sum() == 0, "DataFrame contains NaNs"

    # Save to DATA_PATH
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Real-world Kolkata grid data successfully saved to: {DATA_PATH} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    df = generate_real_grid_data()
    print("\nFirst 5 rows:")
    print(df.head(5))
