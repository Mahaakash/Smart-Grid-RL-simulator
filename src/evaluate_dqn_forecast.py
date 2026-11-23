import os
import sys
import pandas as pd
from stable_baselines3 import DQN

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import EVAL_DAYS, TRAIN_SPLIT_DAY
from src.grid_env_forecast import SmartGridForecastEnv


def evaluate_dqn_forecast(model_path="models/dqn_forecast.zip"):
    """Evaluates the trained DQN+Forecast agent on the held-out evaluation days."""
    os.makedirs("results", exist_ok=True)

    # 1. Load trained DQN+Forecast model
    model = DQN.load(model_path)

    # 2. Create evaluation environment
    env = SmartGridForecastEnv(
        mode="eval",
        episode_days=EVAL_DAYS,
        fixed_start_day=TRAIN_SPLIT_DAY,
    )

    obs, info = env.reset()
    done = False
    records = []

    # 3. Run evaluation episode
    while not done:
        current_state = env.sim.get_state()

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        records.append({
            "index": current_state["index"],
            "hour": current_state["hour"],
            "solar_kw": current_state["solar_kw"],
            "wind_kw": current_state["wind_kw"],
            "demand_kw": current_state["demand_kw"],
            "price": current_state["price"],
            "battery_level": step_info["battery_level"],
            "action": action,
            "bought_kwh": step_info["bought_kwh"],
            "sold_kwh": step_info["sold_kwh"],
            "cost_inr": step_info["cost_inr"],
            "co2_kg": step_info["co2_kg"],
            "unmet_demand_kwh": step_info["unmet_demand_kwh"],
        })

    df = pd.DataFrame(records)

    # 4. Save results to results/dqn_forecast.csv
    output_path = "results/dqn_forecast.csv"
    df.to_csv(output_path, index=False)
    print(f"DQN + Forecast evaluation results saved to: {output_path}")

    # 5. Calculate and print summary metrics
    total_cost_inr = float(df["cost_inr"].sum())
    total_co2_kg = float(df["co2_kg"].sum())
    total_unmet_kwh = float(df["unmet_demand_kwh"].sum())

    print(f"Total Cost (INR): {total_cost_inr:.2f}")
    print(f"Total CO2 (kg): {total_co2_kg:.2f}")
    print(f"Total Unmet Demand (kWh): {total_unmet_kwh:.2f}")

    # 6. Action distribution
    action_labels = {0: "Charge", 1: "Discharge", 2: "Buy", 3: "Sell"}
    print("\nAction Selection Counts:")
    for a in range(4):
        count = int((df["action"] == a).sum())
        print(f"Action {a} ({action_labels[a]}): {count}")

    return df


if __name__ == "__main__":
    evaluate_dqn_forecast()

