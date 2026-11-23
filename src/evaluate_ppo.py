import os
import sys
import pandas as pd
from stable_baselines3 import PPO

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import EVAL_DAYS, TRAIN_SPLIT_DAY
from src.grid_env import SmartGridEnv


def evaluate_ppo(model_path="models/ppo_grid.zip"):
    """Loads a trained PPO model and evaluates it on the evaluation period."""
    os.makedirs("results", exist_ok=True)

    # 1. Load trained PPO model
    model = PPO.load(model_path)

    # 2. Create evaluation environment
    env = SmartGridEnv(mode="eval", episode_days=EVAL_DAYS, fixed_start_day=TRAIN_SPLIT_DAY)

    obs, info = env.reset()
    done = False
    records = []

    # 3. Run one full evaluation episode
    while not done:
        current_state = env.sim.get_state()

        # Predict deterministic action
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Step the environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        # Record hourly metrics using info dict from env
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

    # 4. Save results to CSV
    output_path = "results/ppo.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # 5. Calculate and print summary metrics
    total_cost_inr = float(df["cost_inr"].sum())
    total_co2_kg = float(df["co2_kg"].sum())
    total_unmet_kwh = float(df["unmet_demand_kwh"].sum())

    print(f"Total Cost (INR): {total_cost_inr:.2f}")
    print(f"Total CO2 (kg): {total_co2_kg:.2f}")
    print(f"Total Unmet Demand (kWh): {total_unmet_kwh:.2f}")

    # 6. Print action distribution counts
    action_labels = {0: "Charge", 1: "Discharge", 2: "Buy", 3: "Sell"}
    print("\nAction Selection Counts:")
    for a in range(4):
        count = int((df["action"] == a).sum())
        print(f"Action {a} ({action_labels[a]}): {count}")


if __name__ == "__main__":
    evaluate_ppo()

