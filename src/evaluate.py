import os
import sys
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import EVAL_DAYS, TRAIN_SPLIT_DAY
from src.grid_simulator import MicrogridSimulator
from src.baselines import GridOnly, RuleBased


def run_policy(policy, days=EVAL_DAYS, start_day=TRAIN_SPLIT_DAY):
    """Evaluates a policy on the microgrid simulator for a given number of days."""
    sim = MicrogridSimulator()
    start_index = start_day * 24
    state = sim.reset(start_index)

    records = []
    total_steps = days * 24

    for _ in range(total_steps):
        # Choose action based on current state
        action = policy.act(state)

        # Step the simulator
        result = sim.step(action)

        # Record hourly metrics
        records.append({
            "index": state["index"],
            "hour": state["hour"],
            "solar_kw": state["solar_kw"],
            "wind_kw": state["wind_kw"],
            "demand_kw": state["demand_kw"],
            "price": state["price"],
            "battery_level": result["battery_level"],
            "action": action,
            "bought_kwh": result["bought_kwh"],
            "sold_kwh": result["sold_kwh"],
            "cost_inr": result["cost_inr"],
            "co2_kg": result["co2_kg"],
            "unmet_demand_kwh": result["unmet_demand_kwh"],
        })

        # Update state for next step
        state = result["state"]

    hourly_df = pd.DataFrame(records)

    summary_dict = {
        "total_cost_inr": float(hourly_df["cost_inr"].sum()),
        "total_co2_kg": float(hourly_df["co2_kg"].sum()),
        "total_unmet_kwh": float(hourly_df["unmet_demand_kwh"].sum()),
    }

    return hourly_df, summary_dict


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 1. Run GridOnly baseline
    grid_policy = GridOnly()
    grid_df, grid_summary = run_policy(grid_policy)
    grid_df.to_csv("results/grid_only.csv", index=False)

    # 2. Run RuleBased baseline
    rule_policy = RuleBased()
    rule_df, rule_summary = run_policy(rule_policy)
    rule_df.to_csv("results/rule_based.csv", index=False)

    # 3. Create and save summary table
    summary_df = pd.DataFrame([
        {"policy": "GridOnly", **grid_summary},
        {"policy": "RuleBased", **rule_summary},
    ])
    summary_df.to_csv("results/baseline_summary.csv", index=False)

    # 4. Print summary table
    print("Baseline Evaluation Summary:")
    print(summary_df.to_string(index=False))

