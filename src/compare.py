import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


def compare_results():
    """Loads baseline, DQN, PPO, and DQN+Forecast results, calculates comparisons, and generates charts."""
    os.makedirs("results", exist_ok=True)

    # 1. Load evaluation results
    grid_df = pd.read_csv("results/grid_only.csv")
    rule_df = pd.read_csv("results/rule_based.csv")
    dqn_df = pd.read_csv("results/dqn.csv")

    methods = ["GridOnly", "RuleBased", "DQN"]
    dfs = [grid_df, rule_df, dqn_df]

    # Check and optionally load PPO results
    ppo_path = "results/ppo.csv"
    if os.path.exists(ppo_path):
        ppo_df = pd.read_csv(ppo_path)
        methods.append("PPO")
        dfs.append(ppo_df)

    # Check and optionally load DQN + Forecast results
    forecast_path = "results/dqn_forecast.csv"
    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)
        methods.append("DQN + Forecast")
        dfs.append(forecast_df)

    # Calculate metrics
    costs = [df["cost_inr"].sum() for df in dfs]
    co2s = [df["co2_kg"].sum() for df in dfs]
    unmets = [df["unmet_demand_kwh"].sum() for df in dfs]

    base_cost = costs[0]
    base_co2 = co2s[0]

    # Calculate percentage savings relative to GridOnly
    cost_savings = [
        ((base_cost - c) / base_cost) * 100.0 if base_cost != 0 else 0.0
        for c in costs
    ]
    co2_savings = [
        ((base_co2 - e) / base_co2) * 100.0 if base_co2 != 0 else 0.0
        for e in co2s
    ]

    # Assemble comparison table
    comparison_df = pd.DataFrame({
        "Method": methods,
        "Total Cost (₹)": [round(c, 2) for c in costs],
        "Total CO2 (kg)": [round(e, 2) for e in co2s],
        "Unmet Demand (kWh)": [round(u, 2) for u in unmets],
        "Cost saved vs GridOnly (%)": [round(s, 2) for s in cost_savings],
        "CO2 saved vs GridOnly (%)": [round(s, 2) for s in co2_savings],
    })

    # Save table to CSV and print
    comparison_path = "results/comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("Comparison Summary Table:")
    print(comparison_df.to_string(index=False))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"][:len(methods)]

    # 2. Bar chart: Cost comparison
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, costs, color=colors)
    plt.ylabel("Total Cost (₹)")
    plt.title("Total Cost Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"₹{height:,.0f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig("results/cost_comparison.png")
    plt.close()

    # 3. Bar chart: CO2 comparison
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, co2s, color=colors)
    plt.ylabel("Total CO2 (kg)")
    plt.title("Total CO2 Emissions Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:,.0f} kg",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig("results/co2_comparison.png")
    plt.close()

    # 4. Line chart: Battery level for the first 24 hours (day 1 of evaluation)
    plt.figure(figsize=(12, 6))
    hours = list(range(24))
    plt.plot(hours, grid_df["battery_level"].iloc[:24], label="GridOnly", linestyle="--", color="#1f77b4")
    plt.plot(hours, rule_df["battery_level"].iloc[:24], label="RuleBased", color="#2ca02c")
    plt.plot(hours, dqn_df["battery_level"].iloc[:24], label="DQN", color="#ff7f0e")

    if "PPO" in methods:
        ppo_idx = methods.index("PPO")
        plt.plot(hours, dfs[ppo_idx]["battery_level"].iloc[:24], label="PPO", color="#d62728")

    if "DQN + Forecast" in methods:
        f_idx = methods.index("DQN + Forecast")
        plt.plot(hours, dfs[f_idx]["battery_level"].iloc[:24], label="DQN + Forecast", color="#9467bd", linewidth=2.0)

    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Battery Level (fraction 0-1)")
    plt.title("Battery Level Comparison - Day 1 of Evaluation")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/battery_one_day.png")
    plt.close()

    print("\nSaved charts to results/:")
    print("- results/cost_comparison.png")
    print("- results/co2_comparison.png")
    print("- results/battery_one_day.png")


if __name__ == "__main__":
    compare_results()
