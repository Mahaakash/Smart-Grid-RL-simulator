import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATA_PATH
from src.data_generator import generate_real_grid_data
from src.baselines import GridOnly, RuleBased
from src.evaluate import run_policy
from src.train_dqn import train_dqn
from src.train_ppo import train_ppo
from src.evaluate_rl import evaluate_rl
from src.evaluate_ppo import evaluate_ppo
from src.compare import compare_results
from src.plot_sample_week import plot_sample_week
from src.plot_training_curves import plot_training_curves
import pandas as pd


def run_pipeline(quick=False):
    """Executes the full Smart Grid RL simulator pipeline in sequential order."""
    print("=" * 60)
    print("   SMART GRID RL SIMULATOR - COMPLETE PIPELINE EXECUTION")
    print("=" * 60)

    dqn_steps = 20000 if quick else 200000
    ppo_steps = 20000 if quick else 300000

    if quick:
        print(f"\n[QUICK MODE ENABLED] DQN steps: {dqn_steps}, PPO steps: {ppo_steps}\n")

    # Step 1: Build Data
    print("\n--- STEP 1: Building / Verifying Grid Dataset ---")
    if not os.path.exists(DATA_PATH):
        print(f"'{DATA_PATH}' not found. Downloading and generating dataset...")
        generate_real_grid_data()
    else:
        print(f"Using existing grid dataset at: {DATA_PATH}")

    # Step 2: Evaluate Baselines
    print("\n--- STEP 2: Evaluating Baselines (GridOnly & RuleBased) ---")
    os.makedirs("results", exist_ok=True)
    grid_df, grid_sum = run_policy(GridOnly())
    grid_df.to_csv("results/grid_only.csv", index=False)

    rule_df, rule_sum = run_policy(RuleBased())
    rule_df.to_csv("results/rule_based.csv", index=False)

    baseline_summary = pd.DataFrame([
        {"policy": "GridOnly", **grid_sum},
        {"policy": "RuleBased", **rule_sum},
    ])
    baseline_summary.to_csv("results/baseline_summary.csv", index=False)
    print(baseline_summary.to_string(index=False))

    # Step 3: Train DQN
    print(f"\n--- STEP 3: Training DQN Agent ({dqn_steps} steps) ---")
    train_dqn(timesteps=dqn_steps)

    # Step 4: Train PPO
    print(f"\n--- STEP 4: Training PPO Agent ({ppo_steps} steps) ---")
    train_ppo(timesteps=ppo_steps)

    # Step 5: Evaluate Both Agents
    print("\n--- STEP 5: Evaluating RL Agents ---")
    print("Evaluating DQN...")
    evaluate_rl("models/dqn_grid.zip")
    print("Evaluating PPO...")
    evaluate_ppo("models/ppo_grid.zip")

    # Step 6: Compare All Methods
    print("\n--- STEP 6: Comparing All Methods ---")
    compare_results()

    # Step 7: Generate Plots
    print("\n--- STEP 7: Generating Final Plots ---")
    plot_sample_week()
    plot_training_curves()

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE - ALL ARTIFACTS GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Smart Grid RL Simulator pipeline")
    parser.add_argument("--quick", action="store_true", help="Train DQN and PPO for only 20,000 steps")
    args = parser.parse_args()

    run_pipeline(quick=args.quick)

