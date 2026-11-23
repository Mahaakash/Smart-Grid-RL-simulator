"""
Autonomy Benchmark: Comprehensive Comparative Evaluation of Microgrid Control Strategies.
Evaluates:
1. Grid Only (no battery use, no response to faults)
2. Rule-Based with the supervisor
3. Autonomous Operator with DQN
4. Autonomous Operator with PPO
5. Autonomous Operator with DQN+Forecast (if available)

Outputs:
- results/autonomy_benchmark.csv (summary table with mean ± std across 5 seeds)
- results/cost_bars.png (cost comparison bar chart with error bars)
- results/unmet_demand_bars.png (unmet demand comparison bar chart with error bars)
- results/battery_events_timeline.png (event timeline with battery level of each method)
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RANDOM_SEED,
    BATTERY_CAPACITY_KWH,
    PEAK_PRICE,
    OFFPEAK_PRICE,
    SELL_PRICE,
    GRID_CO2_KG_PER_KWH,
)
from src.grid_simulator import MicrogridSimulator
from src.devices import EnergySystem
from src.scenarios import ScenarioEngine, ScenarioEvent
from src.autonomous_operator import AutonomousOperator

DEFAULT_SEEDS = [42, 101, 2024, 7, 99]

SETUP_COLORS = {
    "Grid Only": "#7f8c8d",
    "Rule-Based": "#2980b9",
    "Autonomous (DQN)": "#e67e22",
    "Autonomous (PPO)": "#27ae60",
    "Autonomous (DQN+Forecast)": "#8e44ad",
}


def run_grid_only(
    scenario_engine: ScenarioEngine,
    hours: int = 168,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes a baseline simulation for Grid Only:
    - Battery is unused (disabled)
    - No supervisor policy or fault response
    - Action is fixed to 2 (BUY from grid)
    """
    sim = MicrogridSimulator()
    sim.reset(0)
    sim.energy_system.battery.disable()  # Battery disabled / unused

    step_history = []
    for hour in range(int(hours)):
        # Inject / remove scenario events at simulated hour
        scenario_engine.apply_hour(hour, sim.energy_system, sim)

        # Action 2 = BUY
        step_result = sim.step(action=2)

        record = {
            "hour": hour,
            "day": hour // 24,
            "action": 2,
            "action_name": "BUY",
            "strategy": "Grid Only",
            "reason": "Grid Only baseline: direct grid import without storage buffering.",
            "cost_inr": step_result["cost_inr"],
            "co2_kg": step_result["co2_kg"],
            "battery_level": step_result["battery_level"],
            "bought_kwh": step_result["bought_kwh"],
            "sold_kwh": step_result["sold_kwh"],
            "unmet_demand_kwh": step_result["unmet_demand_kwh"],
            "load_shed_kwh": 0.0,
            "done": step_result["done"],
        }
        step_history.append(record)
        if step_result["done"]:
            break

    df = pd.DataFrame(step_history)
    if not df.empty:
        df["cum_cost"] = df["cost_inr"].cumsum()
        df["cum_co2"] = df["co2_kg"].cumsum()

    total_cost = float(df["cost_inr"].sum()) if not df.empty else 0.0
    total_co2 = float(df["co2_kg"].sum()) if not df.empty else 0.0
    unmet_demand = float(df["unmet_demand_kwh"].sum()) if not df.empty else 0.0

    kpis = {
        "total_cost": round(total_cost, 2),
        "total_cost_inr": round(total_cost, 2),
        "co2": round(total_co2, 2),
        "total_co2_kg": round(total_co2, 2),
        "unmet_demand": round(unmet_demand, 2),
        "unmet_demand_kwh": round(unmet_demand, 2),
        "load_shed_total_kwh": 0.0,
        "number_of_incidents": 0,
        "incidents_detected": 0,
        "incidents_resolved": 0,
        "average_detection_delay_hours": 0.0,
        "average_time_to_recover_hours": 0.0,
    }

    return df, [], kpis


def get_available_setups() -> List[str]:
    """Identifies which control setups are available based on installed model files."""
    setups = [
        "Grid Only",
        "Rule-Based",
        "Autonomous (DQN)",
        "Autonomous (PPO)",
    ]
    if os.path.exists("models/dqn_forecast.zip"):
        setups.append("Autonomous (DQN+Forecast)")
    return setups


def run_setup(
    setup_name: str,
    seed: int,
    preset: str = "Stress test",
    hours: int = 168,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes a single setup on the specified scenario preset and random seed.
    """
    scenario_engine = ScenarioEngine(seed=seed, total_hours=hours)
    scenario_engine.generate_timeline(preset=preset, seed=seed, total_hours=hours)

    if setup_name == "Grid Only":
        return run_grid_only(scenario_engine, hours=hours)

    # Strategy name mapping for AutonomousOperator
    strategy_map = {
        "Rule-Based": "Rule-Based",
        "Autonomous (DQN)": "DQN",
        "Autonomous (PPO)": "PPO",
        "Autonomous (DQN+Forecast)": "DQN+Forecast",
    }
    strategy = strategy_map.get(setup_name, "Rule-Based")
    operator = AutonomousOperator(strategy=strategy, scenario_engine=scenario_engine)
    df, incidents, kpis = operator.run(hours=hours, scenario=scenario_engine, strategy=strategy)

    # Calculate detected and resolved incidents
    incidents_detected = len(incidents)
    incidents_resolved = sum(1 for i in incidents if i.get("status") == "RESOLVED")
    kpis["incidents_detected"] = incidents_detected
    kpis["incidents_resolved"] = incidents_resolved

    return df, incidents, kpis


def plot_cost_bars(summary_df: pd.DataFrame, output_path: str):
    """Generates and saves the cost comparison bar chart with standard deviation error bars."""
    plt.figure(figsize=(9, 5.5))
    setups = summary_df["Setup"].tolist()
    means = summary_df["Cost_Mean"].tolist()
    stds = summary_df["Cost_Std"].tolist()
    colors = [SETUP_COLORS.get(s, "#34495e") for s in setups]

    bars = plt.bar(setups, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.88, width=0.55)
    plt.title("Operational Cost Comparison across Autonomous Control Strategies", fontsize=13, fontweight="bold", pad=14)
    plt.ylabel("Total Cost (₹)", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.45)

    # Add numeric labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        label_text = f"₹{mean:,.0f}\n(±{std:.0f})" if std > 0 else f"₹{mean:,.0f}"
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (std if std > 0 else 0) + 180, label_text,
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.ylim(0, max(means) * 1.22)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_unmet_demand_bars(summary_df: pd.DataFrame, output_path: str):
    """Generates and saves the unmet demand comparison bar chart with error bars."""
    plt.figure(figsize=(9, 5.5))
    setups = summary_df["Setup"].tolist()
    means = summary_df["Unmet_Demand_Mean"].tolist()
    stds = summary_df["Unmet_Demand_Std"].tolist()
    colors = [SETUP_COLORS.get(s, "#e74c3c") for s in setups]

    bars = plt.bar(setups, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.88, width=0.55)
    plt.title("Unmet Demand Comparison (Campus Resilience & Islanding)", fontsize=13, fontweight="bold", pad=14)
    plt.ylabel("Unmet Demand (kWh)", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.45)

    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        label_text = f"{mean:.1f} kWh\n(±{std:.1f})" if std > 0 else f"{mean:.1f} kWh"
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (std if std > 0 else 0) + 3, label_text,
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.ylim(0, max(means) * 1.25 if max(means) > 0 else 10)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_battery_events_timeline(
    history_dict: Dict[str, pd.DataFrame],
    events: List[ScenarioEvent],
    output_path: str,
):
    """
    Generates a 2-panel synchronized timeline plot:
    Panel 1: Battery level (%) across simulated time for each method.
    Panel 2: Event occurrences with shaded time intervals and descriptive labels.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    # --- Panel 1: Battery Level (%) ---
    line_styles = {
        "Grid Only": ":",
        "Rule-Based": "-",
        "Autonomous (DQN)": "--",
        "Autonomous (PPO)": "-.",
        "Autonomous (DQN+Forecast)": "-",
    }

    for setup, df in history_dict.items():
        if df is not None and not df.empty and "battery_level" in df.columns:
            hours = df["hour"].values
            soc_pct = df["battery_level"].values * 100.0
            color = SETUP_COLORS.get(setup, "#333333")
            lw = 2.4 if "DQN" in setup or "Rule" in setup else 1.8
            ax1.plot(hours, soc_pct, label=setup, color=color, linestyle=line_styles.get(setup, "-"), linewidth=lw, alpha=0.92)

    ax1.set_title("Autonomous Microgrid Timeline: Battery Storage State of Charge & Incident Dynamics", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Battery SoC (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    # Highlight 10% safety reserve on battery plot
    ax1.axhline(10.0, color="#c0392b", linestyle=":", linewidth=1.4, alpha=0.8, label="Safety Reserve (10%)")

    # --- Panel 2: Scenario Events Span ---
    event_colors = {
        "grid outage": "#e74c3c",
        "heatwave": "#e67e22",
        "cloudy spell": "#95a5a6",
        "solar soiling": "#d35400",
        "inverter failure": "#c0392b",
        "battery degradation": "#8e44ad",
        "wind turbine trip": "#16a085",
        "sensor stuck": "#34495e",
        "price spike": "#f1c40f",
    }

    y_pos = 0.5
    for idx, ev in enumerate(events):
        color = event_colors.get(ev.event_type.lower(), "#bdc3c7")
        start = ev.start_hour
        dur = ev.duration
        # Add shaded rectangle on event panel
        ax2.add_patch(plt.Rectangle((start, 0.1), dur, 0.8, facecolor=color, alpha=0.85, edgecolor="black"))
        # Also shade top panel lightly
        ax1.axvspan(start, start + dur, color=color, alpha=0.12)

        # Label on event bar
        label = ev.event_type.title()
        ax2.text(start + (dur / 2.0), 0.5, label, color="black", fontsize=8.5, fontweight="bold",
                 ha="center", va="center", rotation=0 if dur >= 8 else 45)

    ax2.set_title("Active Physical Grid Disruptions & Environmental Stress Events", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Simulated Hour (0 - 168h / 1 Week)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Event Status", fontsize=10, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max(168, max([e.end_hour for e in events] + [168])))
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend for event types
    event_patches = [
        mpatches.Patch(color=event_colors.get(etype, "#bdc3c7"), label=etype.title())
        for etype in set(e.event_type.lower() for e in events)
    ]
    ax2.legend(handles=event_patches, loc="upper right", ncol=min(4, len(event_patches)), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_benchmark(
    preset: str = "Stress test",
    seeds: Optional[List[int]] = None,
    hours: int = 168,
    output_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes benchmark comparison across multiple seeds and generates summary metrics & visual plots.

    Args:
        preset: Scenario preset name ('Stress test', 'Grid outage drill', 'Normal week', 'Random').
        seeds: List of random seeds (defaults to 5 distinct seeds).
        hours: Simulation duration in hours (default 168 for 1 week).
        output_dir: Directory where CSV tables and PNG charts will be saved.

    Returns:
        summary_df: Aggregated DataFrame with mean ± std for each setup.
        raw_df: Detailed DataFrame containing individual run records across all seeds.
    """
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    os.makedirs(output_dir, exist_ok=True)
    available_setups = get_available_setups()

    print("=" * 80)
    print(f"MICROGRID AUTONOMY BENCHMARK SUITE")
    print(f"Scenario Preset : {preset}")
    print(f"Seeds Tested    : {seeds} ({len(seeds)} seeds)")
    print(f"Duration        : {hours} hours (1 week)")
    print(f"Setups Evaluated: {', '.join(available_setups)}")
    print("=" * 80)

    raw_records = []
    representative_history: Dict[str, pd.DataFrame] = {}
    representative_events: List[ScenarioEvent] = []

    # Run simulations across seeds
    for s_idx, seed in enumerate(seeds):
        print(f"\n--- Running Seed {seed} ({s_idx + 1}/{len(seeds)}) ---")
        grid_only_cost = 0.0
        seed_results: Dict[str, Dict[str, Any]] = {}

        # 1. Grid Only run first to establish baseline cost
        go_se = ScenarioEngine(seed=seed, total_hours=hours)
        go_se.generate_timeline(preset=preset, seed=seed, total_hours=hours)
        if s_idx == 0:
            representative_events = go_se.get_timeline()

        go_df, _, go_kpis = run_grid_only(go_se, hours=hours)
        grid_only_cost = go_kpis["total_cost"]
        seed_results["Grid Only"] = go_kpis
        if s_idx == 0:
            representative_history["Grid Only"] = go_df

        raw_records.append({
            "Seed": seed,
            "Setup": "Grid Only",
            "Cost": go_kpis["total_cost"],
            "CO2": go_kpis["co2"],
            "Unmet_Demand": go_kpis["unmet_demand"],
            "Incidents_Detected": 0,
            "Incidents_Resolved": 0,
            "Avg_Recovery_Time": 0.0,
            "Pct_Saved": 0.0,
        })
        print(f"  [Grid Only]                 Cost: ₹{go_kpis['total_cost']:>9,.2f} | CO2: {go_kpis['co2']:>8,.1f} kg | Unmet: {go_kpis['unmet_demand']:>7,.1f} kWh | Saved: 0.00%")

        # 2. Run other setups
        for setup in available_setups:
            if setup == "Grid Only":
                continue

            df, incs, kpis = run_setup(setup, seed=seed, preset=preset, hours=hours)
            if s_idx == 0:
                representative_history[setup] = df

            cost = kpis["total_cost"]
            pct_saved = ((grid_only_cost - cost) / grid_only_cost) * 100.0 if grid_only_cost > 0 else 0.0
            kpis["pct_saved"] = pct_saved
            seed_results[setup] = kpis

            raw_records.append({
                "Seed": seed,
                "Setup": setup,
                "Cost": cost,
                "CO2": kpis["co2"],
                "Unmet_Demand": kpis["unmet_demand"],
                "Incidents_Detected": kpis["incidents_detected"],
                "Incidents_Resolved": kpis["incidents_resolved"],
                "Avg_Recovery_Time": kpis["average_time_to_recover_hours"],
                "Pct_Saved": pct_saved,
            })
            print(f"  [{setup:<25}] Cost: ₹{cost:>9,.2f} | CO2: {kpis['co2']:>8,.1f} kg | Unmet: {kpis['unmet_demand']:>7,.1f} kWh | Saved: {pct_saved:>5.2f}%")

    raw_df = pd.DataFrame(raw_records)

    # -------------------------------------------------------------
    # Performance Regression Check: Print cases where RL is worse than Rule-Based
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RL VS RULE-BASED OPERATIONAL COMPARISON CHECK")
    print("=" * 80)
    worse_cases = []

    for seed in seeds:
        seed_subset = raw_df[raw_df["Seed"] == seed].set_index("Setup")
        if "Rule-Based" not in seed_subset.index:
            continue

        rb_cost = seed_subset.loc["Rule-Based", "Cost"]
        rb_unmet = seed_subset.loc["Rule-Based", "Unmet_Demand"]

        for setup in available_setups:
            if setup in ["Grid Only", "Rule-Based"]:
                continue
            if setup not in seed_subset.index:
                continue

            rl_cost = seed_subset.loc[setup, "Cost"]
            rl_unmet = seed_subset.loc[setup, "Unmet_Demand"]

            cost_worse = rl_cost > (rb_cost + 1e-4)
            unmet_worse = rl_unmet > (rb_unmet + 1e-4)

            if cost_worse or unmet_worse:
                diff_cost = rl_cost - rb_cost
                diff_unmet = rl_unmet - rb_unmet
                worse_cases.append({
                    "seed": seed,
                    "setup": setup,
                    "rl_cost": rl_cost,
                    "rb_cost": rb_cost,
                    "diff_cost": diff_cost,
                    "rl_unmet": rl_unmet,
                    "rb_unmet": rb_unmet,
                    "diff_unmet": diff_unmet,
                })
                print(f"⚠️  [REGRESSION] Seed {seed} | {setup}:")
                if cost_worse:
                    print(f"    - Cost: ₹{rl_cost:,.2f} vs Rule-Based ₹{rb_cost:,.2f} (+₹{diff_cost:,.2f}, +{(diff_cost/rb_cost)*100:.2f}%)")
                if unmet_worse:
                    print(f"    - Unmet Demand: {rl_unmet:.2f} kWh vs Rule-Based {rb_unmet:.2f} kWh (+{diff_unmet:.2f} kWh)")

    if not worse_cases:
        print("✅ No regressions found! All RL operators matched or outperformed Rule-Based heuristic across all tested seeds.")
    else:
        print(f"\nTotal cases where RL was worse than Rule-Based: {len(worse_cases)}")

    # -------------------------------------------------------------
    # Compute Summary Statistics (Mean ± Std)
    # -------------------------------------------------------------
    summary_rows = []
    for setup in available_setups:
        sub = raw_df[raw_df["Setup"] == setup]
        cost_m, cost_s = sub["Cost"].mean(), sub["Cost"].std(ddof=1) if len(sub) > 1 else 0.0
        co2_m, co2_s = sub["CO2"].mean(), sub["CO2"].std(ddof=1) if len(sub) > 1 else 0.0
        unmet_m, unmet_s = sub["Unmet_Demand"].mean(), sub["Unmet_Demand"].std(ddof=1) if len(sub) > 1 else 0.0
        inc_det_m, inc_det_s = sub["Incidents_Detected"].mean(), sub["Incidents_Detected"].std(ddof=1) if len(sub) > 1 else 0.0
        inc_res_m, inc_res_s = sub["Incidents_Resolved"].mean(), sub["Incidents_Resolved"].std(ddof=1) if len(sub) > 1 else 0.0
        recov_m, recov_s = sub["Avg_Recovery_Time"].mean(), sub["Avg_Recovery_Time"].std(ddof=1) if len(sub) > 1 else 0.0
        saved_m, saved_s = sub["Pct_Saved"].mean(), sub["Pct_Saved"].std(ddof=1) if len(sub) > 1 else 0.0

        summary_rows.append({
            "Setup": setup,
            "Cost (₹)": f"₹{cost_m:,.2f} ± {cost_s:,.2f}",
            "CO2 (kg)": f"{co2_m:,.2f} ± {co2_s:,.2f}",
            "Unmet Demand (kWh)": f"{unmet_m:,.2f} ± {unmet_s:,.2f}",
            "Incidents Detected": f"{inc_det_m:.1f} ± {inc_det_s:.1f}",
            "Incidents Resolved": f"{inc_res_m:.1f} ± {inc_res_s:.1f}",
            "Avg Recovery Time (h)": f"{recov_m:.2f} ± {recov_s:.2f}",
            "% Saved vs Grid Only": f"{saved_m:.2f}% ± {saved_s:.2f}%",
            # Numeric fields for charting / programmatic access
            "Cost_Mean": round(cost_m, 2),
            "Cost_Std": round(cost_s, 2),
            "CO2_Mean": round(co2_m, 2),
            "CO2_Std": round(co2_s, 2),
            "Unmet_Demand_Mean": round(unmet_m, 2),
            "Unmet_Demand_Std": round(unmet_s, 2),
            "Incidents_Detected_Mean": round(inc_det_m, 2),
            "Incidents_Detected_Std": round(inc_det_s, 2),
            "Incidents_Resolved_Mean": round(inc_res_m, 2),
            "Incidents_Resolved_Std": round(inc_res_s, 2),
            "Avg_Recovery_Time_Mean": round(recov_m, 2),
            "Avg_Recovery_Time_Std": round(recov_s, 2),
            "Pct_Saved_Mean": round(saved_m, 2),
            "Pct_Saved_Std": round(saved_s, 2),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save summary and raw CSVs
    summary_csv_path = os.path.join(output_dir, "autonomy_benchmark.csv")
    raw_csv_path = os.path.join(output_dir, "autonomy_benchmark_raw.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\n[Artifact] Saved summary table -> {summary_csv_path}")
    print(f"[Artifact] Saved raw seed records -> {raw_csv_path}")

    # Generate charts
    cost_chart_path = os.path.join(output_dir, "cost_bars.png")
    unmet_chart_path = os.path.join(output_dir, "unmet_demand_bars.png")
    timeline_chart_path = os.path.join(output_dir, "battery_events_timeline.png")
    alt_timeline_chart_path = os.path.join(output_dir, "autonomy_timeline.png")

    plot_cost_bars(summary_df, cost_chart_path)
    plot_unmet_demand_bars(summary_df, unmet_chart_path)
    plot_battery_events_timeline(representative_history, representative_events, timeline_chart_path)
    # Also save a copy as autonomy_timeline.png
    plot_battery_events_timeline(representative_history, representative_events, alt_timeline_chart_path)

    print(f"[Artifact] Saved cost bars chart -> {cost_chart_path}")
    print(f"[Artifact] Saved unmet demand bars chart -> {unmet_chart_path}")
    print(f"[Artifact] Saved event timeline plot -> {timeline_chart_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY RESULTS (Mean ± Std over {len(seeds)} Seeds)")
    print("=" * 80)
    display_cols = [
        "Setup",
        "Cost (₹)",
        "CO2 (kg)",
        "Unmet Demand (kWh)",
        "Incidents Detected",
        "Incidents Resolved",
        "Avg Recovery Time (h)",
        "% Saved vs Grid Only",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)

    return summary_df, raw_df


def main():
    parser = argparse.ArgumentParser(description="Microgrid Autonomy Benchmark Suite")
    parser.add_argument("--preset", type=str, default="Stress test", help="Scenario preset name")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--hours", type=int, default=168, help="Simulation hours (default: 168)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for CSV and chart outputs")

    args = parser.parse_args()
    run_benchmark(
        preset=args.preset,
        seeds=args.seeds,
        hours=args.hours,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
