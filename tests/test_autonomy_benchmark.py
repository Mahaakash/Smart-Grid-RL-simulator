"""
Unit tests for autonomy_benchmark.py:
Verifies Grid Only baseline, multi-strategy comparative evaluation,
statistical aggregation across seeds, CSV export, and chart generation.
"""

import os
import shutil
import pytest
import pandas as pd

from src.scenarios import ScenarioEngine
from src.autonomy_benchmark import (
    run_grid_only,
    run_setup,
    run_benchmark,
    get_available_setups,
)


def test_get_available_setups():
    """Confirms available setups include Grid Only, Rule-Based, DQN, PPO, and DQN+Forecast if present."""
    setups = get_available_setups()
    assert "Grid Only" in setups
    assert "Rule-Based" in setups
    assert "Autonomous (DQN)" in setups
    assert "Autonomous (PPO)" in setups
    if os.path.exists("models/dqn_forecast.zip"):
        assert "Autonomous (DQN+Forecast)" in setups


def test_run_grid_only():
    """Verifies that Grid Only baseline takes action 2, doesn't use battery, and calculates KPIs."""
    se = ScenarioEngine(seed=42, total_hours=24)
    se.generate_timeline("Normal week", seed=42, total_hours=24)

    df, incidents, kpis = run_grid_only(se, hours=24)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert (df["action"] == 2).all()
    assert len(incidents) == 0
    assert kpis["total_cost"] > 0
    assert kpis["co2"] > 0
    assert kpis["incidents_detected"] == 0
    assert kpis["incidents_resolved"] == 0


def test_run_setup():
    """Verifies run_setup for both Grid Only and AutonomousOperator with Rule-Based."""
    # Test Grid Only
    df_go, inc_go, kpis_go = run_setup("Grid Only", seed=42, preset="Normal week", hours=24)
    assert len(df_go) == 24
    assert kpis_go["incidents_detected"] == 0

    # Test Rule-Based
    df_rb, inc_rb, kpis_rb = run_setup("Rule-Based", seed=42, preset="Normal week", hours=24)
    assert len(df_rb) == 24
    assert "total_cost" in kpis_rb
    assert "unmet_demand" in kpis_rb


def test_run_benchmark_integration(tmp_path):
    """
    Executes a fast multi-seed benchmark run (2 seeds, 24 hours) into a temporary directory
    to confirm end-to-end table construction, CSV writing, and chart rendering.
    """
    out_dir = str(tmp_path / "test_results")

    summary_df, raw_df = run_benchmark(
        preset="Normal week",
        seeds=[42, 101],
        hours=24,
        output_dir=out_dir,
    )

    # Check summary DataFrame schema
    assert isinstance(summary_df, pd.DataFrame)
    assert "Setup" in summary_df.columns
    assert "Cost (₹)" in summary_df.columns
    assert "CO2 (kg)" in summary_df.columns
    assert "Unmet Demand (kWh)" in summary_df.columns
    assert "Incidents Detected" in summary_df.columns
    assert "Incidents Resolved" in summary_df.columns
    assert "Avg Recovery Time (h)" in summary_df.columns
    assert "% Saved vs Grid Only" in summary_df.columns

    # Check numeric columns
    assert "Cost_Mean" in summary_df.columns
    assert "Cost_Std" in summary_df.columns
    assert "Unmet_Demand_Mean" in summary_df.columns

    # Check raw DataFrame
    assert len(raw_df) == 2 * len(summary_df)

    # Check file artifacts
    assert os.path.exists(os.path.join(out_dir, "autonomy_benchmark.csv"))
    assert os.path.exists(os.path.join(out_dir, "autonomy_benchmark_raw.csv"))
    assert os.path.exists(os.path.join(out_dir, "cost_bars.png"))
    assert os.path.exists(os.path.join(out_dir, "unmet_demand_bars.png"))
    assert os.path.exists(os.path.join(out_dir, "battery_events_timeline.png"))
