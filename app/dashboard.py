import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import DQN, PPO

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    TRAIN_SPLIT_DAY,
    BATTERY_CAPACITY_KWH,
    PEAK_HOURS,
    EVAL_DAYS,
)
from src.grid_simulator import MicrogridSimulator
from src.baselines import GridOnly, RuleBased
from src.grid_env import SmartGridEnv
from src.grid_env_forecast import SmartGridForecastEnv
from src.devices import EnergySystem
from src.command_controller import CommandController
from src.autopilot import AutoPilot, ACTION_NAMES
from src.diagnostics import scan
from plotly.subplots import make_subplots
from src.autonomous_operator import AutonomousOperator
from src.scenarios import ScenarioEngine, ScenarioEvent
from src.autonomy_benchmark import run_grid_only, run_benchmark


@st.cache_resource
def get_cached_model(strategy_name: str):
    """Loads and caches SB3 RL models to maximize runtime performance."""
    path_map = {
        "DQN": "models/dqn_grid.zip",
        "PPO": "models/ppo_grid.zip",
        "DQN+Forecast": "models/dqn_forecast.zip",
        "DQN + Forecast": "models/dqn_forecast.zip",
    }
    path = path_map.get(strategy_name)
    if not path or not os.path.exists(path):
        return None
    try:
        if "PPO" in strategy_name:
            from stable_baselines3 import PPO
            return PPO.load(path)
        else:
            from stable_baselines3 import DQN
            return DQN.load(path)
    except Exception:
        return None


# Page setup
st.set_page_config(
    page_title="Smart Grid RL Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SMART FONTS & MODERN UI STYLING ---
st.markdown(
    r"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">

    <style>
        /* Global typography with smart fonts */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        code, pre, .stMetricValue, .metric-number {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Pulsing status LEDs */
        @keyframes pulse-emerald {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-amber {
            0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
            70% { box-shadow: 0 0 0 7px rgba(245, 158, 11, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        @keyframes pulse-cyan {
            0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.75); }
            70% { box-shadow: 0 0 0 7px rgba(56, 189, 248, 0); }
            100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75); }
            70% { box-shadow: 0 0 0 7px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .led-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .led-green { background-color: #10b981; animation: pulse-emerald 2s infinite; }
        .led-amber { background-color: #f59e0b; animation: pulse-amber 2s infinite; }
        .led-cyan { background-color: #38bdf8; animation: pulse-cyan 2s infinite; }
        .led-red { background-color: #ef4444; animation: pulse-red 2s infinite; }
        .led-gray { background-color: #64748b; }

        /* Modern card design */
        .glass-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
            margin-bottom: 12px;
        }

        /* Flight deck & Control deck strip */
        .control-deck {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(24, 34, 53, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(56, 189, 248, 0.15);
            margin-bottom: 16px;
        }
        .deck-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            padding-bottom: 6px;
        }

        /* SCADA Technical Badge */
        .scada-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 6px;
            padding: 3px 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.5px;
        }

        /* Flow metric box */
        .flow-pill {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .flow-pill-title {
            font-size: 0.82rem;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .flow-pill-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: #38bdf8;
        }

        /* Status badge */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Viva presentation highlight banner */
        .viva-banner {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.45) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-left: 6px solid #38bdf8;
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            margin-bottom: 16px;
        }

        /* Button micro-interactions */
        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.2px;
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.22);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("⚡ Smart Grid RL Simulator")
st.caption("Autonomous AI Microgrid Energy Management & SCADA Optimization Platform")

# --- SIDEBAR CONFIGURATION (SCADA DISPATCH CONSOLE) ---
st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85)); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
            <span style="font-size: 0.74rem; font-weight: 800; color: #38bdf8; letter-spacing: 0.8px;">SCADA DISPATCH BUS</span>
            <span class="led-dot led-green"></span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8; line-height: 1.4;">
            SUBSTATION: WB-KOLKATA-01<br>INTERCONNECT: 415V 50Hz
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("🕹️ Simulation Dispatch")

strategy = st.sidebar.selectbox(
    "Dispatch Strategy",
    ["Grid Only", "Rule-Based", "DQN", "PPO", "DQN + Forecast"],
    index=2,
    help="Select the control agent used to optimize battery storage and grid transactions.",
)

strat_descriptions = {
    "Grid Only": "Baseline grid-tethered import/export without battery storage.",
    "Rule-Based": "Deterministic peak-shaving & renewable capture heuristic.",
    "DQN": "Double Q-learning with experience replay & target network.",
    "PPO": "Clipped surrogate policy gradient with actor-critic value head.",
    "DQN + Forecast": "DQN reinforced with 3h & 6h PyTorch LSTM irradiance predictions.",
}
st.sidebar.caption(f"🧠 *{strat_descriptions.get(strategy, '')}*")

sim_days = st.sidebar.slider("Simulation Horizon (Days)", min_value=1, max_value=30, value=7, step=1)

battery_capacity = st.sidebar.slider(
    "Battery Capacity (kWh)",
    min_value=20,
    max_value=250,
    value=int(BATTERY_CAPACITY_KWH),
    step=10,
    help="Dynamically alters the simulated physical battery capacity.",
)

c_rate = 25.0 / battery_capacity if battery_capacity > 0 else 0.0
st.sidebar.markdown(
    f"""
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 8px 12px; margin-top: -8px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94a3b8;">
            <span>Inverter Rating:</span> <strong style="color: #f8fafc; font-family: 'JetBrains Mono';">25.0 kW</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94a3b8; margin-top: 2px;">
            <span>Effective C-Rate:</span> <strong style="color: #38bdf8; font-family: 'JetBrains Mono';">{c_rate:.2f}C</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94a3b8; margin-top: 2px;">
            <span>Chemistry / Eff:</span> <strong style="color: #10b981; font-family: 'JetBrains Mono';">LFP / 95%</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

run_button = st.sidebar.button("⚡ Execute Microgrid Dispatch", use_container_width=True, type="primary")


def get_action_reason(action, row, peak_hours):
    """Generates an intuitive, plain-English operational reason for the selected action."""
    solar = row["solar_kw"]
    wind = row["wind_kw"]
    renew = solar + wind
    demand = row["demand_kw"]
    battery = row["battery_level"]
    hour = int(row["hour"])
    is_peak = peak_hours[0] <= hour < peak_hours[1]

    if action == 0:  # CHARGE
        if renew > demand and battery < 0.99:
            return "Surplus clean solar/wind generation available and battery is not full, so charging battery."
        elif not is_peak and battery < 0.35:
            return "Electricity price is cheap (off-peak rate); opportunistically charging battery for upcoming peak loads."
        else:
            return "Charging battery to secure clean energy reserves for high-tariff intervals."
    elif action == 1:  # DISCHARGE
        if is_peak:
            return "Peak grid electricity price active (₹10/kWh); discharging battery to save expensive grid costs."
        elif renew < demand:
            return "Campus load exceeds renewable generation; discharging stored energy to avoid grid purchase."
        else:
            return "Discharging stored clean power to meet college demand directly."
    elif action == 2:  # BUY
        if renew < demand:
            return "Renewables and battery capacity insufficient; purchasing balance directly from the grid."
        else:
            return "Supplying campus needs from grid import while preserving storage reserves."
    elif action == 3:  # SELL
        if renew > demand and battery >= 0.98:
            return "Battery storage is fully charged and surplus solar remains; exporting to grid for revenue."
        else:
            return "Exporting surplus renewable electricity to the grid to earn feed-in revenue."
    return "Balanced baseline operation."


def run_live_simulation(strat, days, capacity):
    """Executes the simulation dynamically from trained model weights or baseline policies."""
    import src.grid_simulator
    src.grid_simulator.BATTERY_CAPACITY_KWH = capacity

    start_day = TRAIN_SPLIT_DAY
    total_steps = days * 24
    records = []

    if strat == "Grid Only":
        policy = GridOnly()
        sim = MicrogridSimulator()
        state = sim.reset(start_day * 24)
        for _ in range(total_steps):
            act = policy.act(state)
            res = sim.step(act)
            records.append((state, act, res))
            state = res["state"]

    elif strat == "Rule-Based":
        policy = RuleBased()
        sim = MicrogridSimulator()
        state = sim.reset(start_day * 24)
        for _ in range(total_steps):
            act = policy.act(state)
            res = sim.step(act)
            records.append((state, act, res))
            state = res["state"]

    elif strat == "DQN":
        model_path = "models/dqn_grid.zip"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please train DQN first.")
            return None
        model = DQN.load(model_path)
        env = SmartGridEnv(mode="eval", episode_days=days, fixed_start_day=start_day)
        obs, _ = env.reset()
        for _ in range(total_steps):
            cur_state = env.sim.get_state()
            act, _ = model.predict(obs, deterministic=True)
            act = int(act)
            obs, reward, term, trunc, info = env.step(act)
            records.append((cur_state, act, info))
            if term or trunc:
                break

    elif strat == "PPO":
        model_path = "models/ppo_grid.zip"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please train PPO first.")
            return None
        model = PPO.load(model_path)
        env = SmartGridEnv(mode="eval", episode_days=days, fixed_start_day=start_day)
        obs, _ = env.reset()
        for _ in range(total_steps):
            cur_state = env.sim.get_state()
            act, _ = model.predict(obs, deterministic=True)
            act = int(act)
            obs, reward, term, trunc, info = env.step(act)
            records.append((cur_state, act, info))
            if term or trunc:
                break

    elif strat == "DQN + Forecast":
        model_path = "models/dqn_forecast.zip"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please train DQN + Forecast first.")
            return None
        model = DQN.load(model_path)
        env = SmartGridForecastEnv(mode="eval", episode_days=days, fixed_start_day=start_day)
        obs, _ = env.reset()
        for _ in range(total_steps):
            cur_state = env.sim.get_state()
            act, _ = model.predict(obs, deterministic=True)
            act = int(act)
            obs, reward, term, trunc, info = env.step(act)
            records.append((cur_state, act, info))
            if term or trunc:
                break

    # Format simulation results into a DataFrame
    rows = []
    for cur_state, act, res in records:
        solar = float(cur_state["solar_kw"])
        wind = float(cur_state["wind_kw"])
        renew = solar + wind
        demand = float(cur_state["demand_kw"])
        bought = float(res["bought_kwh"])
        sold = float(res["sold_kwh"])
        net = renew - demand

        solar_ratio = (solar / renew) if renew > 0 else 0.0
        renew_to_college = min(renew, demand)
        solar_to_college = renew_to_college * solar_ratio

        if act == 0:  # CHARGE
            surplus = max(0.0, net)
            solar_to_battery = min(surplus, 25.0) * solar_ratio
            battery_to_college = 0.0
            grid_to_college = max(0.0, demand - renew_to_college)
            college_to_grid = 0.0
        elif act == 1:  # DISCHARGE
            solar_to_battery = 0.0
            battery_to_college = max(0.0, demand - renew_to_college - bought)
            grid_to_college = bought
            college_to_grid = 0.0
        elif act == 2:  # BUY
            solar_to_battery = 0.0
            battery_to_college = 0.0
            grid_to_college = bought
            college_to_grid = 0.0
        elif act == 3:  # SELL
            solar_to_battery = 0.0
            battery_to_college = 0.0
            grid_to_college = bought
            college_to_grid = sold
        else:
            solar_to_battery = battery_to_college = grid_to_college = college_to_grid = 0.0

        rows.append({
            "index": cur_state["index"],
            "hour": cur_state["hour"],
            "day": cur_state["index"] // 24,
            "solar_kw": solar,
            "wind_kw": wind,
            "demand_kw": demand,
            "price": cur_state["price"],
            "battery_level": res["battery_level"],
            "action": act,
            "bought_kwh": bought,
            "sold_kwh": sold,
            "cost_inr": res["cost_inr"],
            "co2_kg": res["co2_kg"],
            "unmet_demand_kwh": res["unmet_demand_kwh"],
            "solar_to_college": round(solar_to_college, 2),
            "solar_to_battery": round(solar_to_battery, 2),
            "battery_to_college": round(battery_to_college, 2),
            "grid_to_college": round(grid_to_college, 2),
            "college_to_grid": round(college_to_grid, 2),
        })

    df = pd.DataFrame(rows)
    df["cum_cost"] = df["cost_inr"].cumsum()
    df["cum_co2"] = df["co2_kg"].cumsum()
    return df


# Check or trigger initial simulation run
if run_button or "live_df" not in st.session_state:
    with st.spinner(f"Simulating {strategy} across {sim_days} days (Battery: {battery_capacity} kWh)..."):
        sim_df = run_live_simulation(strategy, sim_days, battery_capacity)
        if sim_df is not None:
            st.session_state["live_df"] = sim_df
            st.session_state["sim_strategy"] = strategy
            st.session_state["sim_days"] = sim_days
            st.session_state["battery_capacity"] = battery_capacity
            st.session_state["selected_hour"] = 0

# --- TABS ---
tab_live, tab_cmd, tab_auto, tab_compare, tab_about = st.tabs(["⚡ Live Simulation", "🎛️ Command Center", "🤖 Autonomous Simulation", "📊 Comparison", "ℹ️ About"])


def render_hour_display(df, hour_idx):
    """Renders the complete visual interface for the selected simulation hour."""
    row = df.iloc[hour_idx]
    action_names = {0: "CHARGE BATTERY", 1: "DISCHARGE BATTERY", 2: "BUY FROM GRID", 3: "SELL TO GRID"}
    act_name = action_names.get(int(row["action"]), "UNKNOWN")
    reason = get_action_reason(int(row["action"]), row, PEAK_HOURS)

    # 1. Metric Cards for Selected Hour
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("☀️ Solar", f"{row['solar_kw']:.1f} kW")
    m2.metric("🌬️ Wind", f"{row['wind_kw']:.1f} kW")
    m3.metric("🏫 Demand", f"{row['demand_kw']:.1f} kW")

    battery_pct = float(row["battery_level"]) * 100.0
    m4.metric("🔋 Battery", f"{battery_pct:.1f}%")
    m4.progress(min(1.0, max(0.0, float(row["battery_level"]))))

    m5.metric("💰 Price", f"₹{row['price']:.2f}/kWh")
    m6.metric("🕐 Time", f"Day {int(row['day'])}, {int(row['hour']):02d}:00")

    # 2. Big Highlighted "AI DECISION" Box
    action_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#3b82f6"}
    action_emojis = {0: "🟢", 1: "🟡", 2: "🔴", 3: "🔵"}
    theme_color = action_colors.get(int(row["action"]), "#3b82f6")
    badge_emoji = action_emojis.get(int(row["action"]), "⚡")

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #090d16 0%, #111827 100%); border-radius: 12px; border-left: 8px solid {theme_color}; padding: 18px 24px; margin: 15px 0 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">🤖 AI DECISION</span>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {theme_color}; margin-top: 2px;">
                        {badge_emoji} {act_name}
                    </div>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.05); padding: 8px 16px; border-radius: 8px;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Hourly Cost / Yield</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {'#ef4444' if row['cost_inr'] > 0 else '#10b981'};">
                        {'₹+' if row['cost_inr'] > 0 else '₹'}{row['cost_inr']:.2f}
                    </div>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 1.05rem; color: #e2e8f0; font-weight: 500; line-height: 1.4;">
                <em>💡 {reason}</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Energy Flow Summary for the Hour
    st.markdown("##### ⚡ Live Microgrid Energy Flow")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown(
            f"""
            <div class="flow-pill">
                <div class="flow-pill-title">☀️ Solar → 🏫 Campus</div>
                <div class="flow-pill-value">{row['solar_to_college']:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"""
            <div class="flow-pill">
                <div class="flow-pill-title">☀️ Solar → 🔋 Battery</div>
                <div class="flow-pill-value">{row['solar_to_battery']:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f"""
            <div class="flow-pill">
                <div class="flow-pill-title">🔋 Battery → 🏫 Campus</div>
                <div class="flow-pill-value">{row['battery_to_college']:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f4:
        st.markdown(
            f"""
            <div class="flow-pill">
                <div class="flow-pill-title">⚡ Grid → 🏫 Campus</div>
                <div class="flow-pill-value">{row['grid_to_college']:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f5:
        st.markdown(
            f"""
            <div class="flow-pill">
                <div class="flow-pill-title">🏫 Campus → ⚡ Grid</div>
                <div class="flow-pill-value">{row['college_to_grid']:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Interactive Timeline Charts with selected hour marker
    ch1, ch2 = st.columns(2)
    with ch1:
        fig1 = px.line(df, x=df.index, y=["solar_kw", "demand_kw"], title="☀️ Solar Generation vs 🏫 Campus Demand (kW)", color_discrete_map={"solar_kw": "#f59e0b", "demand_kw": "#ef4444"})
        fig1.add_vline(x=hour_idx, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text=f"Hr {hour_idx}")
        fig1.update_layout(legend_title="Metric", margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        fig2 = px.line(df, x=df.index, y="battery_level", title="🔋 Battery State of Charge (Fraction 0.0 - 1.0)", color_discrete_sequence=["#10b981"])
        fig2.add_vline(x=hour_idx, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text=f"Hr {hour_idx}")
        fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        fig3 = px.line(df, x=df.index, y="cum_cost", title="💰 Cumulative Electricity Cost Over Time (₹)", color_discrete_sequence=["#3b82f6"])
        fig3.add_vline(x=hour_idx, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text=f"Hr {hour_idx}")
        fig3.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        fig4 = px.line(df, x=df.index, y="cum_co2", title="🌱 Cumulative CO₂ Emissions Over Time (kg)", color_discrete_sequence=["#a855f7"])
        fig4.add_vline(x=hour_idx, line_dash="dash", line_color="#38bdf8", line_width=2, annotation_text=f"Hr {hour_idx}")
        fig4.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
        st.plotly_chart(fig4, use_container_width=True)


# ==================== TAB 1: LIVE SIMULATION ====================
with tab_live:
    if "live_df" in st.session_state and st.session_state["live_df"] is not None:
        df = st.session_state["live_df"]
        max_hours = len(df)

        # Overview banner
        total_cost = float(df["cost_inr"].sum())
        total_co2 = float(df["co2_kg"].sum())
        total_unmet = float(df["unmet_demand_kwh"].sum())

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Strategy", st.session_state.get("sim_strategy", strategy))
        b2.metric("Total Period Cost", f"₹{total_cost:,.2f}")
        b3.metric("Total CO₂ Emissions", f"{total_co2:,.1f} kg")
        b4.metric("Unmet Load", f"{total_unmet:.2f} kWh")

        st.markdown("---")

        # SCADA Flight Deck Transport Console
        current_h = int(st.session_state.get("selected_hour", 0))
        day_num = current_h // 24
        hour_of_day = current_h % 24

        st.markdown(
            f"""
            <div class="control-deck">
                <div class="deck-header">
                    <span>🎮 Flight Deck Timeline Scrubber & Transport Console</span>
                    <span class="scada-badge">
                        <span class="led-dot led-cyan"></span>HOUR {current_h} / {max_hours-1} &bull; DAY {day_num}, {hour_of_day:02d}:00
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.2, 1.6, 1.2, 1.2, 1.2, 2.2])
        with ctrl_col1:
            prev_btn = st.button("⏮️ -1h Step", key="tab1_prev", use_container_width=True)
        with ctrl_col2:
            start_live = st.button("▶️ Auto-Play Live", type="primary", key="tab1_play", use_container_width=True)
        with ctrl_col3:
            pause_live = st.button("⏸️ Pause", key="tab1_pause", use_container_width=True)
        with ctrl_col4:
            next_btn = st.button("⏭️ +1h Step", key="tab1_next", use_container_width=True)
        with ctrl_col5:
            reset_h_btn = st.button("🔄 Hr 0", key="tab1_reset", use_container_width=True)
        with ctrl_col6:
            anim_speed = st.slider("⏱️ Playback Velocity (s/h)", min_value=0.05, max_value=0.8, value=0.15, step=0.05, key="tab1_speed")

        if prev_btn:
            current_h = max(0, current_h - 1)
            st.session_state["selected_hour"] = current_h

        if next_btn:
            current_h = min(max_hours - 1, current_h + 1)
            st.session_state["selected_hour"] = current_h

        if reset_h_btn:
            current_h = 0
            st.session_state["selected_hour"] = 0

        slider_h = st.slider(
            "Precision Timeline Scrubber (Hours)",
            min_value=0,
            max_value=max_hours - 1,
            value=current_h,
            key="time_scrubber",
        )
        st.session_state["selected_hour"] = slider_h

        st.markdown("---")

        # Live Display Container
        live_area = st.empty()

        if start_live:
            # Live simulation loop: animate through time
            for h in range(st.session_state["selected_hour"], max_hours):
                st.session_state["selected_hour"] = h
                with live_area.container():
                    render_hour_display(df, h)
                time.sleep(anim_speed)
        else:
            with live_area.container():
                render_hour_display(df, st.session_state["selected_hour"])


# ==================== TAB 2: COMMAND CENTER ====================
with tab_cmd:
    st.subheader("🎛️ Microgrid Command Center & Autonomous Operations")

    # Ensure Command Center session state is initialized
    if "cmd_sim" not in st.session_state:
        st.session_state["cmd_sim"] = MicrogridSimulator()
        st.session_state["cmd_sim"].reset(0)

    if "cmd_autopilot" not in st.session_state:
        st.session_state["cmd_autopilot"] = AutoPilot(
            simulator=st.session_state["cmd_sim"],
            strategy="DQN",
            override_hours=2,
        )

    if "cmd_controller" not in st.session_state:
        st.session_state["cmd_controller"] = CommandController(
            autopilot=st.session_state["cmd_autopilot"],
            override_hours=2,
        )

    if "cmd_scan_report" not in st.session_state:
        st.session_state["cmd_scan_report"] = None

    if "cmd_history_df" not in st.session_state:
        st.session_state["cmd_history_df"] = pd.DataFrame()

    cmd_sim = st.session_state["cmd_sim"]
    ap = st.session_state["cmd_autopilot"]
    controller = st.session_state["cmd_controller"]
    es = cmd_sim.energy_system
    sim_state = cmd_sim.get_state()

    # --- 1. TOP CONTROL BAR: Auto-Pilot, Strategy, Mode ---
    st.markdown(
        """
        <div class="control-deck">
            <div class="deck-header">
                <span>🎛️ Supervisory Control & Data Acquisition (SCADA) Core</span>
                <span class="scada-badge"><span class="led-dot led-cyan"></span>DISPATCH BUS ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_top1, col_top2, col_top3 = st.columns([1.5, 2.0, 2.5])

    with col_top1:
        auto_pilot_toggle = st.toggle(
            "🤖 Auto-Pilot",
            value=ap.enabled,
            key="cc_ap_toggle",
            help="Enable or disable autonomous decision-making",
        )
        if auto_pilot_toggle != ap.enabled:
            ap.enabled = auto_pilot_toggle
            controller.auto_mode = auto_pilot_toggle
            st.rerun()

    with col_top2:
        strat_options = ["DQN", "PPO", "DQN+Forecast", "Rule-Based"]
        cur_strat_idx = strat_options.index(ap.strategy) if ap.strategy in strat_options else 0
        selected_strategy = st.selectbox(
            "Dispatch Strategy",
            strat_options,
            index=cur_strat_idx,
            key="cc_strat_select",
            help="AI or heuristic strategy governing Auto-Pilot actions",
        )
        if selected_strategy != ap.strategy:
            ap.strategy = selected_strategy
            ap._load_strategy_model()
            st.rerun()

    with col_top3:
        st.markdown("**System Mode:**")
        if ap.is_override_active():
            mode_html = f'<span style="background-color: rgba(120, 53, 15, 0.4); border: 1px solid #f59e0b; color: #fde68a; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center;"><span class="led-dot led-amber"></span>MANUAL OVERRIDE ({ap.active_override["mode"].upper()}, {ap.active_override["hours_left"]}h left)</span>'
        elif ap.enabled:
            mode_html = '<span style="background-color: rgba(6, 95, 70, 0.4); border: 1px solid #10b981; color: #a7f3d0; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center;"><span class="led-dot led-green"></span>AUTOPILOT ENGAGED</span>'
        else:
            mode_html = '<span style="background-color: rgba(55, 65, 81, 0.4); border: 1px solid #6b7280; color: #d1d5db; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; display: inline-flex; align-items: center;"><span class="led-dot led-gray"></span>MANUAL OPERATOR MODE</span>'
        st.markdown(mode_html, unsafe_allow_html=True)
        if ap.model_notice:
            st.caption(f"⚠️ {ap.model_notice}")

    st.markdown("---")

    # --- 2. DEVICE STATUS PANEL (4 Cards with Indicators & Quick Toggles) ---
    st.subheader("🖥️ Device Status & Actuators")
    c_sol, c_wnd, c_bat, c_grd = st.columns(4)

    def get_status_badge(dev):
        if dev.status == "ON":
            return '<span style="background-color: rgba(6, 95, 70, 0.35); border: 1px solid #10b981; color: #34d399; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; display: inline-flex; align-items: center;"><span class="led-dot led-green"></span>ONLINE</span>'
        elif dev.status == "FAULT":
            fault_txt = f" ({dev.fault_type})" if dev.fault_type else ""
            return f'<span style="background-color: rgba(127, 29, 29, 0.4); border: 1px solid #ef4444; color: #f87171; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; display: inline-flex; align-items: center;"><span class="led-dot led-red"></span>FAULT{fault_txt}</span>'
        else:
            return '<span style="background-color: rgba(55, 65, 81, 0.4); border: 1px solid #6b7280; color: #9ca3af; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; display: inline-flex; align-items: center;"><span class="led-dot led-gray"></span>OFFLINE</span>'

    # Card 1: Solar
    with c_sol:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.05rem;">☀️ Solar PV Array</span>
                    {get_status_badge(es.solar)}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f59e0b; margin-bottom: 4px;">
                    {sim_state['solar_kw']:.1f} kW
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Health: <strong>{es.solar.derate * 100:.0f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_s1, b_s2 = st.columns(2)
        if b_s1.button("Turn ON", key="btn_sol_on", use_container_width=True):
            res = controller.execute("solar on")
            st.session_state["cmd_last_result"] = res
            st.rerun()
        if b_s2.button("Turn OFF", key="btn_sol_off", use_container_width=True):
            res = controller.execute("solar off")
            st.session_state["cmd_last_result"] = res
            st.rerun()

    # Card 2: Wind
    with c_wnd:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.05rem;">🌬️ Wind Turbine</span>
                    {get_status_badge(es.wind)}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #38bdf8; margin-bottom: 4px;">
                    {sim_state['wind_kw']:.1f} kW
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Health: <strong>{es.wind.derate * 100:.0f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_w1, b_w2 = st.columns(2)
        if b_w1.button("Turn ON", key="btn_wnd_on", use_container_width=True):
            res = controller.execute("wind on")
            st.session_state["cmd_last_result"] = res
            st.rerun()
        if b_w2.button("Turn OFF", key="btn_wnd_off", use_container_width=True):
            res = controller.execute("wind off")
            st.session_state["cmd_last_result"] = res
            st.rerun()

    # Card 3: Battery
    with c_bat:
        bat_soc = sim_state['battery_level'] * 100.0
        eff_txt = f"{getattr(es.battery, 'efficiency', 0.95)*100:.0f}% eff"
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.05rem;">🔋 Battery BESS</span>
                    {get_status_badge(es.battery)}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #10b981; margin-bottom: 4px;">
                    {bat_soc:.1f}% SoC
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Health: <strong>{es.battery.derate * 100:.0f}%</strong> ({eff_txt})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_b1, b_b2 = st.columns(2)
        if b_b1.button("Enable", key="btn_bat_on", use_container_width=True):
            res = controller.execute("battery on")
            st.session_state["cmd_last_result"] = res
            st.rerun()
        if b_b2.button("Disable", key="btn_bat_off", use_container_width=True):
            res = controller.execute("battery off")
            st.session_state["cmd_last_result"] = res
            st.rerun()

    # Card 4: Grid
    with c_grd:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.05rem;">⚡ Main Grid PCC</span>
                    {get_status_badge(es.grid)}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #818cf8; margin-bottom: 4px;">
                    ₹{sim_state['price']:.2f}/kWh
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                    Mode: <strong>{'Grid-Tied' if es.grid.status == 'ON' else 'Island Mode'}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_g1, b_g2 = st.columns(2)
        if b_g1.button("Connect", key="btn_grd_on", use_container_width=True):
            res = controller.execute("grid connect")
            st.session_state["cmd_last_result"] = res
            st.rerun()
        if b_g2.button("Disconnect", key="btn_grd_off", use_container_width=True):
            res = controller.execute("grid disconnect")
            st.session_state["cmd_last_result"] = res
            st.rerun()

    st.markdown("---")

    # --- 3. COMMAND BOX (Text input, Quick buttons, Help) ---
    st.subheader("💬 Command Terminal")

    st.caption("Quick Action Presets:")
    qb1, qb2, qb3, qb4, qb5, qb6 = st.columns(6)
    if qb1.button("☀️ Solar Toggle", key="qb_sol", use_container_width=True):
        cmd = "solar off" if es.solar.status == "ON" else "solar on"
        st.session_state["cmd_last_result"] = controller.execute(cmd)
        st.rerun()
    if qb2.button("🌬️ Wind Toggle", key="qb_wnd", use_container_width=True):
        cmd = "wind off" if es.wind.status == "ON" else "wind on"
        st.session_state["cmd_last_result"] = controller.execute(cmd)
        st.rerun()
    if qb3.button("🔋 Charge (2h)", key="qb_chg", use_container_width=True):
        st.session_state["cmd_last_result"] = controller.execute("battery charge")
        st.rerun()
    if qb4.button("🔋 Discharge (2h)", key="qb_dis", use_container_width=True):
        st.session_state["cmd_last_result"] = controller.execute("battery discharge")
        st.rerun()
    if qb5.button("🔋 Hold Battery", key="qb_hld", use_container_width=True):
        st.session_state["cmd_last_result"] = controller.execute("battery hold")
        st.rerun()
    if qb6.button("⚡ Grid Toggle", key="qb_grd", use_container_width=True):
        cmd = "grid disconnect" if es.grid.status == "ON" else "grid connect"
        st.session_state["cmd_last_result"] = controller.execute(cmd)
        st.rerun()

    with st.form("command_center_form", clear_on_submit=True):
        col_c_in, col_c_btn = st.columns([5, 1])
        user_command_str = col_c_in.text_input(
            "Natural Language Command:",
            placeholder="Type command, e.g. 'please switch solar off', 'battery charge', 'clear fault solar', 'status'",
            label_visibility="collapsed",
            key="cc_input_field",
        )
        send_btn = col_c_btn.form_submit_button("Send 🚀", use_container_width=True)

        if send_btn and user_command_str:
            res = controller.execute(user_command_str)
            st.session_state["cmd_last_result"] = res
            st.rerun()

    # Display command feedback
    if "cmd_last_result" in st.session_state and st.session_state["cmd_last_result"] is not None:
        last_res = st.session_state["cmd_last_result"]
        if not last_res.success:
            st.error(f"🛑 {last_res.message}")
        elif last_res.warning:
            st.warning(f"⚠️ {last_res.message}")
        else:
            st.success(f"✅ {last_res.message}")

    with st.expander("📖 Available Commands & Help Reference"):
        st.markdown(
            r"""
            The CommandController supports forgiving natural language parsing (case-insensitive with extra words):
            - **Solar Control**: `solar on`, `solar off`, `"please switch the solar panel off"`
            - **Wind Control**: `wind on`, `wind off`, `"turn off the wind turbine"`
            - **Battery Directives**: `battery charge`, `battery discharge`, `battery hold`, `battery on/off`
            - **Grid Connection**: `grid connect`, `grid disconnect` (warns if island demand cannot be met)
            - **Autonomous Mode**: `auto on`, `auto off`
            - **Faults & Diagnostics**: `scan`, `status`, `clear fault`, `clear fault solar`
            - **Help**: `help`
            """
        )

    st.markdown("---")

    # --- 4. SCAN SECTION ---
    st.subheader("🔍 Diagnostics & Health Scan")
    col_sc1, col_sc2 = st.columns([1.5, 4.5])

    with col_sc1:
        if st.button("🔍 Run Full Diagnostic Scan", key="btn_run_full_scan", use_container_width=True):
            st.session_state["cmd_scan_report"] = scan(
                simulator=cmd_sim,
                energy_system=es,
                history=st.session_state["cmd_history_df"],
            )
            st.rerun()

    scan_rep = st.session_state.get("cmd_scan_report")
    if scan_rep is not None:
        if scan_rep.is_clean:
            st.success("✅ Diagnostic Scan Complete: All system devices healthy and operating at 100% capacity.")
        else:
            st.warning(f"⚠️ Diagnostic Scan Alert: Found {len(scan_rep)} issue(s) requiring attention.")
            scan_table = []
            for issue in scan_rep:
                sev = issue["severity"].lower()
                sev_icon = "🔴 CRITICAL" if sev == "critical" else ("🟡 WARNING" if sev == "warning" else "🔵 INFO")
                scan_table.append({
                    "Severity": sev_icon,
                    "Device": issue["device"].upper(),
                    "Problem Description": issue["description"],
                    "Suggested Action / Fix": issue["suggested_fix"],
                })
            st.dataframe(pd.DataFrame(scan_table), use_container_width=True)

    st.markdown("---")

    # --- 5. DEMO SECTION: Fault Injection & Clear ---
    st.subheader("🧪 Fault Injection Sandbox (Demo & Stress Testing)")
    col_d1, col_d2, col_d3 = st.columns([3, 1.2, 1.2])

    with col_d1:
        fault_to_inject = st.selectbox(
            "Select Fault Condition to Inject:",
            [
                "solar soiling (output reduced 30%)",
                "inverter failure (solar to 0)",
                "battery degradation (efficiency drops)",
                "sensor stuck (solar flatline)",
                "grid outage (island mode)",
            ],
            key="demo_fault_select",
        )

    with col_d2:
        if st.button("⚠️ Inject Fault", key="btn_demo_inject", use_container_width=True):
            es.inject_fault(fault_to_inject)
            st.session_state["cmd_scan_report"] = scan(simulator=cmd_sim, energy_system=es)
            st.warning(f"Injected fault: '{fault_to_inject}'. Check status panel and run scan.")
            st.rerun()

    with col_d3:
        if st.button("✅ Clear All Faults", key="btn_demo_clear", use_container_width=True):
            es.clear_faults()
            controller.execute("clear fault")
            st.session_state["cmd_scan_report"] = scan(simulator=cmd_sim, energy_system=es)
            st.success("All faults cleared! System restored to 100% health.")
            st.rerun()

    st.markdown("---")

    # --- 6. SIMULATION CONTROLS & LIVE CHARTS ---
    st.subheader("🕹️ Simulation Dispatch & Live Telemetry")

    step_idx = int(sim_state["index"])
    st.markdown(
        f"""
        <div class="control-deck">
            <div class="deck-header">
                <span>⏱️ Real-Time Actuation & Time-Step Dispatch Console</span>
                <span class="scada-badge">
                    <span class="led-dot led-cyan"></span>STEP: {step_idx} &bull; DAY {step_idx // 24}, {step_idx % 24:02d}:00
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sim_btn1, col_sim_btn2, col_sim_btn3, col_sim_btn4 = st.columns([1.3, 1.0, 1.4, 1.2])

    with col_sim_btn1:
        step_1h_btn = st.button("⏩ Single Step (1h)", key="btn_cc_step_1h", use_container_width=True)

    with col_sim_btn2:
        run_n_count = st.number_input("Hours", min_value=1, max_value=168, value=24, step=1, label_visibility="collapsed", key="cc_n_hours_input")

    with col_sim_btn3:
        run_nh_btn = st.button(f"▶️ Dispatch {run_n_count}h Batch", type="primary", key="btn_cc_run_nh", use_container_width=True)

    with col_sim_btn4:
        reset_sim_btn = st.button("🔄 Reset Step 0", key="btn_cc_reset", use_container_width=True)

    def record_step_to_history(step_output):
        res = step_output["result"]
        cur_st = step_output["state"]
        new_row = {
            "index": int(cur_st["index"]),
            "hour": int(cur_st["hour"]),
            "day": int(cur_st["index"]) // 24,
            "solar_kw": float(cur_st["solar_kw"]),
            "wind_kw": float(cur_st["wind_kw"]),
            "demand_kw": float(cur_st["demand_kw"]),
            "price": float(cur_st["price"]),
            "battery_level": float(res["battery_level"]),
            "action": int(step_output["action"]),
            "action_name": step_output["action_name"],
            "reason": step_output["reason"],
            "strategy": step_output["strategy_used"],
            "bought_kwh": float(res["bought_kwh"]),
            "sold_kwh": float(res["sold_kwh"]),
            "cost_inr": float(res["cost_inr"]),
            "co2_kg": float(res["co2_kg"]),
            "unmet_demand_kwh": float(res["unmet_demand_kwh"]),
        }
        df_row = pd.DataFrame([new_row])
        if st.session_state["cmd_history_df"].empty:
            st.session_state["cmd_history_df"] = df_row
        else:
            st.session_state["cmd_history_df"] = pd.concat([st.session_state["cmd_history_df"], df_row], ignore_index=True)

    if step_1h_btn:
        out = ap.step()
        record_step_to_history(out)
        st.rerun()

    if run_nh_btn:
        with st.spinner(f"Advancing Auto-Pilot simulation across {run_n_count} hours..."):
            for _ in range(int(run_n_count)):
                out = ap.step()
                record_step_to_history(out)
                if out["result"]["done"]:
                    break
        st.rerun()

    if reset_sim_btn:
        cmd_sim.reset(0)
        st.session_state["cmd_history_df"] = pd.DataFrame()
        ap.decision_log.clear()
        st.session_state["cmd_scan_report"] = None
        st.success("Simulation reset to Hour 0.")
        st.rerun()

    # Display Current Live Telemetry & AI Decision Card
    cur_telemetry = cmd_sim.get_state()
    latest_decision = ap.decision_log[-1] if ap.decision_log else None

    # Metric Cards
    tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
    tm1.metric("☀️ Solar", f"{cur_telemetry['solar_kw']:.1f} kW")
    tm2.metric("🌬️ Wind", f"{cur_telemetry['wind_kw']:.1f} kW")
    tm3.metric("🏫 Demand", f"{cur_telemetry['demand_kw']:.1f} kW")
    tm4.metric("🔋 Battery", f"{cur_telemetry['battery_level']*100:.1f}%")
    tm4.progress(min(1.0, max(0.0, cur_telemetry["battery_level"])))
    tm5.metric("💰 Price", f"₹{cur_telemetry['price']:.2f}/kWh")
    tm6.metric("🕐 Step Hour", f"Hour {cur_telemetry['index']} (Day {cur_telemetry['index']//24})")

    # Big AI Decision Box
    if latest_decision is not None:
        act_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#3b82f6"}
        theme_col = act_colors.get(latest_decision["action"], "#3b82f6")
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 2px solid {theme_col}; border-radius: 12px; padding: 18px 24px; box-shadow: 0 0 25px {theme_col}33; margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">🤖 AI Decision Dispatch ({latest_decision['strategy']})</span>
                    <span style="background-color: {theme_col}26; color: {theme_col}; border: 1px solid {theme_col}66; padding: 4px 12px; border-radius: 9999px; font-weight: 800; font-size: 0.85rem;">
                        {latest_decision['action_name']}
                    </span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc;">
                    "{latest_decision['reason']}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Live Charts
    h_df = st.session_state["cmd_history_df"]
    if not h_df.empty:
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(x=h_df.index, y=h_df["solar_kw"], name="Solar (kW)", line=dict(color="#f59e0b", width=2)))
            fig_power.add_trace(go.Scatter(x=h_df.index, y=h_df["wind_kw"], name="Wind (kW)", line=dict(color="#38bdf8", width=2)))
            fig_power.add_trace(go.Scatter(x=h_df.index, y=h_df["demand_kw"], name="Demand (kW)", line=dict(color="#ef4444", width=2, dash="dash")))
            fig_power.update_layout(title="⚡ Renewable Generation vs Campus Demand", height=320, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_power, use_container_width=True)

        with ch2:
            fig_bat = go.Figure()
            fig_bat.add_trace(go.Scatter(x=h_df.index, y=h_df["battery_level"] * 100, name="Battery SoC (%)", line=dict(color="#10b981", width=2.5), fill="tozeroy"))
            fig_bat.add_hline(y=10, line_dash="dot", line_color="#ef4444", annotation_text="10% Safety Floor")
            fig_bat.update_layout(title="🔋 Battery State of Charge (%) & Safety Limits", height=320, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), yaxis_range=[0, 105])
            st.plotly_chart(fig_bat, use_container_width=True)

    st.markdown("---")

    # --- 7. LOG PANEL (Scrollable timestamped history) ---
    st.subheader("📜 System Audit & Activity Logs")
    l_tab1, l_tab2, l_tab3 = st.tabs(["🤖 Auto-Pilot Decisions", "💬 Command History", "🚨 Safety & Diagnostics Alerts"])

    with l_tab1:
        if ap.decision_log:
            dec_df = pd.DataFrame(ap.decision_log).iloc[::-1]
            st.dataframe(dec_df, use_container_width=True)
        else:
            st.info("No Auto-Pilot decisions logged yet. Click 'Step 1 Hour' or 'Run N Hours' to begin.")

    with l_tab2:
        if controller.command_log:
            cmd_df = pd.DataFrame(controller.command_log).iloc[::-1]
            st.dataframe(cmd_df, use_container_width=True)
        else:
            st.info("No text commands executed yet.")

    with l_tab3:
        all_events = controller.event_log + ap.event_log
        if all_events:
            ev_df = pd.DataFrame(all_events).iloc[::-1]
            st.dataframe(ev_df, use_container_width=True)
        else:
            st.info("No safety refusals or critical diagnostic events recorded.")


# ==================== TAB 3: AUTONOMOUS SIMULATION ====================
def render_autonomous_view(op: AutonomousOperator, total_hours: int, auto_preset: str, auto_seed: int, auto_strat: str):
    """Renders the comprehensive live interface for the autonomous supervisory microgrid simulation."""
    cur_h = op.current_hour
    history = op.step_history
    sim_state = op.simulator.get_state()
    es = op.energy_system

    if not history:
        st.info("Simulation initialized at Hour 0. Press **'▶️ Start Simulation'** or **'⏭️ Step 1h'** to launch autonomous operations.")
        return

    latest_rec = history[-1]
    day = int(latest_rec.get("day", cur_h // 24))
    hour_of_day = int(latest_rec.get("hour", 0)) % 24

    # 1. System Health Assessment (Green / Amber / Red)
    active_faults = [
        f"{name.title()}: {dev.status} ({dev.fault_type or 'degraded'})"
        for name, dev in es.devices.items()
        if dev.status != "ON" or dev.derate < 0.95 or getattr(dev, "sensor_stuck", False)
    ]
    unmet_now = latest_rec.get("unmet_demand_kwh", 0.0)

    if es.grid.status != "ON" or unmet_now > 0.0 or any(dev.status == "FAULT" and dev.derate == 0.0 for dev in es.devices.values()):
        health_color = "#ef4444"
        health_text = "🔴 CRITICAL - GRID OUTAGE / FAULT ACTIVE"
        health_badge_bg = "rgba(239, 68, 68, 0.2)"
    elif active_faults or float(sim_state["price"]) > 15.0 or es.price_multiplier > 1.0:
        health_color = "#f59e0b"
        health_text = "🟡 DEGRADED - SUBSYSTEM IMPAIRMENT / TARIFF SPIKE"
        health_badge_bg = "rgba(245, 158, 11, 0.2)"
    else:
        health_color = "#10b981"
        health_text = "🟢 HEALTHY - ALL ASSETS NOMINAL"
        health_badge_bg = "rgba(16, 185, 129, 0.2)"

    # --- Live Mission Control Header ---
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
                    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 14px 20px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">Simulated Date & Time</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 800; color: #f8fafc;">
                        📅 Day {day}, {hour_of_day:02d}:00 <span style="font-size: 0.95rem; color: #94a3b8; font-weight: 500;">(Hour {cur_h} / {total_hours})</span>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <div style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; border-radius: 8px; padding: 6px 14px; text-align: center;">
                        <span style="font-size: 0.72rem; color: #93c5fd; text-transform: uppercase; font-weight: 700;">Control Mode</span>
                        <div style="font-size: 0.95rem; font-weight: 800; color: #60a5fa;">🤖 AUTONOMOUS</div>
                    </div>
                    <div style="background: rgba(147, 51, 234, 0.2); border: 1px solid #9333ea; border-radius: 8px; padding: 6px 14px; text-align: center;">
                        <span style="font-size: 0.72rem; color: #d8b4fe; text-transform: uppercase; font-weight: 700;">Strategy</span>
                        <div style="font-size: 0.95rem; font-weight: 800; color: #c084fc;">{auto_strat}</div>
                    </div>
                    <div style="background: {health_badge_bg}; border: 1px solid {health_color}; border-radius: 8px; padding: 6px 14px; text-align: center;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Health Status</span>
                        <div style="font-size: 0.95rem; font-weight: 800; color: {health_color};">{health_text}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 2. Metric Cards ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("☀️ Solar", f"{float(latest_rec.get('solar_kw', sim_state['solar_kw'])):.1f} kW")
    m2.metric("🌬️ Wind", f"{float(latest_rec.get('wind_kw', sim_state['wind_kw'])):.1f} kW")
    m3.metric("🏫 Demand", f"{float(latest_rec.get('demand_kw', sim_state['demand_kw'])):.1f} kW")

    bat_pct = float(latest_rec.get("battery_level", sim_state["battery_level"])) * 100.0
    m4.metric("🔋 Battery", f"{bat_pct:.1f}%")
    m4.progress(min(1.0, max(0.0, float(latest_rec.get("battery_level", sim_state["battery_level"])))))

    m5.metric("💰 Tariff", f"₹{float(latest_rec.get('price', sim_state['price'])):.2f}/kWh")
    grid_status_str = "🟢 CONNECTED" if es.grid.status == "ON" else "🔴 ISLAND MODE"
    m6.metric("⚡ Grid Status", grid_status_str)

    # --- 3. Big "🤖 AI DECISION" Box + Last 5 Decisions Feed ---
    act_idx = int(latest_rec.get("action", 2))
    act_name = ACTION_NAMES.get(act_idx, "UNKNOWN")
    action_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#3b82f6"}
    action_emojis = {0: "🟢", 1: "🟡", 2: "🔴", 3: "🔵"}
    theme_color = action_colors.get(act_idx, "#3b82f6")
    badge_emoji = action_emojis.get(act_idx, "⚡")
    reason = latest_rec.get("reason", "Proactive energy balance optimization.")
    cost_step = latest_rec.get("cost_inr", 0.0)

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #090d16 0%, #111827 100%);
                    border-radius: 12px; border-left: 8px solid {theme_color}; padding: 18px 24px; margin: 15px 0 20px 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 1.05rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">🤖 AI DECISION</span>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {theme_color}; margin-top: 2px;">
                        {badge_emoji} {act_name}
                    </div>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.05); padding: 8px 16px; border-radius: 8px;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Hourly Cost Impact</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {'#ef4444' if cost_step > 0 else '#10b981'};">
                        {'₹+' if cost_step > 0 else '₹'}{cost_step:.2f}
                    </div>
                </div>
            </div>
            <p style="margin: 12px 0 0 0; font-size: 1.05rem; color: #e2e8f0; font-weight: 500; line-height: 1.45;">
                <em>💡 {reason}</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini feed: Last 5 decisions
    st.markdown("**⏱️ Last 5 Decisions Mini Feed:**")
    last_5 = history[-5:][::-1]
    cols = st.columns(len(last_5))
    for idx, (col, r) in enumerate(zip(cols, last_5)):
        r_act = int(r.get("action", 2))
        r_color = action_colors.get(r_act, "#94a3b8")
        r_name = ACTION_NAMES.get(r_act, "BUY")
        with col:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; color: #94a3b8;">
                        <span>Hour {r.get('hour', 0)}</span>
                        <span style="color: {r_color};">{r_name.split()[0]}</span>
                    </div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{r.get('reason')}">
                        {r.get('reason')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- 4. Live Charts with Event Markers ---
    hist_df = pd.DataFrame(history)
    if not hist_df.empty and "hour" in hist_df.columns:
        c_ch1, c_ch2 = st.columns(2)

        with c_ch1:
            fig_power = go.Figure()
            if "solar_kw" in hist_df.columns:
                fig_power.add_trace(go.Scatter(x=hist_df["hour"], y=hist_df["solar_kw"], name="Solar (kW)", line=dict(color="#f59e0b", width=2)))
            if "wind_kw" in hist_df.columns:
                fig_power.add_trace(go.Scatter(x=hist_df["hour"], y=hist_df["wind_kw"], name="Wind (kW)", line=dict(color="#06b6d4", width=1.8)))
            if "demand_kw" in hist_df.columns:
                fig_power.add_trace(go.Scatter(x=hist_df["hour"], y=hist_df["demand_kw"], name="Demand (kW)", line=dict(color="#ef4444", width=2.2)))

            fig_power.update_layout(
                title="⚡ Solar & Wind Generation vs Campus Demand",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                template="plotly_dark",
                height=320,
                margin=dict(l=30, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_power, use_container_width=True)

        with c_ch2:
            fig_bat = go.Figure()
            bat_pct_series = hist_df["battery_level"] * 100.0
            fig_bat.add_trace(go.Scatter(x=hist_df["hour"], y=bat_pct_series, name="Battery SoC (%)", line=dict(color="#10b981", width=2.4)))
            fig_bat.add_hline(y=10.0, line_dash="dash", line_color="#ef4444", annotation_text="Safety Reserve (10%)", annotation_position="bottom right")

            # Add event markers (vertical spans) where faults occurred
            if op.scenario_engine is not None:
                for ev in op.scenario_engine.timeline:
                    if ev.start_hour <= cur_h:
                        x1 = min(cur_h, ev.end_hour)
                        fig_bat.add_vrect(
                            x0=ev.start_hour,
                            x1=x1,
                            fillcolor="rgba(239, 68, 68, 0.18)",
                            line_width=1,
                            line_color="#ef4444",
                            line_dash="dot",
                            annotation_text=ev.event_type.title(),
                            annotation_position="top left",
                        )

            fig_bat.update_layout(
                title="🔋 Battery State of Charge (%) & Disruption Timeline",
                xaxis_title="Hour",
                yaxis_title="SoC (%)",
                yaxis_range=[-5, 105],
                template="plotly_dark",
                height=320,
                margin=dict(l=30, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bat, use_container_width=True)

        # Cumulative Cost & CO2
        hist_df["cum_cost"] = hist_df["cost_inr"].cumsum()
        hist_df["cum_co2"] = hist_df["co2_kg"].cumsum()

        fig_cost_co2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cost_co2.add_trace(go.Scatter(x=hist_df["hour"], y=hist_df["cum_cost"], name="Cumulative Cost (₹)", line=dict(color="#3b82f6", width=2.2)), secondary_y=False)
        fig_cost_co2.add_trace(go.Scatter(x=hist_df["hour"], y=hist_df["cum_co2"], name="Cumulative CO₂ (kg)", line=dict(color="#8b5cf6", width=2.0, dash="dot")), secondary_y=True)

        if op.scenario_engine is not None:
            for ev in op.scenario_engine.timeline:
                if ev.start_hour <= cur_h:
                    fig_cost_co2.add_vline(x=ev.start_hour, line_dash="dash", line_color="#ef4444", annotation_text=ev.event_type.title(), annotation_position="top left")

        fig_cost_co2.update_layout(
            title="📈 Cumulative Cost (₹) & Carbon Emissions (kg CO₂)",
            xaxis_title="Hour",
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_cost_co2.update_yaxes(title_text="Cost (₹)", secondary_y=False)
        fig_cost_co2.update_yaxes(title_text="CO₂ (kg)", secondary_y=True)
        st.plotly_chart(fig_cost_co2, use_container_width=True)

    # --- 5. Incident Panel ---
    st.markdown("---")
    st.subheader("🚨 Incident Resilience & Autonomous Response Panel")
    inc_col1, inc_col2, inc_col3 = st.columns([2, 2.5, 1.2])

    with inc_col1:
        st.markdown("**Active Faults / Physical Disruptions:**")
        if active_faults:
            for af in active_faults:
                st.error(f"⚠️ {af}")
        else:
            st.success("✅ All physical generation & storage assets nominal.")

    with inc_col2:
        st.markdown("**Autonomous Supervisor Actions:**")
        recent_supervisor = [e for e in op.controller.event_log if e.get("event_type") in ["SOLAR_RESTART", "LOAD_SHEDDING_ACTIVE", "LOAD_SHEDDING_OFF", "SOLAR_MAINTENANCE", "FALLBACK_ENGAGED", "FALLBACK_DISENGAGED", "SOLAR_LOST_GEN"]]
        if recent_supervisor:
            for ev in recent_supervisor[-3:][::-1]:
                st.info(f"🛡️ **{ev.get('event_type')}:** {ev.get('details')}")
        else:
            st.write("AI maintaining standard proactive dispatch and battery peak shaving.")

    with inc_col3:
        resolved_count = sum(1 for inc in op.incident_log if inc.get("status") == "RESOLVED")
        total_inc = len(op.incident_log)
        st.metric("Incidents Handled", f"{resolved_count} / {total_inc} Resolved", delta=f"{resolved_count} cleared" if resolved_count > 0 else None)

    # --- 6. Scrolling Log of AI Actions ---
    with st.expander("📜 Scrolling Log of AI Actions & Audit Records", expanded=False):
        disp_cols = ["hour", "day", "action_name", "strategy", "reason", "cost_inr", "co2_kg", "unmet_demand_kwh", "load_shed_kwh"]
        available_disp = [c for c in disp_cols if c in hist_df.columns]
        st.dataframe(hist_df.iloc[::-1][available_disp], use_container_width=True)

    # --- 7. Finished Summary Report Card ---
    if cur_h >= total_hours and history:
        st.markdown("---")
        st.success("🎯 **Autonomous Simulation Completed!** Performance Summary Report:")

        tot_cost = float(hist_df["cost_inr"].sum())
        tot_co2 = float(hist_df["co2_kg"].sum())
        tot_unmet = float(hist_df["unmet_demand_kwh"].sum())
        resolved_cnt = sum(1 for inc in op.incident_log if inc.get("status") == "RESOLVED")
        tot_inc = len(op.incident_log)

        # Compute % saved vs Grid Only baseline
        go_se = ScenarioEngine(seed=int(auto_seed), total_hours=total_hours)
        go_se.generate_timeline(preset=auto_preset, seed=int(auto_seed), total_hours=total_hours)
        _, _, go_kpis = run_grid_only(go_se, hours=total_hours)
        go_cost = go_kpis["total_cost"]
        pct_saved = ((go_cost - tot_cost) / go_cost) * 100.0 if go_cost > 0 else 0.0

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Total Operating Cost", f"₹{tot_cost:,.2f}", delta=f"-₹{go_cost - tot_cost:,.2f} vs Grid Only" if tot_cost < go_cost else None)
        r2.metric("Total CO₂ Emissions", f"{tot_co2:,.1f} kg")
        r3.metric("Unmet Campus Demand", f"{tot_unmet:.2f} kWh", delta=f"-{go_kpis['unmet_demand'] - tot_unmet:.1f} kWh vs Grid" if tot_unmet < go_kpis['unmet_demand'] else None)
        r4.metric("Incidents Handled", f"{resolved_cnt} / {tot_inc} Resolved")
        r5.metric("% Saved vs Grid Only", f"{pct_saved:.2f}%")

        col_dl, col_bench = st.columns(2)
        with col_dl:
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Log as CSV",
                data=csv_bytes,
                file_name=f"autonomous_sim_{auto_strat}_{auto_preset}_seed{auto_seed}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_dl_csv",
            )

        with col_bench:
            compare_clicked = st.button("📊 Compare with Other Strategies", use_container_width=True, key="btn_compare_auto")

        if compare_clicked or st.session_state.get("auto_bench_df") is not None:
            if compare_clicked:
                with st.spinner(f"Running comparative benchmark across all strategies for Seed {auto_seed}..."):
                    sum_df, raw_df = run_benchmark(preset=auto_preset, seeds=[int(auto_seed)], hours=total_hours, output_dir="results")
                    st.session_state["auto_bench_df"] = sum_df

            if st.session_state.get("auto_bench_df") is not None:
                st.markdown("### 📊 Benchmark Comparison across All Control Strategies")
                bench_df = st.session_state["auto_bench_df"]
                disp_bench_cols = [
                    "Setup", "Cost (₹)", "CO2 (kg)", "Unmet Demand (kWh)",
                    "Incidents Detected", "Incidents Resolved", "Avg Recovery Time (h)", "% Saved vs Grid Only"
                ]
                st.dataframe(bench_df[disp_bench_cols], use_container_width=True)

                cost_p = "results/cost_bars.png"
                unmet_p = "results/unmet_demand_bars.png"
                if os.path.exists(cost_p) and os.path.exists(unmet_p):
                    b_img1, b_img2 = st.columns(2)
                    b_img1.image(cost_p, caption="Operational Cost Comparison", use_container_width=True)
                    b_img2.image(unmet_p, caption="Unmet Demand Resilience Comparison", use_container_width=True)


with tab_auto:
    st.subheader("🤖 Autonomous Supervisory Microgrid Simulation")
    st.caption("SCADA Mission Control: Continuous observation, algorithmic diagnostics, RL/rule-based dispatch, and autonomous incident resilience.")

    # 0. One-Click Viva Demo Banner & Lite Mode Toggle
    st.markdown(
        """
        <div class="viva-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="led-dot led-cyan"></span>
                        <span style="font-weight: 800; font-size: 0.95rem; color: #38bdf8; letter-spacing: 0.8px;">EXECUTIVE VIVA DEMONSTRATION RUNNER</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">
                        One-click evaluation: 3-Day (72h) Stress Test with deep solar soiling, inverter failure, and grid outage resilience.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_c1, top_c2 = st.columns([3, 2])
    with top_c1:
        run_demo_btn = st.button(
            "🚀 ▶ Run Demo (3-Day Viva Stress Test)",
            type="primary",
            use_container_width=True,
            key="btn_run_demo",
            help="One-click viva presentation demo: runs a 3-day (72h) Stress Test with the best agent at fast speed with no manual setup.",
        )
    with top_c2:
        lite_mode = st.toggle(
            "⚡ Lite Mode (Precomputed Replay)",
            value=False,
            key="auto_lite_mode_toggle",
            help="Instantly loads pre-computed simulation runs from results/ for lightweight hosting on Streamlit Cloud without live stepping overhead.",
        )

    if run_demo_btn:
        best_agent = "DQN+Forecast" if os.path.exists("models/dqn_forecast.zip") else "DQN"
        st.session_state["auto_preset_select"] = "Stress test"
        st.session_state["auto_strat_select"] = best_agent
        st.session_state["auto_days_slider"] = 3
        st.session_state["auto_seed_input"] = 42
        st.session_state["auto_speed_select"] = "fast"
        st.session_state["auto_running"] = True
        st.rerun()

    # 1. Parameter Control Matrices (Environmental Disruption & Autonomous Dispatch)
    st.markdown(
        """
        <div class="control-deck" style="margin-bottom: 8px;">
            <div class="deck-header">
                <span>⚙️ Mission Configuration Matrices</span>
                <span class="scada-badge"><span class="led-dot led-green"></span>SYNCHRONIZED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_mat1, c_mat2 = st.columns(2)
    with c_mat1:
        st.markdown(
            """
            <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                ⚡ Environmental Disruption Architecture
            </div>
            """,
            unsafe_allow_html=True,
        )
        sub_c1, sub_c2, sub_c3 = st.columns([2.0, 1.4, 1.4])
        with sub_c1:
            auto_preset = st.selectbox(
                "Scenario Preset",
                ["Stress test", "Grid outage drill", "Normal week", "Random"],
                index=0,
                key="auto_preset_select",
                help="Select environmental and physical disruption scenario timeline",
            )
        with sub_c2:
            auto_days = st.slider("Horizon (Days)", min_value=1, max_value=30, value=7, step=1, key="auto_days_slider")
        with sub_c3:
            auto_seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=42, step=1, key="auto_seed_input")

    with c_mat2:
        st.markdown(
            """
            <div style="font-size: 0.8rem; font-weight: 700; color: #a855f7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                🤖 Autonomous Dispatch Dynamics
            </div>
            """,
            unsafe_allow_html=True,
        )
        sub_d1, sub_d2 = st.columns([2.0, 1.5])
        with sub_d1:
            auto_strat = st.selectbox(
                "Dispatch Strategy",
                ["Rule-Based", "DQN", "PPO", "DQN+Forecast"],
                index=1,
                key="auto_strat_select",
                help="Autonomous dispatch policy for battery storage",
            )
        with sub_d2:
            auto_speed = st.selectbox("Clock Velocity", ["normal", "slow", "fast"], index=0, key="auto_speed_select")

    speed_delay = 0.05 if auto_speed == "normal" else (0.22 if auto_speed == "slow" else 0.005)
    total_hours = auto_days * 24

    # State initialization / configuration tracking
    auto_config = (auto_preset, auto_strat, int(auto_seed), total_hours)
    if "auto_op" not in st.session_state or st.session_state.get("auto_last_cfg") != auto_config:
        auto_se = ScenarioEngine(seed=int(auto_seed), total_hours=total_hours)
        auto_se.generate_timeline(preset=auto_preset, seed=int(auto_seed), total_hours=total_hours)
        auto_op = AutonomousOperator(strategy=auto_strat, scenario_engine=auto_se)
        c_model = get_cached_model(auto_strat)
        if c_model is not None:
            auto_op.autopilot.model = c_model
            auto_op.autopilot.model_notice = None

        st.session_state["auto_op"] = auto_op
        st.session_state["auto_last_cfg"] = auto_config
        st.session_state["auto_running"] = False
        st.session_state["auto_bench_df"] = None

    op = st.session_state["auto_op"]
    cur_h = op.current_hour if op else 0
    progress_pct = (cur_h / total_hours * 100.0) if total_hours > 0 else 0.0
    status_label = "MISSION IN PROGRESS" if st.session_state.get("auto_running") else ("MISSION COMPLETE" if cur_h >= total_hours else "READY / STANDBY")
    led_class = "led-cyan" if st.session_state.get("auto_running") else ("led-green" if cur_h >= total_hours else "led-amber")

    # 2. Mission Transport Console & Progress Bar
    st.markdown(
        f"""
        <div class="control-deck" style="margin-top: 10px;">
            <div class="deck-header">
                <span>🎮 Autonomous Mission Transport Console</span>
                <span class="scada-badge">
                    <span class="led-dot {led_class}"></span>{status_label} &bull; HOUR {cur_h} / {total_hours} ({progress_pct:.1f}%)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.5, 1.2, 1.2, 1.4])
    with btn_c1:
        start_auto_btn = st.button("▶️ Launch Mission", type="primary", use_container_width=True, key="btn_start_auto")
    with btn_c2:
        pause_auto_btn = st.button("⏸️ Pause Execution", use_container_width=True, key="btn_pause_auto")
    with btn_c3:
        step_auto_btn = st.button("⏭️ Single Step (1h)", use_container_width=True, key="btn_step_auto")
    with btn_c4:
        reset_auto_btn = st.button("🔄 Reset Subsystems", use_container_width=True, key="btn_reset_auto")

    st.progress(min(1.0, max(0.0, cur_h / total_hours)))

    # Button handlers
    if reset_auto_btn:
        st.session_state["auto_running"] = False
        auto_se = ScenarioEngine(seed=int(auto_seed), total_hours=total_hours)
        auto_se.generate_timeline(preset=auto_preset, seed=int(auto_seed), total_hours=total_hours)
        op.scenario_engine = auto_se
        op.reset(0)
        st.session_state["auto_bench_df"] = None
        st.rerun()

    if pause_auto_btn:
        st.session_state["auto_running"] = False
        st.rerun()

    if start_auto_btn:
        if op.current_hour >= total_hours:
            op.reset(0)
        st.session_state["auto_running"] = True
        st.rerun()

    if step_auto_btn:
        st.session_state["auto_running"] = False
        if op.current_hour < total_hours:
            op.run_step()
        st.rerun()

    # Live Area Placeholder
    live_container = st.empty()

    if lite_mode:
        p_slug = auto_preset.lower().replace(" ", "_")
        s_slug = auto_strat.lower().replace("+", "_").replace(" ", "_")
        candidates = [
            f"results/precomputed_{p_slug}_{s_slug}.csv",
            f"results/precomputed_{p_slug}.csv",
            "results/precomputed_demo_3day.csv" if auto_days == 3 else None,
        ]
        precomputed_file = next((c for c in candidates if c and os.path.exists(c)), None)
        if precomputed_file:
            lite_df = pd.read_csv(precomputed_file)
            target_len = min(len(lite_df), total_hours)
            op.step_history = lite_df.iloc[:target_len].to_dict(orient="records")
            op.current_hour = target_len
            st.info(f"⚡ **Lite Mode Active:** Displaying instant pre-computed replay from `{precomputed_file}` ({target_len} hours).")
        else:
            st.warning("Pre-computed file not found for this scenario/strategy. Run once with 'Start Simulation' or click '▶ Run Demo' to generate.")

        with live_container.container():
            render_autonomous_view(op, total_hours, auto_preset, int(auto_seed), auto_strat)
    elif st.session_state.get("auto_running", False):
        while op.current_hour < total_hours and st.session_state.get("auto_running", False):
            rec = op.run_step()
            with live_container.container():
                render_autonomous_view(op, total_hours, auto_preset, int(auto_seed), auto_strat)
            if speed_delay > 0:
                time.sleep(speed_delay)
            if rec.get("done", False):
                break

        st.session_state["auto_running"] = False
        st.rerun()
    else:
        with live_container.container():
            render_autonomous_view(op, total_hours, auto_preset, int(auto_seed), auto_strat)


# ==================== TAB 4: COMPARISON ====================
with tab_compare:
    st.subheader("📊 Multi-Method Performance Benchmarks")

    comp_path = "results/comparison.csv"
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Comparison table not found. Run `python src/compare.py` to generate.")

    st.markdown("---")
    st.subheader("📉 Cost & Carbon Emissions Comparison")
    col_c1, col_c2 = st.columns(2)

    cost_img = "results/cost_comparison.png"
    co2_img = "results/co2_comparison.png"
    if os.path.exists(cost_img):
        col_c1.image(cost_img, caption="Total Operating Cost (₹) by Strategy", use_container_width=True)
    if os.path.exists(co2_img):
        col_c2.image(co2_img, caption="Total CO₂ Emissions (kg) by Strategy", use_container_width=True)

    st.markdown("---")
    st.subheader("🏋️ Training Reward Curves (SB3 Monitor)")
    t1, t2 = st.columns(2)

    dqn_curve = "results/dqn_reward_curve.png"
    ppo_curve = "results/ppo_reward_curve.png"
    if os.path.exists(dqn_curve):
        t1.image(dqn_curve, caption="DQN Episode Reward Learning Curve", use_container_width=True)
    if os.path.exists(ppo_curve):
        t2.image(ppo_curve, caption="PPO Episode Reward Learning Curve", use_container_width=True)

    comb_curve = "results/training_reward_curves.png"
    if os.path.exists(comb_curve):
        st.image(comb_curve, caption="Combined Training Convergence (DQN vs PPO)", use_container_width=True)


# ==================== TAB 3: ABOUT ====================
with tab_about:
    st.subheader("ℹ️ System Architecture & Optimization Flow")

    st.markdown(
        r"""
        ```
        🌞 Generate (Solar & Wind) ───► 🏫 Demand (Campus Load) ───► 🔋 Store (Battery ESS) ───► 🤖 AI Decides ───► 💰/🌱 Optimize Cost & Carbon
        ```

        ---

        ### 🧠 Reinforcement Learning Formulation

        #### 1. Observation Space (State)
        At every hour $t$, the agent receives a normalized vector containing:
        - **Solar Generation** ($kW$ normalized to `SOLAR_CAPACITY_KW`)
        - **Wind Generation** ($kW$ normalized to `WIND_CAPACITY_KW`)
        - **Campus Demand** ($kW$ normalized to 100 kW)
        - **Battery Level** (State of Charge fraction $0.0 - 1.0$)
        - **Grid Tariff** (Time-of-Use price normalized to `PEAK_PRICE`)
        - **Cyclical Time** ($\sin(2\pi h / 24)$ and $\cos(2\pi h / 24)$)
        - **Day of Week** ($0 - 6$ normalized)
        - **Peak Window Indicator** ($1.0$ if 6 PM – 10 PM, else $0.0$)
        - *(In Forecast Variant)* **3h & 6h Future Solar Predictions** from PyTorch LSTM

        #### 2. Action Space (Discrete Choices)
        - **0: CHARGE** — Directs clean renewable surplus into the battery; imports from the grid if necessary.
        - **1: DISCHARGE** — Discharges stored clean energy to satisfy campus demand and avoid peak grid tariffs.
        - **2: BUY** — Satisfies deficits by importing directly from the grid while conserving battery reserves.
        - **3: SELL** — Exports surplus clean energy to the grid to earn feed-in tariff revenue.

        #### 3. Reward Objective
        $$\text{Reward} = -(\text{Cost Weight} \times \text{INR} + \text{CO}_2\text{ Weight} \times \text{kg CO}_2 + \text{Unmet Weight} \times \text{Unmet kWh}) - \text{Penalty}$$
        """
    )
