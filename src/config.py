# Battery
BATTERY_CAPACITY_KWH = 100
MAX_RATE_KW = 25            # max charge or discharge per hour
BATTERY_EFFICIENCY = 0.95   # applied on charging and on discharging
INITIAL_BATTERY_LEVEL = 0.5 # fraction 0-1

# Grid prices and emissions
PEAK_PRICE = 10             # ₹/kWh
OFFPEAK_PRICE = 5           # ₹/kWh
SELL_PRICE = 3              # ₹/kWh
PEAK_HOURS = (18, 22)       # 6PM to 10PM
GRID_CO2_KG_PER_KWH = 0.8

# Generation capacity
SOLAR_CAPACITY_KW = 100
WIND_CAPACITY_KW = 20

# Data
DATA_PATH = "data/grid_data.csv"
TRAIN_SPLIT_DAY = 270       # days 0-269 for training, rest for evaluation
RANDOM_SEED = 42

# RL environment
EPISODE_DAYS = 7
COST_WEIGHT = 1.0
CO2_WEIGHT = 0.5
UNMET_WEIGHT = 100.0
INVALID_ACTION_PENALTY = 5.0

# Evaluation
EVAL_DAYS = 30

