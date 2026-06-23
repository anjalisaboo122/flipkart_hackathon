import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "police_violations.parquet")

H3_RESOLUTION      = 8      # ~460m hexagons
MIN_VIOLATIONS     = 10     # minimum for a zone to qualify
TOP_ZONES_PROPHET  = 50     # train Prophet on top N zones by volume
FORECAST_DAYS      = 7      # days ahead to predict
PATROL_ZONES       = 10     # zones per patrol route

PEAK_HOURS = list(range(7, 11)) + list(range(17, 21))

VIOLATION_SEVERITY = {
    "DOUBLE PARKING":                              2.0,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE":  1.9,
    "PARKING IN A MAIN ROAD":                      1.7,
    "PARKING NEAR ROAD CROSSING":                  1.6,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":     1.4,
    "NO PARKING":                                  1.2,
    "WRONG PARKING":                               1.0,
}

VEHICLE_WEIGHT = {
    "TANKER":         2.5,
    "BUS":            2.5,
    "LORRY":          2.2,
    "TRUCK":          2.2,
    "VAN":            1.6,
    "MAXI-CAB":       1.5,
    "CAR":            1.2,
    "PASSENGER AUTO": 1.0,
    "GOODS AUTO":     1.0,
    "MOTOR CYCLE":    0.7,
    "SCOOTER":        0.7,
}

# ---------------------------------------------------------------------------
# Traffic Disruption Model parameters
# ---------------------------------------------------------------------------

# Fraction of a lane physically blocked by each vehicle type
# Source: IRC:106-1990 vehicle dimensions + HCM lane-blockage equivalents
LANE_BLOCKAGE = {
    "TANKER":         1.0,
    "BUS":            1.0,
    "LORRY":          1.0,
    "TRUCK":          1.0,
    "VAN":            0.8,
    "MAXI-CAB":       0.7,
    "CAR":            0.6,
    "PASSENGER AUTO": 0.4,
    "GOODS AUTO":     0.4,
    "MOTOR CYCLE":    0.2,
    "SCOOTER":        0.2,
}

# Peak hour multiplier — violations during rush hour affect far more vehicles
PEAK_MULTIPLIER    = 2.5

# Road type factor — junction zones funnel merging traffic, amplifying delays
ROAD_FACTOR_JUNCTION    = 2.5
ROAD_FACTOR_MAIN_ROAD   = 2.0
ROAD_FACTOR_OTHER       = 1.0

# Bengaluru arterial traffic volume (vehicles/hour, conservative estimate)
# Source: BBMP Traffic Engineering studies (~2,000 veh/hr on major roads)
TRAFFIC_VOLUME_PER_HOUR = 2000

# MoRTH value of travel time for Indian cities (₹/vehicle-hour)
VALUE_OF_TIME_INR = 75

# ---------------------------------------------------------------------------
# API keys (set in .env)
MAPPLS_TOKEN      = os.getenv("MAPPLS_TOKEN", "")
TOMTOM_API_KEY    = os.getenv("TOMTOM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
