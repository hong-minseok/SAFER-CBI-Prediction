"""Public, result-free contracts shared by all calculation stages."""

from __future__ import annotations

HORIZONS = ("1hr", "3hr", "6hr", "12hr", "18hr", "24hr")
HORIZON_HOURS = tuple(int(horizon.removesuffix("hr")) for horizon in HORIZONS)
LABEL_COLUMNS = tuple(f"y_{horizon}_true" for horizon in HORIZONS)

PRIMARY_HORIZON = "24hr"
BENCHMARK_HORIZON = "6hr"

MODEL_ORDER = ("multitask", "gru", "single", "lgbm")
MODEL_HORIZONS = {
    "multitask": HORIZONS,
    "gru": HORIZONS,
    "single": (PRIMARY_HORIZON,),
    "lgbm": HORIZONS,
}

FEATURE_COLUMNS = (
    "Age", "Sex", "Day_of_week", "nonwearing",
    "ENMO_min", "ENMO_std", "ENMO_max", "ENMO_mean", "ENMO_median",
    "ENMO_nunique", "HR_min", "HR_std", "HR_max", "HR_mean", "HR_median",
    "HR_nunique", "Entropy_Day_Norm", "Entropy_Day",
    "Location_Variability", "Distance_Traveled", "Place_other",
    "Place_hallway", "ENMO_Cos_MESOR", "ENMO_Cos_Amplitude",
    "ENMO_Cos_Phase", "HR_Cos_MESOR", "HR_Cos_Amplitude", "HR_Cos_Phase",
    "STEP_delta",
)

CATEGORY_MAP = {
    "Basics": ("Age", "Sex", "Day_of_week", "nonwearing"),
    "ENMO": (
        "ENMO_min", "ENMO_std", "ENMO_max", "ENMO_mean", "ENMO_median",
        "ENMO_nunique",
    ),
    "Circadian": (
        "ENMO_Cos_MESOR", "ENMO_Cos_Amplitude", "ENMO_Cos_Phase",
        "HR_Cos_MESOR", "HR_Cos_Amplitude", "HR_Cos_Phase",
    ),
    "Heartrate": (
        "HR_min", "HR_std", "HR_max", "HR_mean", "HR_median", "HR_nunique",
    ),
    "Location": (
        "Entropy_Day_Norm", "Location_Variability", "Entropy_Day",
        "Place_other", "Place_hallway", "Distance_Traveled",
    ),
    "Step": ("STEP_delta",),
}

OUTPUT_COLUMNS = (
    "time", "key",
    "ENMO_mean", "ENMO_std", "ENMO_min", "ENMO_max", "ENMO_median",
    "ENMO_nunique", "HR_mean", "HR_std", "HR_min", "HR_max", "HR_median",
    "HR_nunique", "HEARTBEAT_count", "BATTERY_mean", "institution",
    "STEP_delta", "CALORIES_delta", "DISTANCE_delta", "SLEEP_delta",
    "nonwearing", "Latitude", "Longitude", "cluster_label", "Entropy_Day",
    "Entropy_Day_Norm", "Distance_Traveled", "Location_Variability",
    "place_room", "Place_hallway", "Place_other", "ENMO_Cos_MESOR",
    "ENMO_Cos_Amplitude", "ENMO_Cos_Phase", "ENMO_Phase_unwrapped",
    "HR_Cos_MESOR", "HR_Cos_Amplitude", "HR_Cos_Phase",
    "HEARTBEAT_Phase_unwrapped", "Age", "Sex", *LABEL_COLUMNS,
    "is_zero_padded", "event_index", "is_case", "window_length",
    "is_imputed_location", "Day_of_week",
)

if len(FEATURE_COLUMNS) != 29 or len(set(FEATURE_COLUMNS)) != 29:
    raise RuntimeError("The public feature contract must contain 29 unique columns")
if len(OUTPUT_COLUMNS) != 54 or len(set(OUTPUT_COLUMNS)) != 54:
    raise RuntimeError("The public output contract must contain 54 unique columns")
if set(FEATURE_COLUMNS) != {
    feature for features in CATEGORY_MAP.values() for feature in features
}:
    raise RuntimeError("Feature order and category contracts disagree")
if not set(FEATURE_COLUMNS).issubset(OUTPUT_COLUMNS):
    raise RuntimeError("Every model feature must be present in the output schema")
