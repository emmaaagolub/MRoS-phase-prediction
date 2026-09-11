"""Settings and helper functions shared by the model and evaluation scripts.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: F401,E402  (REGIONS is re-exported)
    OUTPUT_DIR, REGIONS, interpolation_dir,
)

# Interpolation product used as the source of the gridded predictors.
INTERP_TYPE = "kriging"

# Phase codes as they appear in the observation tables.
SNOW_CODE, RAIN_CODE, MIX_CODE = 0, 1, 2
FULL_CLASS_NAMES = ["snow", "rain", "mix"]

# The model is fitted on pure phases only, as a rain-vs-snow problem.
BINARY_LABEL_MAP = {RAIN_CODE: 0, SNOW_CODE: 1}
BINARY_LABEL_NAMES = ["rain", "snow"]
PURE_PHASE_CODES = [SNOW_CODE, RAIN_CODE]
TARGET_FULL = "phase_full"

RANDOM_SEED = 42
TRAIN_FRAC = 0.70  # remainder is split evenly between validation and test

# Which predictors to use.
USE_TEMP_AIR = True
USE_TEMP_DEW = True
USE_TEMP_WET = True
USE_RH = False
USE_IMERG_PLP = True
USE_ELEV = True
USE_MROS_LOOCV = True

# Predictors read off the gridded cube at each observation point.
GRID_FEATURES = [
    name for name, use in [
        ("temp_air", USE_TEMP_AIR),
        ("temp_dew", USE_TEMP_DEW),
        ("temp_wet", USE_TEMP_WET),
        ("rh", USE_RH),
        ("imerg_plp", USE_IMERG_PLP),
        ("elev", USE_ELEV),
    ] if use
]

# Leave-one-out MRoS indicators come from the observation table, not the grid.
MROS_LOOCV_FEATURES = ["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]
FEATURES = GRID_FEATURES + (MROS_LOOCV_FEATURES if USE_MROS_LOOCV else [])

# Training.
BASE_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,
    "tree_method": "hist",
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

# Candidate values for the scale_pos_weight sweep.
SCALE_POS_WEIGHT_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0,
                         1.25, 1.5, 2.0]

# Wet-bulb threshold separating the near-freezing and clear-phase regimes.
# Probabilities are calibrated separately on each side. Value from
# Sims & Liu (2015).
CLEAR_PHASE_TWET_C = 2.0

# Mix is assigned by a band around p(snow) = 0.5 whose half-width varies with
# wet-bulb temperature:
#     half_band(Twet) = base + extra * exp(-Twet^2 / (2 * sigma^2))
# Predictions inside the band are labelled mix. The grids below are searched
# for the parameters maximising the objective in optimize_threshold_params.
BASE_HALF_BAND_GRID = (0.05, 0.10, 0.15, 0.20)
EXTRA_HALF_BAND_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
SIGMA_GRID = (1.0, 1.5, 2.0)  # degC, range from Sims & Liu (2015)
MAX_TOTAL_HALF_BAND = 0.40
MIN_PURE_COVERAGE = 0.50
MIN_NF_PURE_COVERAGE = 0.35
MIX_CAPTURE_WEIGHT = 0.5


def model_paths(region_id, interp_type=INTERP_TYPE):
    """Input and output locations for one region and interpolation product."""
    if interp_type not in ("IDW", "kriging"):
        raise ValueError(f"Unknown interpolation type: {interp_type}")

    interp_dir = interpolation_dir(region_id, interp_type)
    stem = "IDW" if interp_type == "IDW" else "indicator_kriging"
    model_dir = OUTPUT_DIR / "model" / region_id

    return {
        "interp_grid": interp_dir / f"hourly_predictors_1km_{stem}.nc",
        "mros_loocv": interp_dir / f"mros_loocv_point_predictions_{interp_type}.parquet",
        "imerg": OUTPUT_DIR / "resampled_grids" / region_id / "imerg_hourly_1km.nc",
        "model_dir": model_dir,
        "graphics_dir": model_dir / "graphics",
        "master_table": model_dir / "ml_input_points.parquet",
        # The evaluation experiments reuse this table so their train/val/test
        # split is identical to the model's.
        "split_table": model_dir / "ml_input_points_split.parquet",
    }


def make_output_dirs(paths):
    for key in ("model_dir", "graphics_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Uncertainty band
# ---------------------------------------------------------------------------

def gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma):
    """Half-width of the uncertainty band at a given wet-bulb temperature."""
    t = np.asarray(temp_wet, dtype=float)
    return np.clip(base_half_band + extra_half_band * np.exp(-(t ** 2) / (2.0 * sigma ** 2)),
                   0.0, 0.5)


def classify_phase_gaussian_band(p_snow, temp_wet, base_half_band, extra_half_band, sigma):
    """Turn calibrated p(snow) into snow / rain / mix using the band."""
    p_snow = np.asarray(p_snow, dtype=float)
    half_band = gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma)
    pred = np.full(len(p_snow), MIX_CODE, dtype=int)
    pred[p_snow <= 0.5 - half_band] = RAIN_CODE
    pred[p_snow >= 0.5 + half_band] = SNOW_CODE
    return pred


def transition_score_from_psnow(p_snow):
    """Distance of p(snow) from 0 or 1: 1 at p=0.5, 0 at p=0 or p=1."""
    return 1.0 - np.abs(2.0 * np.asarray(p_snow) - 1.0)
