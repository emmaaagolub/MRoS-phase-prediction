"""
ML_XGBoost_binary_uncertainty — Benchmarking vs. Jennings et al. (2025)
=======================================================================

Compares the XGBoost binary-uncertainty model (baseline_full feature set)
against the benchmark precipitation phase partitioning methods (PPMs) used in:

  Jennings, K. S. et al. (2025). Machine learning shows a limit to rain-snow
  partitioning accuracy when using near-surface meteorology.
  Nature Communications 16. https://doi.org/10.1038/s41467-025-58234-2

Benchmark methods (their Table 1; thresholds from Jennings et al. 2018,
Nat. Commun. 9:1148):
  - Static air temperature thresholds:   T_a 1.0 °C, T_a 1.5 °C
  - Dew point temperature thresholds:    T_d 0.0 °C, T_d 0.5 °C
  - Wet bulb temperature thresholds:     T_w 0.0 °C, T_w 0.5 °C, T_w 1.0 °C
  - Binary logistic regression (Jennings et al. 2018, Northern Hemisphere):
        p(snow) = 1 / (1 + exp(-10.04 + 1.41*T_a + 0.09*RH))
    with T_a in deg C and RH in %; snow if p(snow) >= 0.5
  - (optional) Same logistic form re-fitted to this domain's training split

All benchmarks are deterministic given the gridded predictor fields, so they
are evaluated directly on the SAME held-out test split used by the ablation
study (identical cache + identical RANDOM_SEED -> identical split).  The
XGBoost baseline is retrained here with the same configuration as the
ablation `baseline_full` run for a self-contained, reproducible comparison.

Following the paper, the headline benchmark comparison is binary
(rain vs. snow, observer-reported mixed excluded), scored with accuracy,
snow bias, and rain bias — overall and by air temperature bin (their Fig. 1)
plus relative accuracy improvement of the ML model vs. the best and average
benchmarks (their Fig. 2).  Mixed-phase performance is reported separately
(their Fig. 3 / mixed section): benchmarks cannot predict mixed, while the
model's Gaussian half-band abstention provides a mix class.

Outputs are written to:
  DATA_DIR / ML_pipeline / model_artifacts / {REGION} / benchmarking_v1/

Per-observation test predictions for every method are saved to
`benchmark_predictions_test.parquet` — this is the input for later
resampling / bootstrap confidence interval analysis.

Usage
-----
  python ML_XGBoost_binary_uncertainty_benchmarking.py
  python ML_XGBoost_binary_uncertainty_benchmarking.py --dry-run
  python ML_XGBoost_binary_uncertainty_benchmarking.py --replot
"""
# conda activate "C:\Users\EmmaGolub\Desktop\MRoS_local\venv"

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from betacal import BetaCalibration
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# =============================================================================
# 1.  BENCHMARK CONFIG
# =============================================================================

INTERP_TYPE = "kriging"   # "IDW" or "kriging"
REGION      = "CO"

# Threshold benchmarks: (name, predictor column, threshold deg C).
# Decision rule (Jennings et al. 2018/2025): snow if T <= threshold else rain.
THRESHOLD_BENCHMARKS: list[tuple[str, str, float]] = [
    ("ta_1.0", "temp_air", 1.0),
    ("ta_1.5", "temp_air", 1.5),
    ("td_0.0", "temp_dew", 0.0),
    ("td_0.5", "temp_dew", 0.5),
    ("tw_0.0", "temp_wet", 0.0),
    ("tw_0.5", "temp_wet", 0.5),
    ("tw_1.0", "temp_wet", 1.0),
]

# Jennings et al. (2018) Northern Hemisphere binary logistic regression
# coefficients: p(snow) = 1 / (1 + exp(a + b*T_air + g*RH))
BINLOG_A     = -10.04
BINLOG_B     = 1.41
BINLOG_G     = 0.09
# Also fit the same logistic form (T_air + RH) on this domain's train split.
INCLUDE_FITTED_BINLOG = True

# XGBoost model config — must match the ablation `baseline_full` run.
MODEL_NAME     = "xgboost_mros"
MODEL_FEATURES = ["temp_air", "temp_dew", "temp_wet", "imerg_plp", "elev",
                  "mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]

# Air-temperature binning for the Fig. 1 / Fig. 2 style plots.
TAIR_BIN_WIDTH   = 1.0     # deg C (paper used 0.5; 1.0 is safer at our n)
TAIR_BIN_MIN     = -8.0
TAIR_BIN_MAX     = 8.0
MIN_BIN_N        = 20      # min pure obs per bin for accuracy
MIN_BIN_PHASE_N  = 10      # min obs of a phase per bin for its bias

# Near-freezing regime definition (consistent with the ablation study).
NEARFREEZE_TWET_C = 2.0

# =============================================================================
# 2.  SHARED SETUP  (identical to the ablation script — do not change,
#     or the train/val/test split will no longer match)
# =============================================================================

BASE_DIR = Path().resolve().parent
DATA_DIR = Path(r"C:\Users\EmmaGolub\Desktop\MRoS_local\mros-precipitation-phase-product-prototype\outputs")

PATHS = {
    "IDW": {
        "interp_grid": DATA_DIR / f"interpolated/{REGION}/IDW_refactored/hourly_predictors_1km_IDW.nc",
        "mros_loocv":  DATA_DIR / f"interpolated/{REGION}/IDW_refactored/mros_loocv_point_predictions_IDW.parquet",
    },
    "kriging": {
        "interp_grid": DATA_DIR / f"interpolated/{REGION}/indicator_kriging_refactored/hourly_predictors_1km_indicator_kriging.nc",
        "mros_loocv":  DATA_DIR / f"interpolated/{REGION}/indicator_kriging_refactored/mros_loocv_point_predictions_kriging.parquet",
    },
    "imerg": DATA_DIR / f"resampled_grids/{REGION}/imerg_hourly_1km.nc",
}

interp_type_folder = "results_binaryXGB_withIDW_v2" if INTERP_TYPE == "IDW" else "results_binaryXGB_withKriging_v2"
SETUP_DIR  = DATA_DIR / f"ML_pipeline/compiled_input_predictors/{REGION}" / interp_type_folder
BENCH_ROOT = DATA_DIR / f"ML_pipeline/model_artifacts/{REGION}/benchmarking_v1"
SETUP_DIR.mkdir(parents=True, exist_ok=True)
BENCH_ROOT.mkdir(parents=True, exist_ok=True)

# Training (identical to ablation)
TRAIN_FRAC            = 0.70
RANDOM_SEED           = 42
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND       = 2000

# Phase codes
SNOW_CODE        = 0
RAIN_CODE        = 1
MIX_CODE         = 2
BINARY_LABEL_MAP = {RAIN_CODE: 0, SNOW_CODE: 1}

SCALE_POS_WEIGHT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0]

CLEAR_PHASE_TWET_C = 2.0

BASE_HALF_BAND_GRID  = (0.05, 0.10, 0.15, 0.20)
EXTRA_HALF_BAND_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
MAX_TOTAL_HALF_BAND  = 0.40
SIGMA_GRID           = (1.0, 1.5, 2.0)
MIN_PURE_COVERAGE    = 0.50
MIN_NF_PURE_COVERAGE = 0.35
MIX_CAPTURE_WEIGHT   = 0.5

BASE_XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "max_depth": 6, "eta": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 5,
    "lambda": 1.0, "tree_method": "hist",
}

ALL_GRID_FEATURES  = ["temp_air", "temp_dew", "temp_wet", "rh", "imerg_plp", "elev"]
ALL_LOOCV_FEATURES = ["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]

# Plot styling: one colour family per benchmark family (paper Fig. 1 style)
METHOD_STYLE = {
    "ta_1.0":            dict(color="#e08214", ls="--", lw=1.4),
    "ta_1.5":            dict(color="#b35806", ls="-",  lw=1.4),
    "td_0.0":            dict(color="#7fbc41", ls="--", lw=1.4),
    "td_0.5":            dict(color="#4d9221", ls="-",  lw=1.4),
    "tw_0.0":            dict(color="#92c5de", ls=":",  lw=1.6),
    "tw_0.5":            dict(color="#4393c3", ls="--", lw=1.6),
    "tw_1.0":            dict(color="#2166ac", ls="-",  lw=1.6),
    "binlog_jennings18": dict(color="#9970ab", ls="-",  lw=1.6),
    "binlog_fitted":     dict(color="#762a83", ls="--", lw=1.6),
    MODEL_NAME:          dict(color="black",   ls="-",  lw=2.6),
}
METHOD_LABEL = {
    "ta_1.0": "$T_{a}$ 1.0 °C", "ta_1.5": "$T_{a}$ 1.5 °C",
    "td_0.0": "$T_{d}$ 0.0 °C", "td_0.5": "$T_{d}$ 0.5 °C",
    "tw_0.0": "$T_{w}$ 0.0 °C", "tw_0.5": "$T_{w}$ 0.5 °C", "tw_1.0": "$T_{w}$ 1.0 °C",
    "binlog_jennings18": "Bin. logistic (Jennings et al. 2018)",
    "binlog_fitted":     "Bin. logistic (fitted, this domain)",
    MODEL_NAME:          "XGBoost + MRoS (this study)",
}

# =============================================================================
# 3.  HELPERS  (verbatim from the ablation script where shared)
# =============================================================================

PHASE_NAME_TO_CODE = {"snow": 0, "rain": 1, "mix": 2, "mixed": 2, "mixed_phase": 2}
PHASE_CODE_TO_NAME = {0: "snow", 1: "rain", 2: "mix"}


def candidate_column(df, names, label):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Could not find {label}. Tried: {list(names)}")


def optional_column(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_phase_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        key = v.strip().lower()
        if key in PHASE_NAME_TO_CODE:
            return PHASE_NAME_TO_CODE[key]
        try:
            iv = int(float(key))
            if iv in PHASE_CODE_TO_NAME:
                return iv
        except Exception:
            return None
        return None
    try:
        iv = int(v)
        if iv in PHASE_CODE_TO_NAME:
            return iv
    except Exception:
        return None
    return None


def get_dataset_crs(ds, fallback="EPSG:26911"):
    try:
        if hasattr(ds, "rio") and ds.rio.crs is not None:
            return ds.rio.crs
    except Exception:
        pass
    for key in ["crs", "spatial_ref"]:
        if key in ds.attrs and ds.attrs[key]:
            return ds.attrs[key]
    return fallback


def project_lonlat_to_grid(lon, lat, grid_crs):
    transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, float), np.asarray(y, float)


def prep_loocv_table(df, ds_interp):
    df = df.copy()
    time_col = candidate_column(df, ["time","hour_utc","datetime","timestamp"], "time")
    x_col    = optional_column(df, ["x","x_proj","grid_x"])
    y_col    = optional_column(df, ["y","y_proj","grid_y"])
    lon_col  = optional_column(df, ["lon","longitude","x_lon"])
    lat_col  = optional_column(df, ["lat","latitude","y_lat"])
    if (x_col is None or y_col is None) and (lon_col is None or lat_col is None):
        raise KeyError("Could not find projected x/y or lon/lat in LOOCV parquet.")
    obs_col  = candidate_column(df, ["observed_phase","phase_obs","raw_phase","phase","obs_phase","phase_label"], "observed phase")
    snow_col = candidate_column(df, ["mros_p_snow_loocv","p_snow_loocv","p_snow_cv","p_snow"], "LOOCV snow prob")
    mix_col  = candidate_column(df, ["mros_p_mix_loocv","p_mix_loocv","p_mix_cv","p_mix"],   "LOOCV mix prob")
    rain_col = candidate_column(df, ["mros_p_rain_loocv","p_rain_loocv","p_rain_cv","p_rain"],"LOOCV rain prob")

    out = pd.DataFrame({
        "time":              pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None),
        "phase_full":        df[obs_col].map(normalize_phase_value),
        "mros_p_snow_loocv": pd.to_numeric(df[snow_col], errors="coerce"),
        "mros_p_mix_loocv":  pd.to_numeric(df[mix_col],  errors="coerce"),
        "mros_p_rain_loocv": pd.to_numeric(df[rain_col], errors="coerce"),
    })
    if lon_col: out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    if lat_col: out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    if x_col and y_col:
        out["x"] = pd.to_numeric(df[x_col], errors="coerce")
        out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    else:
        crs = get_dataset_crs(ds_interp, fallback="EPSG:26911")
        xp, yp = project_lonlat_to_grid(out["lon"].to_numpy(float), out["lat"].to_numpy(float), crs)
        out["x"] = xp; out["y"] = yp

    for c in ["hour_utc","elev","pred_phase","obs_p_snow","obs_p_mix","obs_p_rain",
              "pred_max_prob","pred_entropy","pred_correct","station_id","obs_id","source"]:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    if "elev" in df.columns and "obs_elev" not in out.columns:
        out["obs_elev"] = pd.to_numeric(df["elev"], errors="coerce")

    out["phase_full"] = out["phase_full"].astype("Int64")
    probs = out[["mros_p_snow_loocv","mros_p_mix_loocv","mros_p_rain_loocv"]].to_numpy(float)
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1)
    valid = row_sum > 0
    probs[valid]  = probs[valid] / row_sum[valid, None]
    probs[~valid] = np.nan
    out[["mros_p_snow_loocv","mros_p_mix_loocv","mros_p_rain_loocv"]] = probs
    out = out.dropna(subset=["time","x","y","phase_full",
                              "mros_p_snow_loocv","mros_p_mix_loocv","mros_p_rain_loocv"]).copy()
    out["phase_full"] = out["phase_full"].astype(int)
    return out


def build_predictor_cube(ds_interp, ds_imerg,
                          use_temp_air=True, use_temp_dew=True, use_temp_wet=True,
                          use_rh=False, use_elev=True, use_imerg_plp=True):
    interp_keep = [n for flag, n in [(use_temp_air,"temp_air"),(use_temp_dew,"temp_dew"),
                                      (use_temp_wet,"temp_wet"),(use_rh,"rh"),(use_elev,"elev")]
                   if flag and n in ds_interp]
    imerg_keep  = ["imerg_plp"] if use_imerg_plp and "imerg_plp" in ds_imerg else []
    parts = ([ds_interp[interp_keep]] if interp_keep else []) + \
            ([ds_imerg[imerg_keep]]   if imerg_keep  else [])
    if not parts:
        raise ValueError("No predictor variables selected.")
    return xr.merge(parts, compat="override", join="exact")


def nearest_index_1d(coord_vals, query_vals):
    coord_vals = np.asarray(coord_vals); query_vals = np.asarray(query_vals)
    ascending  = coord_vals[0] < coord_vals[-1]
    cw  = coord_vals if ascending else coord_vals[::-1]
    idx = np.clip(np.searchsorted(cw, query_vals), 1, len(cw) - 1)
    out = np.where(np.abs(query_vals - cw[idx]) < np.abs(query_vals - cw[idx-1]), idx, idx-1)
    return ((len(coord_vals) - 1) - out if not ascending else out).astype(np.int64)


def sample_predictor_cube_to_points_batched(points_df, ds_pred, predictor_vars, verbose=True):
    pts = points_df.copy().reset_index(drop=True)
    pts["time"] = pd.to_datetime(pts["time"]).dt.floor("h")
    predictor_vars = [v for v in predictor_vars if v in ds_pred.data_vars]
    if not predictor_vars:
        raise ValueError("No valid predictor_vars found in ds_pred.")
    x_min, x_max = float(ds_pred["x"].min()), float(ds_pred["x"].max())
    y_min, y_max = float(ds_pred["y"].min()), float(ds_pred["y"].max())
    in_domain = (pts["x"].between(min(x_min,x_max), max(x_min,x_max)) &
                 pts["y"].between(min(y_min,y_max), max(y_min,y_max)))
    if (~in_domain).any():
        pts = pts.loc[in_domain].copy().reset_index(drop=True)
    tvals = pd.to_datetime(ds_pred["time"].values)
    xvals = ds_pred["x"].values; yvals = ds_pred["y"].values
    t_idx = nearest_index_1d(tvals.astype("datetime64[ns]").astype("int64"),
                              pd.to_datetime(pts["time"]).to_numpy().astype("datetime64[ns]").astype("int64"))
    pts["_t_idx"] = t_idx; pts["_orig_row"] = np.arange(len(pts))
    chunks = []
    for tidx in np.sort(pts["_t_idx"].unique()):
        chunk   = pts.loc[pts["_t_idx"] == tidx].copy()
        x_idx   = nearest_index_1d(xvals, chunk["x"].to_numpy(float))
        y_idx   = nearest_index_1d(yvals, chunk["y"].to_numpy(float))
        ds_hour = ds_pred.isel(time=int(tidx))
        sampled = pd.DataFrame({v: ds_hour[v].values[y_idx, x_idx] for v in predictor_vars},
                               index=chunk.index)
        chunks.append(pd.concat([chunk, sampled], axis=1))
    return pd.concat(chunks).sort_values("_orig_row").reset_index(drop=True).drop(columns=["_t_idx","_orig_row"])


def gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma):
    t = np.asarray(temp_wet, float)
    return np.clip(base_half_band + extra_half_band * np.exp(-(t**2) / (2.0 * sigma**2)), 0.0, 0.5)


def classify_phase_gaussian_band(p_snow, temp_wet, base_half_band, extra_half_band, sigma):
    p_snow = np.asarray(p_snow, float)
    hb     = gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma)
    pred   = np.full(len(p_snow), MIX_CODE, dtype=int)
    pred[p_snow <= 0.5 - hb] = RAIN_CODE
    pred[p_snow >= 0.5 + hb] = SNOW_CODE
    return pred


def optimize_threshold_params(p_snow_cal, temp_wet, y_true_phase):
    p_snow = np.asarray(p_snow_cal, float); t_wet = np.asarray(temp_wet, float)
    y_true = np.asarray(y_true_phase, int)
    true_mix  = y_true == MIX_CODE
    true_pure = np.isin(y_true, [SNOW_CODE, RAIN_CODE])
    records = []
    for base_hb, extra_hb, sigma in itertools.product(BASE_HALF_BAND_GRID, EXTRA_HALF_BAND_GRID, SIGMA_GRID):
        if base_hb + extra_hb >= MAX_TOTAL_HALF_BAND:
            continue
        pred      = classify_phase_gaussian_band(p_snow, t_wet, base_hb, extra_hb, sigma)
        pred_conf = pred != MIX_CODE
        mix_cap   = float(np.mean(pred[true_mix]  == MIX_CODE))  if true_mix.any()  else np.nan
        pure_cov  = float(np.mean(pred_conf[true_pure]))          if true_pure.any() else np.nan
        mask_cp   = true_pure & pred_conf
        pure_acc  = float(np.mean(pred[mask_cp] == y_true[mask_cp])) if mask_cp.any() else np.nan
        composite = (MIX_CAPTURE_WEIGHT * mix_cap + (1 - MIX_CAPTURE_WEIGHT) * pure_acc
                     if not (np.isnan(mix_cap) or np.isnan(pure_acc)) else np.nan)
        nf_pure   = true_pure & (np.abs(t_wet) <= 1.0)
        nf_cov    = float(np.mean(pred_conf[nf_pure])) if nf_pure.any() else np.nan
        feasible  = (not np.isnan(pure_cov)) and (pure_cov >= MIN_PURE_COVERAGE) and \
                    (not np.isnan(nf_cov))   and (nf_cov   >= MIN_NF_PURE_COVERAGE)
        records.append(dict(base_half_band=base_hb, extra_half_band=extra_hb, sigma=sigma,
                            mix_capture_rate=mix_cap, pure_coverage=pure_cov,
                            pure_conf_accuracy=pure_acc, composite_score=composite, feasible=feasible))
    results_df  = pd.DataFrame(records).sort_values(["feasible","composite_score"], ascending=[False,False]).reset_index(drop=True)
    feasible_df = results_df[results_df["feasible"]]
    best = feasible_df.iloc[0] if not feasible_df.empty else results_df.iloc[0]
    return {k: float(best[k]) for k in ["base_half_band","extra_half_band","sigma",
                                         "composite_score","mix_capture_rate",
                                         "pure_conf_accuracy","pure_coverage"]}


def sweep_scale_pos_weight(X_tr, y_tr, X_vl, y_vl, candidates, base_params,
                            probe_rounds=350, early_stopping=30, random_seed=42):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dval   = xgb.DMatrix(X_vl, label=y_vl, feature_names=list(X_vl.columns))
    records = []
    for spw in candidates:
        p     = {**base_params, "scale_pos_weight": spw, "seed": random_seed}
        probe = xgb.train(params=p, dtrain=dtrain, num_boost_round=probe_rounds,
                          evals=[(dval,"val")], early_stopping_rounds=early_stopping,
                          verbose_eval=False)
        y_pred   = (probe.predict(dval) >= 0.5).astype(int)
        snow_rec = recall_score(y_vl, y_pred, pos_label=1, zero_division=0)
        rain_rec = recall_score(y_vl, y_pred, pos_label=0, zero_division=0)
        records.append(dict(scale_pos_weight=spw,
                            snow_recall=round(snow_rec,4), rain_recall=round(rain_rec,4),
                            recall_gap=round(abs(snow_rec-rain_rec),4),
                            balanced_accuracy=round(balanced_accuracy_score(y_vl,y_pred),4),
                            macro_f1=round(f1_score(y_vl,y_pred,average="macro",zero_division=0),4)))
    df = pd.DataFrame(records).sort_values("recall_gap").reset_index(drop=True)
    return float(df.iloc[0]["scale_pos_weight"]), df


# =============================================================================
# 4.  ONE-TIME DATA LOAD  (verbatim)
# =============================================================================

def load_and_sync_datasets():
    print("\n" + "="*60)
    print("ONE-TIME: Opening and time-syncing datasets")
    print("="*60)
    ds_interp    = xr.open_dataset(PATHS[INTERP_TYPE]["interp_grid"])
    ds_imerg     = xr.open_dataset(PATHS["imerg"])
    df_loocv_raw = pd.read_parquet(PATHS[INTERP_TYPE]["mros_loocv"])

    ds_interp = ds_interp.assign_coords(time=pd.to_datetime(ds_interp.time.values).floor("h"))
    ds_imerg  = ds_imerg.assign_coords(time=pd.to_datetime(ds_imerg.time.values).floor("h"))

    common_times = np.intersect1d(ds_interp.time.values, ds_imerg.time.values)
    print(f"Common timesteps: {len(common_times)}")

    ds_interp = ds_interp.sel(time=common_times)
    ds_imerg  = ds_imerg.sel(time=common_times)
    if ds_imerg.y.values[0] > ds_imerg.y.values[-1]:
        ds_imerg = ds_imerg.isel(y=slice(None, None, -1))

    return ds_interp, ds_imerg, df_loocv_raw, common_times


# =============================================================================
# 5.  ONE-TIME SAMPLING  (verbatim — shares the ablation cache)
# =============================================================================

def get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times) -> pd.DataFrame:
    cache_path = SETUP_DIR / f"ml_input_points_{INTERP_TYPE}_binary_uncertainty_random_split.parquet"

    if cache_path.exists():
        print(f"\nONE-TIME SAMPLING: Cache found — loading\n  {cache_path}")
        master_df = pd.read_parquet(cache_path)
        print(f"  Shape: {master_df.shape}")
        return master_df

    print("\nONE-TIME SAMPLING: No cache found — sampling predictor cube now...")

    ds_pred_full = build_predictor_cube(
        ds_interp=ds_interp, ds_imerg=ds_imerg,
        use_temp_air=True, use_temp_dew=True, use_temp_wet=True,
        use_rh=True, use_elev=True, use_imerg_plp=True,
    )
    loocv_df = prep_loocv_table(df_loocv_raw, ds_interp=ds_pred_full)
    loocv_df = loocv_df[loocv_df["time"].isin(pd.to_datetime(common_times))].copy()
    print(f"  LOOCV rows after time intersection: {len(loocv_df)}")

    present_grid = [f for f in ALL_GRID_FEATURES if f in ds_pred_full.data_vars]
    t0 = time.time()
    master_df = sample_predictor_cube_to_points_batched(
        loocv_df, ds_pred=ds_pred_full, predictor_vars=present_grid, verbose=True,
    )
    print(f"  Sampling done in {(time.time()-t0)/60:.1f} min")

    master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
    master_df = master_df.dropna(subset=present_grid + ["phase_full"]).copy()
    master_df["phase_full"] = master_df["phase_full"].astype(int)
    master_df.to_parquet(cache_path, index=False)
    print(f"  Cached to: {cache_path}")
    return master_df


# =============================================================================
# 6.  SHARED TRAIN / VAL / TEST SPLIT  (verbatim — identical to ablation)
# =============================================================================

def make_split(master_df: pd.DataFrame) -> pd.DataFrame:
    train_df, valtest_df = train_test_split(
        master_df, test_size=1.0 - TRAIN_FRAC,
        random_state=RANDOM_SEED, stratify=master_df["phase_full"],
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=0.5,
        random_state=RANDOM_SEED, stratify=valtest_df["phase_full"],
    )
    for df, label in [(train_df,"train"), (val_df,"val"), (test_df,"test")]:
        df["split"] = label
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"\nSplit: { {s: int((split_df['split']==s).sum()) for s in ['train','val','test']} }")
    return split_df


# =============================================================================
# 7.  XGBOOST BASELINE MODEL  (condensed ablation baseline_full run)
# =============================================================================

def train_xgboost_baseline(split_df: pd.DataFrame, out_dir: Path) -> dict:
    """Train the baseline_full XGBoost model exactly as in the ablation study
    and return calibrated test-set predictions (binary + 3-class band)."""
    print(f"\n{'='*62}\n  MODEL: {MODEL_NAME} (baseline_full configuration)\n{'='*62}")
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in MODEL_FEATURES if f not in split_df.columns]
    if missing:
        raise KeyError(f"Missing model features in master table: {missing}")

    train_full = split_df[split_df["split"] == "train"].copy()
    val_full   = split_df[split_df["split"] == "val"].copy()
    test_full  = split_df[split_df["split"] == "test"].copy()

    pure = [SNOW_CODE, RAIN_CODE]
    train_fit = train_full[train_full["phase_full"].isin(pure)].copy()
    val_fit   = val_full[val_full["phase_full"].isin(pure)].copy()
    test_fit  = test_full[test_full["phase_full"].isin(pure)].copy()

    X_train = train_fit[MODEL_FEATURES]; y_train = train_fit["phase_full"].map(BINARY_LABEL_MAP).astype(int)
    X_val   = val_fit[MODEL_FEATURES];   y_val   = val_fit["phase_full"].map(BINARY_LABEL_MAP).astype(int)

    best_spw, spw_df = sweep_scale_pos_weight(
        X_train, y_train, X_val, y_val,
        SCALE_POS_WEIGHT_GRID, BASE_XGB_PARAMS,
        probe_rounds=350, early_stopping=30, random_seed=RANDOM_SEED,
    )
    spw_df.to_csv(out_dir / "scale_pos_weight_sweep.csv", index=False)
    print(f"  Best scale_pos_weight: {best_spw}")

    params  = {**BASE_XGB_PARAMS, "scale_pos_weight": best_spw, "seed": RANDOM_SEED}
    dtrain  = xgb.DMatrix(X_train, label=y_train, feature_names=MODEL_FEATURES)
    dval_dm = xgb.DMatrix(X_val,   label=y_val,   feature_names=MODEL_FEATURES)
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain,"train"),(dval_dm,"val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=200,
    )
    booster.save_model(out_dir / "xgb_binary_phase_model.bin")

    # Beta calibration (two-regime, fitted on train+val pure obs)
    trainval_pure = pd.concat([train_fit, val_fit]).reset_index(drop=True)
    tw_tv    = trainval_pure["temp_wet"].values
    p_raw_tv = booster.predict(xgb.DMatrix(trainval_pure[MODEL_FEATURES], feature_names=MODEL_FEATURES))
    y_tv     = trainval_pure["phase_full"].map(BINARY_LABEL_MAP).astype(int)

    clear_mask_tv = np.abs(tw_tv) > CLEAR_PHASE_TWET_C
    cal_clear = BetaCalibration(parameters="abm")
    cal_clear.fit(p_raw_tv[clear_mask_tv].reshape(-1,1), y_tv.values[clear_mask_tv])
    cal_nf = BetaCalibration(parameters="ab")
    cal_nf.fit(p_raw_tv[~clear_mask_tv].reshape(-1,1), y_tv.values[~clear_mask_tv])

    with open(out_dir / "beta_calibration_models.pkl", "wb") as f:
        pickle.dump({"method":"beta_calibration","clear_phase":cal_clear,
                     "near_freezing":cal_nf,"clear_phase_twet_threshold":CLEAR_PHASE_TWET_C}, f)

    def calibrate(p_raw, twet):
        p_raw = np.asarray(p_raw, float); twet = np.asarray(twet, float)
        out   = np.empty_like(p_raw)
        nf    = np.abs(twet) <= CLEAR_PHASE_TWET_C
        if (~nf).any(): out[~nf] = cal_clear.predict(p_raw[~nf].reshape(-1,1))
        if nf.any():    out[nf]  = cal_nf.predict(p_raw[nf].reshape(-1,1))
        return np.clip(out, 1e-4, 1-1e-4)

    # Gaussian half-band optimised on the validation set (full phase set)
    p_valf_cal = calibrate(
        booster.predict(xgb.DMatrix(val_full[MODEL_FEATURES], feature_names=MODEL_FEATURES)),
        val_full["temp_wet"].to_numpy())
    thr = optimize_threshold_params(p_valf_cal, val_full["temp_wet"].to_numpy(),
                                    val_full["phase_full"].astype(int).to_numpy())
    print(f"  Band params: base={thr['base_half_band']}, extra={thr['extra_half_band']}, sigma={thr['sigma']}")

    # Test-set predictions (full phase set)
    p_testf_raw = booster.predict(xgb.DMatrix(test_full[MODEL_FEATURES], feature_names=MODEL_FEATURES))
    p_testf_cal = calibrate(p_testf_raw, test_full["temp_wet"].to_numpy())
    pred_test_band = classify_phase_gaussian_band(
        p_testf_cal, test_full["temp_wet"].to_numpy(),
        thr["base_half_band"], thr["extra_half_band"], thr["sigma"])
    pred_test_bin05 = np.where(p_testf_cal >= 0.5, SNOW_CODE, RAIN_CODE)

    return dict(
        booster=booster, band_params=thr, best_spw=best_spw,
        test_full=test_full,
        p_test_cal=p_testf_cal,
        pred_test_bin05=pred_test_bin05,     # binary 0.5 decision (benchmark-comparable)
        pred_test_band=pred_test_band,       # 3-class with mix abstention band
    )


# =============================================================================
# 8.  BENCHMARK METHODS
# =============================================================================

def binlog_p_snow(temp_air, rh, a=BINLOG_A, b=BINLOG_B, g=BINLOG_G):
    """Jennings et al. (2018) binary logistic regression probability of snow.
    temp_air in deg C, rh in %."""
    z = a + b * np.asarray(temp_air, float) + g * np.asarray(rh, float)
    return 1.0 / (1.0 + np.exp(z))


def fit_domain_binlog(train_fit: pd.DataFrame):
    """Refit the T_air + RH logistic form on this domain's training split
    (pure snow/rain obs). Returns a fitted sklearn model or None."""
    if "rh" not in train_fit.columns or train_fit["rh"].isna().all():
        return None
    d = train_fit.dropna(subset=["temp_air", "rh"])
    X = d[["temp_air", "rh"]].to_numpy(float)
    y = (d["phase_full"] == SNOW_CODE).astype(int).to_numpy()
    lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularised
    lr.fit(X, y)
    print(f"  Fitted binlog coefficients: intercept={lr.intercept_[0]:.3f}, "
          f"T_air={lr.coef_[0][0]:.3f}, RH={lr.coef_[0][1]:.3f}")
    return lr


def benchmark_predictions(df: pd.DataFrame, fitted_binlog=None) -> dict[str, np.ndarray]:
    """Apply every benchmark PPM to a dataframe with temp_air/temp_dew/temp_wet
    (+ rh). Returns {method_name: predicted phase codes (snow/rain only)}."""
    preds = {}
    for name, col, thresh in THRESHOLD_BENCHMARKS:
        if col not in df.columns:
            print(f"  WARNING: {col} missing — skipping benchmark {name}")
            continue
        t = df[col].to_numpy(float)
        preds[name] = np.where(t <= thresh, SNOW_CODE, RAIN_CODE)

    if "rh" in df.columns and df["rh"].notna().any():
        rh = df["rh"].to_numpy(float)
        # RH sanity: Jennings et al. use percent (0-100). Rescale if fractional.
        if np.nanmax(rh) <= 1.5:
            rh = rh * 100.0
        p_snow = binlog_p_snow(df["temp_air"].to_numpy(float), rh)
        preds["binlog_jennings18"] = np.where(p_snow >= 0.5, SNOW_CODE, RAIN_CODE)
        if fitted_binlog is not None:
            X = np.column_stack([df["temp_air"].to_numpy(float), df["rh"].to_numpy(float)])
            p_fit = fitted_binlog.predict_proba(X)[:, 1]
            preds["binlog_fitted"] = np.where(p_fit >= 0.5, SNOW_CODE, RAIN_CODE)
    else:
        print("  WARNING: rh missing — skipping binary logistic regression benchmarks")
    return preds


# =============================================================================
# 9.  EVALUATION — paper-style metrics
# =============================================================================

def snow_rain_bias(y_true, y_pred):
    """Jennings et al. bias: 100 * (n_predicted / n_observed - 1) per phase."""
    out = {}
    for code, label in [(SNOW_CODE, "snow"), (RAIN_CODE, "rain")]:
        n_obs  = int(np.sum(y_true == code))
        n_pred = int(np.sum(y_pred == code))
        out[f"{label}_bias_pct"] = round(100.0 * (n_pred / n_obs - 1.0), 2) if n_obs > 0 else np.nan
    return out


def overall_metrics(y_true, y_pred, temp_wet) -> dict:
    """Binary metrics on pure snow/rain observations (paper convention)."""
    m = {
        "n": int(len(y_true)),
        "accuracy_pct": round(100.0 * float(np.mean(y_pred == y_true)), 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=[SNOW_CODE, RAIN_CODE],
                                          average="macro", zero_division=0)), 4),
        "snow_recall": round(float(np.mean(y_pred[y_true == SNOW_CODE] == SNOW_CODE)), 4)
                        if (y_true == SNOW_CODE).any() else np.nan,
        "rain_recall": round(float(np.mean(y_pred[y_true == RAIN_CODE] == RAIN_CODE)), 4)
                        if (y_true == RAIN_CODE).any() else np.nan,
    }
    m.update(snow_rain_bias(y_true, y_pred))
    nf = np.abs(np.asarray(temp_wet, float)) <= NEARFREEZE_TWET_C
    if nf.sum() >= MIN_BIN_N:
        m["nearfreeze_n"] = int(nf.sum())
        m["nearfreeze_accuracy_pct"] = round(100.0 * float(np.mean(y_pred[nf] == y_true[nf])), 2)
        m["nearfreeze_macro_f1"] = round(float(f1_score(
            y_true[nf], y_pred[nf], labels=[SNOW_CODE, RAIN_CODE],
            average="macro", zero_division=0)), 4)
        m.update({f"nearfreeze_{k}": v for k, v in snow_rain_bias(y_true[nf], y_pred[nf]).items()})
    return m


def binned_profile(temp_air, y_true, y_pred) -> pd.DataFrame:
    """Accuracy / snow bias / rain bias by air temperature bin (paper Fig. 1)."""
    edges = np.arange(TAIR_BIN_MIN, TAIR_BIN_MAX + TAIR_BIN_WIDTH, TAIR_BIN_WIDTH)
    t = np.asarray(temp_air, float)
    records = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (t >= lo) & (t < hi)
        n = int(mask.sum())
        if n < MIN_BIN_N:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rec = {"t_mid": (lo + hi) / 2, "n": n,
               "accuracy_pct": 100.0 * float(np.mean(yp == yt))}
        for code, label in [(SNOW_CODE, "snow"), (RAIN_CODE, "rain")]:
            n_obs = int(np.sum(yt == code))
            rec[f"{label}_bias_pct"] = (100.0 * (int(np.sum(yp == code)) / n_obs - 1.0)
                                        if n_obs >= MIN_BIN_PHASE_N else np.nan)
        records.append(rec)
    return pd.DataFrame(records)


def evaluate_all(test_full: pd.DataFrame, bench_preds: dict[str, np.ndarray],
                 model_pred_bin05: np.ndarray, model_p_cal: np.ndarray,
                 model_pred_band: np.ndarray, band_params: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Evaluate benchmarks + model on the pure-phase test subset. Returns
    (comparison table, per-method binned profiles, mixed-phase summary)."""
    y_full = test_full["phase_full"].astype(int).to_numpy()
    pure   = np.isin(y_full, [SNOW_CODE, RAIN_CODE])
    y_pure = y_full[pure]
    tair   = test_full["temp_air"].to_numpy(float)
    twet   = test_full["temp_wet"].to_numpy(float)

    all_preds = {name: p[pure] for name, p in bench_preds.items()}
    all_preds[MODEL_NAME] = model_pred_bin05[pure]

    rows, profiles = [], {}
    for name, yp in all_preds.items():
        m = {"method": name, "label": METHOD_LABEL.get(name, name),
             "uses_humidity": name not in ("ta_1.0", "ta_1.5"),
             **overall_metrics(y_pure, yp, twet[pure])}
        if name == MODEL_NAME:
            y_bin = (y_pure == SNOW_CODE).astype(int)
            m["roc_auc_cal"] = round(float(roc_auc_score(y_bin, model_p_cal[pure])), 4)
            m["brier_cal"]   = round(float(brier_score_loss(y_bin, model_p_cal[pure])), 4)
        rows.append(m)
        profiles[name] = binned_profile(tair[pure], y_pure, yp)

    table = pd.DataFrame(rows).sort_values("accuracy_pct", ascending=False).reset_index(drop=True)

    # ── Mixed-phase summary (paper's mixed section) ───────────────────────────
    mix_summary = {}
    n_mix = int(np.sum(y_full == MIX_CODE))
    mix_summary["n_test_obs"] = int(len(y_full))
    mix_summary["n_mix_obs"] = n_mix
    mix_summary["mix_fraction_pct"] = round(100.0 * n_mix / len(y_full), 2)
    mix_summary["note"] = ("Benchmark PPMs are binary and predict 0% of mixed obs "
                           "(mixed bias = -100%), as do the paper's ML methods "
                           "(XGBoost/ANN: 0% mix capture; RF: 9.3%).")
    if n_mix > 0:
        mix_mask = y_full == MIX_CODE
        mix_summary[f"{MODEL_NAME}_band_mix_capture"] = round(
            float(np.mean(model_pred_band[mix_mask] == MIX_CODE)), 4)
        mix_summary[f"{MODEL_NAME}_3class_accuracy_pct"] = round(
            100.0 * float(np.mean(model_pred_band == y_full)), 2)
        mix_summary[f"{MODEL_NAME}_frac_pred_mix"] = round(
            float(np.mean(model_pred_band == MIX_CODE)), 4)
        mix_summary["band_params"] = {k: band_params[k] for k in
                                      ["base_half_band", "extra_half_band", "sigma"]}
    return table, profiles, mix_summary


# =============================================================================
# 10.  FIGURES
# =============================================================================

def plot_fig1_by_tair(profiles: dict[str, pd.DataFrame], out_path: Path):
    """Paper Fig. 1 analogue: accuracy, snow bias, rain bias vs air temperature
    for every benchmark + the XGBoost model (bold black)."""
    panel_specs = [
        ("accuracy_pct",  "Accuracy (%)",  (0, 102)),
        ("snow_bias_pct", "Snow bias (%)", (-105, 105)),
        ("rain_bias_pct", "Rain bias (%)", (-105, 105)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    fig.suptitle(f"{REGION} test split — benchmark PPMs vs. XGBoost + MRoS\n"
                 f"(pure rain/snow obs; {TAIR_BIN_WIDTH:g} °C air-temperature bins, "
                 f"bins with n ≥ {MIN_BIN_N})", fontsize=12)

    order = [n for n in METHOD_STYLE if n in profiles]
    for ax, (col, ylabel, ylim) in zip(axes, panel_specs):
        for name in order:
            prof = profiles[name]
            if prof.empty or col not in prof.columns:
                continue
            valid = prof[col].notna()
            style = METHOD_STYLE[name]
            ax.plot(prof.loc[valid, "t_mid"], np.clip(prof.loc[valid, col], *ylim),
                    marker="o", ms=4 if name == MODEL_NAME else 3,
                    label=METHOD_LABEL.get(name, name),
                    zorder=5 if name == MODEL_NAME else 2, **style)
        ax.axvline(0, color="grey", ls="--", lw=1.0, alpha=0.7)
        if "bias" in col:
            ax.axhline(0, color="grey", lw=0.8, alpha=0.7)
        ax.set(ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Air temperature (°C)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_relative_improvement(profiles: dict[str, pd.DataFrame],
                                   table: pd.DataFrame, out_path: Path):
    """Paper Fig. 2 analogue: model accuracy minus (best / average) benchmark
    accuracy, by air temperature bin. 'Best benchmark' = highest overall
    accuracy among benchmarks (fixed method, as in the paper)."""
    bench_names = [n for n in profiles if n != MODEL_NAME]
    if not bench_names or MODEL_NAME not in profiles:
        return
    bench_table = table[table["method"] != MODEL_NAME]
    best_name = bench_table.sort_values("accuracy_pct", ascending=False)["method"].iloc[0]

    model_prof = profiles[MODEL_NAME].set_index("t_mid")
    best_prof  = profiles[best_name].set_index("t_mid")
    avg_prof   = (pd.concat([profiles[n].set_index("t_mid")["accuracy_pct"].rename(n)
                             for n in bench_names], axis=1).mean(axis=1))

    common_best = model_prof.index.intersection(best_prof.index)
    common_avg  = model_prof.index.intersection(avg_prof.index)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(common_best, model_prof.loc[common_best, "accuracy_pct"] - best_prof.loc[common_best, "accuracy_pct"],
            "--o", color="black", lw=2, ms=5,
            label=f"vs. best benchmark ({METHOD_LABEL.get(best_name, best_name)})")
    ax.plot(common_avg, model_prof.loc[common_avg, "accuracy_pct"] - avg_prof.loc[common_avg],
            "-o", color="#d62728", lw=2, ms=5, label="vs. average benchmark")
    ax.axhline(0, color="grey", lw=1.0)
    ax.axvline(0, color="grey", ls="--", lw=1.0, alpha=0.7)
    ax.axvspan(0, 4, alpha=0.06, color="orange")
    ax.set(xlabel="Air temperature (°C)",
           ylabel="Δ Accuracy (percentage points)",
           title=f"{REGION} — XGBoost + MRoS accuracy relative to benchmarks\n"
                 f"(positive = model better; shaded 0–4 °C = benchmark performance-dip zone)")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall_bars(table: pd.DataFrame, out_path: Path):
    """Overall + near-freezing accuracy bars, benchmark vs model."""
    t = table.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * len(t))), sharey=True)
    for ax, col, title in [
        (axes[0], "accuracy_pct",            "Overall accuracy (%)"),
        (axes[1], "nearfreeze_accuracy_pct", f"Near-freezing accuracy (%)\n(|T_wet| ≤ {NEARFREEZE_TWET_C:g} °C)"),
    ]:
        if col not in t.columns:
            continue
        sub = t.dropna(subset=[col]).sort_values(col)
        colors = ["black" if m == MODEL_NAME else
                  METHOD_STYLE.get(m, {}).get("color", "grey") for m in sub["method"]]
        ax.barh(sub["label"], sub[col], color=colors, alpha=0.85)
        for _, r in sub.iterrows():
            ax.text(r[col] + 0.3, r["label"], f"{r[col]:.1f}", va="center", fontsize=8)
        ax.set(title=title, xlim=(0, 105))
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{REGION} test split — benchmark comparison (pure rain/snow obs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all_outputs(test_full, bench_preds, model_out, out_root: Path):
    """Evaluate, save tables/parquet, draw figures."""
    table, profiles, mix_summary = evaluate_all(
        test_full, bench_preds,
        model_out["pred_test_bin05"], model_out["p_test_cal"],
        model_out["pred_test_band"], model_out["band_params"],
    )

    table.to_csv(out_root / "benchmark_comparison.csv", index=False)
    with open(out_root / "benchmark_mixed_phase_summary.json", "w") as f:
        json.dump(mix_summary, f, indent=2)
    for name, prof in profiles.items():
        prof.to_csv(out_root / f"profile_by_tair_{name}.csv", index=False)

    # Per-observation predictions parquet — input for bootstrap CIs later.
    keep_meta = [c for c in ["time", "x", "y", "lon", "lat", "elev", "station_id",
                             "temp_air", "temp_dew", "temp_wet", "rh", "imerg_plp",
                             "phase_full"] if c in test_full.columns]
    pred_df = test_full[keep_meta].reset_index(drop=True).copy()
    for name, p in bench_preds.items():
        pred_df[f"pred_{name}"] = p
    pred_df[f"pred_{MODEL_NAME}_bin05"] = model_out["pred_test_bin05"]
    pred_df[f"pred_{MODEL_NAME}_band"]  = model_out["pred_test_band"]
    pred_df[f"p_snow_cal_{MODEL_NAME}"] = model_out["p_test_cal"]
    pred_df.to_parquet(out_root / "benchmark_predictions_test.parquet", index=False)

    graphics = out_root / "graphics"
    graphics.mkdir(exist_ok=True)
    plot_fig1_by_tair(profiles, graphics / "fig1_accuracy_bias_by_tair.png")
    plot_fig2_relative_improvement(profiles, table, graphics / "fig2_relative_improvement.png")
    plot_overall_bars(table, graphics / "fig3_overall_accuracy_bars.png")

    print("\n" + "=" * 78)
    print("BENCHMARK COMPARISON (pure rain/snow test obs)")
    print("=" * 78)
    show_cols = [c for c in ["label", "n", "accuracy_pct", "snow_bias_pct", "rain_bias_pct",
                             "macro_f1", "nearfreeze_accuracy_pct", "nearfreeze_macro_f1"]
                 if c in table.columns]
    print(table[show_cols].to_string(index=False))
    print(f"\nMixed-phase summary: {json.dumps(mix_summary, indent=2)}")
    print(f"\nOutputs saved to: {out_root}")
    return table, profiles, mix_summary


# =============================================================================
# 11.  MAIN
# =============================================================================

def replot_from_saved(out_root: Path):
    """Regenerate tables/figures from the saved per-observation parquet
    (no retraining, no data-cube access needed)."""
    pq = out_root / "benchmark_predictions_test.parquet"
    if not pq.exists():
        print(f"No saved predictions found at {pq} — run the full pipeline first.")
        return
    pred_df = pd.read_parquet(pq)
    bench_cols = [c for c in pred_df.columns
                  if c.startswith("pred_") and MODEL_NAME not in c]
    bench_preds = {c.replace("pred_", ""): pred_df[c].to_numpy(int) for c in bench_cols}
    model_out = dict(
        pred_test_bin05=pred_df[f"pred_{MODEL_NAME}_bin05"].to_numpy(int),
        pred_test_band=pred_df[f"pred_{MODEL_NAME}_band"].to_numpy(int),
        p_test_cal=pred_df[f"p_snow_cal_{MODEL_NAME}"].to_numpy(float),
        band_params=dict(base_half_band=np.nan, extra_half_band=np.nan, sigma=np.nan),
    )
    make_all_outputs(pred_df, bench_preds, model_out, out_root)


def main():
    parser = argparse.ArgumentParser(description="Benchmark analysis vs. Jennings et al. (2025)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print benchmark method list without running")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate tables/figures from saved per-obs predictions")
    args = parser.parse_args()

    print(f"\nBENCHMARK ANALYSIS — Region: {REGION} | Interp: {INTERP_TYPE}")
    print(f"Output: {BENCH_ROOT}\n")
    print("Benchmark methods (Jennings et al. 2018 / 2025):")
    for name, col, thresh in THRESHOLD_BENCHMARKS:
        print(f"  {name:22s} snow if {col} <= {thresh:.1f} °C")
    print(f"  {'binlog_jennings18':22s} p(snow) = 1/(1+exp({BINLOG_A:+.2f} + {BINLOG_B:.2f}·T_a + {BINLOG_G:.2f}·RH))")
    if INCLUDE_FITTED_BINLOG:
        print(f"  {'binlog_fitted':22s} same form, coefficients fitted on this domain's train split")
    print(f"  {MODEL_NAME:22s} XGBoost baseline_full ({len(MODEL_FEATURES)} features), "
          f"beta-calibrated, binary 0.5 decision (+ mix band reported separately)")

    if args.dry_run:
        return
    if args.replot:
        replot_from_saved(BENCH_ROOT)
        return

    # 1. Data (identical pipeline + cache + split as the ablation study)
    ds_interp, ds_imerg, df_loocv_raw, common_times = load_and_sync_datasets()
    master_df = get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times)
    split_df  = make_split(master_df)

    if "rh" not in split_df.columns:
        warnings.warn("Master cache has no 'rh' column — logistic regression "
                      "benchmarks will be skipped. Delete the cache parquet and "
                      "re-run to resample with rh included.")

    # 2. XGBoost baseline model
    model_out = train_xgboost_baseline(split_df, BENCH_ROOT / "model")
    test_full = model_out["test_full"]

    # 3. Benchmark methods on the same test split
    print(f"\n{'='*62}\n  BENCHMARK PPMs\n{'='*62}")
    fitted_binlog = None
    if INCLUDE_FITTED_BINLOG and "rh" in split_df.columns:
        train_fit = split_df[(split_df["split"] == "train") &
                             (split_df["phase_full"].isin([SNOW_CODE, RAIN_CODE]))]
        fitted_binlog = fit_domain_binlog(train_fit)
    bench_preds = benchmark_predictions(test_full, fitted_binlog=fitted_binlog)
    print(f"  Benchmarks evaluated: {list(bench_preds.keys())}")

    # 4. Evaluation, tables, figures
    make_all_outputs(test_full, bench_preds, model_out, BENCH_ROOT)


if __name__ == "__main__":
    main()
