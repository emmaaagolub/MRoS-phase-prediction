"""
ML_XGBoost_binary_uncertainty — Ablation Study Runner
======================================================

Runs a configurable set of ablation experiments over the feature-toggle axes
defined in the base notebook.  The gridded predictor cube is sampled to MRoS
points once (or loaded from a cached parquet on subsequent runs) and shared
across all experiments — the expensive resampling step is never repeated.

Each experiment writes its own uniquely-named subfolder under:
  DATA_DIR / ML_pipeline / model_artifacts / {REGION} / ablations/

Structure
---------
  1.  ABLATION CONFIG
  2.  SHARED SETUP
  3.  HELPERS
  4.  ONE-TIME DATA LOAD
  5.  ONE-TIME SAMPLING  (cached to parquet)
  6.  SHARED TRAIN / VAL / TEST SPLIT
  7.  run_experiment()
  8.  ABLATION LOOP + CROSS-EXPERIMENT PLOTS

Usage
-----
  python ML_XGBoost_binary_uncertainty_ablation_v2.py
  python ML_XGBoost_binary_uncertainty_ablation_v2.py --configs 0,1,2
  python ML_XGBoost_binary_uncertainty_ablation_v2.py --dry-run

  NOTE: This version employs beta calibration, which is better used for the CA dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xarray as xr
import xgboost as xgb
from betacal import BetaCalibration
from pyproj import Transformer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score, 
    brier_score_loss, 
    f1_score,
    log_loss, 
    recall_score, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
import matplotlib.ticker as mticker

# =============================================================================
# 1.  ABLATION CONFIG
# =============================================================================
# Each entry is a dict of feature-toggle overrides.  Any toggle not listed
# falls back to BASE_FEATURE_CONFIG.  Add / remove entries freely.

BASE_FEATURE_CONFIG = dict(
    use_temp_air   = True,
    use_temp_dew   = True,
    use_temp_wet   = True,
    use_rh         = False,
    use_imerg_plp  = True,
    use_elev       = True,
    use_mros_loocv = True,
)

ABLATION_CONFIGS: list[dict] = [
    # baseline
    dict(name="baseline_full"),
    # leave-one-out
    dict(name="no_mros_loocv", use_mros_loocv=False),
    dict(name="no_temp_air",   use_temp_air=False),
    dict(name="no_temp_dew",   use_temp_dew=False),
    dict(name="no_temp_wet",   use_temp_wet=False),
    dict(name="no_imerg_plp",  use_imerg_plp=False),
    dict(name="no_elev", use_elev=False),
    # compound
    dict(name="thermo_only",   use_imerg_plp=False, use_elev=False, use_mros_loocv=False),
    # dict(name="with_rh",       use_rh=True),
    dict(name="min_core_noplp",  use_temp_air=False, use_temp_dew=False, use_imerg_plp=False),
    dict(name="min_core_wplp",  use_temp_air=False, use_temp_dew=False),
]

INTERP_TYPE = "kriging"   # "IDW" or "kriging"
REGION      = "CA"


# =============================================================================
# 2.  SHARED SETUP
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
SETUP_DIR      = DATA_DIR / f"ML_pipeline/compiled_input_predictors/{REGION}" / interp_type_folder
ABLATION_ROOT  = DATA_DIR / f"ML_pipeline/model_artifacts/{REGION}/ablations_v2"
SETUP_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

# Training
TRAIN_FRAC            = 0.70
RANDOM_SEED           = 42
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND       = 2000

# Phase codes
SNOW_CODE        = 0
RAIN_CODE        = 1
MIX_CODE         = 2
BINARY_LABEL_MAP = {RAIN_CODE: 0, SNOW_CODE: 1}
FULL_CLASS_NAMES = ["snow", "rain", "mix"]

# scale_pos_weight sweep
SCALE_POS_WEIGHT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0]

# Calibration
CLEAR_PHASE_TWET_C = 2.0

# Gaussian half-band grids
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

# SHAP / T_wet bin config
TWET_BIN_EDGES  = [-np.inf, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, np.inf]
TWET_BIN_LABELS = ["<-6","-6–-5","-5–-4","-4–-3","-3–-2","-2–-1",
                   "-1–0","0–1","1–2","2–3","3–4","4–5","5–6",">6"]
NEAR_FREEZE_THRESH  = 2.0
TOP_N_SHAP_FEATURES = 5
NF_BIN_LABELS       = ["-2–-1", "-1–0", "0–1", "1–2"]

PHASE_COLORS = {"snow": "#1f77b4", "rain": "#2ca02c", "mix": "#e377c2"}


# =============================================================================
# 3.  HELPERS  (verbatim from base notebook)
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


def expected_calibration_error(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc  = float(np.mean((p_pred[mask] >= 0.5).astype(int) == y_true[mask]))
        conf = float(np.mean(p_pred[mask]))
        ece += mask.sum() * abs(acc - conf)
    return round(ece / max(len(y_true), 1), 4)

def plot_per_experiment_stories(
    name, graphics_dir,
    y_val_fit_phase, pred_val_bin05, y_test_fit_phase, pred_test_bin05,
    y_val_full_phase, pred_val_full, y_test_full_phase, pred_test_full,
    y_val_bin, p_val_cal, y_test_bin, p_test_cal,
    p_val_raw, p_test_raw,         
    val_full_df, test_full_df,
    base_hb, extra_hb, sigma,
):
    def _norm_cm(ax, y_true, y_pred, labels, display_labels, title, cmap="Blues"):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums,
                            out=np.zeros_like(cm, dtype=float),
                            where=row_sums > 0)
        n = len(labels)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(display_labels, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(display_labels, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=9); ax.grid(False)
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.0%}\n({cm[i,j]})",
                        ha="center", va="center", fontsize=7, color=color)

    t_wet_val  = val_full_df["temp_wet"].to_numpy()
    t_wet_test = test_full_df["temp_wet"].to_numpy()
    p_valf_cal  = val_full_df["p_snow_cal"].to_numpy()
    p_testf_cal = test_full_df["p_snow_cal"].to_numpy()

    # ── Story 1 ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"{name} — Story 1: Binary discrimination", fontsize=12, y=1.01)

    ax00 = fig.add_subplot(2, 4, 1)
    ax01 = fig.add_subplot(2, 4, 2)
    ax02 = fig.add_subplot(2, 4, 3)
    ax03 = fig.add_subplot(2, 4, 4)

    _norm_cm(ax00, y_val_fit_phase,  pred_val_bin05,
             [SNOW_CODE, RAIN_CODE], ["snow","rain"], "Val — binary (0.5 thr)")
    _norm_cm(ax01, y_test_fit_phase, pred_test_bin05,
             [SNOW_CODE, RAIN_CODE], ["snow","rain"], "Test — binary (0.5 thr)")
    _norm_cm(ax02, y_val_full_phase,  pred_val_full,
             [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES, "Val — 3-class (band)")
    _norm_cm(ax03, y_test_full_phase, pred_test_full,
             [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES, "Test — 3-class (band)")

    ax_roc = fig.add_subplot(2, 4, (5, 6))
    ax_pr  = fig.add_subplot(2, 4, (7, 8))
    split_colors = {"Validation": "#8338ec", "Test": "#ff006e"}
    for y_true_b, p_cal, split in [
        (y_val_bin,  p_val_cal,  "Validation"),
        (y_test_bin, p_test_cal, "Test"),
    ]:
        color = split_colors[split]
        fpr, tpr, _ = roc_curve(y_true_b, p_cal)
        prec, rec, _ = precision_recall_curve(y_true_b, p_cal)
        auc = roc_auc_score(y_true_b, p_cal)
        ap  = average_precision_score(y_true_b, p_cal)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{split}  AUC={auc:.3f}")
        ax_pr.plot(rec, prec,  color=color, lw=2, label=f"{split}  AP={ap:.3f}")
    ax_roc.plot([0,1],[0,1],"--",color="grey",lw=1,alpha=0.6)
    ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC"); ax_roc.legend(fontsize=9)
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="PR"); ax_pr.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(graphics_dir / "story1_discrimination.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

 # ── Story 2 ───────────────────────────────────────────────────────────────
    band_at_zero  = gaussian_half_band(np.array([0.0]), base_hb, extra_hb, sigma)[0]
    rain_thresh_0 = 0.5 - band_at_zero
    snow_thresh_0 = 0.5 + band_at_zero
    split_colors_2 = {"Validation": "#8338ec", "Test": "#ff006e"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{name} — Story 2: Probability calibration", fontsize=12)

    for col, (y_true_b, p_raw, p_cal, split) in enumerate([
        (y_val_bin,  p_val_raw, p_val_cal,  "Validation"),
        (y_test_bin, p_test_raw, p_test_cal, "Test"),
    ]):
        ax_rel  = axes[0, col]
        ax_hist = axes[1, col]
        color   = split_colors_2[split]

        if len(y_true_b) < 10:
            ax_rel.text(0.5, 0.5, "insufficient data",
                        ha="center", va="center", transform=ax_rel.transAxes)
            continue

        frac_raw, mean_raw = calibration_curve(y_true_b, p_raw, n_bins=15, strategy="quantile")
        frac_cal, mean_cal = calibration_curve(y_true_b, p_cal, n_bins=15, strategy="quantile")

        ax_rel.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange",
                       label=f"Band at T_wet=0°C ({rain_thresh_0:.2f}–{snow_thresh_0:.2f})")
        ax_rel.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, alpha=0.7,
                    label="Perfect calibration")
        ax_rel.plot(mean_raw, frac_raw, "o--", color="#aaaaaa", lw=1.5, ms=5,
                    label="Raw XGBoost")
        ax_rel.plot(mean_cal, frac_cal, "o-",  color=color, lw=2, ms=6,
                    label="Calibrated (beta)")
        ax_rel.set(xlim=(0,1), ylim=(0,1),
                   ylabel="Observed snow frequency", title=split)
        ax_rel.legend(fontsize=9); ax_rel.grid(alpha=0.25)

        bs_raw = brier_score_loss(y_true_b, p_raw)
        bs_cal = brier_score_loss(y_true_b, p_cal)
        ax_rel.text(0.03, 0.92, f"Brier  raw={bs_raw:.3f}  cal={bs_cal:.3f}",
                    transform=ax_rel.transAxes, fontsize=9, color="dimgrey")

        ax_hist.hist(p_cal, bins=30, color=color, alpha=0.7, edgecolor="none")
        ax_hist.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.15, color="orange")
        ax_hist.set(xlabel="Predicted p(snow), calibrated", ylabel="Count")
        ax_hist.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(graphics_dir / "story2_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 3 ───────────────────────────────────────────────────────────────
    def _twet_profile(df_f, p_cal_f, y_true_f, bin_edges):
        records = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (df_f["temp_wet"].to_numpy() >= lo) & (df_f["temp_wet"].to_numpy() < hi)
            if mask.sum() < 10:
                continue
            yt = y_true_f[mask]
            yp = classify_phase_gaussian_band(p_cal_f[mask],
                                              df_f["temp_wet"].to_numpy()[mask],
                                              base_hb, extra_hb, sigma)
            def _f1(code):
                tp = np.sum((yt==code)&(yp==code)); fp = np.sum((yt!=code)&(yp==code))
                fn = np.sum((yt==code)&(yp!=code))
                p  = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
                return 2*p*r/(p+r) if p+r else 0
            tm = yt == MIX_CODE
            records.append({"t_mid": (lo+hi)/2, "n": mask.sum(),
                             "f1_snow": _f1(SNOW_CODE), "f1_rain": _f1(RAIN_CODE),
                             "mix_capture": float(np.mean(yp[tm]==MIX_CODE)) if tm.any() else np.nan})
        return pd.DataFrame(records)

    bin_edges = np.arange(-6, 7, 1)
    prof_val  = _twet_profile(val_full_df,  p_valf_cal,  y_val_full_phase,  bin_edges)
    prof_test = _twet_profile(test_full_df, p_testf_cal, y_test_full_phase, bin_edges)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{name} — Story 3: Performance vs T_wet", fontsize=12)
    for ax, col, ylabel in zip(axes,
            ["f1_snow","f1_rain","mix_capture"],
            ["F1 — snow","F1 — rain","Mix capture rate"]):
        color = PHASE_COLORS.get(ylabel.split("—")[-1].strip().replace(" ","").lower(),
                                  "#555555")
        for prof, split, ls in [(prof_val,"Val","--"),(prof_test,"Test","-")]:
            if col not in prof.columns or prof.empty:
                continue
            ax.plot(prof["t_mid"], prof[col], ls, color=color, lw=2, label=split,
                    marker="o", ms=4)
        ax.axvspan(-1, 1, alpha=0.08, color="orange")
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        ax.set(xlabel="T_wet (°C)", ylabel=ylabel, title=ylabel,
               xlim=(bin_edges[0], bin_edges[-1]), ylim=(0, 1.05))
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(graphics_dir / "story3_twet_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 4 ───────────────────────────────────────────────────────────────
    t_grid = np.linspace(-6, 6, 300)
    hb_grid = gaussian_half_band(t_grid, base_hb, extra_hb, sigma)
    snow_boundary = 0.5 + hb_grid
    rain_boundary = 0.5 - hb_grid

    df_combo = pd.concat([
        val_full_df.assign(split="val"),
        test_full_df.assign(split="test"),
    ], ignore_index=True)
    mix_mask = df_combo["phase_full"] == MIX_CODE
    df_mix   = df_combo[mix_mask].copy()
    if len(df_mix) > 0:
        hb_obs = gaussian_half_band(df_mix["temp_wet"].to_numpy(), base_hb, extra_hb, sigma)
        df_mix["inside_band"] = (
            (df_mix["p_snow_cal"] > 0.5 - hb_obs) &
            (df_mix["p_snow_cal"] < 0.5 + hb_obs)
        )
        capture_pct = 100 * df_mix["inside_band"].mean()
    else:
        capture_pct = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(f"{name} — Story 4: Uncertainty band placement", fontsize=12, y=1.01)

    # Panel A
    ax_a = axes[0]
    ax_a.fill_between(t_grid, rain_boundary, snow_boundary, alpha=0.10, color="orange")
    ax_a.plot(t_grid, snow_boundary, color="black", lw=2, ls="-",  label="Snow threshold")
    ax_a.plot(t_grid, rain_boundary, color="black", lw=2, ls="--", label="Rain threshold")
    if len(df_mix) > 0:
        for flag, label, color, marker in [
            (False, "Missed", "#cc3311", "x"),
            (True,  "Captured", "#009988", "o"),
        ]:
            sub = df_mix[df_mix["inside_band"] == flag]
            ax_a.scatter(sub["temp_wet"], sub["p_snow_cal"], c=color, marker=marker,
                         s=40, alpha=0.65, linewidths=1.0,
                         label=f"{label} (n={len(sub)})", zorder=3+flag)
    ax_a.axvline(0, color="grey", lw=0.8, alpha=0.4)
    ax_a.axhline(0.5, color="grey", lw=0.8, alpha=0.4)
    ax_a.set(xlim=(-6,6), ylim=(-0.04,1.04),
             xlabel="T_wet (°C)", ylabel="Calibrated p(snow)",
             title=f"A — True-mix band capture (n={len(df_mix)}, {capture_pct:.0f}% inside)")
    ax_a.legend(fontsize=9)

    # Panel B — violin by phase across 2°C bins
    ax_b = axes[1]
    violin_bins = [(-6,-4),(-4,-2),(-2,0),(0,2),(2,4),(4,6)]
    bin_labels  = [f"{lo}–{hi}" for lo,hi in violin_bins]
    for b_idx, (lo, hi) in enumerate(violin_bins):
        mask_bin = (df_combo["temp_wet"] >= lo) & (df_combo["temp_wet"] < hi)
        sub_bin  = df_combo[mask_bin]
        for p_idx, (code, cname) in enumerate(zip(
                [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES)):
            vals = sub_bin.loc[sub_bin["phase_full"]==code, "p_snow_cal"].to_numpy()
            x_pos = b_idx + (p_idx - 1) * 0.27
            if len(vals) < 4:
                if len(vals) > 0:
                    ax_b.plot(x_pos, np.median(vals), "_",
                              color=PHASE_COLORS[cname], ms=10, mew=2)
                continue
            parts = ax_b.violinplot(vals, positions=[x_pos], widths=0.23,
                                     showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(PHASE_COLORS[cname]); pc.set_edgecolor("none"); pc.set_alpha(0.7)
            parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    ax_b.axhspan(rain_boundary.min(), snow_boundary.max(), alpha=0.06, color="orange")
    ax_b.axhline(0.5, color="grey", lw=0.8, alpha=0.4)
    ax_b.set_xticks(range(len(violin_bins))); ax_b.set_xticklabels(bin_labels, fontsize=8)
    ax_b.set(xlabel="T_wet bin (°C)", ylabel="Calibrated p(snow)", ylim=(-0.04,1.04),
             title="B — p(snow) by true phase (2°C bins, val+test)")
    legend_els = [Patch(facecolor=PHASE_COLORS[c], label=c, alpha=0.75) for c in FULL_CLASS_NAMES]
    ax_b.legend(handles=legend_els, fontsize=9)
    plt.tight_layout()
    fig.savefig(graphics_dir / "story4_band_placement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 5 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{name} — Story 5: Near-freezing deep-dive", fontsize=12)
    for (df_f, y_true_f, pred_f, split, flag_col, flag_label), (r, c) in zip([
        (val_full_df,  y_val_full_phase,  pred_val_full,  "Val",  "near_freezing_air_2C", "|T_air| ≤ 2°C"),
        (val_full_df,  y_val_full_phase,  pred_val_full,  "Val",  "near_freezing_wet_2C", "|T_wet| ≤ 2°C"),
        (test_full_df, y_test_full_phase, pred_test_full, "Test", "near_freezing_air_2C", "|T_air| ≤ 2°C"),
        (test_full_df, y_test_full_phase, pred_test_full, "Test", "near_freezing_wet_2C", "|T_wet| ≤ 2°C"),
    ], [(0,0),(0,1),(1,0),(1,1)]):
        ax = axes[r, c]
        # Derive the mask from temp columns since flag columns may not exist
        if flag_col == "near_freezing_air_2C":
            if "temp_air" in df_f.columns:
                mask = np.abs(df_f["temp_air"].to_numpy()) <= 2.0
            else:
                ax.set_visible(False); continue
        else:
            mask = np.abs(df_f["temp_wet"].to_numpy()) <= 2.0
        if mask.sum() < 5:
            ax.set_visible(False); continue
        _norm_cm(ax, y_true_f[mask], pred_f[mask],
                 [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES,
                 f"{split} — {flag_label} (n={mask.sum()})")
    plt.tight_layout()
    fig.savefig(graphics_dir / "story5_near_freezing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Stories saved to: {graphics_dir}")

# =============================================================================
# 4.  ONE-TIME DATA LOAD
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
# 5.  ONE-TIME SAMPLING
# =============================================================================

def get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times) -> pd.DataFrame:
    """
    Sample the full-feature predictor cube to MRoS points once and cache to
    parquet.  All ablation runs select columns from this cached table —
    the gridded resampling never runs again after the first call.
    """
    cache_path = SETUP_DIR / f"ml_input_points_{INTERP_TYPE}_binary_uncertainty_random_split.parquet"

    if cache_path.exists():
        print(f"\nONE-TIME SAMPLING: Cache found — loading\n  {cache_path}")
        master_df = pd.read_parquet(cache_path)
        print(f"  Shape: {master_df.shape}")
        return master_df

    print("\nONE-TIME SAMPLING: No cache found — sampling predictor cube now...")
    print("  (Runs once; cached for all subsequent ablation experiments)")

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
# 6.  SHARED TRAIN / VAL / TEST SPLIT
# =============================================================================

def make_split(master_df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 split, computed once and shared across all runs."""
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
# 7.  run_experiment()
# =============================================================================

def build_feature_list(cfg: dict) -> list[str]:
    grid = [n for flag, n in [
        (cfg.get("use_temp_air",  BASE_FEATURE_CONFIG["use_temp_air"]),  "temp_air"),
        (cfg.get("use_temp_dew",  BASE_FEATURE_CONFIG["use_temp_dew"]),  "temp_dew"),
        (cfg.get("use_temp_wet",  BASE_FEATURE_CONFIG["use_temp_wet"]),  "temp_wet"),
        (cfg.get("use_rh",        BASE_FEATURE_CONFIG["use_rh"]),        "rh"),
        (cfg.get("use_imerg_plp", BASE_FEATURE_CONFIG["use_imerg_plp"]), "imerg_plp"),
        (cfg.get("use_elev",      BASE_FEATURE_CONFIG["use_elev"]),      "elev"),
    ] if flag]
    if cfg.get("use_mros_loocv", BASE_FEATURE_CONFIG["use_mros_loocv"]):
        grid.extend(ALL_LOOCV_FEATURES)
    return grid


def run_experiment(cfg: dict, split_df: pd.DataFrame, out_dir: Path) -> dict:
    name = cfg.get("name", "unnamed")
    print(f"\n{'='*62}\n  EXPERIMENT: {name}\n{'='*62}")

    out_dir.mkdir(parents=True, exist_ok=True)
    graphics_dir = out_dir / "graphics"
    graphics_dir.mkdir(exist_ok=True)

    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=2)

    FEATURES = build_feature_list(cfg)

    missing = [f for f in FEATURES if f not in split_df.columns]
    if missing:
        print(f"  SKIP — missing features: {missing}")
        return {"name": name, "status": "skipped", "missing_features": missing}

    TARGET_FULL = "phase_full"
    train_full  = split_df[split_df["split"] == "train"].copy()
    val_full    = split_df[split_df["split"] == "val"].copy()
    test_full   = split_df[split_df["split"] == "test"].copy()

    pure = [SNOW_CODE, RAIN_CODE]
    train_fit = train_full[train_full[TARGET_FULL].isin(pure)].copy()
    val_fit   = val_full[val_full[TARGET_FULL].isin(pure)].copy()
    test_fit  = test_full[test_full[TARGET_FULL].isin(pure)].copy()

    X_train     = train_fit[FEATURES]
    X_val       = val_fit[FEATURES]
    X_test      = test_fit[FEATURES]
    X_val_full  = val_full[FEATURES]
    X_test_full = test_full[FEATURES]

    y_train = train_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)
    y_val   = val_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)
    y_test  = test_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)

    y_val_full_phase  = val_full[TARGET_FULL].astype(int).to_numpy()
    y_test_full_phase = test_full[TARGET_FULL].astype(int).to_numpy()
    y_val_bin         = y_val.to_numpy()
    y_test_bin        = y_test.to_numpy()

    # ── scale_pos_weight sweep ────────────────────────────────────────────────
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    best_spw, spw_df = sweep_scale_pos_weight(
        X_train, y_train, X_val, y_val,
        SCALE_POS_WEIGHT_GRID, BASE_XGB_PARAMS,
        probe_rounds=350, early_stopping=30, random_seed=RANDOM_SEED,
    )
    spw_df.to_csv(out_dir / "scale_pos_weight_sweep.csv", index=False)
    print(f"  Best scale_pos_weight: {best_spw}")

    # ── Full model fit ────────────────────────────────────────────────────────
    params   = {**BASE_XGB_PARAMS, "scale_pos_weight": best_spw, "seed": RANDOM_SEED}
    dtrain   = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval_dm  = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURES)
    evals_result = {}
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain,"train"),(dval_dm,"val")], evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=200,
    )
    booster.save_model(out_dir / "xgb_binary_phase_model.bin")
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evals_result["train"]["logloss"], label="train")
    ax.plot(evals_result["val"]["logloss"],   label="val")
    ax.axvline(booster.best_iteration, ls="--", alpha=0.6, label=f"best={booster.best_iteration}")
    ax.set(xlabel="Iteration", ylabel="Logloss", title=f"{name} — training curve")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(graphics_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Beta calibration ──────────────────────────────────────────────────────
    trainval_pure = pd.concat([train_fit, val_fit]).reset_index(drop=True)
    tw_tv         = trainval_pure["temp_wet"].values
    p_raw_tv      = booster.predict(xgb.DMatrix(trainval_pure[FEATURES], feature_names=FEATURES))
    y_tv          = trainval_pure[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)

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

    # ── Probabilities ─────────────────────────────────────────────────────────
    p_val_raw   = booster.predict(xgb.DMatrix(X_val,       feature_names=FEATURES))
    p_test_raw  = booster.predict(xgb.DMatrix(X_test,      feature_names=FEATURES))
    p_valf_raw  = booster.predict(xgb.DMatrix(X_val_full,  feature_names=FEATURES))
    p_testf_raw = booster.predict(xgb.DMatrix(X_test_full, feature_names=FEATURES))

    p_val_cal   = calibrate(p_val_raw,   val_fit["temp_wet"].to_numpy())
    p_test_cal  = calibrate(p_test_raw,  test_fit["temp_wet"].to_numpy())
    p_valf_cal  = calibrate(p_valf_raw,  val_full["temp_wet"].to_numpy())
    p_testf_cal = calibrate(p_testf_raw, test_full["temp_wet"].to_numpy())

    # ── Gaussian half-band ────────────────────────────────────────────────────
    thr = optimize_threshold_params(p_valf_cal, val_full["temp_wet"].to_numpy(), y_val_full_phase)
    BASE_HB  = thr["base_half_band"]
    EXTRA_HB = thr["extra_half_band"]
    SIGMA    = thr["sigma"]

    def classify(p_snow, twet):
        return classify_phase_gaussian_band(p_snow, twet, BASE_HB, EXTRA_HB, SIGMA)

    pred_val_full  = classify(p_valf_cal,  val_full["temp_wet"].to_numpy())
    pred_test_full = classify(p_testf_cal, test_full["temp_wet"].to_numpy())
    pred_val_bin05  = np.where(p_val_cal  >= 0.5, SNOW_CODE, RAIN_CODE)
    pred_test_bin05 = np.where(p_test_cal >= 0.5, SNOW_CODE, RAIN_CODE)

    # ── Score distribution plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, p_raw, p_cal, split_label in [
        (axes[0], p_val_raw,  p_val_cal,  "Val"),
        (axes[1], p_test_raw, p_test_cal, "Test"),
    ]:
        ax.hist(p_raw, bins=40, alpha=0.55, color="grey",    label="raw",        density=True)
        ax.hist(p_cal, bins=40, alpha=0.55, color="#3a86ff", label="calibrated", density=True)
        ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set(xlabel="p(snow)", ylabel="Density", title=f"{name} — {split_label} score dist.")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphics_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _safe(fn, *a, **kw):
        try:    return round(float(fn(*a, **kw)), 4)
        except: return None

    metrics = {
        "name": name, "features": FEATURES, "n_features": len(FEATURES),
        "best_iteration": int(booster.best_iteration), "best_spw": float(best_spw),
        "band_base_hb": float(BASE_HB), "band_extra_hb": float(EXTRA_HB), "band_sigma": float(SIGMA),
        # probability
        "val_roc_auc_cal":  _safe(roc_auc_score, y_val_bin,  p_val_cal),
        "test_roc_auc_cal": _safe(roc_auc_score, y_test_bin, p_test_cal),
        "val_brier_cal":    _safe(brier_score_loss, y_val_bin,  p_val_cal),
        "test_brier_cal":   _safe(brier_score_loss, y_test_bin, p_test_cal),
        "val_logloss_cal":  _safe(log_loss, y_val_bin,  p_val_cal, labels=[0,1]),
        "test_logloss_cal": _safe(log_loss, y_test_bin, p_test_cal, labels=[0,1]),
        "val_ece":          expected_calibration_error(y_val_bin,  p_val_cal),
        "test_ece":         expected_calibration_error(y_test_bin, p_test_cal),
        # hard metrics
        "val_macro_f1_binary":      _safe(f1_score, val_fit[TARGET_FULL].to_numpy(), pred_val_bin05,  average="macro", zero_division=0),
        "test_macro_f1_binary":     _safe(f1_score, test_fit[TARGET_FULL].to_numpy(), pred_test_bin05, average="macro", zero_division=0),
        "val_macro_f1_3class":      _safe(f1_score, y_val_full_phase,  pred_val_full,  average="macro", zero_division=0),
        "test_macro_f1_3class":     _safe(f1_score, y_test_full_phase, pred_test_full, average="macro", zero_division=0),
        "val_balanced_acc_3class":  _safe(balanced_accuracy_score, y_val_full_phase,  pred_val_full),
        "test_balanced_acc_3class": _safe(balanced_accuracy_score, y_test_full_phase, pred_test_full),
        # mix capture
        "val_mix_capture":  float(np.mean(pred_val_full[y_val_full_phase   == MIX_CODE] == MIX_CODE)) if (y_val_full_phase  == MIX_CODE).any() else None,
        "test_mix_capture": float(np.mean(pred_test_full[y_test_full_phase == MIX_CODE] == MIX_CODE)) if (y_test_full_phase == MIX_CODE).any() else None,
        # confidence fractions
        "test_frac_pred_snow": round(float(np.mean(pred_test_full == SNOW_CODE)), 4),
        "test_frac_pred_rain": round(float(np.mean(pred_test_full == RAIN_CODE)), 4),
        "test_frac_pred_mix":  round(float(np.mean(pred_test_full == MIX_CODE)),  4),
        # near-freezing regime
        "status": "ok",
    }

    # near-freezing regime metrics
    for split_name, df_full, p_cal_full, y_full, df_fit, p_cal_fit, y_fit_bin in [
        ("val",  val_full,  p_valf_cal,  y_val_full_phase,  val_fit,  p_val_cal,  y_val_bin),
        ("test", test_full, p_testf_cal, y_test_full_phase, test_fit, p_test_cal, y_test_bin),
    ]:
        for regime in ["nearfreeze", "clearphase"]:
            mf_full = (np.abs(df_full["temp_wet"].to_numpy()) <= 2.0) if regime == "nearfreeze" \
                      else (np.abs(df_full["temp_wet"].to_numpy()) > 2.0)
            mf_fit  = (np.abs(df_fit["temp_wet"].to_numpy())  <= 2.0) if regime == "nearfreeze" \
                      else (np.abs(df_fit["temp_wet"].to_numpy())  > 2.0)
            if mf_full.sum() >= 5:
                pred_r = classify(p_cal_full[mf_full], df_full["temp_wet"].to_numpy()[mf_full])
                metrics[f"{split_name}_{regime}_macro_f1_3class"] = _safe(
                    f1_score, y_full[mf_full], pred_r, average="macro", zero_division=0)
                if (y_full[mf_full] == MIX_CODE).any():
                    metrics[f"{split_name}_{regime}_mix_capture"] = round(
                        float(np.mean(pred_r[y_full[mf_full] == MIX_CODE] == MIX_CODE)), 4)
            if mf_fit.sum() >= 5:
                metrics[f"{split_name}_{regime}_roc_auc"] = _safe(
                    roc_auc_score, y_fit_bin[mf_fit], p_cal_fit[mf_fit])

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Feature importance ────────────────────────────────────────────────────
    imp_gain   = booster.get_score(importance_type="gain")
    imp_weight = booster.get_score(importance_type="weight")
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "gain":    [imp_gain.get(f,   0.0) for f in FEATURES],
        "weight":  [imp_weight.get(f, 0.0) for f in FEATURES],
    }).sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES)*0.5)))
    ax.barh(imp_df["feature"], imp_df["gain"], color="steelblue")
    ax.set(xlabel="Gain", title=f"{name} — XGBoost feature importance")
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(graphics_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("  Computing SHAP values...")

    # Build combined table (all splits, full phase set)
    df_all_full = pd.concat([
        train_full.assign(split="train"),
        val_full.assign(split="val"),
        test_full.assign(split="test"),
    ], ignore_index=True)

    p_all_raw = booster.predict(xgb.DMatrix(df_all_full[FEATURES], feature_names=FEATURES))
    p_all_cal = calibrate(p_all_raw, df_all_full["temp_wet"].to_numpy())
    pred_phase_all = classify(p_all_cal, df_all_full["temp_wet"].to_numpy())

    # from plot_mros_loocv_conflict import make_conflict_figure
    # make_conflict_figure(
    #     df_all=df_all_full,
    #     p_snow_cal=p_all_cal,
    #     p_snow_raw=p_all_raw,
    #     out_path=out_dir / "graphics" / "mros_loocv_conflict_diagnostic.png",
    #     experiment_name=name,
    # )
    
    # Stratified background sample from train split
    train_rows = df_all_full[df_all_full["split"] == "train"]
    N_BG = 500
    background = (
        train_rows[FEATURES]
        .groupby(train_rows["phase_full"], group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), max(1, int(N_BG * len(g) / len(train_rows)))),
            random_state=RANDOM_SEED,
        ))
        .reset_index(drop=True)
    )

    explainer   = shap.TreeExplainer(booster, data=background,
                                      feature_perturbation="interventional",
                                      model_output="probability")
    shap_values = explainer.shap_values(df_all_full[FEATURES])

    # Build shap_df
    meta_cols = [c for c in ["time","x","y","phase_full","temp_wet","temp_air","temp_dew","elev"]
                if c in df_all_full.columns]
    shap_df = df_all_full[meta_cols].copy()
    shap_df["p_snow_cal"]                   = p_all_cal
    shap_df["p_snow_raw"]                   = p_all_raw
    shap_df["split"]      = df_all_full["split"].values
    shap_df["prediction_phase_uncertainty"] = pred_phase_all
    shap_df["phase_label"] = shap_df["prediction_phase_uncertainty"].map(
        {SNOW_CODE:"snow", RAIN_CODE:"rain", MIX_CODE:"mix"})
    shap_cols = [f"shap_{f}" for f in FEATURES]
    for i, feat in enumerate(FEATURES):
        shap_df[f"shap_{feat}"] = shap_values[:, i]

    shap_df.to_parquet(out_dir / "shap_values_all.parquet")

    # Run this immediately after run_experiment() returns, while
    low_psnow_mask = (p_all_cal < 0.05) & (df_all_full["phase_full"] == SNOW_CODE)
    low_psnow = df_all_full[low_psnow_mask].copy()
    low_psnow["p_snow_cal"] = p_all_cal[low_psnow_mask]

    print(f"n = {len(low_psnow)}  ({100*len(low_psnow)/len(df_all_full):.1f}% of all obs)")
    print(f"\nSplit breakdown:")
    print(low_psnow["split"].value_counts())

    print(f"\nThermodynamic fields:")
    print(low_psnow[["temp_wet", "temp_air", "temp_dew", "elev"]].describe().round(2))

    print(f"\nLOOCV predictors:")
    print(low_psnow[["mros_p_snow_loocv", "mros_p_rain_loocv", "mros_p_mix_loocv"]].describe().round(3))

    if "imerg_plp" in low_psnow.columns:
        print(f"\nIMERG PLP: mean={low_psnow['imerg_plp'].mean():.3f}, "
            f"median={low_psnow['imerg_plp'].median():.3f}")

    # Compare to the full snow population
    all_snow = df_all_full[df_all_full["phase_full"] == SNOW_CODE]
    print(f"\n--- Contrast: full snow population (n={len(all_snow)}) ---")
    print(all_snow[["temp_wet", "elev", "mros_p_snow_loocv", "mros_p_rain_loocv"]].describe().round(3))
    
    # Are these concentrated at specific stations / locations?
    print("Spatial clustering:")
    print(low_psnow[["x", "y", "elev"]].describe().round(1))

    # Are they concentrated in certain months / storm types?
    low_psnow["month"] = pd.to_datetime(low_psnow["time"]).dt.month
    print("\nMonthly distribution:")
    print(low_psnow["month"].value_counts().sort_index())

    # How does the LOOCV conflict look — is mros_p_rain_loocv consistently near 1?
    print("\nLOOCV conflict severity:")
    print((low_psnow["mros_p_rain_loocv"] > 0.8).sum(), 
        "of 42 have mros_p_rain_loocv > 0.8")
    print((low_psnow["mros_p_rain_loocv"] > 0.9).sum(),
        "of 42 have mros_p_rain_loocv > 0.9")

    # Is there a station_id column that could reveal if it's one or two observers?
    if "station_id" in low_psnow.columns:
        print("\nUnique stations:", low_psnow["station_id"].nunique())
        print(low_psnow["station_id"].value_counts())

    # ── SHAP plot 1: mean |SHAP| by phase (bar chart) ─────────────────────────
    phase_shap = (
        shap_df.groupby("phase_label")[shap_cols]
        .apply(lambda d: d.abs().mean())
        .T
        .rename(index=lambda c: c.replace("shap_", ""))
        .rename_axis("feature")
        .reset_index()
    )
    phase_shap["all"] = pd.DataFrame(np.abs(shap_values), columns=FEATURES).mean(axis=0).values
    phase_shap = phase_shap.sort_values("all", ascending=False)
    phase_shap.to_csv(out_dir / "shap_summary_by_phase.csv", index=False)

    phase_plot_cols = [c for c in ["snow","rain","mix"] if c in phase_shap.columns]
    x = np.arange(len(FEATURES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, color) in enumerate(zip(phase_plot_cols,
                                          [PHASE_COLORS["snow"], PHASE_COLORS["rain"], PHASE_COLORS["mix"]])):
        vals = phase_shap.set_index("feature").reindex(FEATURES)[col].values
        ax.bar(x + i*width, vals, width, label=col, color=color, alpha=0.85)
    ax.set_xticks(x + width); ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set(ylabel="Mean |SHAP| (probability units)",
           title=f"{name} — feature importance by predicted phase")
    ax.legend(title="Predicted phase"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(graphics_dir / "shap_mean_by_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── SHAP plot 2: top-N features across T_wet bins (line plot) ─────────────
    shap_df["twet_bin"] = pd.cut(shap_df["temp_wet"], bins=TWET_BIN_EDGES,
                                  labels=TWET_BIN_LABELS, right=True)
    shap_by_bin = (
        shap_df.groupby("twet_bin", observed=True)[shap_cols]
        .apply(lambda d: d.abs().mean())
        .T
        .rename(index=lambda c: c.replace("shap_", ""))
        .rename_axis("feature")
    )
    shap_by_bin.to_csv(out_dir / "shap_by_wetbulb_bin.csv")

    # Rank by near-freezing importance
    nf_mask_shap = shap_df["twet_bin"].isin(NF_BIN_LABELS)
    top_feats = (
        shap_df[nf_mask_shap][shap_cols].abs().mean()
        .rename(index=lambda c: c.replace("shap_", ""))
        .nlargest(TOP_N_SHAP_FEATURES).index.tolist()
    )
    valid_bins = [b for b in TWET_BIN_LABELS if b in shap_by_bin.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    for feat, color in zip(top_feats, plt.cm.tab10(np.linspace(0, 0.9, len(top_feats)))):
        if feat not in shap_by_bin.index:
            continue
        ax.plot(valid_bins, shap_by_bin.loc[feat, valid_bins].values.astype(float),
                marker="o", label=feat, color=color, linewidth=1.8)
    nf_idx = [valid_bins.index(b) for b in NF_BIN_LABELS if b in valid_bins]
    if nf_idx:
        ax.axvspan(min(nf_idx)-0.5, max(nf_idx)+0.5, alpha=0.10,
                   color="steelblue", label="|T_wet| ≤ 2°C zone")
    ax.set(xlabel="Wet-bulb temperature bin (°C)", ylabel="Mean |SHAP|",
           title=f"{name} — top {TOP_N_SHAP_FEATURES} features across T_wet bins")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right"); fig.tight_layout()
    fig.savefig(graphics_dir / "shap_wetbulb_lineplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def plot_calibration_detail(y_true_bin, p_cal, name, out_path, n_bins=15):
        """
        Three-panel calibration deep-dive:
        Left:   reliability diagram with bin-level ECE contribution
        Centre: histogram of predicted probabilities (sharpness)
        Right:  ECE contribution per bin (which bins hurt most)
        """
        bins      = np.linspace(0, 1, n_bins + 1)
        bin_accs  = []
        bin_confs = []
        bin_ns    = []
        bin_eces  = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_cal >= lo) & (p_cal < hi)
            n    = mask.sum()
            if n == 0:
                bin_accs.append(np.nan); bin_confs.append((lo+hi)/2)
                bin_ns.append(0);        bin_eces.append(0.0)
                continue
            acc  = float(np.mean((p_cal[mask] >= 0.5).astype(int) == y_true_bin[mask]))
            conf = float(np.mean(p_cal[mask]))
            bin_accs.append(acc); bin_confs.append(conf)
            bin_ns.append(n);     bin_eces.append(n * abs(acc - conf) / len(y_true_bin))

        bin_mids = [(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{name} — calibration deep-dive (test set)", fontsize=12)

        # Panel 1: reliability diagram, points sized by n
        ax = axes[0]
        ax.plot([0,1],[0,1],"--", color="grey", lw=1.2, alpha=0.6, label="Perfect")
        sizes = [max(20, n/max(bin_ns)*400) for n in bin_ns]
        sc = ax.scatter(bin_confs, bin_accs, s=sizes, c=bin_eces,
                        cmap="YlOrRd", vmin=0, vmax=max(bin_eces or [0.01]),
                        zorder=3, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="ECE contribution")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean predicted p(snow)",
            ylabel="Observed fraction snow", title="Reliability diagram\n(dot size = n obs, colour = ECE contribution)")
        ax.grid(alpha=0.25)

        # Panel 2: sharpness histogram
        ax = axes[1]
        ax.hist(p_cal, bins=30, color="#3a86ff", alpha=0.7, edgecolor="none")
        ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
        # shade near-freezing band if band params are accessible via closure
        ax.set(xlabel="Calibrated p(snow)", ylabel="Count",
            title="Sharpness (predicted probability distribution)")
        ax.grid(alpha=0.25)

        # Panel 3: ECE contribution by bin
        ax = axes[2]
        bar_colors = plt.cm.YlOrRd(
            np.array(bin_eces) / max(max(bin_eces), 1e-6)
        )
        ax.bar(bin_mids, bin_eces, width=(bins[1]-bins[0])*0.85,
            color=bar_colors, edgecolor="none")
        ax.set(xlabel="Predicted p(snow) bin", ylabel="ECE contribution",
            title="ECE contribution by probability bin\n(which bins drive miscalibration)")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_calibration_detail(
        y_test_bin, p_test_cal, name,
        graphics_dir / "calibration_detail.png",
    )

    # Store top SHAP features in metrics for comparison table
    metrics["shap_top_feature_overall"]    = str(phase_shap.iloc[0]["feature"]) if not phase_shap.empty else ""
    metrics["shap_top_feature_nearfreeze"] = str(top_feats[0]) if top_feats else ""

    print(f"  test ROC AUC (cal):      {metrics['test_roc_auc_cal']}")
    print(f"  test macro F1 (3-class): {metrics['test_macro_f1_3class']}")
    print(f"  test mix capture:        {metrics['test_mix_capture']}")
    print(f"  test ECE:                {metrics['test_ece']}")
    print(f"  SHAP top (overall):      {metrics['shap_top_feature_overall']}")
    print(f"  SHAP top (near-freeze):  {metrics['shap_top_feature_nearfreeze']}")
    print(f"  Saved to: {out_dir}")

    # ── Per-experiment story plots ──────────────────────
    # Attach p_snow_cal to full-split DataFrames so story helper is self-contained
    _val_full_story  = val_full.copy();  _val_full_story["p_snow_cal"]  = p_valf_cal
    _test_full_story = test_full.copy(); _test_full_story["p_snow_cal"] = p_testf_cal
    plot_per_experiment_stories(
        name           = name,
        graphics_dir   = graphics_dir,
        y_val_fit_phase   = val_fit[TARGET_FULL].to_numpy(),
        pred_val_bin05    = pred_val_bin05,
        y_test_fit_phase  = test_fit[TARGET_FULL].to_numpy(),
        pred_test_bin05   = pred_test_bin05,
        y_val_full_phase  = y_val_full_phase,
        pred_val_full     = pred_val_full,
        y_test_full_phase = y_test_full_phase,
        pred_test_full    = pred_test_full,
        y_val_bin   = y_val_bin,
        p_val_cal   = p_val_cal,
        y_test_bin  = y_test_bin,
        p_test_cal  = p_test_cal,
        p_val_raw   = p_val_raw,
        p_test_raw  = p_test_raw,
        val_full_df  = _val_full_story,
        test_full_df = _test_full_story,
        base_hb      = BASE_HB,
        extra_hb     = EXTRA_HB,
        sigma        = SIGMA,
    )

    return metrics

# =============================================================================
# 8.  ABLATION LOOP + CROSS-EXPERIMENT PLOTS
# =============================================================================

def load_all_metrics(ablation_root: Path) -> list[dict]:
    """
    Scan all subfolders of ablation_root for metrics_summary.json
    and return a list of metric dicts.  Subfolders without a
    metrics_summary.json are silently skipped.
    """
    all_metrics = []
    for run_dir in sorted(ablation_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        all_metrics.append(m)
        print(f"  Loaded: {run_dir.name}  (status={m.get('status','?')})")
    return all_metrics


def save_cross_experiment_plots(all_metrics: list[dict], ablation_root: Path) -> None:
    """
    Generate and save all cross-experiment comparison outputs.
    Can be called after a fresh run or standalone via --replot.
    """
    comparison_cols = [
        "name", "n_features", "status",
        "test_roc_auc_cal", "test_ece", "test_brier_cal", "test_logloss_cal",
        "test_macro_f1_binary", "test_macro_f1_3class", "test_balanced_acc_3class",
        "test_mix_capture", "test_frac_pred_mix",
        "test_nearfreeze_macro_f1_3class", "test_nearfreeze_mix_capture",
        "test_nearfreeze_roc_auc", "test_clearphase_macro_f1_3class",
        "val_roc_auc_cal", "val_macro_f1_3class",
        "shap_top_feature_overall", "shap_top_feature_nearfreeze",
    ]
    comparison_df = pd.DataFrame(all_metrics)
    present_cols  = [c for c in comparison_cols if c in comparison_df.columns]
    comparison_df[present_cols].to_csv(ablation_root / "ablation_comparison.csv", index=False)
    print(f"\nComparison table saved ({len(comparison_df)} runs).")
    print(comparison_df[present_cols].to_string(index=False))

    ok = comparison_df[comparison_df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful runs found — skipping plots.")
        return

    # ── Bar chart: F1 and ROC AUC ─────────────────────────────────────────────
    ok_sorted = ok.sort_values("test_macro_f1_3class", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ok)*0.45)))
    for ax, col, title in [
        (axes[0], "test_macro_f1_3class", "Test macro F1 (3-class)"),
        (axes[1], "test_roc_auc_cal",     "Test ROC AUC (calibrated)"),
    ]:
        if col not in ok_sorted.columns:
            continue
        colors = ["#2dc653" if r["name"] == "baseline_full" else "#3a86ff"
                  for _, r in ok_sorted.iterrows()]
        ax.barh(ok_sorted["name"], ok_sorted[col], color=colors)
        ax.set(xlabel=title, title=title)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Ablation comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(ablation_root / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ECE bar chart ─────────────────────────────────────────────────────────
    if "test_ece" in ok.columns:
        ok_ece = ok.sort_values("test_ece")
        fig, ax = plt.subplots(figsize=(8, max(4, len(ok)*0.45)))
        colors  = ["#2dc653" if r["name"] == "baseline_full" else "#e07b39"
                   for _, r in ok_ece.iterrows()]
        ax.barh(ok_ece["name"], ok_ece["test_ece"], color=colors)
        ax.set(xlabel="Expected Calibration Error (lower = better)",
               title="Ablation — test ECE")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ablation_root / "ablation_calibration_ece.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Near-freeze vs clear-phase F1 scatter ─────────────────────────────────
    nf_col = "test_nearfreeze_macro_f1_3class"
    cp_col = "test_clearphase_macro_f1_3class"
    if nf_col in ok.columns and cp_col in ok.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_ok = ok.dropna(subset=[nf_col, cp_col])
        sc = ax.scatter(scatter_ok[cp_col], scatter_ok[nf_col],
                        c=scatter_ok["test_roc_auc_cal"],
                        cmap="viridis", s=80, zorder=3,
                        vmin=scatter_ok["test_roc_auc_cal"].min(),
                        vmax=scatter_ok["test_roc_auc_cal"].max())
        plt.colorbar(sc, ax=ax, label="Test ROC AUC (cal)")
        for _, row in scatter_ok.iterrows():
            ax.annotate(row["name"], (row[cp_col], row[nf_col]),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.plot([0,1],[0,1],"--", color="grey", lw=1, alpha=0.5)
        ax.set(xlabel="Clear-phase macro F1", ylabel="Near-freeze macro F1",
               title="Near-freeze vs. clear-phase F1\n(diagonal = equal performance in both regimes)")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(ablation_root / "ablation_nearfreeze_vs_clearphase.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Mix capture vs F1 scatter ─────────────────────────────────────────────
    if "test_mix_capture" in ok.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_ok = ok.dropna(subset=["test_mix_capture", "test_macro_f1_3class"])
        sc = ax.scatter(scatter_ok["test_mix_capture"], scatter_ok["test_macro_f1_3class"],
                        c=scatter_ok["test_frac_pred_mix"] if "test_frac_pred_mix" in scatter_ok.columns else "steelblue",
                        cmap="plasma", s=80, zorder=3)
        if "test_frac_pred_mix" in scatter_ok.columns:
            plt.colorbar(sc, ax=ax, label="Fraction predicted as mix")
        for _, row in scatter_ok.iterrows():
            ax.annotate(row["name"], (row["test_mix_capture"], row["test_macro_f1_3class"]),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.set(xlabel="Mix capture rate (true mix → predicted mix)",
               ylabel="Test macro F1 (3-class)",
               title="Mix capture vs. overall F1 trade-off\n(colour = fraction of all predictions assigned to mix)")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(ablation_root / "ablation_mix_tradeoff.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Parallel coordinates ──────────────────────────────────────────────────
    radar_cols = [
        "test_roc_auc_cal", "test_macro_f1_3class",
        "test_nearfreeze_macro_f1_3class", "test_mix_capture", "test_frac_pred_mix",
    ]
    plot_cols = [c for c in radar_cols if c in ok.columns and ok[c].notna().any()]
    if plot_cols:
        ok_norm = ok[["name"] + plot_cols].copy()
        for col in plot_cols:
            col_min = ok_norm[col].min(); col_max = ok_norm[col].max()
            if col_max > col_min:
                ok_norm[col] = (ok_norm[col] - col_min) / (col_max - col_min)
        fig, ax = plt.subplots(figsize=(12, 5))
        parallel_coordinates(ok_norm, "name", colormap="tab20", ax=ax, alpha=0.75)
        ax.set_xticklabels(plot_cols, rotation=20, ha="right", fontsize=9)
        ax.set_title("Ablation — normalised metrics (parallel coordinates)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.25); fig.tight_layout()
        fig.savefig(ablation_root / "ablation_parallel_coords.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll cross-experiment plots saved to: {ablation_root}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated indices of ABLATION_CONFIGS to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment list without running")
    parser.add_argument("--replot", action="store_true",
                        help="Skip training; reload all existing metrics_summary.json "
                             "files from subfolders and regenerate cross-experiment plots")
    args = parser.parse_args()

    if args.replot:
        print(f"\nREPLOT MODE — scanning {ABLATION_ROOT} for completed runs...")
        all_metrics = load_all_metrics(ABLATION_ROOT)
        if not all_metrics:
            print("No metrics_summary.json files found. Run experiments first.")
            return
        save_cross_experiment_plots(all_metrics, ABLATION_ROOT)
        return

    selected_idx = [int(x) for x in args.configs.split(",") if x.strip().isdigit()] \
                   if args.configs else None
    configs_to_run = ([ABLATION_CONFIGS[i] for i in selected_idx if i < len(ABLATION_CONFIGS)]
                      if selected_idx else ABLATION_CONFIGS)

    print(f"\nABLATION STUDY — {len(configs_to_run)} experiment(s)")
    print(f"  Region: {REGION}  |  Interp: {INTERP_TYPE}  |  Output: {ABLATION_ROOT}\n")
    for i, cfg in enumerate(configs_to_run):
        print(f"  [{i:2d}] {cfg.get('name','unnamed'):30s}  "
              f"{{{', '.join(f'{k}={v}' for k,v in cfg.items() if k!='name')}}}")

    if args.dry_run:
        return

    ds_interp, ds_imerg, df_loocv_raw, common_times = load_and_sync_datasets()
    master_df = get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times)
    split_df  = make_split(master_df)

    all_metrics = []
    for cfg in configs_to_run:
        full_cfg = {**BASE_FEATURE_CONFIG, **cfg}
        try:
            m = run_experiment(full_cfg, split_df, ABLATION_ROOT / full_cfg.get("name","unnamed"))
        except Exception as exc:
            print(f"  ERROR in {cfg.get('name','unnamed')}: {exc}")
            m = {"name": cfg.get("name","unnamed"), "status": f"error: {exc}"}
        all_metrics.append(m)

    # After the run, merge with any other completed runs already on disk
    # so the comparison plots are always complete
    print("\nMerging with any other completed runs on disk...")
    all_on_disk   = load_all_metrics(ABLATION_ROOT)
    names_just_run = {m["name"] for m in all_metrics}
    merged = all_metrics + [m for m in all_on_disk if m["name"] not in names_just_run]

    save_cross_experiment_plots(merged, ABLATION_ROOT)


if __name__ == "__main__":
    main()