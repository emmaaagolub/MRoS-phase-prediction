"""Shared code for the evaluation experiment runners.

Holds the settings, path helpers and functions used by both ablation.py and
benchmarking.py: reading the interpolated grid and leave-one-out observations,
sampling the grid at each observation, sweeping the class weight, and fitting
the uncertainty band.

Nothing here is specific to a region or to one experiment.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: F401,E402  (REGIONS is re-exported)
    OUTPUT_DIR, REGIONS, interpolation_dir,
)

# ---------------------------------------------------------------------------
# Settings shared by every experiment
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 2000

SNOW_CODE = 0
RAIN_CODE = 1
MIX_CODE = 2
BINARY_LABEL_MAP = {RAIN_CODE: 0, SNOW_CODE: 1}
FULL_CLASS_NAMES = ["snow", "rain", "mix"]

# Class weights searched for the one that balances snow and rain recall.
SCALE_POS_WEIGHT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50,
                         0.75, 1.0, 1.25, 1.5, 2.0]

# Wet-bulb temperature separating near-freezing from clear-phase conditions.
CLEAR_PHASE_TWET_C = 2.0

# Grids searched for the uncertainty band around p(snow) = 0.5.
BASE_HALF_BAND_GRID = (0.05, 0.10, 0.15, 0.20)
EXTRA_HALF_BAND_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
MAX_TOTAL_HALF_BAND = 0.40
SIGMA_GRID = (1.0, 1.5, 2.0)
MIN_PURE_COVERAGE = 0.50
MIN_NF_PURE_COVERAGE = 0.35
MIX_CAPTURE_WEIGHT = 0.5

BASE_XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "max_depth": 6, "eta": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 5,
    "lambda": 1.0, "tree_method": "hist",
}

ALL_GRID_FEATURES = ["temp_air", "temp_dew", "temp_wet", "rh", "imerg_plp", "elev"]
ALL_LOOCV_FEATURES = ["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]

# One-degree wet-bulb bins with open-ended tails.
TWET_BIN_EDGES = [-np.inf, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, np.inf]
TWET_BIN_LABELS = ["<-6", "-6–-5", "-5–-4", "-4–-3", "-3–-2", "-2–-1",
                   "-1–0", "0–1", "1–2", "2–3", "3–4", "4–5", "5–6", ">6"]
NF_BIN_LABELS = ["-2–-1", "-1–0", "0–1", "1–2"]
NEAR_FREEZE_THRESH = 2.0
TOP_N_SHAP_FEATURES = 5

PHASE_COLORS = {"snow": "#1f77b4", "rain": "#2ca02c", "mix": "#e377c2"}


def interp_paths(region, interp_type):
    """Where the interpolated grid, leave-one-out table and IMERG grid live."""
    if interp_type not in ("IDW", "kriging"):
        raise ValueError(f"Unknown interpolation type: {interp_type}")
    stem = "IDW" if interp_type == "IDW" else "indicator_kriging"
    interp_dir = interpolation_dir(region, interp_type)
    return {
        "interp_grid": interp_dir / f"hourly_predictors_1km_{stem}.nc",
        "mros_loocv": interp_dir / f"mros_loocv_point_predictions_{interp_type}.parquet",
        "imerg": OUTPUT_DIR / "resampled_grids" / region / "imerg_hourly_1km.nc",
    }


def model_dir(region):
    """The trained model's directory, which also holds the shared split table."""
    return OUTPUT_DIR / "model" / region


def experiment_dir(region, name):
    """Where one experiment family (ablation, benchmarking) writes its results."""
    return OUTPUT_DIR / "evaluation" / region / name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
