"""Interpolate the point observations onto the 1 km grid, hour by hour.

Produces two outputs:

1. Gridded predictor surfaces.
   Temperature variables are regressed on elevation each hour and the residual
   is kriged with ordinary kriging. Humidity is kriged with elevation as an
   external drift. The MRoS phase categories are converted to snow/mix/rain
   indicators, kriged separately, then rescaled to sum to one.

2. A leave-one-out table of MRoS predictions.
   Each MRoS observation is predicted by kriging from the other observations
   in the same hour.

Inputs:  outputs/compiled/<REGION>/hourly_data/{stations,mros}_hourly.parquet
         Data/Elevation/<REGION>_DEM_AOI_1km.tif
Output:  outputs/interpolated/<REGION>/indicator_kriging/
           hourly_predictors_1km_indicator_kriging.nc
           mros_loocv_point_predictions_kriging.{parquet,csv}
           mros_loocv_summary_kriging.csv
           variogram_calibration/

Results are written one day at a time and appended to the output file in
batches, so an interrupted run resumes from the last completed day.
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from itertools import groupby
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pyproj import CRS, Transformer
from rasterio.transform import rowcol as rio_rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REGIONS, WY_END, WY_START, parse_region_args, region_paths  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PHASE_ORDER = ("snow", "mix", "rain")
PHASE_TO_PROB = {"snow": "p_snow", "mix": "p_mix", "rain": "p_rain"}
PHASE_COL_CANDIDATES = ("phase", "mros_phase", "phase_class", "ptype",
                        "precip_phase", "phase_label")
PHASE_ALIASES = {
    "snow": "snow", "s": "snow", "sn": "snow", "solid": "snow",
    "mix": "mix", "mixed": "mix", "transition": "mix", "m": "mix",
    "rain": "rain", "r": "rain", "liquid": "rain",
    "rain_snow": "mix", "snow_rain": "mix", "rainsnow": "mix",
}

# Days interpolated before results are appended to the output file.
FLUSH_EVERY_DAYS = 30

SETTINGS = {
    # Minimum stations or observations needed before a variable is interpolated.
    "min_points_temp": 4,
    "min_points_rh": 4,
    "min_points_indicator": 4,
    "min_points_lapse": 5,

    # Lapse rate used when the hourly fit is unavailable, and the range the
    # fitted value is held within.
    "default_lapse_degC_per_m": -0.005,
    "lapse_bounds_degC_per_m": (-0.009, 0.002),

    # Variogram fitting. Empirical variograms from a sample of hours are
    # pooled and one model is fitted to the pooled result.
    "variogram_model": "spherical",
    "max_hours_for_variogram": 300,
    "max_points_per_hour_for_variogram": 30,
    "n_lags": 14,
    "pair_distance_quantile": 0.95,
    "variogram_min_pairs": 40,
    "variogram_min_bins": 5,
    "variogram_eps": 1e-6,
    "eps": 1e-6,

    "temp_vars": ("temp_air", "temp_dew", "temp_wet"),
    "station_uk_vars": ("rh",),
    "mros_phase_probs": ("p_snow", "p_mix", "p_rain"),

    "save_processed_parquet": True,
    "reuse_processed": True,
}


def build_config(region_id):
    paths = region_paths(region_id)
    cfg = dict(SETTINGS)
    cfg.update({
        "region": region_id,
        "label": REGIONS[region_id]["label"],
        "wy_start": WY_START,
        "wy_end": WY_END,
        "mros_active_months": REGIONS[region_id]["mros_active_months"],
        "proj_fallback": REGIONS[region_id]["utm_crs"],
        "dem_path": paths["dem_1km"],
        "stations_parquet": paths["compiled_dir"] / "hourly_data" / "stations_hourly.parquet",
        "mros_parquet": paths["compiled_dir"] / "hourly_data" / "mros_hourly.parquet",
        "out_dir": paths["interpolated_dir"],
    })
    for sub in ("", "processed_inputs", "variogram_calibration"):
        (cfg["out_dir"] / sub).mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Grid and coordinate helpers
# ---------------------------------------------------------------------------

def to_utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.floor("h")


def hourly_index(start_iso, end_iso):
    return pd.date_range(start=pd.to_datetime(start_iso), end=pd.to_datetime(end_iso),
                         freq="h", tz="UTC")


def ensure_columns(df, required, df_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def choose_projected_crs_from_dem(dem_path, fallback):
    with rio.open(dem_path) as src:
        crs = CRS.from_user_input(src.crs)
        if crs.is_projected:
            return crs
    return CRS.from_user_input(fallback)


def load_dem(path, target_crs):
    """Read the DEM, reprojecting only if it is not already in the working CRS."""
    with rio.open(path) as src:
        src_crs = CRS.from_user_input(src.crs)
        tgt_crs = CRS.from_user_input(target_crs)

        if src_crs == tgt_crs:
            profile = src.profile.copy()
            profile["crs"] = tgt_crs.to_wkt()
            return src.read(1).astype(np.float32), profile, tgt_crs

        transform, width, height = calculate_default_transform(
            src_crs, tgt_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update({"crs": tgt_crs.to_wkt(), "transform": transform,
                        "width": width, "height": height, "dtype": "float32"})
        dem = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rio.band(src, 1), destination=dem,
            src_transform=src.transform, src_crs=src_crs,
            dst_transform=transform, dst_crs=tgt_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
        return dem, profile, tgt_crs


def grid_centers(profile):
    transform = profile["transform"]
    xs = transform.c + (np.arange(profile["width"]) + 0.5) * transform.a
    ys = transform.f + (np.arange(profile["height"]) + 0.5) * transform.e
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([x_mesh.ravel(), y_mesh.ravel()])


def sample_dem_at_xy(xs, ys, dem_data, dem_profile):
    rows, cols = rio_rowcol(dem_profile["transform"], xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = ((rows >= 0) & (rows < dem_data.shape[0])
              & (cols >= 0) & (cols < dem_data.shape[1]))
    out = np.full(len(xs), np.nan, dtype=np.float32)
    out[inside] = dem_data[rows[inside], cols[inside]]
    nodata = dem_profile.get("nodata")
    if nodata is not None:
        out[np.isclose(out, nodata)] = np.nan
    return out


def attach_projected_xy_and_dem(df, proj_crs, dem_data, dem_profile):
    """Add projected coordinates, and DEM elevation wherever it is missing."""
    out = df.copy()
    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    out["x"], out["y"] = transformer.transform(out["lon"].values, out["lat"].values)

    if "elev" not in out.columns:
        out["elev"] = np.nan
    need = out["elev"].isna()
    if need.any():
        out.loc[need, "elev"] = sample_dem_at_xy(
            out.loc[need, "x"].to_numpy(), out.loc[need, "y"].to_numpy(),
            dem_data, dem_profile,
        )
    return out


# ---------------------------------------------------------------------------
# Preparing the observations
# ---------------------------------------------------------------------------

def canonicalize_phase(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return PHASE_ALIASES.get(text, text if text in PHASE_ORDER else None)


def phase_from_legacy_proxy(value):
    """Older files stored phase as 0 / 50 / 100 instead of a label."""
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isclose(numeric, 0.0):
        return "snow"
    if np.isclose(numeric, 50.0):
        return "mix"
    if np.isclose(numeric, 100.0):
        return "rain"
    return None


def prepare_mros_indicators(mros_df):
    """Turn the phase label into three 0/1 indicator columns."""
    df = mros_df.copy()
    phase_col = next((c for c in PHASE_COL_CANDIDATES if c in df.columns), None)

    if phase_col is not None:
        df["mros_phase"] = df[phase_col].map(canonicalize_phase)
    elif "mros_plp_proxy" in df.columns:
        warnings.warn("No phase column found; using the legacy numeric proxy.")
        df["mros_phase"] = df["mros_plp_proxy"].map(phase_from_legacy_proxy)
    else:
        raise ValueError(f"No usable phase column. Columns present: {list(df.columns)}")

    for phase, out_col in PHASE_TO_PROB.items():
        df[out_col] = (df["mros_phase"] == phase).astype(float)
    return df


def dedupe_station_hourly(df, value_cols):
    """One row per station location per hour, averaging any duplicates."""
    def reducer(group):
        out = {"hour_utc": group.name[0], "lon": group.name[1], "lat": group.name[2]}
        for col in value_cols:
            vals = (pd.to_numeric(group[col], errors="coerce").dropna()
                    if col in group.columns else pd.Series(dtype=float))
            out[col] = float(vals.mean()) if len(vals) else np.nan
        return pd.Series(out)

    return (df.groupby(["hour_utc", "lon", "lat"], dropna=False, sort=False)
              .apply(reducer).reset_index(drop=True))


def dedupe_mros_hourly(df):
    """One row per location per hour; repeated reports become fractional votes."""
    def reducer(group):
        out = {"hour_utc": group.name[0], "lon": group.name[1], "lat": group.name[2]}
        phases = group["mros_phase"].dropna().tolist()
        if not phases:
            out.update({"mros_phase": None, "is_soft_duplicate": False,
                        "p_snow": np.nan, "p_mix": np.nan, "p_rain": np.nan})
            return pd.Series(out)

        counts = pd.Series(phases).value_counts(normalize=True)
        out["mros_phase"] = counts.idxmax()
        out["is_soft_duplicate"] = len(counts) > 1
        out["p_snow"] = float(counts.get("snow", 0.0))
        out["p_mix"] = float(counts.get("mix", 0.0))
        out["p_rain"] = float(counts.get("rain", 0.0))
        return pd.Series(out)

    return (df.groupby(["hour_utc", "lon", "lat"], dropna=False, sort=False)
              .apply(reducer).reset_index(drop=True))


def prepare_inputs(cfg, force_rebuild=False):
    """Load, clean and cache the station and MRoS inputs."""
    proc_dir = cfg["out_dir"] / "processed_inputs"
    st_proc, mros_proc = proc_dir / "stations_processed.parquet", proc_dir / "mros_processed.parquet"

    proj_crs = choose_projected_crs_from_dem(cfg["dem_path"], cfg["proj_fallback"])
    dem_data, dem_profile, proj_crs = load_dem(cfg["dem_path"], proj_crs)

    if cfg["reuse_processed"] and not force_rebuild and st_proc.exists() and mros_proc.exists():
        print("  reusing cached processed inputs")
        return (pd.read_parquet(st_proc), pd.read_parquet(mros_proc),
                dem_data, dem_profile, proj_crs)

    st = pd.read_parquet(cfg["stations_parquet"])
    mros = pd.read_parquet(cfg["mros_parquet"])
    ensure_columns(st, ["hour_utc", "lon", "lat"], "stations parquet")
    ensure_columns(mros, ["hour_utc", "lon", "lat"], "MRoS parquet")

    st["hour_utc"] = to_utc(st["hour_utc"])
    mros["hour_utc"] = to_utc(mros["hour_utc"])
    start, end = pd.to_datetime(cfg["wy_start"]), pd.to_datetime(cfg["wy_end"])
    st = st.loc[st["hour_utc"].between(start, end)].copy()
    mros = mros.loc[mros["hour_utc"].between(start, end)].copy()

    station_vars = [c for c in list(cfg["temp_vars"]) + list(cfg["station_uk_vars"])
                    if c in st.columns]
    if not station_vars:
        raise ValueError("None of the configured station variables are present.")

    st = st[["hour_utc", "lon", "lat"] + station_vars].copy()
    st = dedupe_station_hourly(st, station_vars)
    st = attach_projected_xy_and_dem(st, proj_crs, dem_data, dem_profile)

    mros = prepare_mros_indicators(mros)
    mros = mros[["hour_utc", "lon", "lat", "mros_phase", "p_snow", "p_mix", "p_rain"]].copy()
    mros = dedupe_mros_hourly(mros)
    mros = attach_projected_xy_and_dem(mros, proj_crs, dem_data, dem_profile)

    st = st.dropna(subset=["x", "y"])
    mros = mros.dropna(subset=["x", "y"])

    if cfg["save_processed_parquet"]:
        st.to_parquet(st_proc, index=False)
        mros.to_parquet(mros_proc, index=False)

    return st, mros, dem_data, dem_profile, proj_crs


# ---------------------------------------------------------------------------
# Variograms
# ---------------------------------------------------------------------------

def spherical_gamma(h, sill, rng, nugget):
    h = np.asarray(h, dtype=float)
    rng = max(float(rng), 1e-12)
    hr = h / rng
    return np.where(h <= rng, nugget + sill * (1.5 * hr - 0.5 * hr ** 3), nugget + sill)


def exponential_gamma(h, sill, rng, nugget):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1.0 - np.exp(-h / max(float(rng), 1e-12)))


def gaussian_gamma(h, sill, rng, nugget):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1.0 - np.exp(-(h / max(float(rng), 1e-12)) ** 2))


MODEL_FUNCS = {
    "spherical": spherical_gamma,
    "exponential": exponential_gamma,
    "gaussian": gaussian_gamma,
}


def empirical_variogram_xy(x, y, z, n_lags, quantile):
    """Average squared difference against separation distance, in distance bins."""
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = np.asarray(x)[keep], np.asarray(y)[keep], np.asarray(z)[keep]
    if len(z) < 3:
        return None

    dists = pdist(np.column_stack([x, y]))
    if len(dists) == 0:
        return None
    semis = 0.5 * pdist(z.reshape(-1, 1), metric="sqeuclidean")

    dmax = np.quantile(dists, quantile)
    if not np.isfinite(dmax) or dmax <= 0:
        return None

    bins = np.linspace(0.0, dmax, n_lags + 1)
    lag_ids = np.digitize(dists, bins) - 1

    mids, gamma_emp, n_pairs = [], [], []
    for i in range(n_lags):
        mask = lag_ids == i
        if not mask.sum():
            continue
        mids.append(0.5 * (bins[i] + bins[i + 1]))
        gamma_emp.append(np.nanmean(semis[mask]))
        n_pairs.append(int(mask.sum()))

    if not mids:
        return None
    return {"dist_mid": np.asarray(mids), "gamma_emp": np.asarray(gamma_emp),
            "n_pairs": np.asarray(n_pairs)}


def fit_variogram_model(emp, model_name, var_name, eps):
    """Fit sill, range and nugget to the binned variogram."""
    mids = np.asarray(emp["dist_mid"], dtype=float)
    gamma = np.asarray(emp["gamma_emp"], dtype=float)
    n_pairs = np.asarray(emp["n_pairs"], dtype=float)
    if len(mids) < 3:
        return None

    sill_guess = max(float(np.nanquantile(gamma, 0.9)), eps)
    range_guess = max(float(np.nanquantile(mids, 0.75)), eps)
    nugget_guess = max(float(np.nanmin(gamma)), 0.0)

    try:
        popt, _ = curve_fit(
            MODEL_FUNCS[model_name], mids, gamma,
            sigma=np.maximum(1.0 / np.sqrt(np.maximum(n_pairs, 1.0)), eps),
            p0=[sill_guess, range_guess, nugget_guess],
            bounds=([eps, eps, 0.0],
                    [10.0 * max(sill_guess, 1.0), mids.max() * 3.0 + eps,
                     max(sill_guess * 0.95, eps)]),
            maxfev=5000,
        )
        sill, rng, nugget = map(float, popt)

        # A nugget at or above the sill is a degenerate fit.
        if nugget >= sill:
            warnings.warn(f"Degenerate variogram fit for '{var_name}'; using initial guesses.")
            sill, rng, nugget = sill_guess, range_guess, nugget_guess
    except Exception as exc:
        warnings.warn(f"Variogram fit failed for '{var_name}' ({exc}); using initial guesses.")
        sill, rng, nugget = sill_guess, range_guess, nugget_guess

    return {"variable": var_name, "model": model_name,
            "sill": max(sill, eps), "range": max(rng, eps),
            "nugget": max(nugget, 0.0), "psill": max(sill - nugget, eps)}


def pooled_hourly_variogram_fit(df, value_col, cfg, fit_name=None):
    """Pool empirical variograms across a sample of hours, then fit one model."""
    fit_name = fit_name or value_col
    hours = pd.Series(df["hour_utc"].dropna().sort_values().unique())
    if len(hours) == 0:
        return None, None
    if len(hours) > cfg["max_hours_for_variogram"]:
        hours = (hours.sample(cfg["max_hours_for_variogram"], random_state=42)
                      .sort_values().reset_index(drop=True))

    mids_all, gamma_all, pairs_all = [], [], []
    for hour in hours:
        sub = df[df["hour_utc"] == hour].dropna(subset=["x", "y", value_col])
        if len(sub) > cfg["max_points_per_hour_for_variogram"]:
            sub = sub.sample(cfg["max_points_per_hour_for_variogram"], random_state=42)
        if len(sub) < 3:
            continue

        emp = empirical_variogram_xy(sub["x"].to_numpy(), sub["y"].to_numpy(),
                                     sub[value_col].to_numpy(),
                                     cfg["n_lags"], cfg["pair_distance_quantile"])
        if emp is None:
            continue

        valid = emp["n_pairs"] >= cfg["variogram_min_pairs"]
        if valid.sum() < cfg["variogram_min_bins"]:
            continue
        mids_all.append(emp["dist_mid"][valid])
        gamma_all.append(emp["gamma_emp"][valid])
        pairs_all.append(emp["n_pairs"][valid])

    if not mids_all:
        return None, None

    mids = np.concatenate(mids_all)
    gamma = np.concatenate(gamma_all)
    n_pairs = np.concatenate(pairs_all)

    # Re-bin the pooled points, weighting each hour by how many pairs it had.
    bins = np.linspace(mids.min(), mids.max() + cfg["variogram_eps"], cfg["n_lags"] + 1)
    lag_ids = np.digitize(mids, bins) - 1
    mid_final, gamma_final, pair_final = [], [], []
    for i in range(cfg["n_lags"]):
        mask = lag_ids == i
        if not np.any(mask):
            continue
        weights = n_pairs[mask]
        mid_final.append(float(np.average(mids[mask], weights=weights)))
        gamma_final.append(float(np.average(gamma[mask], weights=weights)))
        pair_final.append(int(np.sum(weights)))

    pooled = {"dist_mid": np.asarray(mid_final), "gamma_emp": np.asarray(gamma_final),
              "n_pairs": np.asarray(pair_final)}
    params = fit_variogram_model(pooled, cfg["variogram_model"], fit_name, cfg["variogram_eps"])
    return params, pooled


def calibrate_variograms(st_hr, mros_hr, cfg):
    """Fit one variogram per interpolated variable and save the parameters."""
    params, summary_rows, bins_dict = {}, [], {}

    for var in [c for c in cfg["temp_vars"] if c in st_hr.columns]:
        rows = []
        for hour in pd.Series(st_hr["hour_utc"].dropna().unique()):
            pts, _, _ = estimate_lapse_rate_and_residuals(
                st_hr[st_hr["hour_utc"] == hour], var, cfg
            )
            if len(pts):
                rows.append(pts[["hour_utc", "x", "y", "resid"]]
                            .rename(columns={"resid": f"{var}_resid"}))
        if not rows:
            continue

        fit_df = pd.concat(rows, ignore_index=True)

        # Lower the pair thresholds to what the available hours support.
        n_median = fit_df.groupby("hour_utc").size().median()
        max_pairs = int(n_median * (n_median - 1) / 2 / cfg["n_lags"])
        sparse_cfg = {**cfg,
                      "variogram_min_pairs": min(cfg["variogram_min_pairs"],
                                                 max(max_pairs // 2, 3)),
                      "variogram_min_bins": 3}

        fit, emp = pooled_hourly_variogram_fit(fit_df, f"{var}_resid", sparse_cfg,
                                               fit_name=f"{var}_resid")
        if fit is not None:
            params[f"{var}_resid"] = fit
            bins_dict[f"{var}_resid"] = emp
            summary_rows.append(fit)

    for var in [c for c in cfg["station_uk_vars"] if c in st_hr.columns]:
        fit, emp = pooled_hourly_variogram_fit(st_hr, var, cfg, fit_name=var)
        if fit is not None:
            params[var] = fit
            bins_dict[var] = emp
            summary_rows.append(fit)

    # Fitted on hours with observations only, with lower pair thresholds.
    mros_active = mros_hr[mros_hr["mros_phase"].isin(PHASE_ORDER)]
    mros_cfg = {**cfg, "variogram_min_pairs": 5, "variogram_min_bins": 3}
    for prob_col in cfg["mros_phase_probs"]:
        fit, emp = pooled_hourly_variogram_fit(mros_active, prob_col, mros_cfg,
                                               fit_name=prob_col)
        if fit is not None:
            params[prob_col] = fit
            bins_dict[prob_col] = emp
            summary_rows.append(fit)

    summary = pd.DataFrame(summary_rows)
    calib_dir = cfg["out_dir"] / "variogram_calibration"
    with open(calib_dir / "variogram_params.json", "w") as handle:
        json.dump(params, handle, indent=2)
    summary.to_csv(calib_dir / "variogram_summary.csv", index=False)

    return params, summary, bins_dict


# ---------------------------------------------------------------------------
# Kriging
# ---------------------------------------------------------------------------

def pykrige_variogram_dict(params):
    # pykrige wants the partial sill (sill minus nugget), not the total sill.
    return {
        "variogram_model": params["model"],
        "variogram_parameters": {"psill": params["psill"], "range": params["range"],
                                 "nugget": params["nugget"]},
    }


def ordinary_krige_grid(x, y, z, target_xy, params):
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[keep], y[keep], z[keep]
    if len(z) < 3:
        return np.full(len(target_xy), np.nan, dtype=np.float32)
    try:
        ok = OrdinaryKriging(x, y, z, **pykrige_variogram_dict(params),
                             enable_plotting=False, verbose=False)
        est, _ = ok.execute("points", target_xy[:, 0], target_xy[:, 1])
        return np.asarray(est, dtype=np.float32)
    except Exception:
        return np.full(len(target_xy), np.nan, dtype=np.float32)


def ordinary_krige_point(x, y, z, target_x, target_y, params):
    pred = ordinary_krige_grid(np.asarray(x), np.asarray(y), np.asarray(z),
                               np.array([[target_x, target_y]], dtype=float), params)
    return float(pred[0]) if len(pred) else np.nan


def universal_krige_dem_grid(x, y, z, drift_train, target_xy, drift_target, params):
    """Kriging with elevation as an external drift term."""
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(drift_train)
    x, y, z, drift_train = x[keep], y[keep], z[keep], drift_train[keep]
    if len(z) < 3:
        return np.full(len(target_xy), np.nan, dtype=np.float32)
    try:
        uk = UniversalKriging(x, y, z, drift_terms=["specified"],
                              specified_drift=[drift_train],
                              **pykrige_variogram_dict(params),
                              enable_plotting=False, verbose=False)
        est, _ = uk.execute("points", target_xy[:, 0], target_xy[:, 1],
                            specified_drift_arrays=[drift_target])
        return np.asarray(est, dtype=np.float32)
    except Exception:
        return np.full(len(target_xy), np.nan, dtype=np.float32)


def probability_closure(prob_cube, eps=1e-6):
    """Rescale the three phase probabilities so they sum to one.

    Cells where any phase was missing stay missing rather than being filled by
    the others.
    """
    raw = np.asarray(prob_cube, dtype=float)
    arr = np.clip(raw, 0.0, 1.0)
    any_nan = ~np.all(np.isfinite(raw), axis=0)
    arr[~np.isfinite(raw)] = 0.0

    denom = np.sum(arr, axis=0)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = (denom > eps) & ~any_nan
    out[:, valid] = arr[:, valid] / denom[valid]
    return out


# ---------------------------------------------------------------------------
# Per-hour interpolation
# ---------------------------------------------------------------------------

def estimate_lapse_rate_and_residuals(hour_points, value_col, cfg):
    """Fit the variable against elevation and return the leftover residuals."""
    pts = hour_points.dropna(subset=["elev", value_col, "x", "y"]).copy()
    if len(pts) == 0:
        return pts, cfg["default_lapse_degC_per_m"], 0.0

    slope = cfg["default_lapse_degC_per_m"]
    intercept = float(np.nanmean(pts[value_col]))

    if len(pts) >= cfg["min_points_lapse"]:
        try:
            model = LinearRegression().fit(pts[["elev"]].to_numpy(),
                                           pts[value_col].to_numpy())
            lo, hi = cfg["lapse_bounds_degC_per_m"]
            slope = float(np.clip(model.coef_[0], lo, hi))
            intercept = float(model.intercept_)
        except Exception:
            pass

    pts["resid"] = pts[value_col] - (intercept + slope * pts["elev"])
    return pts, slope, intercept


def interpolate_temperature_hour(st_t, var, grid_xy_valid, dem_valid, variogram_params, cfg):
    """Elevation trend plus kriged residual, evaluated on the grid."""
    missing = np.full(len(grid_xy_valid), np.nan, dtype=np.float32)
    if f"{var}_resid" not in variogram_params:
        return missing

    pts, slope, intercept = estimate_lapse_rate_and_residuals(st_t, var, cfg)
    if "resid" not in pts.columns or len(pts) == 0:
        return missing
    pts = pts.dropna(subset=["x", "y", "resid"])
    if len(pts) < cfg["min_points_temp"]:
        return missing

    resid_pred = ordinary_krige_grid(
        pts["x"].to_numpy(), pts["y"].to_numpy(), pts["resid"].to_numpy(),
        grid_xy_valid, variogram_params[f"{var}_resid"],
    )
    return np.clip(intercept + slope * dem_valid + resid_pred, -60.0, 60.0).astype(np.float32)


def interpolate_rh_hour(st_t, var, grid_xy_valid, dem_valid, variogram_params, cfg):
    missing = np.full(len(grid_xy_valid), np.nan, dtype=np.float32)
    if var not in variogram_params:
        return missing

    sub = st_t.dropna(subset=["x", "y", "elev", var])
    if len(sub) < cfg["min_points_rh"]:
        return missing

    result = universal_krige_dem_grid(
        sub["x"].to_numpy(), sub["y"].to_numpy(), sub[var].to_numpy(),
        sub["elev"].to_numpy(), grid_xy_valid, dem_valid, variogram_params[var],
    )
    return np.clip(result, 0.0, 100.0).astype(np.float32)


def interpolate_mros_indicator_hour(mros_t, grid_xy_valid, variogram_params, cfg):
    """Krige each phase indicator, then rescale the three to sum to one."""
    missing = lambda: np.full(len(grid_xy_valid), np.nan, dtype=np.float32)  # noqa: E731
    sub = mros_t.dropna(subset=["x", "y"])

    if len(sub) < cfg["min_points_indicator"]:
        return {col: missing() for col in cfg["mros_phase_probs"]}

    pred = {}
    for col in cfg["mros_phase_probs"]:
        col_sub = sub.dropna(subset=[col])
        if len(col_sub) < cfg["min_points_indicator"] or col not in variogram_params:
            pred[col] = missing()
            continue
        pred[col] = ordinary_krige_grid(
            col_sub["x"].to_numpy(), col_sub["y"].to_numpy(), col_sub[col].to_numpy(),
            grid_xy_valid, variogram_params[col],
        )

    cube = probability_closure(
        np.vstack([pred["p_snow"], pred["p_mix"], pred["p_rain"]]), eps=cfg["eps"]
    )
    pred["p_snow"], pred["p_mix"], pred["p_rain"] = [cube[i].astype(np.float32) for i in range(3)]
    return pred


# ---------------------------------------------------------------------------
# Leave-one-out MRoS predictions
# ---------------------------------------------------------------------------

def krige_mros_at_point(train_df, target_row, variogram_params, cfg):
    sub = train_df.dropna(subset=["x", "y"])
    if len(sub) < cfg["min_points_indicator"]:
        return {col: np.nan for col in cfg["mros_phase_probs"]}

    pred = {}
    for col in cfg["mros_phase_probs"]:
        col_sub = sub.dropna(subset=[col])
        if len(col_sub) < cfg["min_points_indicator"] or col not in variogram_params:
            pred[col] = np.nan
            continue
        pred[col] = ordinary_krige_point(
            col_sub["x"].to_numpy(), col_sub["y"].to_numpy(), col_sub[col].to_numpy(),
            float(target_row["x"]), float(target_row["y"]), variogram_params[col],
        )

    cube = probability_closure(
        np.array([[pred["p_snow"]], [pred["p_mix"]], [pred["p_rain"]]], dtype=float),
        eps=cfg["eps"],
    )
    return {"p_snow": float(cube[0, 0]), "p_mix": float(cube[1, 0]),
            "p_rain": float(cube[2, 0])}


def loocv_mros_hour(mros_t, variogram_params, cfg):
    """Predict each observation in the hour from all the others."""
    cols_needed = ["x", "y", "elev", "mros_phase", "p_snow", "p_mix", "p_rain"]
    df = mros_t[mros_t["mros_phase"].isin(PHASE_ORDER)].dropna(subset=cols_needed)

    blank = {"n_points": len(df), "n_eval": 0, "accuracy": np.nan,
             "macro_f1": np.nan, "multiclass_logloss": np.nan}
    if len(df) < max(cfg["min_points_indicator"] + 1, 3):
        return pd.DataFrame(), blank

    rows = []
    for i in range(len(df)):
        test = df.iloc[i]
        pred = krige_mros_at_point(df.drop(df.index[i]), test, variogram_params, cfg)
        if any(pd.isna(list(pred.values()))):
            continue

        probs = [pred["p_snow"], pred["p_mix"], pred["p_rain"]]
        clipped = np.clip(probs, cfg["eps"], 1.0)
        pred_phase = PHASE_ORDER[int(np.argmax(probs))]
        rows.append({
            "lon": float(test["lon"]), "lat": float(test["lat"]),
            "x": float(test["x"]), "y": float(test["y"]), "elev": float(test["elev"]),
            "obs_phase": test["mros_phase"], "pred_phase": pred_phase,
            "mros_p_snow_loocv": pred["p_snow"],
            "mros_p_mix_loocv": pred["p_mix"],
            "mros_p_rain_loocv": pred["p_rain"],
            "obs_p_snow": float(test["p_snow"]),
            "obs_p_mix": float(test["p_mix"]),
            "obs_p_rain": float(test["p_rain"]),
            "pred_max_prob": float(np.max(probs)),
            "pred_entropy": float(-np.sum(clipped * np.log(clipped))),
            "pred_correct": int(pred_phase == test["mros_phase"]),
        })

    details = pd.DataFrame(rows)
    if details.empty:
        return details, blank

    metrics = {
        "n_points": len(df), "n_eval": len(details),
        "accuracy": accuracy_score(details["obs_phase"], details["pred_phase"]),
        "macro_f1": f1_score(details["obs_phase"], details["pred_phase"],
                             labels=PHASE_ORDER, average="macro"),
    }
    try:
        metrics["multiclass_logloss"] = log_loss(
            details[["obs_p_snow", "obs_p_mix", "obs_p_rain"]].values,
            details[["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]].values,
        )
    except Exception:
        metrics["multiclass_logloss"] = np.nan
    return details, metrics


def _score_block(block):
    """Accuracy, macro F1 and log loss for one set of leave-one-out predictions."""
    row = {
        "accuracy": accuracy_score(block["obs_phase"], block["pred_phase"]),
        "macro_f1": f1_score(block["obs_phase"], block["pred_phase"],
                             labels=PHASE_ORDER, average="macro"),
    }
    try:
        row["multiclass_logloss"] = log_loss(
            block[["obs_p_snow", "obs_p_mix", "obs_p_rain"]].values,
            block[["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]].values,
        )
    except Exception:
        row["multiclass_logloss"] = np.nan
    return row


def summarize_loocv(points_df):
    """Overall scores plus one row per hour."""
    if points_df.empty:
        return pd.DataFrame([{"scope": "overall", "n_eval": 0, "accuracy": np.nan,
                              "macro_f1": np.nan, "multiclass_logloss": np.nan}])

    overall = {"scope": "overall", "n_eval": len(points_df), **_score_block(points_df)}
    by_hour = [
        {"scope": "hourly", "hour_utc": hour, "n_eval": len(group), **_score_block(group)}
        for hour, group in points_df.dropna(subset=["hour_utc"]).groupby("hour_utc")
    ]
    return pd.concat([pd.DataFrame([overall]), pd.DataFrame(by_hour)], ignore_index=True)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def _read_loocv_csv(path):
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.empty or "hour_utc" not in df.columns:
            return None
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce")
        return df
    except Exception as exc:
        warnings.warn(f"Could not read {path.name}: {exc}")
        return None


def interpolate_all_hours(st_hr, mros_hr, dem_data, dem_profile, proj_crs,
                          variogram_params, cfg):
    x_centers, y_centers, grid_xy = grid_centers(dem_profile)

    # xarray writes rows south-to-north, so flip the DEM if it is stored the
    # other way round and rebuild the coordinates to match.
    if y_centers[0] > y_centers[-1]:
        y_centers = y_centers[::-1]
        dem_data = dem_data[::-1, :]
        x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
        grid_xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    height, width = dem_data.shape
    valid_points = np.isfinite(dem_data).ravel()
    grid_xy_valid = grid_xy[valid_points]
    dem_valid = dem_data.ravel()[valid_points].astype(np.float32)
    times = hourly_index(cfg["wy_start"], cfg["wy_end"])

    out_vars = ([c for c in cfg["temp_vars"] if c in st_hr.columns]
                + [c for c in cfg["station_uk_vars"] if c in st_hr.columns]
                + list(cfg["mros_phase_probs"]))

    ckpt_dir = cfg["out_dir"] / "hourly_chunks"
    loocv_dir = cfg["out_dir"] / "loocv_chunks"
    ckpt_dir.mkdir(exist_ok=True)
    loocv_dir.mkdir(exist_ok=True)
    final_nc = cfg["out_dir"] / "hourly_predictors_1km_indicator_kriging.nc"

    # An output file left over from a run with a different time window would be
    # appended to incorrectly, so check it before continuing.
    if final_nc.exists():
        try:
            with xr.open_dataset(final_nc) as existing:
                existing_start = pd.Timestamp(existing.time.values[0])
                existing_end = pd.Timestamp(existing.time.values[-1])
            expected_start = pd.Timestamp(cfg["wy_start"])
            expected_end = pd.Timestamp(cfg["wy_end"])
            if existing_start != expected_start or existing_end.date() != expected_end.date():
                print(f"  existing output covers {existing_start.date()} to "
                      f"{existing_end.date()}, not the configured window — rebuilding")
                final_nc.unlink()
        except Exception:
            print("  existing output is unreadable — rebuilding")
            final_nc.unlink()

    days_written = len(list(ckpt_dir.glob("*.nc")))
    if days_written:
        print(f"  resuming with {days_written} unflushed day chunk(s)")

    def flush_chunks():
        """Append the pending day chunks to the output file and delete them."""
        pending = sorted(ckpt_dir.glob("*.nc"))
        if not pending:
            return
        try:
            if not final_nc.exists():
                opened = [xr.open_dataset(f) for f in pending]
                ds_batch = xr.concat(opened, dim="time").sortby("time")
                encoding = {v: {"zlib": True, "complevel": 4} for v in out_vars}
                encoding["time"] = {"units": "hours since 2022-10-01", "calendar": "standard"}
                ds_batch.to_netcdf(final_nc, unlimited_dims=["time"], encoding=encoding)
                ds_batch.close()
                for handle in opened:
                    handle.close()

                # Write a CF grid mapping so GDAL and rioxarray pick up the CRS.
                with nc4.Dataset(final_nc, "a") as ds_nc:
                    crs_var = ds_nc.createVariable("spatial_ref", "i4")
                    crs_var.crs_wkt = proj_crs.to_wkt()
                    crs_var.grid_mapping_name = proj_crs.to_cf()["grid_mapping_name"]
                    crs_var.spatial_ref = proj_crs.to_wkt()
                    for var in out_vars:
                        if var in ds_nc.variables:
                            ds_nc.variables[var].grid_mapping = "spatial_ref"
                    ds_nc.crs = cfg["proj_fallback"]
                    ds_nc.region = cfg["region"]
                    ds_nc.region_label = cfg["label"]
                    ds_nc.description = ("Hourly station interpolation and "
                                         "categorical MRoS indicator kriging")
                    ds_nc.mros_note = ("p_snow/p_mix/p_rain are predictor surfaces "
                                       "only; raw MRoS points remain the labels.")
            else:
                # Append in place rather than reading the whole file into memory.
                origin = pd.Timestamp("2022-10-01")
                with nc4.Dataset(final_nc, "a") as dst:
                    current_len = len(dst.variables["time"])
                    for path in pending:
                        with nc4.Dataset(path, "r") as src:
                            n = len(src.variables["time"])
                            with xr.open_dataset(path) as chunk:
                                times_in_chunk = chunk.time.values
                            hours_since_origin = np.array(
                                [(pd.Timestamp(t) - origin).total_seconds() / 3600
                                 for t in times_in_chunk], dtype=np.float64
                            )
                            dst.variables["time"][current_len:current_len + n] = hours_since_origin
                            for var in out_vars:
                                if var in src.variables:
                                    dst.variables[var][current_len:current_len + n] = \
                                        src.variables[var][:]
                            current_len += n

            for path in pending:
                path.unlink()
            print(f"  flushed {len(pending)} day chunk(s)")
        except Exception as exc:
            print(f"  flush failed ({exc}) — chunks kept for the next run")

    days = [(date, list(hour_group))
            for date, hour_group in groupby(times, key=lambda t: pd.Timestamp(t).date())]

    for date, hour_group in tqdm(days, desc=f"{cfg['region']} days"):
        stamp = date.strftime("%Y%m%d")
        nc_path = ckpt_dir / f"{stamp}.nc"
        csv_path = loocv_dir / f"{stamp}.csv"

        if csv_path.exists() and not nc_path.exists():
            continue  # already flushed into the output file
        if nc_path.exists() and csv_path.exists():
            days_written += 1  # written but not yet flushed
            if days_written % FLUSH_EVERY_DAYS == 0:
                flush_chunks()
            continue

        day_slices, day_loocv, day_times = [], [], []

        for hour in hour_group:
            st_t = st_hr[st_hr["hour_utc"] == hour]
            mros_t = mros_hr[mros_hr["hour_utc"] == hour]
            slice_vars = {}

            for var in [c for c in cfg["temp_vars"] if c in st_hr.columns]:
                vals = interpolate_temperature_hour(st_t, var, grid_xy_valid, dem_valid,
                                                    variogram_params, cfg)
                full = np.full(height * width, np.nan, dtype=np.float32)
                full[valid_points] = vals
                slice_vars[var] = full.reshape(height, width)

            for var in [c for c in cfg["station_uk_vars"] if c in st_hr.columns]:
                vals = interpolate_rh_hour(st_t, var, grid_xy_valid, dem_valid,
                                           variogram_params, cfg)
                full = np.full(height * width, np.nan, dtype=np.float32)
                full[valid_points] = vals
                slice_vars[var] = full.reshape(height, width)

            # MRoS surfaces and scoring are limited to the active months.
            in_season = pd.Timestamp(hour).month in cfg["mros_active_months"]
            if in_season:
                pred = interpolate_mros_indicator_hour(mros_t, grid_xy_valid,
                                                       variogram_params, cfg)
            else:
                pred = {col: np.full(len(grid_xy_valid), np.nan, dtype=np.float32)
                        for col in cfg["mros_phase_probs"]}

            for col in cfg["mros_phase_probs"]:
                full = np.full(height * width, np.nan, dtype=np.float32)
                full[valid_points] = pred[col]
                slice_vars[col] = full.reshape(height, width)

            if in_season:
                details, _ = loocv_mros_hour(mros_t, variogram_params, cfg)
                if len(details):
                    details["hour_utc"] = hour
                    day_loocv.append(details)

            day_slices.append(slice_vars)
            day_times.append(pd.Timestamp(hour).tz_localize(None))

        # Write to a temporary name first so an interrupted write is not
        # mistaken for a finished day.
        encoding = {k: {"zlib": True, "complevel": 4} for k in out_vars}
        nc_tmp = nc_path.with_suffix(".tmp")
        xr.Dataset(
            {k: (("time", "y", "x"), np.stack([s[k] for s in day_slices]))
             for k in out_vars},
            coords={"time": day_times, "y": y_centers, "x": x_centers},
        ).to_netcdf(nc_tmp, encoding=encoding)

        loocv_day = pd.concat(day_loocv, ignore_index=True) if day_loocv else pd.DataFrame()
        loocv_day.to_csv(csv_path, index=False)
        nc_tmp.rename(nc_path)

        days_written += 1
        if days_written % FLUSH_EVERY_DAYS == 0:
            flush_chunks()

    flush_chunks()

    try:
        with xr.open_dataset(final_nc) as check:
            print(f"  output verified: {len(check.time)} timesteps")
    except Exception as exc:
        raise RuntimeError(
            f"Output file failed verification: {exc}\n"
            f"Leave-one-out chunks kept at {loocv_dir} — do not delete them."
        )

    parts = [df for path in sorted(loocv_dir.glob("*.csv"))
             if (df := _read_loocv_csv(path)) is not None]
    loocv_points = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not loocv_points.empty:
        loocv_points["hour_utc"] = pd.to_datetime(loocv_points["hour_utc"], utc=True,
                                                  errors="coerce")
        loocv_points.to_parquet(
            cfg["out_dir"] / "mros_loocv_point_predictions_kriging.parquet", index=False
        )
        loocv_points.to_csv(
            cfg["out_dir"] / "mros_loocv_point_predictions_kriging.csv", index=False
        )

    loocv_summary = summarize_loocv(loocv_points)
    loocv_summary.to_csv(cfg["out_dir"] / "mros_loocv_summary_kriging.csv", index=False)

    # Only clean up once everything above has been written successfully.
    shutil.rmtree(loocv_dir)
    remaining = list(ckpt_dir.glob("*.nc"))
    if remaining:
        print(f"  {len(remaining)} chunk(s) left in {ckpt_dir} — not deleted")
    else:
        ckpt_dir.rmdir()

    return xr.open_dataset(final_nc), loocv_summary, loocv_points


def process_region(region_id, force_rebuild=False):
    cfg = build_config(region_id)
    print(f"\n=== {region_id}: {cfg['label']} ===")

    for label, path in [("DEM", cfg["dem_path"]),
                        ("stations parquet", cfg["stations_parquet"]),
                        ("MRoS parquet", cfg["mros_parquet"])]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    st_hr, mros_hr, dem_data, dem_profile, proj_crs = prepare_inputs(cfg, force_rebuild)
    print(f"  stations {len(st_hr):,} rows | MRoS {len(mros_hr):,} rows")

    variogram_params, variogram_summary, _ = calibrate_variograms(st_hr, mros_hr, cfg)
    print(variogram_summary.to_string(index=False))

    expected = ([f"{v}_resid" for v in cfg["temp_vars"] if v in st_hr.columns]
                + [v for v in cfg["station_uk_vars"] if v in st_hr.columns]
                + list(cfg["mros_phase_probs"]))
    missing = [k for k in expected if k not in variogram_params]
    if missing:
        raise RuntimeError(f"No variogram fitted for {missing}; these would come out empty.")

    _, loocv_summary, _ = interpolate_all_hours(
        st_hr, mros_hr, dem_data, dem_profile, proj_crs, variogram_params, cfg
    )
    print(loocv_summary.head(1).to_string(index=False))
    print(f"  outputs written to {cfg['out_dir']}")


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
