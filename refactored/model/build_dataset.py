"""Step 9 — Assemble the point-level table the model is trained on.

Each MRoS observation gets:
  - its phase label (snow / rain / mix), which is the only supervised target
  - the interpolated meteorological predictors, read off the 1 km grid at that
    location and hour
  - IMERG liquid-precipitation probability, likewise
  - the leave-one-out MRoS indicators, which describe what nearby observers
    reported without ever using the observation itself

Inputs:  outputs/interpolated/<REGION>/indicator_kriging/*
         outputs/resampled_grids/<REGION>/imerg_hourly_1km.nc
Output:  outputs/model/<REGION>/ml_input_points.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES, GRID_FEATURES, INTERP_TYPE, REGIONS, TARGET_FULL,
    make_output_dirs, model_paths,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import parse_region_args  # noqa: E402

PHASE_NAME_TO_CODE = {"snow": 0, "rain": 1, "mix": 2, "mixed": 2, "mixed_phase": 2}
VALID_PHASE_CODES = {0, 1, 2}


def first_present(df, names, label=None):
    """First of several candidate column names that exists in the frame."""
    for name in names:
        if name in df.columns:
            return name
    if label is None:
        return None
    raise KeyError(f"Could not find {label}. Tried: {list(names)}")


def normalize_phase_value(value):
    """Accept phase as a name or a code and return the numeric code."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in PHASE_NAME_TO_CODE:
            return PHASE_NAME_TO_CODE[key]
        try:
            code = int(float(key))
        except ValueError:
            return None
        return code if code in VALID_PHASE_CODES else None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    return code if code in VALID_PHASE_CODES else None


def get_dataset_crs(ds, fallback):
    try:
        if hasattr(ds, "rio") and ds.rio.crs is not None:
            return ds.rio.crs
    except Exception:
        pass
    for key in ("crs", "spatial_ref"):
        if ds.attrs.get(key):
            return ds.attrs[key]
    return fallback


def prep_loocv_table(df, grid_crs):
    """Standardise the leave-one-out table into the columns the model expects."""
    time_col = first_present(df, ["time", "hour_utc", "datetime", "timestamp"], "time column")
    obs_col = first_present(
        df, ["observed_phase", "phase_obs", "raw_phase", "phase", "obs_phase", "phase_label"],
        "observed phase column",
    )
    snow_col = first_present(df, ["mros_p_snow_loocv", "p_snow_loocv", "p_snow_cv", "p_snow"],
                             "leave-one-out snow probability")
    mix_col = first_present(df, ["mros_p_mix_loocv", "p_mix_loocv", "p_mix_cv", "p_mix"],
                            "leave-one-out mix probability")
    rain_col = first_present(df, ["mros_p_rain_loocv", "p_rain_loocv", "p_rain_cv", "p_rain"],
                             "leave-one-out rain probability")

    x_col = first_present(df, ["x", "x_proj", "grid_x"])
    y_col = first_present(df, ["y", "y_proj", "grid_y"])
    lon_col = first_present(df, ["lon", "longitude", "x_lon"])
    lat_col = first_present(df, ["lat", "latitude", "y_lat"])
    if (x_col is None or y_col is None) and (lon_col is None or lat_col is None):
        raise KeyError("Leave-one-out table has neither projected x/y nor lon/lat.")

    out = pd.DataFrame({
        "time": (pd.to_datetime(df[time_col], errors="coerce", utc=True)
                   .dt.floor("h").dt.tz_localize(None)),
        "phase_full": df[obs_col].map(normalize_phase_value),
        "mros_p_snow_loocv": pd.to_numeric(df[snow_col], errors="coerce"),
        "mros_p_mix_loocv": pd.to_numeric(df[mix_col], errors="coerce"),
        "mros_p_rain_loocv": pd.to_numeric(df[rain_col], errors="coerce"),
    })

    if lon_col is not None:
        out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    if lat_col is not None:
        out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")

    if x_col is not None and y_col is not None:
        out["x"] = pd.to_numeric(df[x_col], errors="coerce")
        out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    else:
        transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
        out["x"], out["y"] = transformer.transform(
            out["lon"].to_numpy(dtype=float), out["lat"].to_numpy(dtype=float)
        )

    for col in ["hour_utc", "elev", "pred_phase", "obs_p_snow", "obs_p_mix", "obs_p_rain",
                "pred_max_prob", "pred_entropy", "pred_correct", "station_id", "obs_id",
                "source"]:
        if col in df.columns and col not in out.columns:
            out[col] = df[col]
    if "elev" in df.columns:
        out["obs_elev"] = pd.to_numeric(df["elev"], errors="coerce")

    # Guard against rounding leaving the three probabilities off unity.
    prob_cols = ["mros_p_snow_loocv", "mros_p_mix_loocv", "mros_p_rain_loocv"]
    probs = np.clip(out[prob_cols].to_numpy(dtype=float), 0.0, 1.0)
    row_sum = probs.sum(axis=1)
    valid = row_sum > 0
    probs[valid] = probs[valid] / row_sum[valid, None]
    probs[~valid] = np.nan
    out[prob_cols] = probs

    out = out.dropna(subset=["time", "x", "y", "phase_full"] + prob_cols).copy()
    out["phase_full"] = out["phase_full"].astype(int)
    return out


def build_predictor_cube(ds_interp, ds_imerg):
    """Merge the interpolated and IMERG grids into one aligned cube."""
    parts = []
    interp_keep = [v for v in GRID_FEATURES if v in ds_interp]
    if interp_keep:
        parts.append(ds_interp[interp_keep])
    if "imerg_plp" in GRID_FEATURES and "imerg_plp" in ds_imerg:
        parts.append(ds_imerg[["imerg_plp"]])
    if not parts:
        raise ValueError("No predictor variables available for the cube.")
    return xr.merge(parts, compat="override", join="exact")


def nearest_index_1d(coord_vals, query_vals):
    """Index of the nearest coordinate value, for ascending or descending axes."""
    coord_vals = np.asarray(coord_vals)
    query_vals = np.asarray(query_vals)
    ascending = coord_vals[0] < coord_vals[-1]
    work = coord_vals if ascending else coord_vals[::-1]

    idx = np.clip(np.searchsorted(work, query_vals), 1, len(work) - 1)
    choose_right = np.abs(query_vals - work[idx]) < np.abs(query_vals - work[idx - 1])
    out = np.where(choose_right, idx, idx - 1)
    if not ascending:
        out = (len(coord_vals) - 1) - out
    return out.astype(np.int64)


def sample_cube_at_points(points_df, ds_pred, predictor_vars, verbose=True):
    """Read the cube at each observation, one hourly slice at a time.

    Only the hours that actually contain observations are opened, which keeps
    memory flat over a multi-year cube.
    """
    pts = points_df.copy().reset_index(drop=True)
    pts["time"] = pd.to_datetime(pts["time"]).dt.floor("h")

    predictor_vars = [v for v in predictor_vars if v in ds_pred.data_vars]
    if not predictor_vars:
        raise ValueError("None of the requested predictors are in the cube.")

    x_lo, x_hi = float(ds_pred["x"].min()), float(ds_pred["x"].max())
    y_lo, y_hi = float(ds_pred["y"].min()), float(ds_pred["y"].max())
    in_domain = (pts["x"].between(min(x_lo, x_hi), max(x_lo, x_hi))
                 & pts["y"].between(min(y_lo, y_hi), max(y_lo, y_hi)))
    if (~in_domain).any():
        print(f"  dropping {(~in_domain).sum()} points outside the grid")
        pts = pts.loc[in_domain].reset_index(drop=True)

    times = pd.to_datetime(ds_pred["time"].values)
    xvals, yvals = ds_pred["x"].values, ds_pred["y"].values

    pts["_t_idx"] = nearest_index_1d(
        times.astype("datetime64[ns]").astype("int64"),
        pd.to_datetime(pts["time"]).to_numpy().astype("datetime64[ns]").astype("int64"),
    )
    pts["_order"] = np.arange(len(pts))

    unique_tidx = np.sort(pts["_t_idx"].unique())
    if verbose:
        print(f"  sampling {len(pts):,} points across {len(unique_tidx):,} hours")

    chunks = []
    for t_idx in unique_tidx:
        chunk = pts.loc[pts["_t_idx"] == t_idx].copy()
        x_idx = nearest_index_1d(xvals, chunk["x"].to_numpy(dtype=float))
        y_idx = nearest_index_1d(yvals, chunk["y"].to_numpy(dtype=float))

        ds_hour = ds_pred.isel(time=int(t_idx))
        sampled = pd.DataFrame(
            {v: ds_hour[v].values[y_idx, x_idx] for v in predictor_vars},
            index=chunk.index,
        )
        chunks.append(pd.concat([chunk, sampled], axis=1))

    return (pd.concat(chunks, axis=0).sort_values("_order")
              .reset_index(drop=True).drop(columns=["_t_idx", "_order"]))


def build_region(region_id, interp_type=INTERP_TYPE):
    paths = model_paths(region_id, interp_type)
    make_output_dirs(paths)
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    ds_interp = xr.open_dataset(paths["interp_grid"])
    ds_imerg = xr.open_dataset(paths["imerg"])
    df_loocv_raw = pd.read_parquet(paths["mros_loocv"])

    # Put both grids on whole hours and keep only the hours they share.
    ds_interp = ds_interp.assign_coords(time=pd.to_datetime(ds_interp.time.values).floor("h"))
    ds_imerg = ds_imerg.assign_coords(time=pd.to_datetime(ds_imerg.time.values).floor("h"))
    common_times = np.intersect1d(ds_interp.time.values, ds_imerg.time.values)
    print(f"  {len(common_times):,} hours shared by the interpolated grid and IMERG")
    ds_interp = ds_interp.sel(time=common_times)
    ds_imerg = ds_imerg.sel(time=common_times)

    # IMERG is written north-to-south; flip it so the two grids line up.
    if ds_imerg.y.values[0] > ds_imerg.y.values[-1]:
        ds_imerg = ds_imerg.isel(y=slice(None, None, -1))

    ds_pred = build_predictor_cube(ds_interp, ds_imerg)

    loocv_df = prep_loocv_table(
        df_loocv_raw, get_dataset_crs(ds_pred, REGIONS[region_id]["utm_crs"])
    )
    loocv_df = loocv_df[loocv_df["time"].isin(pd.to_datetime(common_times))].copy()
    print(f"  {len(loocv_df):,} observations within the shared hours")
    print(loocv_df["phase_full"].value_counts().sort_index().to_string())

    master_df = sample_cube_at_points(loocv_df, ds_pred, GRID_FEATURES)

    required = FEATURES + [TARGET_FULL]
    missing = [c for c in required if c not in master_df.columns]
    if missing:
        raise KeyError(f"Missing columns after sampling: {missing}")

    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    master_df = master_df.dropna(subset=required).copy()
    master_df[TARGET_FULL] = master_df[TARGET_FULL].astype(int)

    master_df.to_parquet(paths["master_table"], index=False)
    print(f"  {len(master_df):,} usable observations -> {paths['master_table'].name}")
    print(master_df[TARGET_FULL].value_counts(normalize=True).sort_index().round(4).to_string())
    return master_df


def main(regions=None, interp_type=INTERP_TYPE):
    for region_id in regions or REGIONS:
        build_region(region_id, interp_type)


if __name__ == "__main__":
    main(parse_region_args())
