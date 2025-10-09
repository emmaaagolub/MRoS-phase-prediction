# %%
"""
Hourly Data Assimilation & Spatial Interpolation using **Kriging** (IDW-structured)
- Mirrors the finalized IDW gridding procedure structure
- Adds **dynamic lapse rate** via `estimate_lapse_rate()` per-hour using station data
- Elevation handled through detrend (to reference elevation) + retrend (per grid cell)
- Ordinary Kriging with auto variogram fitting (sampled for efficiency) and safe fallbacks

Outputs:
- CF-compliant NetCDF with hourly predictor stacks on the DEM grid
- Quicklook PNG maps per sampled hour with stations & MRoS markers overlayed

Notes:
- Variables: temp_air, temp_dew, temp_wet, rh (stations); mros_plp_proxy (MRoS); plp (IMERG)
- Dynamic/variable-specific min_points supported (e.g., PLP/MRoS allow lower threshold)
"""

# ============================ IMPORTS ============================
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import xy as rio_xy
import xarray as xr
import rioxarray  # noqa: F401
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

# Kriging
try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except Exception:
    PYKRIGE_AVAILABLE = False

# %%
# ============================ CONFIG ============================
BASE_DIR = Path().resolve().parent

CONFIG: Dict = {
    # --- Time window ---
    "wy_start": "2024-10-01T00:00:00Z",
    "wy_end":   "2025-05-31T23:59:59Z",
    # test window (subset for dev runs)
    "test_start": "2025-03-30T00:00:00Z",
    "test_end":   "2025-04-01T00:00:00Z",

    # --- IO paths --- (adjust as needed)
    "station_meta_csv": BASE_DIR / "Data/Stations/station_metadata_20241001_20250531.csv",
    "station_hourly_parquet": BASE_DIR / "outputs/hourly_pipeline/stations_hourly.parquet",
    "imerg_hourly_parquet":   BASE_DIR / "outputs/hourly_pipeline/imerg_hourly.parquet",
    "mros_hourly_parquet":    BASE_DIR / "outputs/hourly_pipeline/mros_hourly.parquet",
    "dem_path":               BASE_DIR / "DEM_1km.tif",
    "out_dir":                BASE_DIR / "outputs/hourly_pipeline",

    # --- Method switch (kriging | idw) -> kept for parity/experiments ---
    "method": "kriging",

    # --- Interpolation common params ---
    "min_points_default": 3,          # baseline min points for most variables
    "min_points_plp": 1,              # allow sparse PLP/MRoS
    "k_nearest": 8,                   # used by IDW fallback only
    "idw_power": 2.0,                 # used by IDW fallback only

    # --- Kriging params ---
    "variogram_model": "spherical",  # 'linear'|'power'|'gaussian'|'spherical'|'exponential'
    "variogram_parameters": {         # if any are None -> auto-fit
        "sill": None,
        "range": None,
        "nugget": 0.0,
    },
    "max_points_for_variogram": 100,  # sample size to fit variogram (controls O(n^2))
    "enable_plotting": False,
    "chunk_size": 2_000,              # predict in chunks to manage memory
    "return_variance": False,         # set True to store variance (larger files)

    # --- Lapse options ---
    # lapse is dynamic per-hour via estimate_lapse_rate(); this is only a fallback
    "fallback_lapse_C_per_m": -0.005,

    # --- Projection fallback ---
    "proj_fallback": "EPSG:3310",

    # --- Output filenames ---
    "out_nc_name": "hourly_predictors_1km_kriging.nc",
    "quicklook_every": 6,  # sample every N frames for quicklook maps
}

OUT_DIR = Path(CONFIG["out_dir"]); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variables and their sources & whether to apply lapse correction
VARIABLES: List[Tuple[str, str, bool]] = [
    ("temp_air",       "station", True),
    ("temp_dew",       "station", True),
    ("temp_wet",       "station", True),
    ("rh",             "station", False),
    ("mros_plp_proxy", "mros",    False),
    ("plp",            "imerg",   False),
]

# Variable-specific min_points overrides
MIN_POINTS_BY_VAR: Dict[str, int] = {
    "plp":  CONFIG["min_points_plp"],
    "mros_plp_proxy": CONFIG["min_points_plp"],
}

# %%
# ============================ UTILITIES ============================

def hourly_index(start_iso: str, end_iso: str) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.to_datetime(start_iso), end=pd.to_datetime(end_iso),
                         freq="H", tz="UTC")

def print_time(ts) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%MZ")

# --- transforms ---

def build_transformer(src_epsg: str, dst_crs) -> Transformer:
    return Transformer.from_crs(src_epsg, dst_crs, always_xy=True)

# --- grid helpers ---

def grid_centers(profile: dict) -> np.ndarray:
    T = profile["transform"]
    xs = T.c + (np.arange(profile["width"]) + 0.5) * T.a
    ys = T.f + (np.arange(profile["height"]) + 0.5) * T.e
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])

# %%
# ============================ DATA LOADING ============================

# DEM
with rio.open(CONFIG["dem_path"]) as src:
    DEM_PROFILE = src.profile
    DEM_DATA = src.read(1)
    DEM_CRS = src.crs
    DEM_TRANSFORM = src.transform

GRID_XY = grid_centers(DEM_PROFILE)
GRID_ELEV = DEM_DATA.ravel().astype(float)
PROJ_CRS = DEM_PROFILE["crs"] or CRS.from_user_input(CONFIG["proj_fallback"])  # type: ignore

# Hourly datasets (already preprocessed elsewhere in your pipeline)
st_hr   = pd.read_parquet(CONFIG["station_hourly_parquet"])  # must contain: id, lon, lat, elev, hour_utc, temp_air/temp_dew/temp_wet/rh
imerg_hr = pd.read_parquet(CONFIG["imerg_hourly_parquet"])   # must contain: lon, lat, hour_utc, plp
mros     = pd.read_parquet(CONFIG["mros_hourly_parquet"])    # must contain: lon, lat, hour_utc, mros_plp_proxy

# Normalize/align time
for df in (st_hr, imerg_hr, mros):
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True, errors="coerce").dt.floor("H")

H, W = DEM_PROFILE["height"], DEM_PROFILE["width"]
rows = np.arange(H); cols = np.arange(W)
X_CENTERS = np.array([rio_xy(DEM_TRANSFORM, 0.5, c + 0.5, offset="center")[0] for c in cols])
Y_CENTERS = np.array([rio_xy(DEM_TRANSFORM, r + 0.5, 0.5, offset="center")[1] for r in rows])

# %%
# FUNCTIONS

# ====================== DYNAMIC LAPSE ESTIMATION ======================

def estimate_lapse_rate(st_hour: pd.DataFrame,
                        temp_col: str = "temp_air",
                        elev_col: str = "elev",
                        min_points: int = 3,
                        fallback: float = CONFIG["fallback_lapse_C_per_m"]) -> float:
    """Estimate per-hour temperature lapse rate (K/m) from station data using
    robust linear regression slope of temp vs elevation.
    Returns fallback if insufficient data or ill-conditioned.
    """
    df = st_hour.dropna(subset=[temp_col, elev_col]).copy()
    if len(df) < max(min_points, 3):
        return fallback
    x = df[elev_col].to_numpy(dtype=float)
    y = df[temp_col].to_numpy(dtype=float)
    # center to improve conditioning
    x0 = x - np.nanmean(x)
    y0 = y - np.nanmean(y)
    # simple least squares slope (y = a*x + b) ; slope a has units K/m
    denom = np.dot(x0, x0)
    if not np.isfinite(denom) or denom <= 0:
        return fallback
    slope = float(np.dot(x0, y0) / denom)
    if not np.isfinite(slope) or abs(slope) > 0.02:  # sanity bounds ~ |20 K/km|
        return fallback
    return slope

# ====================== IDW (FALLBACK) ===============================

def idw_grid_from_points(hour_points: pd.DataFrame,
                         grid_xy: np.ndarray,
                         grid_elev: np.ndarray,
                         proj_crs,
                         idw_power: float = CONFIG["idw_power"],
                         k: int = CONFIG["k_nearest"],
                         min_points: int = CONFIG["min_points_default"],
                         value_col: str = "temp_air",
                         station_elev_col: str = "elev",
                         apply_lapse: bool = False,
                         lapse: float = CONFIG["fallback_lapse_C_per_m"],) -> np.ndarray:
    pts = hour_points.dropna(subset=[value_col, "lon", "lat"]).copy()
    if pts.empty or pts[value_col].notna().sum() < min_points:
        return np.full(grid_elev.shape, np.nan, dtype=np.float32)

    tf = build_transformer("EPSG:4326", proj_crs)
    px, py = tf.transform(pts["lon"].values, pts["lat"].values)
    P = np.column_stack([px, py])

    values = pts[value_col].to_numpy(dtype=float)
    stn_elev = pts[station_elev_col].to_numpy(dtype=float) if station_elev_col in pts else np.zeros_like(values)

    tree = cKDTree(P)
    dists, idxs = tree.query(grid_xy, k=min(k, len(P)))
    if dists.ndim == 1:  # ensure 2D
        dists = dists[:, None]
        idxs = idxs[:, None]

    v_neighbors = values[idxs]
    if apply_lapse:
        zc = grid_elev[:, None]
        zj = stn_elev[idxs]
        v_neighbors = v_neighbors + lapse * (zc - zj)

    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(dists, idw_power)
    w[np.isinf(w)] = 1e12
    w[~np.isfinite(w)] = 0.0
    w_sum = w.sum(axis=1, keepdims=True)
    w_norm = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 0)

    valid_counts = np.sum(w > 0, axis=1)
    grid_vals = np.sum(w_norm * v_neighbors, axis=1)
    grid_vals[valid_counts < min_points] = np.nan
    return grid_vals.astype(np.float32)

# ====================== KRIGING CORE ================================

def _coerce_variogram_params(params):
    if params is None:
        return None
    if isinstance(params, dict):
        return params
    try:
        arr = np.asarray(params, dtype=float).ravel()
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.tolist()


def kriging_grid_from_points(hour_points: pd.DataFrame,
                             grid_xy: np.ndarray,
                             grid_elev: np.ndarray,
                             proj_crs,
                             min_points: int,
                             value_col: str,
                             station_elev_col: str = "elev",
                             apply_lapse: bool = False,
                             lapse: float = CONFIG["fallback_lapse_C_per_m"],
                             variogram_model: str = CONFIG["variogram_model"],
                             variogram_params: Optional[Dict] = CONFIG["variogram_parameters"],
                             max_points: int = CONFIG["max_points_for_variogram"],
                             enable_plotting: bool = CONFIG["enable_plotting"],
                             chunk_size: int = CONFIG["chunk_size"],
                             return_variance: bool = CONFIG["return_variance" ],) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with optional lapse detrend/retrend and safe fallbacks."""
    if not PYKRIGE_AVAILABLE:
        z = idw_grid_from_points(hour_points, grid_xy, grid_elev, proj_crs,
                                 min_points=min_points, value_col=value_col,
                                 station_elev_col=station_elev_col,
                                 apply_lapse=apply_lapse, lapse=lapse)
        if return_variance:
            return z, np.full_like(z, np.nan, dtype=np.float32)
        return z

    pts = hour_points.dropna(subset=[value_col, "lon", "lat"]).reset_index(drop=True)
    if pts.empty or pts[value_col].notna().sum() < min_points:
        if return_variance:
            nan_arr = np.full(grid_elev.shape, np.nan, dtype=np.float32)
            return nan_arr, nan_arr.copy()
        return np.full(grid_elev.shape, np.nan, dtype=np.float32)

    # transform once
    tf = build_transformer("EPSG:4326", proj_crs)
    px_all, py_all = tf.transform(pts["lon"].values, pts["lat"].values)
    values_all = pts[value_col].to_numpy(dtype=float)

    # sample for variogram fit
    if len(pts) > max_points:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(pts), size=max_points, replace=False)
    else:
        sample_idx = np.arange(len(pts), dtype=int)
    px_s = px_all[sample_idx]
    py_s = py_all[sample_idx]
    values_s = values_all[sample_idx]

    # Detrend to reference elevation
    if apply_lapse and (station_elev_col in pts.columns):
        stn_elev_all = pts[station_elev_col].to_numpy(dtype=float)
        if grid_elev is not None and np.isfinite(grid_elev).any():
            ref_elev = float(np.nanmean(grid_elev))
        else:
            ref_elev = float(np.nanmean(stn_elev_all))
        if not np.isfinite(ref_elev):
            ref_elev = 0.0
        values_all = values_all + lapse * (ref_elev - stn_elev_all)
        values_s   = values_s   + lapse * (ref_elev - stn_elev_all[sample_idx])
    else:
        ref_elev = 0.0

    # Auto-fit vs provided variogram params
    use_auto = (
        variogram_params is None or
        (isinstance(variogram_params, dict) and any(v is None for v in variogram_params.values()))
    )

    try:
        if use_auto:
            OK_fit = OrdinaryKriging(px_s, py_s, values_s,
                                     variogram_model=variogram_model,
                                     verbose=False,
                                     enable_plotting=enable_plotting,
                                     coordinates_type='euclidean')
            fitted_params = getattr(OK_fit, "variogram_model_parameters",
                             getattr(OK_fit, "variogram_parameters", None))
            fitted_params = _coerce_variogram_params(fitted_params)
        else:
            fitted_params = _coerce_variogram_params(variogram_params)

        if fitted_params is None:
            OK = OrdinaryKriging(px_all, py_all, values_all,
                                 variogram_model=variogram_model,
                                 verbose=False,
                                 enable_plotting=enable_plotting,
                                 coordinates_type='euclidean')
        else:
            OK = OrdinaryKriging(px_all, py_all, values_all,
                                 variogram_model=variogram_model,
                                 variogram_parameters=fitted_params,
                                 verbose=False,
                                 enable_plotting=enable_plotting,
                                 coordinates_type='euclidean')

        n = len(grid_xy)
        n_chunks = (n + chunk_size - 1) // chunk_size
        z_pred = np.full(n, np.nan, dtype=np.float32)
        v_pred = np.full(n, np.nan, dtype=np.float32) if return_variance else None

        for i in range(n_chunks):
            s = i * chunk_size
            e = min((i + 1) * chunk_size, n)
            chunk_xy = grid_xy[s:e]
            try:
                cz, cv = OK.execute('points', chunk_xy[:, 0], chunk_xy[:, 1])
                z_pred[s:e] = np.asarray(cz, dtype=np.float32)
                if return_variance:
                    v_pred[s:e] = np.asarray(cv, dtype=np.float32)
            except Exception:
                pass

        # Retrend from ref elevation to each grid cell
        if apply_lapse and grid_elev is not None:
            z_pred = z_pred + lapse * (grid_elev - ref_elev)
        return (z_pred, v_pred) if return_variance else z_pred

    except Exception:
        z = idw_grid_from_points(pts, grid_xy, grid_elev, proj_crs,
                                 min_points=min_points, value_col=value_col,
                                 station_elev_col=station_elev_col,
                                 apply_lapse=apply_lapse, lapse=lapse)
        if return_variance:
            return z, np.full_like(z, np.nan, dtype=np.float32)
        return z

# %%
# ====================== HOURLY LOOP ===============================

hours = hourly_index(CONFIG["test_start"], CONFIG["test_end"])

coords = {
    "time": hours,
    "y": Y_CENTERS,
    "x": X_CENTERS,
}

# Pre-allocate arrays
DATA_VARS = {name: np.full((len(hours), H, W), np.nan, dtype=np.float32) for (name, _, _) in VARIABLES}

# Helper: per-hour min_points
get_min_points = lambda var: MIN_POINTS_BY_VAR.get(var, CONFIG["min_points_default"])  # noqa: E731

# Helper: assemble point dataframe for each variable/source

def points_for_var(src: str, var: str,
                   st_t: pd.DataFrame,
                   mros_t: pd.DataFrame,
                   imerg_t: pd.DataFrame) -> pd.DataFrame:
    if src == "station":
        if var not in st_t.columns:
            return pd.DataFrame(columns=["lon","lat","elev",var])
        return st_t[["lon","lat","elev", var]].dropna(subset=[var, "lon", "lat"])
    if src == "mros":
        if "mros_plp_proxy" not in mros_t.columns:
            return pd.DataFrame(columns=["lon","lat","elev","val"])  
        return mros_t.rename(columns={"mros_plp_proxy":"val"})[["lon","lat","val"]].assign(elev=0.0)
    if src == "imerg":
        if "plp" not in imerg_t.columns:
            return pd.DataFrame(columns=["lon","lat","elev","val"])  
        return imerg_t.rename(columns={"plp":"val"})[["lon","lat","val"]].assign(elev=0.0)
    raise ValueError(f"Unknown source {src}")

# Iterate hours
for ti, t in enumerate(tqdm(hours, desc="Hourly surfaces", ncols=96)):
    st_t   = st_hr.loc[st_hr["hour_utc"] == t]
    mros_t = mros.loc[mros["hour_utc"] == t]
    imerg_t= imerg_hr.loc[imerg_hr["hour_utc"] == t]

    # --- Dynamic lapse per-hour from stations (for temps) ---
    # If insufficient stations for the target temp var, the estimate function will return fallback
    lapse_air = estimate_lapse_rate(st_t, temp_col="temp_air",
                                    elev_col="elev",
                                    min_points=get_min_points("temp_air"),
                                    fallback=CONFIG["fallback_lapse_C_per_m"])

    for (name, src, use_lapse) in VARIABLES:
        min_points = get_min_points(name)
        pts = points_for_var(src, name, st_t, mros_t, imerg_t)
        if pts.empty or (name in pts.columns and pts[name].notna().sum() < min_points) or (
            "val" in pts.columns and pts["val"].notna().sum() < min_points):
            continue

        value_col = name if src == "station" else "val"

        # choose lapse to apply
        apply_lapse = bool(use_lapse)
        lapse = lapse_air if apply_lapse else 0.0

        # Select method (kept for parity; here we use kriging)
        if CONFIG["method"] == "kriging":
            vals = kriging_grid_from_points(
                pts, GRID_XY, GRID_ELEV, PROJ_CRS,
                min_points=min_points,
                value_col=value_col,
                station_elev_col="elev",
                apply_lapse=apply_lapse,
                lapse=lapse,
                variogram_model=CONFIG["variogram_model"],
                variogram_params=CONFIG["variogram_parameters"],
                max_points=CONFIG["max_points_for_variogram"],
                enable_plotting=CONFIG["enable_plotting"],
                chunk_size=CONFIG["chunk_size"],
                return_variance=CONFIG["return_variance"],
            )
            if CONFIG["return_variance"]:
                vals, _var = vals  # ignore variance unless later stored
        else:
            vals = idw_grid_from_points(
                pts, GRID_XY, GRID_ELEV, PROJ_CRS,
                min_points=min_points,
                value_col=value_col,
                station_elev_col="elev",
                apply_lapse=apply_lapse,
                lapse=lapse,
            )

        assert vals.size == H * W, f"Interpolation returned {vals.size}, expected {H*W}"
        DATA_VARS[name][ti, :, :] = vals.reshape(H, W)

# ====================== BUILD DATASET & SAVE ========================

ds = xr.Dataset(
    {
        **{k: xr.DataArray(v, coords=coords, dims=("time","y","x")) for k, v in DATA_VARS.items()},
        "elev": xr.DataArray(DEM_DATA.astype(np.float32), coords={"y": Y_CENTERS, "x": X_CENTERS}, dims=("y","x")),
    },
    attrs={
        "title": "Hourly predictor stacks on 1-km grid (Kriging)",
        "interpolation_method": CONFIG["method"],
        "variogram_model": CONFIG["variogram_model"],
        "variogram_parameters": json.dumps(CONFIG["variogram_parameters"]),
        "min_points_default": CONFIG["min_points_default"],
        "min_points_plp": CONFIG["min_points_plp"],
        "max_points_for_variogram": CONFIG["max_points_for_variogram"],
        "dynamic_lapse": True,
    }
)

# coordinate metadata
ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
ds = ds.rio.write_crs(DEM_PROFILE["crs"])  # spatial_ref
ds = ds.rio.write_transform(DEM_TRANSFORM)
for v in ds.data_vars:
    ds[v].attrs.setdefault("grid_mapping", "spatial_ref")

# %%
# reasonable chunking/encoding

def _chunks_for(da: xr.DataArray):
    if da.dims == ("time","y","x"):
        return (min(24, da.sizes["time"]), min(256, da.sizes["y"]), min(256, da.sizes["x"]))
    if da.dims == ("y","x"):
        return (min(256, da.sizes["y"]), min(256, da.sizes["x"]))
    return None

encoding = {}
for name, da in ds.data_vars.items():
    ch = _chunks_for(da)
    encoding[name] = ({"zlib": True, "complevel": 4, "chunksizes": ch}
                      if ch is not None else {"zlib": True, "complevel": 4} if da.ndim > 0 else {})

out_nc = OUT_DIR / CONFIG["out_nc_name"]
# ensure naive time for netcdf4
if hasattr(ds.indexes.get("time", None), "tz") and ds.indexes["time"].tz is not None:
    ds = ds.assign_coords(time=ds.indexes["time"].tz_localize(None))

ds.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)
print(f"Wrote {out_nc}")


# %%
# ====================== QUICKLOOK MAPS =============================

def quicklook_hour(ds: xr.Dataset, t: np.datetime64,
                   st_t: pd.DataFrame, mros_t: pd.DataFrame,
                   out_png: Path,
                   vars_to_show=("plp","mros_plp_proxy","temp_air","temp_dew","temp_wet","rh")):
    if t not in ds.time.values:
        return
    xvals = ds["x"].values; yvals = ds["y"].values
    extent = [xvals.min(), xvals.max(), yvals.min(), yvals.max()]

    keep = [v for v in vars_to_show if v in ds.data_vars]
    if not keep:
        return
    n = len(keep); ncols = 3; nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.8*nrows), squeeze=False)
    fig.suptitle(f"Quicklook @ {print_time(t)}", fontsize=14)

    target_crs = ds.rio.crs or CRS.from_user_input(CONFIG["proj_fallback"])  # type: ignore
    tf = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    st_x = st_y = mo_x = mo_y = []
    if len(st_t):
        st_x, st_y = tf.transform(st_t["lon"].values,  st_t["lat"].values)
    if len(mros_t):
        mo_x, mo_y = tf.transform(mros_t["lon"].values, mros_t["lat"].values)

    ti = int(np.where(ds.time.values == np.datetime64(t))[0][0])

    for i, var in enumerate(keep):
        ax = axes[i // ncols, i % ncols]
        arr = ds[var].isel(time=ti).values
        if var in ("plp", "mros_plp_proxy", "rh"):
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="equal", vmin=0, vmax=100)
        else:
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="equal")
        ax.set_title(var)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if len(st_x):
            ax.scatter(st_x, st_y, s=15, c="white", edgecolor="k", marker="o", linewidths=0.5, label="Stations")
        if len(mo_x):
            ax.scatter(mo_x, mo_y, s=25, c="red", edgecolor="k", marker="^", linewidths=0.6, label="MRoS")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for j in range(n, nrows*ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=200); plt.close(fig)

# sample a few hours
quick_dir = OUT_DIR / "maps"; quick_dir.mkdir(parents=True, exist_ok=True)
sample = slice(None, None, max(1, len(hours)//CONFIG["quicklook_every"]))
for t in pd.to_datetime(ds.time.values)[sample]:
    t_utc = pd.to_datetime(t).tz_localize("UTC").floor("H")
    st_t = st_hr[st_hr["hour_utc"].dt.floor("H") == t_utc]
    mros_t = mros[mros["hour_utc"].dt.floor("H") == t_utc]
    quicklook_hour(ds, t, st_t, mros_t, out_png=quick_dir / f"Kriging_quick_{print_time(t).replace(':','-')}.png")
