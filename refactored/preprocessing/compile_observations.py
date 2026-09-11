"""Compile station, IMERG and MRoS observations onto a common hourly
grid for each region.

Stations
  Reads the per-station CSVs, converts local timestamps to UTC, and averages
  to one row per station-hour. Applies four filters: hard physical bounds, a
  per-station interquartile range test, an hour-to-hour step limit, and a
  repeated-value run test. Missing dewpoint and humidity are filled by
  inverse-distance interpolation from other stations with an elevation
  correction, then wet-bulb temperature is derived.

IMERG
  Reshapes the daily wide tables to one row per cell-hour and averages the two
  half-hourly probability values within each hour.

MRoS
  Keeps the last report per observer per hour and maps the reported phase to a
  numeric proxy (snow 0, mix 50, rain 100).

All three are given DEM elevation and clipped to the study-area polygon.

Output: outputs/compiled/<REGION>/hourly_data/{stations,imerg,mros}_hourly.parquet
"""

import math
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytz
import rasterio as rio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CRS_WGS84, REGIONS, WY_END, WY_START, parse_region_args, region_paths,
)

# Hard bounds; readings outside these are set to missing.
PHYSICAL_CUTOFFS = {"temp_air": (-45, 45), "temp_dew": (-50, 25), "rh": (0, 100)}

# Quality-control thresholds for the hourly station series.
IQR_MULTIPLIER = 3.0          # per-station outlier width
SPIKE_THRESHOLDS = {"temp_air": 10.0, "temp_dew": 8.0, "rh": 40.0}  # max change per hour
MIN_STUCK_RUN = 6             # identical consecutive values flagged as a stuck sensor

MROS_PHASE_TO_PROXY = {"snow": 0.0, "mix": 50.0, "rain": 100.0}

# Column patterns in the IMERG parquets.
TIME_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?$")
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PLP_SLOT_RE = re.compile(r"^plp_(\d{2})$")


# ---------------------------------------------------------------------------
# Meteorology
# ---------------------------------------------------------------------------

def esat_hpa(temp_c):
    """Saturation vapor pressure over water (hPa), Magnus/Tetens form. Bolton's coefficients."""
    return 6.112 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def td_from_ta_rh(temp_c, rh):
    """Dewpoint (degC) from air temperature and relative humidity."""
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 1e-6, 100.0)
    a, b = 17.27, 237.3
    gamma = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def rh_from_ta_td(temp_c, dew_c):
    """Relative humidity (%) from air temperature and dewpoint."""
    temp_c = np.asarray(temp_c, dtype=float)
    dew_c = np.asarray(dew_c, dtype=float)
    a, b = 17.27, 237.3
    rh = 100.0 * np.exp((a * dew_c) / (b + dew_c) - (a * temp_c) / (b + temp_c))
    return np.clip(rh, 0.0, 100.0)


def tw_stull(temp_c, rh):
    """Wet-bulb temperature (degC), Stull (2011) empirical fit."""
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 1e-6, 100.0)
    return (temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(temp_c + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def psychrometric_tw(temp_c, rh, pressure_hpa=1013.25):
    """Wet-bulb temperature by iterating the energy balance.

    Used only where the Stull fit falls outside its valid range.
    """
    temp_c, rh = float(temp_c), float(rh)
    if not math.isfinite(temp_c) or not math.isfinite(rh):
        return np.nan

    vapor_pressure = (rh / 100.0) * esat_hpa(temp_c)
    cp, latent_heat, mw_ratio = 1004.0, 2.5e6, 0.622
    gamma = cp * pressure_hpa / (latent_heat * mw_ratio) / 100.0

    tw = temp_c
    for _ in range(50):
        tw_new = tw + 0.2 * ((temp_c - tw) * gamma - (esat_hpa(tw) - vapor_pressure))
        if abs(tw_new - tw) < 0.01:
            break
        tw = tw_new
    return tw


def fill_station_row_vars(df):
    """Make air temperature, dewpoint, humidity and wet-bulb mutually consistent."""
    has = {"temp_air", "temp_dew", "rh"} <= set(df.columns)

    if has:
        missing_rh = df["rh"].isna() & df["temp_air"].notna() & df["temp_dew"].notna()
        df.loc[missing_rh, "rh"] = rh_from_ta_td(
            df.loc[missing_rh, "temp_air"], df.loc[missing_rh, "temp_dew"]
        )
        missing_td = df["temp_dew"].isna() & df["temp_air"].notna() & df["rh"].notna()
        df.loc[missing_td, "temp_dew"] = td_from_ta_rh(
            df.loc[missing_td, "temp_air"], df.loc[missing_td, "rh"]
        )

    if "rh" in df:
        df["rh"] = np.clip(df["rh"].astype(float), 0, 100)

    if "temp_air" in df and "rh" in df:
        ta = df["temp_air"].astype(float)
        rh = df["rh"].astype(float)
        tw = tw_stull(ta, rh)
        df["temp_wet"] = tw

        # Wet-bulb must sit between dewpoint and air temperature; where it does
        # not, fall back to the iterative solution.
        if "temp_dew" in df:
            bad = (tw < df["temp_dew"].astype(float)) | (tw > ta)
        else:
            bad = tw > ta
        if bad.any():
            df.loc[bad, "temp_wet"] = [
                psychrometric_tw(t, h) for t, h in zip(ta[bad], rh[bad])
            ]

    return df


def estimate_lapse_rate(stations, default=-0.005):
    """Fit degC per metre from the stations reporting this hour.

    Falls back to the default whenever the fit is unavailable or implausible.
    """
    if len(stations) < 5 or stations["temp_air"].isna().all():
        return default
    elev = stations[["elev"]].values
    temp = stations["temp_air"].values
    if np.all(np.isfinite(elev)) and np.all(np.isfinite(temp)):
        slope = LinearRegression().fit(elev, temp).coef_[0]
        if -0.009 < slope < -0.003:
            return slope
    return default


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def apply_physical_cutoffs(df, verbose=False):
    """Null readings that lie outside hard physical bounds."""
    for col, (lo, hi) in PHYSICAL_CUTOFFS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.any():
            if verbose:
                print(f"  physical bounds: nulled {mask.sum()} {col} values")
            df.loc[mask, col] = np.nan
    return df


def add_elev_from_dem(df, dem_path, lon_col="lon", lat_col="lat"):
    """Attach DEM elevation at each point."""
    with rio.open(dem_path) as src:
        transformer = Transformer.from_crs(CRS_WGS84, src.crs, always_xy=True)
        xs, ys = transformer.transform(df[lon_col].values, df[lat_col].values)
        elev = np.array([v[0] for v in src.sample(zip(xs, ys))])
        if src.nodata is not None:
            elev = np.where(elev == src.nodata, np.nan, elev)
    return df.assign(elev=elev)


def filter_points_to_aoi(df, aoi_poly):
    """Keep only points inside the study-area polygon."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84
    )
    return df.loc[gdf.within(aoi_poly).values]


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def load_station_meta(meta_csv):
    meta = pd.read_csv(meta_csv)
    missing = {"id", "lat", "lon", "elev", "timezone_lst"} - set(meta.columns)
    if missing:
        raise ValueError(f"Station metadata missing columns: {missing}")
    meta["id"] = meta["id"].astype(str)
    return meta


def load_station_timeseries(station_dir, meta):
    """Read every station CSV and put all timestamps on UTC."""
    files = [p for p in Path(station_dir).glob("*.csv") if "meta" not in p.name.lower()]
    keep = ["id", "datetime", "temp_air", "temp_dew", "rh"]
    frames, tz_failed = [], []

    for path in files:
        df = pd.read_csv(path, low_memory=False)
        if "id" not in df.columns:
            df["id"] = path.stem
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        df = df[keep]
        df["id"] = df["id"].astype(str)

        # Repeated header rows appear when downloads are appended.
        df = df[df["datetime"] != "datetime"]
        for col in ["temp_air", "temp_dew", "rh"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = apply_physical_cutoffs(df)

        tz_vals = meta.loc[meta["id"] == df["id"].iloc[0], "timezone_lst"].values
        dt_local = pd.to_datetime(df["datetime"], errors="coerce")
        if len(tz_vals) == 1:
            try:
                tz = pytz.timezone(tz_vals[0])
                if getattr(dt_local.dt, "tz", None) is None:
                    df["datetime"] = dt_local.dt.tz_localize(
                        tz, ambiguous="NaT", nonexistent="NaT"
                    ).dt.tz_convert("UTC")
                else:
                    df["datetime"] = dt_local.dt.tz_convert("UTC")
            except Exception as exc:
                tz_failed.append(df["id"].iloc[0])
                print(f"  timezone '{tz_vals}' failed for {df['id'].iloc[0]}: {exc}")
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

        frames.append(df)

    if tz_failed:
        print(f"  {len(tz_failed)} station(s) assumed UTC — check these: {tz_failed}")
    if not frames:
        return pd.DataFrame(columns=keep)
    return pd.concat(frames, ignore_index=True)


def hourly_station_agg(st_df, meta):
    """Average each station to hourly values and screen out bad readings."""
    df = st_df.merge(meta, on="id", how="left")
    df["hour_utc"] = df["datetime"].dt.floor("h")

    agg = df.groupby(["id", "hour_utc"], as_index=False).agg(
        temp_air=("temp_air", "mean"),
        temp_dew=("temp_dew", "mean"),
        rh=("rh", "mean"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        elev=("elev", "first"),
    )
    agg = apply_physical_cutoffs(agg, verbose=True)

    # Per-station interquartile range test.
    for col in ["temp_air", "temp_dew"]:
        q1 = agg.groupby("id")[col].transform("quantile", 0.25)
        q3 = agg.groupby("id")[col].transform("quantile", 0.75)
        iqr = q3 - q1
        mask = (agg[col] < q1 - IQR_MULTIPLIER * iqr) | (agg[col] > q3 + IQR_MULTIPLIER * iqr)
        if mask.any():
            print(f"  outlier filter: nulled {mask.sum()} {col} values")
        agg.loc[mask, col] = np.nan

    # Hour-to-hour step limit.
    agg = agg.sort_values(["id", "hour_utc"]).reset_index(drop=True)
    for col, thresh in SPIKE_THRESHOLDS.items():
        delta = agg.groupby("id")[col].diff().abs()
        if (delta > thresh).any():
            print(f"  spike filter: nulled {(delta > thresh).sum()} {col} values")
        agg.loc[delta > thresh, col] = np.nan

    # Runs of identical consecutive values.
    for col in ["temp_air", "temp_dew"]:
        is_same = agg.groupby("id")[col].transform(lambda s: s == s.shift(1))
        agg["_run_id"] = (~is_same).groupby(agg["id"]).cumsum()
        run_len = is_same.groupby([agg["id"], agg["_run_id"]]).transform("sum") + 1
        agg = agg.drop(columns=["_run_id"])
        stuck = (run_len >= MIN_STUCK_RUN) & is_same
        if stuck.any():
            print(f"  stuck-sensor filter: nulled {stuck.sum()} {col} values")
        agg.loc[stuck, col] = np.nan

    # Dewpoint cannot exceed air temperature.
    agg.loc[agg["temp_dew"] > agg["temp_air"], "temp_dew"] = np.nan
    return agg


def lapse_adjusted_idw(st_hr, utm_crs, value_cols=("temp_dew", "rh"),
                       k=6, power=2.0, fallback_lapse=-0.0065):
    """Fill station gaps from neighbouring stations, corrected for elevation.

    Each hour: fit a lapse rate, normalise the reporting stations to a common
    elevation, interpolate by inverse distance weighting, then put the
    interpolated values back at each station's own elevation. Non-temperature
    variables skip the elevation correction.
    """
    df = st_hr.copy().sort_values(["id", "hour_utc"]).reset_index(drop=True)
    thermal_vars = {"temp_air", "temp_dew", "temp_wet"}

    transformer = Transformer.from_crs(CRS_WGS84, utm_crs, always_xy=True)
    df["_px"], df["_py"] = transformer.transform(df["lon"].values, df["lat"].values)

    modeled = {v: np.full(len(df), np.nan) for v in value_cols}

    bad_coords = df[~(np.isfinite(df["_px"]) & np.isfinite(df["_py"]))]["id"].unique()
    if len(bad_coords):
        print(f"  skipping {len(bad_coords)} station(s) with bad coordinates: {bad_coords}")

    for _, group in df.groupby("hour_utc"):
        lapse_source = group[group["temp_air"].notna() & group["elev"].notna()]
        lapse = (estimate_lapse_rate(lapse_source, default=fallback_lapse)
                 if len(lapse_source) >= 5 else fallback_lapse)
        z_ref = np.nanmean(group["elev"].values)

        valid = np.isfinite(group["_px"].values) & np.isfinite(group["_py"].values)
        if not valid.any():
            continue
        gx, gy = group["_px"].values[valid], group["_py"].values[valid]
        elev_valid = group["elev"].values[valid]
        valid_idx = group.index[valid]

        for var in value_cols:
            avail = group[group[var].notna() & group["elev"].notna()
                          & group["_px"].notna() & group["_py"].notna()]
            if len(avail) < 4:
                continue

            tree = cKDTree(np.c_[avail["_px"].values, avail["_py"].values])
            k_actual = min(k, len(avail))
            dists, idxs = tree.query(np.c_[gx, gy], k=k_actual, workers=-1)
            if k_actual == 1:
                dists, idxs = dists[:, None], idxs[:, None]

            weights = 1.0 / np.maximum(dists, 1e-6) ** power
            weights /= weights.sum(axis=1, keepdims=True)

            if var in thermal_vars:
                normed = avail[var].values - lapse * (avail["elev"].values - z_ref)
                pred = np.sum(normed[idxs] * weights, axis=1) + lapse * (elev_valid - z_ref)
            else:
                pred = np.sum(avail[var].values[idxs] * weights, axis=1)

            modeled[var][valid_idx] = pred

    for var in value_cols:
        df[f"{var}_modeled"] = modeled[var]
        print(f"  filled {np.sum(~np.isnan(modeled[var])):,} hourly {var} values")

    return df.drop(columns=["_px", "_py"])


def build_stations(paths, utm_crs):
    meta = load_station_meta(paths["station_meta"])
    st_hr = hourly_station_agg(load_station_timeseries(paths["station_dir"], meta), meta)

    st_hr = st_hr[(st_hr["hour_utc"] >= pd.to_datetime(WY_START))
                  & (st_hr["hour_utc"] <= pd.to_datetime(WY_END))]
    print(f"  station-hours in window: {len(st_hr):,}")

    value_cols = ("temp_dew", "rh")
    st_hr = lapse_adjusted_idw(st_hr, utm_crs, value_cols=value_cols)
    for var in value_cols:
        st_hr[var] = st_hr[var].where(st_hr[var].notna(), st_hr[f"{var}_modeled"])
    st_hr = st_hr.drop(columns=[f"{v}_modeled" for v in value_cols])

    st_hr = fill_station_row_vars(st_hr)
    print("  remaining missing fraction:")
    print(st_hr[["temp_dew", "rh", "temp_wet"]].isna().mean().round(3).to_string())
    return st_hr


# ---------------------------------------------------------------------------
# IMERG
# ---------------------------------------------------------------------------

def read_imerg_wide_to_long(path):
    """Melt one daily IMERG file into rows of (time, lat, lon, plp)."""
    df = pq.read_table(path).to_pandas().rename(columns={"x": "lon", "y": "lat"})

    # Older files name their columns by timestamp; current ones use plp_01..plp_48.
    time_cols = [c for c in df.columns if TIME_COL_RE.match(str(c))]
    if not time_cols:
        time_cols = [c for c in df.columns
                     if re.match(r"^\d{4}-\d{2}-\d{2}", str(c))
                     and not DATE_ONLY_RE.match(str(c))]

    if time_cols:
        long = df.melt(id_vars=["lat", "lon"], value_vars=time_cols,
                       var_name="time_str", value_name="plp_raw")
        long["time_utc"] = pd.to_datetime(long["time_str"], utc=True, errors="coerce")
    else:
        slot_cols = sorted(
            [c for c in df.columns if PLP_SLOT_RE.match(str(c))],
            key=lambda c: int(PLP_SLOT_RE.match(c).group(1)),
        )
        if not slot_cols:
            raise ValueError(f"No time or plp_NN columns in {path.name}")
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if not date_match:
            raise ValueError(f"Cannot read a date from {path.name}")
        file_date = pd.Timestamp(date_match.group(1), tz="UTC")

        long = df.melt(id_vars=["lat", "lon"], value_vars=slot_cols,
                       var_name="slot", value_name="plp_raw")
        slot_num = long["slot"].str.extract(r"plp_(\d+)")[0].astype(int)
        long["time_utc"] = file_date + pd.to_timedelta((slot_num - 1) * 30, unit="m")

    plp = pd.to_numeric(long["plp_raw"], errors="coerce").astype(float)
    if np.nanmax(plp) <= 1.0:  # some files store a fraction rather than a percent
        plp *= 100.0
    long["plp"] = plp

    return long.loc[long["time_utc"].notna(), ["time_utc", "lat", "lon", "plp"]]


def build_imerg(imerg_dir):
    """Combine the daily files and average each pixel's two slots per hour."""
    files = list(Path(imerg_dir).rglob("*.parquet"))
    empty = pd.DataFrame(columns=["hour_utc", "lat", "lon", "plp"])
    if not files:
        print(f"  no IMERG files under {imerg_dir}")
        return empty

    start_ts, end_ts = pd.to_datetime(WY_START, utc=True), pd.to_datetime(WY_END, utc=True)
    frames = []
    for path in files:
        df = read_imerg_wide_to_long(path)
        df = df[(df["time_utc"] >= start_ts) & (df["time_utc"] <= end_ts)]
        if not df.empty:
            frames.append(df)

    if not frames:
        print("  IMERG files found but none within the study window")
        return empty

    imerg = pd.concat(frames, ignore_index=True)
    imerg["hour_utc"] = imerg["time_utc"].dt.floor("h")
    return imerg.groupby(["hour_utc", "lat", "lon"], as_index=False).agg(plp=("plp", "mean"))


# ---------------------------------------------------------------------------
# MRoS observations
# ---------------------------------------------------------------------------

def parse_mros_datetime(date_series, time_series):
    """Parse the submitted date and time, allowing 2- or 4-digit years."""
    combined = date_series.str.strip() + " " + time_series.str.strip()
    dt = pd.to_datetime(combined, format="%m/%d/%Y %H:%M:%S", errors="coerce", utc=True)
    retry = dt.isna()
    if retry.any():
        dt[retry] = pd.to_datetime(combined[retry], format="%m/%d/%y %H:%M:%S",
                                   errors="coerce", utc=True)
    return dt


def build_mros(mros_parquet, aoi_lonlat):
    """Load reports, keep the latest per observer-hour, add the numeric proxy."""
    df = pq.read_table(mros_parquet).to_pandas()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    lons = [p[0] for p in aoi_lonlat]
    lats = [p[1] for p in aoi_lonlat]
    df = df.loc[df["longitude"].between(min(lons), max(lons))
                & df["latitude"].between(min(lats), max(lats))].copy()

    if "datetime_utc" not in df.columns:
        df["datetime_utc"] = parse_mros_datetime(df["date_submitted_utc"],
                                                 df["time_submitted_utc"])
        print(f"  parsed {df['datetime_utc'].notna().sum():,} / {len(df):,} timestamps")

    df["hour_utc"] = df["datetime_utc"].dt.floor("h")
    df["phase"] = df["phase"].str.lower()

    # A second report in the same hour is treated as a correction of the first.
    key_cols = ["hour_utc"]
    if "observer_id" in df.columns:
        key_cols.append("observer_id")
    else:
        df["lat_bin"] = df["latitude"].round(4)
        df["lon_bin"] = df["longitude"].round(4)
        key_cols += ["lat_bin", "lon_bin"]

    last = df.sort_values("datetime_utc").groupby(key_cols, as_index=False).tail(1)
    last["mros_plp_proxy"] = last["phase"].map(MROS_PHASE_TO_PROXY).astype(float)
    last = last[(last["hour_utc"] >= pd.to_datetime(WY_START))
                & (last["hour_utc"] <= pd.to_datetime(WY_END))]

    return last.rename(columns={"latitude": "lat", "longitude": "lon"})[
        ["hour_utc", "lat", "lon", "mros_plp_proxy", "phase"]
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_region(region_id):
    cfg = REGIONS[region_id]
    paths = region_paths(region_id)
    out_dir = paths["compiled_dir"] / "hourly_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {region_id}: {cfg['label']} ===")

    print("Stations...")
    st_hr = build_stations(paths, cfg["utm_crs"])

    print("IMERG...")
    imerg_hr = add_elev_from_dem(build_imerg(paths["imerg_dir"]), paths["dem_1km"])
    print(f"  {len(imerg_hr):,} cell-hours")

    print("MRoS...")
    mros_hr = add_elev_from_dem(
        build_mros(paths["mros_parquet"], cfg["aoi_lonlat"]), paths["dem_1km"]
    )
    print(f"  {len(mros_hr):,} reports")

    aoi_poly = Polygon(cfg["aoi_lonlat"])
    st_hr = filter_points_to_aoi(st_hr, aoi_poly)
    imerg_hr = filter_points_to_aoi(imerg_hr, aoi_poly)
    mros_hr = filter_points_to_aoi(mros_hr, aoi_poly)
    print(f"Inside study area — stations {len(st_hr):,}, "
          f"IMERG {len(imerg_hr):,}, MRoS {len(mros_hr):,}")

    st_hr.to_parquet(out_dir / "stations_hourly.parquet", index=False)
    imerg_hr.to_parquet(out_dir / "imerg_hourly.parquet", index=False)
    mros_hr.to_parquet(out_dir / "mros_hourly.parquet", index=False)
    print(f"Wrote {out_dir}")

    return st_hr, imerg_hr, mros_hr


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
