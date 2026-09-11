"""Resample PRISM and IMERG onto the 1 km hourly grid.

PRISM is daily at 800 m. It is regridded to the DEM grid, then expanded to
hourly: air temperature is interpolated from the daily minimum and maximum
using a fixed diurnal curve, and the remaining variables repeat their daily
value at each hour.

IMERG is already hourly and is only regridded.

Inputs:  Data/PRISM/<REGION>/combined_prism_*.parquet
         outputs/compiled/<REGION>/hourly_data/imerg_hourly.parquet
         Data/Elevation/<REGION>_DEM_AOI_1km.tif   (defines the grid)
Output:  outputs/resampled_grids/<REGION>/{prism,imerg}_hourly_1km.nc
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CRS_WGS84, REGIONS, parse_region_args, region_paths  # noqa: E402

PRISM_VARS = ["tmin", "tmax", "tmean", "tdmean", "ppt"]


def daily_to_hourly_temperature(tmin, tmax, hour):
    """Air temperature at a given hour, assuming the daily minimum falls near
    06:00 and the maximum near 15:00."""
    phase = (hour - 6) / 24.0 * 2 * np.pi
    return tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - phase))


def frame_to_grid(df, var, lats, lons, name=None):
    """Reshape one timestep of a regular lon/lat table into a georeferenced grid."""
    values = (df.sort_values(["lat", "lon"], ascending=[False, True])[var]
                .to_numpy().reshape(len(lats), len(lons)))
    da = xr.DataArray(values, dims=("lat", "lon"),
                      coords={"lat": lats, "lon": lons}, name=name or var)
    return (da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
              .rio.write_crs(CRS_WGS84, inplace=False))


def load_grid_template(dem_path):
    with rasterio.open(dem_path) as src:
        return src.crs, src.transform, src.height, src.width


def resample_prism(prism_parquet, template):
    """Load PRISM, clean it, regrid to 1 km, then expand to hourly."""
    crs, transform, ny, nx = template

    prism = pd.read_parquet(prism_parquet)
    prism["date"] = pd.to_datetime(prism["date"])
    prism = prism.dropna(subset=["lon", "lat", "date"])
    prism.loc[prism["ppt"] < 0, "ppt"] = 0
    prism.loc[(prism["tmin"] < -60) | (prism["tmax"] > 50),
              ["tmin", "tmean", "tmax", "tdmean"]] = np.nan

    lons = np.sort(prism["lon"].unique())
    lats = np.sort(prism["lat"].unique())[::-1]  # north to south

    daily = []
    for date in np.sort(prism["date"].unique()):
        df_day = prism[prism["date"] == date]
        day_vars = {
            var: frame_to_grid(df_day, var, lats, lons).rio.reproject(
                dst_crs=crs, transform=transform, shape=(ny, nx),
                resampling=Resampling.bilinear,
            )
            for var in PRISM_VARS
        }
        daily.append(xr.Dataset(day_vars).assign_coords(time=date))

    prism_daily = xr.concat(daily, dim="time")

    hourly = []
    for hour in range(24):
        ds_hour = xr.Dataset({
            # Reconstructed from the day's real minimum and maximum.
            "prism_tair": daily_to_hourly_temperature(
                prism_daily["tmin"], prism_daily["tmax"], hour
            ),
            # These carry the daily value through every hour.
            "prism_tmean": prism_daily["tmean"],
            "prism_tdmean": prism_daily["tdmean"],
            "prism_ppt": prism_daily["ppt"],
        })
        hourly.append(ds_hour.assign_coords(
            time=prism_daily["time"] + np.timedelta64(hour, "h")
        ))

    return xr.concat(hourly, dim="time")


def resample_imerg(imerg_parquet, template):
    """Regrid the hourly IMERG probabilities onto the 1 km grid."""
    crs, transform, ny, nx = template

    imerg = pd.read_parquet(imerg_parquet)
    imerg["hour_utc"] = pd.to_datetime(imerg["hour_utc"]).dt.floor("h")

    lons = np.sort(imerg["lon"].unique())
    lats = np.sort(imerg["lat"].unique())[::-1]

    slices = []
    for hour in np.sort(imerg["hour_utc"].unique()):
        da = frame_to_grid(imerg[imerg["hour_utc"] == hour], "plp", lats, lons,
                           name="imerg_plp")
        da_1km = da.rio.reproject(dst_crs=crs, transform=transform, shape=(ny, nx),
                                  resampling=Resampling.bilinear)
        slices.append(da_1km.to_dataset().assign_coords(time=hour))

    return xr.concat(slices, dim="time")


def write_netcdf(ds, path):
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, format="NETCDF4", encoding=encoding)
    print(f"  wrote {path}")


def process_region(region_id):
    paths = region_paths(region_id)
    out_dir = paths["resampled_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")
    template = load_grid_template(paths["dem_1km"])

    prism_hourly = resample_prism(paths["prism_parquet"], template)
    imerg_hourly = resample_imerg(
        paths["compiled_dir"] / "hourly_data" / "imerg_hourly.parquet", template
    )

    # Drop timezone information so the two datasets can be aligned.
    prism_hourly = prism_hourly.assign_coords(
        time=pd.to_datetime(prism_hourly.time.values).tz_localize(None)
    )
    imerg_hourly = imerg_hourly.assign_coords(
        time=pd.to_datetime(imerg_hourly.time.values).tz_localize(None)
    )
    prism_hourly, imerg_hourly = xr.align(prism_hourly, imerg_hourly, join="inner")

    if prism_hourly.sizes["time"] == 0:
        raise ValueError(f"{region_id}: PRISM and IMERG share no timesteps.")
    print(f"  {prism_hourly.sizes['time']:,} shared hours")

    write_netcdf(prism_hourly, out_dir / "prism_hourly_1km.nc")
    write_netcdf(imerg_hourly, out_dir / "imerg_hourly_1km.nc")


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
