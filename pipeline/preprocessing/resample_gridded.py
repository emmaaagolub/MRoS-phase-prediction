"""Resample the hourly IMERG phase probability onto the 1 km grid.

IMERG is already hourly; it is only regridded (bilinear) onto the grid defined
by the 1 km DEM.

Inputs:  data/interim/hourly_compiled/<REGION>/<run_id>/imerg_hourly.parquet
         data/interim/dem_1km/<REGION>_DEM_AOI_1km.tif   (defines the grid)
Output:  data/interim/resampled_1km/<REGION>/<run_id>/imerg_hourly_1km.nc
"""

import sys
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CRS_WGS84, REGIONS, finish_step, input_run, open_run, parse_region_args, region_paths,
)


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


# Hours regridded and written to the output file at a time. The full CA cube is
# ~20 GB in memory (31,000 hours x 333 x 235 cells, float64), so it is written
# in batches instead of being assembled first.
BATCH_HOURS = 744
TIME_UNITS = "hours since 2022-10-01"


def iter_imerg_batches(imerg_parquet, template, batch_hours=BATCH_HOURS):
    """Regrid the hourly IMERG probabilities onto the 1 km grid, yielding one
    xarray Dataset per batch of hours."""
    crs, transform, ny, nx = template

    imerg = pd.read_parquet(imerg_parquet, columns=["hour_utc", "lat", "lon", "plp"])
    imerg["hour_utc"] = pd.to_datetime(imerg["hour_utc"]).dt.floor("h")
    if imerg["hour_utc"].dt.tz is not None:  # plain UTC datetime64, not objects
        imerg["hour_utc"] = imerg["hour_utc"].dt.tz_convert(None)
    if not imerg["hour_utc"].is_monotonic_increasing:  # compile writes it sorted
        imerg = imerg.sort_values("hour_utc", kind="stable", ignore_index=True)

    lons = np.sort(imerg["lon"].unique())
    lats = np.sort(imerg["lat"].unique())[::-1]

    hours = imerg["hour_utc"].to_numpy()
    starts = np.flatnonzero(np.r_[True, hours[1:] != hours[:-1]])
    unique_hours = hours[starts]
    bounds = list(starts) + [len(imerg)]
    del hours

    slices = []
    for i, hour in enumerate(unique_hours):
        rows = imerg.iloc[bounds[i]:bounds[i + 1]]
        da = frame_to_grid(rows, "plp", lats, lons, name="imerg_plp")
        da_1km = da.rio.reproject(dst_crs=crs, transform=transform, shape=(ny, nx),
                                  resampling=Resampling.bilinear)
        slices.append(da_1km.to_dataset().assign_coords(time=hour))
        if len(slices) == batch_hours or i == len(unique_hours) - 1:
            batch = xr.concat(slices, dim="time")
            slices = []
            yield batch.assign_coords(
                time=pd.to_datetime(batch.time.values).tz_localize(None))


def write_netcdf(batches, path):
    """Write the first batch with xarray, then append the rest in place.
    Chunks hold one hour each, so later stages can read single hours quickly."""
    n_hours = 0
    for batch in batches:
        if n_hours == 0:
            ny, nx = batch.sizes["y"], batch.sizes["x"]
            encoding = {v: {"zlib": True, "complevel": 4, "chunksizes": (1, ny, nx)}
                        for v in batch.data_vars if batch[v].ndim == 3}
            encoding["time"] = {"units": TIME_UNITS, "calendar": "standard",
                                "dtype": "float64"}
            batch.to_netcdf(path, format="NETCDF4", encoding=encoding,
                            unlimited_dims=["time"])
        else:
            origin = pd.Timestamp(TIME_UNITS.split("since ")[1])
            with nc4.Dataset(path, "a") as dst:
                start = len(dst.variables["time"])
                k = batch.sizes["time"]
                dst.variables["time"][start:start + k] = (
                    (pd.to_datetime(batch.time.values) - origin) / pd.Timedelta(hours=1)
                ).to_numpy(dtype=np.float64)
                for v in batch.data_vars:
                    if batch[v].ndim == 3:
                        dst.variables[v][start:start + k] = batch[v].transpose(
                            "time", "y", "x").values
        n_hours += batch.sizes["time"]
        print(f"    {n_hours:,} hours written", end="\r", flush=True)
    print(f"\n  wrote {path}")
    return n_hours


def process_region(region_id):
    paths = region_paths(region_id)
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    compiled = input_run("hourly_compiled", region_id)
    imerg_parquet = compiled / "imerg_hourly.parquet"
    out_dir = open_run("resampled_1km", region_id, step="resample_gridded",
                       inputs={"hourly_compiled": compiled, "dem_1km": paths["dem_1km"]})

    template = load_grid_template(paths["dem_1km"])
    n_hours = write_netcdf(iter_imerg_batches(imerg_parquet, template),
                           out_dir / "imerg_hourly_1km.nc")
    print(f"  {n_hours:,} hours")
    finish_step(out_dir, "resample_gridded")


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
