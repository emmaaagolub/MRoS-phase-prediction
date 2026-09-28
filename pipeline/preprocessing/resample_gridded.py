"""Resample the hourly IMERG phase probability onto the 1 km grid.

IMERG is already hourly; it is only regridded (bilinear) onto the grid defined
by the 1 km DEM.

Inputs:  data/interim/hourly_compiled/<REGION>/<run_id>/imerg_hourly.parquet
         data/interim/dem_1km/<REGION>_DEM_AOI_1km.tif   (defines the grid)
Output:  data/interim/resampled_1km/<REGION>/<run_id>/imerg_hourly_1km.nc
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
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    compiled = input_run("hourly_compiled", region_id)
    imerg_parquet = compiled / "imerg_hourly.parquet"
    out_dir = open_run("resampled_1km", region_id, step="resample_gridded",
                       inputs={"hourly_compiled": compiled, "dem_1km": paths["dem_1km"]})

    template = load_grid_template(paths["dem_1km"])
    imerg_hourly = resample_imerg(imerg_parquet, template)
    imerg_hourly = imerg_hourly.assign_coords(
        time=pd.to_datetime(imerg_hourly.time.values).tz_localize(None)
    )
    print(f"  {imerg_hourly.sizes['time']:,} hours")

    write_netcdf(imerg_hourly, out_dir / "imerg_hourly_1km.nc")
    finish_step(out_dir, "resample_gridded")


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
