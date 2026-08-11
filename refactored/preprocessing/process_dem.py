"""Step 5 — Turn the raw 10 m DEMs into the 1 km grid the rest of the pipeline
uses.

For each region: reproject to the local UTM zone, average down to 1 km, then
cut to the study-area polygon. Cells outside the polygon become NaN. Nothing
intermediate is written to disk.

Input:  Data/Elevation/<region>_DEM_AOI_TNM_10m.tif  (get_elevation.R)
Output: Data/Elevation/{CA,CO}_DEM_AOI_1km.tif
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import Polygon, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CRS_WGS84, REGIONS, parse_region_args, region_paths  # noqa: E402

TARGET_RES_M = 1000


def project_polygon(lonlat_vertices, dst_crs):
    """Move the study-area polygon from lat/lon into the target projection."""
    transformer = Transformer.from_crs(CRS_WGS84, dst_crs, always_xy=True)
    return Polygon([transformer.transform(lon, lat) for lon, lat in lonlat_vertices])


def process_region(region_id, target_res=TARGET_RES_M):
    cfg = REGIONS[region_id]
    paths = region_paths(region_id)
    input_tif, output_tif = paths["dem_10m"], paths["dem_1km"]
    dst_crs = cfg["utm_crs"]

    print(f"\n=== {cfg['label']} ===")
    if not input_tif.exists():
        raise FileNotFoundError(f"{input_tif} not found — run get_elevation.R first.")

    # Reproject to UTM at the DEM's native resolution.
    with rasterio.open(input_tif) as src:
        print(f"  source: {src.crs}, {src.res}, {src.height} x {src.width}")
        utm_transform, utm_width, utm_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dem_utm = np.empty((utm_height, utm_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=utm_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    # Average down to the 1 km grid.
    new_width = max(1, int(round(utm_width * abs(utm_transform.a) / target_res)))
    new_height = max(1, int(round(utm_height * abs(utm_transform.e) / target_res)))
    km_transform = rasterio.Affine(
        target_res, 0, utm_transform.c,
        0, -target_res, utm_transform.f,
    )

    dem_1km = np.empty((new_height, new_width), dtype=np.float32)
    reproject(
        source=dem_utm,
        destination=dem_1km,
        src_transform=utm_transform,
        src_crs=dst_crs,
        dst_transform=km_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,
    )
    del dem_utm

    km_profile = {
        "driver": "GTiff",
        "height": new_height,
        "width": new_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": km_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    # Cut to the study-area polygon.
    aoi_utm = project_polygon(cfg["aoi_lonlat"], dst_crs)
    with MemoryFile() as memfile:
        with memfile.open(**km_profile) as mem_ds:
            mem_ds.write(dem_1km, 1)
        with memfile.open() as mem_ds:
            out_image, out_transform = mask(
                mem_ds, [mapping(aoi_utm)], crop=True, filled=True, nodata=np.nan
            )
            out_profile = mem_ds.profile.copy()
            out_profile.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": np.nan,
                "dtype": "float32",
            })

    clipped = out_image[0]
    valid = clipped[~np.isnan(clipped)]
    if valid.size == 0:
        raise ValueError(f"{cfg['label']}: no valid cells — does the AOI overlap the DEM?")

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    if output_tif.exists():
        output_tif.unlink()
    with rasterio.open(output_tif, "w", **out_profile) as dst:
        dst.write(clipped, 1)

    print(f"  {clipped.shape[0]} x {clipped.shape[1]} @ {target_res} m, {dst_crs}")
    print(f"  elevation {valid.min():.0f}–{valid.max():.0f} m "
          f"(mean {valid.mean():.0f}, sd {valid.std():.0f})")
    print(f"  valid cells: {valid.size:,} / {clipped.size:,}")
    print(f"  wrote {output_tif}")
    return output_tif


def main(regions=None):
    for region_id in regions or REGIONS:
        process_region(region_id)


if __name__ == "__main__":
    main(parse_region_args())
