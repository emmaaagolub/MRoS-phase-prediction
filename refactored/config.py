"""Shared settings for the precipitation-phase pipeline.

Region definitions, file paths and the study period are defined here and
imported by every stage.
"""

from pathlib import Path

# Found relative to this file so the scripts work from any working directory.
PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent

# Downloaded source data is shared with the rest of the repository.
DATA_DIR = REPO_ROOT / "Data"

# All pipeline output is written inside this folder.
OUTPUT_DIR = PIPELINE_ROOT / "outputs"

# Study period, shared by every stage.
WY_START = "2022-10-01T00:00:00Z"
WY_END = "2026-05-01T23:59:59Z"

# Suffix on the station files written by download_station_data.R.
STATION_DATE_SUFFIX = "20221001_20260501"

CRS_WGS84 = "EPSG:4326"

# The two study areas. Polygon vertices are (lon, lat) in WGS84 and are the same
# ones used by get_elevation.R to cut the DEMs.
REGIONS = {
    "CA": {
        "label": "California: Sierra Nevada / Lake Tahoe",
        "utm_crs": "EPSG:26911",
        "dem_10m": "california_DEM_AOI_TNM_10m.tif",
        "dem_1km": "CA_DEM_AOI_1km.tif",
        # Months for which MRoS surfaces are interpolated.
        "mros_active_months": (10, 11, 12, 1, 2, 3, 4, 5),
        "aoi_lonlat": [
            (-119.45505750721992, 39.65343608043361),
            (-121.27878797084242, 39.66189413918429),
            (-119.11448133630248, 36.726935737063016),
            (-118.49924696303225, 37.235952484988736),
            (-119.46604383531404, 38.37304030164334),
        ],
    },
    "CO": {
        "label": "Colorado Mountains",
        "utm_crs": "EPSG:32613",
        "dem_10m": "colorado_DEM_AOI_TNM_10m.tif",
        "dem_1km": "CO_DEM_AOI_1km.tif",
        "mros_active_months": (9, 10, 11, 12, 1, 2, 3, 4, 5, 6),
        "aoi_lonlat": [
            (-105.19885928678391, 40.62046076499234),
            (-106.88927700287375, 40.555465783967925),
            (-107.66078646416787, 38.79540171139857),
            (-104.87856373593310, 38.77382201116306),
        ],
    },
}

REGION_IDS = tuple(REGIONS)


def interpolation_dir(region_id, interp_type="kriging"):
    """Where one region's interpolated grids live."""
    name = "indicator_kriging" if interp_type == "kriging" else "IDW"
    return OUTPUT_DIR / "interpolated" / region_id / name


def region_paths(region_id):
    """All input and output locations for one region.

    Inputs come from the repository's shared Data folder; everything the
    pipeline writes goes under refactored/outputs/.
    """
    return {
        # --- inputs ---
        "dem_10m": DATA_DIR / "Elevation" / REGIONS[region_id]["dem_10m"],
        "dem_1km": DATA_DIR / "Elevation" / REGIONS[region_id]["dem_1km"],
        "station_dir": DATA_DIR / "Stations" / region_id,
        "station_meta": DATA_DIR / "Stations" / region_id
        / f"station_metadata_{STATION_DATE_SUFFIX}.csv",
        "imerg_dir": DATA_DIR / "IMERG" / region_id,
        "prism_parquet": DATA_DIR / "PRISM" / region_id
        / "combined_prism_20221001_20260501.parquet",
        "mros_parquet": DATA_DIR / "observations"
        / "mros_ca_co_20221001_20260501.parquet",

        # --- outputs ---
        "compiled_dir": OUTPUT_DIR / "compiled" / region_id,
        "resampled_dir": OUTPUT_DIR / "resampled_grids" / region_id,
        "interpolated_dir": interpolation_dir(region_id),
        "model_dir": OUTPUT_DIR / "model" / region_id,
        "evaluation_dir": OUTPUT_DIR / "evaluation" / region_id,
        "figures_dir": OUTPUT_DIR / "figures" / region_id,
    }


def parse_region_args(argv=None):
    """Shared --regions flag. Defaults to running every region."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGION_IDS,
        default=list(REGION_IDS),
        help="Regions to process (default: all).",
    )
    return parser.parse_args(argv).regions
