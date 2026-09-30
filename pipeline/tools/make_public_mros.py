"""Make the location-coarsened MRoS file that is published on Zenodo.

The raw MRoS reports carry the observer's position to the mm, in the
latitude/longitude columns and again in geohash12 and the exact distances to
named weather stations. This script writes a public copy in which:

  latitude, longitude   rounded to 4 decimal places (~11 m N-S, ~8-9 m E-W)
  geohash12             replaced by geohash7 (~150 m cell), computed from the
                        rounded position
  *_nearest_dist,       distances to weather stations, rounded to the nearest
  *_avg_dist            1000 m
  elevation             rounded to the nearest 10 m
  comment               dropped (free text written by observers)

Every other column is kept unchanged. Rounding (not truncating) is used so
the coordinates are not all shifted in the same direction.

Input:  data/raw/mros/mros_ca_co_20221001_20260501.parquet     (raw.mros)
Output: data/raw/mros/public/mros_ca_co_20221001_20260501.parquet
        data/raw/mros/public/README_public_version.txt

package_for_zenodo.py uploads the public file under the raw.mros path, so
anyone who downloads the data gets it where the pipeline expects it.

  python pipeline/tools/make_public_mros.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import raw_path  # noqa: E402

DECIMALS = 4          # latitude / longitude
XY_ROUND_M = 10       # projected (UTM) x / y, metres; matches ~4 decimals
DIST_ROUND_M = 1000
ELEV_ROUND_M = 10
GEOHASH_LEN = 7
DROP = ["comment"]

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

ACCESS_NOTE = (
    "Full-precision MRoS observation locations can be shared on request, subject to "
    "approval by the Mountain Rain or Snow project team."
)


def public_mros_path() -> Path:
    raw = raw_path("mros")
    return raw.parent / "public" / raw.name


def geohash(lat, lon, length=GEOHASH_LEN):
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    lat_rng, lon_rng = [-90.0, 90.0], [-180.0, 180.0]
    out, bits, ch, even = [], 0, 0, True
    while len(out) < length:
        rng, val = (lon_rng, lon) if even else (lat_rng, lat)
        mid = (rng[0] + rng[1]) / 2
        if val >= mid:
            ch = (ch << 1) | 1
            rng[0] = mid
        else:
            ch <<= 1
            rng[1] = mid
        even = not even
        bits += 1
        if bits == 5:
            out.append(_BASE32[ch])
            bits, ch = 0, 0
    return "".join(out)


def round_numeric(s, step=None, decimals=None):
    """Round a column stored as text or numbers; keeps missing values missing."""
    x = pd.to_numeric(s, errors="coerce")
    if decimals is not None:
        x = x.round(decimals)
    else:
        x = (x / step).round() * step
    return x


def coarsen_point_table(src, dst):
    """Copy a table of individual MRoS reports with its coordinates rounded the
    same way as the public MRoS file: lat/lon to DECIMALS places, projected x/y
    to the nearest XY_ROUND_M metres. Other columns are unchanged. Handles
    .parquet and .csv."""
    src, dst = Path(src), Path(dst)
    csv = src.suffix.lower() == ".csv"
    df = pd.read_csv(src, low_memory=False) if csv else pd.read_parquet(src)
    for col in df.columns:
        key = col.lower()
        if key in ("lat", "lon", "latitude", "longitude"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round(DECIMALS)
        elif key in ("x", "y"):
            df[col] = round_numeric(df[col], step=XY_ROUND_M)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if csv:
        df.to_csv(dst, index=False)
    else:
        df.to_parquet(dst)
    return dst


def main():
    src = raw_path("mros")
    dst = public_mros_path()
    df = pd.read_parquet(src)
    n = len(df)

    lat0 = pd.to_numeric(df["latitude"], errors="coerce")
    lon0 = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = lat0.round(DECIMALS)
    df["longitude"] = lon0.round(DECIMALS)

    if "geohash12" in df.columns:
        pos = df.columns.get_loc("geohash12")
        df = df.drop(columns="geohash12")
        df.insert(pos, f"geohash{GEOHASH_LEN}",
                  [geohash(a, b) for a, b in zip(df["latitude"], df["longitude"])])

    dist_cols = [c for c in df.columns if c.endswith("_nearest_dist") or c.endswith("_avg_dist")]
    for c in dist_cols:
        df[c] = round_numeric(df[c], step=DIST_ROUND_M)
    if "elevation" in df.columns:
        df["elevation"] = round_numeric(df["elevation"], step=ELEV_ROUND_M)

    dropped = [c for c in DROP if c in df.columns]
    df = df.drop(columns=dropped)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)

    shift_m = np.hypot((df["latitude"] - lat0) * 111_000,
                       (df["longitude"] - lon0) * 111_000 * np.cos(np.radians(lat0)))
    notes = f"""Mountain Rain or Snow reports, public version
Source file: {src.name} ({n} reports, CA and CO, 2022-10-01 to 2026-05-01)

To protect observers' locations, this copy differs from the raw export:
  latitude, longitude  rounded to {DECIMALS} decimal places (~11 m N-S, ~8-9 m E-W);
                       median shift {np.nanmedian(shift_m):.0f} m, max {np.nanmax(shift_m):.0f} m
  geohash12            replaced by geohash{GEOHASH_LEN}, computed from the rounded position
  {', '.join(dist_cols)}
                       rounded to the nearest {DIST_ROUND_M} m
  elevation            rounded to the nearest {ELEV_ROUND_M} m
  dropped columns      {', '.join(dropped) or 'none'}
All other columns are unchanged. Made by pipeline/tools/make_public_mros.py.

Results computed from this file will differ slightly from those computed from
the full-precision data (a few reports can fall in a neighbouring 1 km grid
cell). {ACCESS_NOTE}
"""
    (dst.parent / "README_public_version.txt").write_text(notes)
    print(notes)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
