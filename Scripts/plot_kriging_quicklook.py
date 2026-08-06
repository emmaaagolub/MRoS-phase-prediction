#!/usr/bin/env python3
"""
Quicklook figures from saved indicator-kriging outputs.

This script does NOT re-run any kriging / variogram fitting. It simply loads
the already-saved products from a previous run of
`kriging_interpolation_updated_multiregion.ipynb`:

  - processed station / MRoS parquets      (outputs/interpolated/<REGION>/.../processed_inputs/)
  - the gridded hourly predictor NetCDF    (outputs/interpolated/<REGION>/.../hourly_predictors_1km_indicator_kriging.nc)

and reproduces the "quicklook" panel figure (phase_pred, p_snow, p_mix,
p_rain, temp_air, temp_dew, temp_wet, rh) for the hour with the most MRoS
observations, saving it to disk instead of just calling plt.show().

Usage
-----
    python plot_kriging_quicklook.py [--region CA|CO] [--date YYYY-MM-DD] [--base-dir PATH]

If --date is omitted, the script auto-picks the hour with the most MRoS
observations (same behavior as the notebook).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm, LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from pyproj import CRS, Transformer

try:
    import rasterio as rio
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False

try:
    import contextily as ctx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

# ====================================================================
# Region config (mirrors the notebook's REGION_CONFIG)
# ====================================================================
REGION_CONFIG = {
    "CA": {
        "label": "California: Sierra Nevada / Lake Tahoe",
        "utm_crs": "EPSG:26911",
        "dem_file": "CA_DEM_AOI_1km.tif",
    },
    "CO": {
        "label": "Colorado Mountains",
        "utm_crs": "EPSG:32613",
        "dem_file": "CO_DEM_AOI_1km.tif",
    },
}

# Each phase gets its own white -> phase-color gradient, so P(snow)/P(mix)/
# P(rain) are each read as "how much of this phase," and the color always
# means the same phase across the figure.
PHASE_COLORS = {"snow": "#2f6fb5", "mix": "#e0559c", "rain": "#2e8b57"}
PHASE_PROB_CMAPS = {
    phase: LinearSegmentedColormap.from_list(f"white_to_{phase}", ["#ffffff", color])
    for phase, color in PHASE_COLORS.items()
}

# Manuscript-style plotting defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PANEL_LABELS = list("abcdefgh")

PHASE_ORDER = ("snow", "mix", "rain")

def build_paths(region: str, base_dir: Path) -> dict:
    """Locations of the already-saved data products for a region."""
    out_dir = base_dir / f"mros-precipitation-phase-product-prototype/outputs/interpolated/{region}/indicator_kriging_refactored"
    dem_file = REGION_CONFIG[region]["dem_file"]
    return {
        "out_dir": out_dir,
        "processed_dir": out_dir / "processed_inputs",
        "stations_processed": out_dir / "processed_inputs" / "stations_processed.parquet",
        "mros_processed": out_dir / "processed_inputs" / "mros_processed.parquet",
        "predictors_nc": out_dir / "hourly_predictors_1km_indicator_kriging.nc",
        "dem_path": base_dir / f"mros-precipitation-phase-product-prototype/Data/elevation/{dem_file}",
    }


def load_hillshade(dem_path: Path, target_crs: CRS, azimuth: float = 315, altitude: float = 45):
    """Read the DEM (read-only, already-saved raster) and compute a hillshade
    to use as terrain context behind the kriged surfaces. Returns None if
    rasterio isn't available or the DEM file can't be found — hillshade is a
    nice-to-have, not a hard requirement for the quicklook."""
    if not HAVE_RASTERIO or not dem_path.exists():
        print(f"  Skipping hillshade (DEM not found or rasterio unavailable): {dem_path}")
        return None, None

    with rio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        bounds = src.bounds
        dem_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    hillshade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), vert_exag=1)
    return hillshade, dem_extent


def load_saved_data(paths: dict):
    """Load the already-processed station/MRoS tables and gridded predictors.

    Everything here is read-only: no kriging, no variogram fitting, no
    re-deriving anything — just reading what a prior pipeline run wrote out.
    """
    for label, p in [
        ("processed stations parquet", paths["stations_processed"]),
        ("processed MRoS parquet", paths["mros_processed"]),
        ("gridded predictors NetCDF", paths["predictors_nc"]),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {label}: {p}\n"
                "Run the interpolation pipeline first to generate saved outputs."
            )

    st_hr = pd.read_parquet(paths["stations_processed"])
    mros_hr = pd.read_parquet(paths["mros_processed"])
    ds = xr.open_dataset(paths["predictors_nc"])

    st_hr["hour_utc"] = pd.to_datetime(st_hr["hour_utc"], utc=True)
    mros_hr["hour_utc"] = pd.to_datetime(mros_hr["hour_utc"], utc=True)

    return st_hr, mros_hr, ds


def pick_timestamp(ds: xr.Dataset, mros_hr: pd.DataFrame, target_date: str | None):
    """Pick the timestep to plot: an explicit date, or the hour with the most
    MRoS observations (same fallback logic as the notebook)."""
    times = pd.to_datetime(ds.time.values).floor("h")

    if target_date is not None:
        mask = times.normalize() == pd.to_datetime(target_date)
        if not mask.any():
            raise ValueError(f"No timestep found for {target_date}")
        ti = np.where(mask)[0][0]
        return ti, times[ti]

    mros_counts = (
        mros_hr[mros_hr["mros_phase"].isin(PHASE_ORDER)]
        .groupby("hour_utc")
        .size()
        .sort_values(ascending=False)
    )
    if mros_counts.empty:
        raise ValueError("No MRoS observations found to auto-select a timestamp.")

    best_ts = mros_counts.index[0]
    print("Top 10 hours by MRoS observation count:")
    print(mros_counts.head(10))
    print(f"Auto-selected timestamp: {best_ts}")

    ti = int(np.abs(ds.time.values - best_ts.to_datetime64()).argmin())
    return ti, times[ti]


def make_quicklook_figure(
    ds: xr.Dataset,
    st_hr: pd.DataFrame,
    mros_hr: pd.DataFrame,
    ti: int,
    t_plot: pd.Timestamp,
    region_crs: str,
    out_path: Path,
    dem_path: Path | None = None,
    region: str | None = None,
):
    xmin, xmax = ds.x.min().item(), ds.x.max().item()
    ymin, ymax = ds.y.min().item(), ds.y.max().item()
    extent = [xmin, xmax, ymin, ymax]

    # View limits match the actual kriging/AOI grid extent exactly (no
    # padding) — this keeps the Easting/Northing axes here numerically
    # identical to the phase-locations-coverage panel in
    # produce_manuscript_figures.py, which also plots the DEM's native
    # bounds with no margin. The basemap tile below will still render some
    # area past the grid edge at low zoom, but the axis itself reflects only
    # the true modeled domain.
    view_xmin, view_xmax = xmin, xmax
    view_ymin, view_ymax = ymin, ymax

    hillshade, hs_extent = (None, None)
    if dem_path is not None:
        hillshade, hs_extent = load_hillshade(dem_path, region_crs)

    if "spatial_ref" in ds and "crs_wkt" in ds["spatial_ref"].attrs:
        ds_crs = ds["spatial_ref"].attrs["crs_wkt"]
    elif "crs" in ds.attrs:
        ds_crs = ds.attrs["crs"]
    else:
        ds_crs = region_crs
        print(f"  No CRS in dataset attrs — using region fallback: {ds_crs}")

    tf = Transformer.from_crs("EPSG:4326", CRS.from_user_input(ds_crs), always_xy=True)

    st_time = st_hr["hour_utc"].dt.tz_convert(None) if st_hr["hour_utc"].dt.tz is not None else st_hr["hour_utc"]
    mros_time = mros_hr["hour_utc"].dt.tz_convert(None) if mros_hr["hour_utc"].dt.tz is not None else mros_hr["hour_utc"]

    st_t = st_hr[st_time.dt.floor("h") == t_plot].copy()
    mros_t = mros_hr[mros_time.dt.floor("h") == t_plot].copy()

    st_x, st_y = np.array([]), np.array([])
    if len(st_t):
        sx, sy = tf.transform(st_t["lon"].values, st_t["lat"].values)
        sx, sy = np.asarray(sx), np.asarray(sy)
        m = (sx >= xmin) & (sx <= xmax) & (sy >= ymin) & (sy <= ymax)
        st_x, st_y = sx[m], sy[m]

    mo_x, mo_y = np.array([]), np.array([])
    if len(mros_t):
        mx, my = tf.transform(mros_t["lon"].values, mros_t["lat"].values)
        mx, my = np.asarray(mx), np.asarray(my)
        m = (mx >= xmin) & (mx <= xmax) & (my >= ymin) & (my <= ymax)
        mo_x, mo_y = mx[m], my[m]

    print(f"Stations in grid: {len(st_x)}")
    print(f"MRoS in grid:     {len(mo_x)}")

    p_snow = ds["p_snow"].isel(time=ti).values
    p_mix = ds["p_mix"].isel(time=ti).values
    p_rain = ds["p_rain"].isel(time=ti).values

    # AOI outline: boundary of the valid (non-NaN) footprint, from any of the
    # continuous surfaces — drawn as a thin black outline instead of an
    # unexplained white cutoff
    valid_mask = np.isfinite(p_snow).astype(float)

    var_specs = [
        ("p_snow", p_snow, dict(cmap=PHASE_PROB_CMAPS["snow"], vmin=0, vmax=1), "Probability", "continuous"),
        ("p_mix", p_mix, dict(cmap=PHASE_PROB_CMAPS["mix"], vmin=0, vmax=1), "Probability", "continuous"),
        ("p_rain", p_rain, dict(cmap=PHASE_PROB_CMAPS["rain"], vmin=0, vmax=1), "Probability", "continuous"),
        ("temp_air", ds["temp_air"].isel(time=ti).values, dict(cmap="RdBu_r", vmin=-10, vmax=10), "Temperature (°C)", "continuous"),
        ("temp_dew", ds["temp_dew"].isel(time=ti).values, dict(cmap="RdBu_r", vmin=-10, vmax=10), "Temperature (°C)", "continuous"),
        ("temp_wet", ds["temp_wet"].isel(time=ti).values, dict(cmap="RdBu_r", vmin=-10, vmax=10), "Temperature (°C)", "continuous"),
        ("rh", ds["rh"].isel(time=ti).values, dict(cmap="viridis", vmin=0, vmax=100), "Relative humidity (%)", "continuous"),
    ]

    panel_titles = {
        "p_snow": "P(snow)",
        "p_mix": "P(mix)",
        "p_rain": "P(rain)",
        "temp_air": "Air temperature",
        "temp_dew": "Dewpoint temperature",
        "temp_wet": "Wet-bulb temperature",
        "rh": "Relative humidity",
    }

    # Custom grid: top row holds the 3 phase-probability panels, centered
    # over a 4-panel-wide bottom row of the other predictors. Gaps between
    # panels are explicit spacer columns (not just gridspec wspace), so the
    # gap is guaranteed wide enough for a colorbar + 4-digit UTM tick labels
    # no matter how much text the neighboring axis needs.
    PANEL_W, GAP_W = 6, 3
    n_bottom = 4
    total_cols = n_bottom * PANEL_W + (n_bottom - 1) * GAP_W
    block_w_top = 3 * PANEL_W + 2 * GAP_W
    top_offset = (total_cols - block_w_top) // 2

    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(2, total_cols, hspace=0.45, wspace=0)
    top_axes = [
        fig.add_subplot(gs[0, top_offset + j * (PANEL_W + GAP_W): top_offset + j * (PANEL_W + GAP_W) + PANEL_W])
        for j in range(3)
    ]
    bottom_axes = [
        fig.add_subplot(gs[1, i * (PANEL_W + GAP_W): i * (PANEL_W + GAP_W) + PANEL_W])
        for i in range(n_bottom)
    ]
    axes = top_axes + bottom_axes
    fig.suptitle(f"Interpolated (Kriged) Predictor Surfaces — Timestamp {t_plot:%Y-%m-%d %H:%M UTC}", fontsize=14, y=0.99)

    for i, (name, arr, imshow_kwargs, cbar_label, kind) in enumerate(var_specs):
        ax = axes[i]

        # widen the view beyond the AOI grid first, so the basemap tile
        # fetch below covers the actual visible area
        ax.set_xlim(view_xmin, view_xmax)
        ax.set_ylim(view_ymin, view_ymax)

        # real basemap tile (OpenStreetMap) as background context — gives
        # roads, place names, and terrain shading around the AOI for free,
        # no hardcoded city list needed. Falls back to a DEM hillshade if
        # there's no internet access, then to plain white as a last resort.
        basemap_drawn = False
        if HAVE_CONTEXTILY:
            try:
                ctx.add_basemap(ax, crs=ds_crs, source=ctx.providers.OpenStreetMap.Mapnik,
                                 attribution=False, zorder=0)
                basemap_drawn = True
            except Exception as exc:
                print(f"  Basemap tile fetch failed ({exc}); falling back to hillshade/DEM.")

        if not basemap_drawn and hillshade is not None:
            ax.imshow(hillshade, origin="upper", extent=hs_extent, cmap="gray", vmin=0, vmax=1, aspect="equal", zorder=0)

        data_alpha = 0.85 if basemap_drawn or hillshade is not None else 1.0

        im = ax.imshow(arr, origin="lower", extent=extent, aspect="equal",
                        alpha=data_alpha, zorder=1, **imshow_kwargs)

        # AOI outline instead of an unexplained white cutoff
        ax.contour(valid_mask, levels=[0.5], colors="black", linewidths=0.8,
                   extent=extent, origin="lower", zorder=2)

        ax.set_title(panel_titles[name], fontsize=11)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.ticklabel_format(style="plain")
        ax.set_xticklabels([f"{x/1000:.0f}" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{y/1000:.0f}" for y in ax.get_yticks()])

        # panel label (a)-(h) for manuscript cross-referencing
        ax.text(0.02, 0.97, f"({PANEL_LABELS[i]})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        if len(st_x):
            ax.scatter(st_x, st_y, s=12, c="white", edgecolor="black", linewidth=0.4, zorder=3)
        if len(mo_x):
            ax.scatter(mo_x, mo_y, s=25, c="red", marker="^", edgecolor="black", linewidth=0.4, zorder=3)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # scale bar + north arrow on the first panel only (repeating on all 8
        # panels is visual noise; readers only need it once since all
        # panels share the same extent)
        if i == 0:
            bar_len = 50_000  # meters
            bx0 = xmin + 0.06 * (xmax - xmin)
            by0 = ymin + 0.06 * (ymax - ymin)
            ax.plot([bx0, bx0 + bar_len], [by0, by0], color="black", lw=2.5, zorder=4)
            ax.text(bx0 + bar_len / 2, by0 + 0.015 * (ymax - ymin), "50 km",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=4)

            nx0 = xmax - 0.10 * (xmax - xmin)
            ny0 = ymin + 0.08 * (ymax - ymin)
            arrow_len = 0.06 * (ymax - ymin)
            ax.add_patch(FancyArrow(nx0, ny0, 0, arrow_len, width=arrow_len * 0.06,
                                     head_width=arrow_len * 0.35, head_length=arrow_len * 0.35,
                                     color="black", zorder=4))
            ax.text(nx0, ny0 + arrow_len * 1.15, "N", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", zorder=4)

    # single shared legend for station/MRoS markers instead of repeating it
    # in all 8 panels
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Weather stations"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="MRoS observations"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10)

    # note: no plt.tight_layout() here — it fights with the explicit
    # GridSpec column widths used to center the top row; the manual
    # wspace/hspace above plus bbox_inches="tight" on save handle spacing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")
    print(f"Saved vector figure: {out_path.with_suffix('.pdf')}")

    # QC prints (unchanged from notebook)
    print("\n" + "=" * 80)
    print("QUICK QC")
    print("=" * 80)
    for v in ["temp_air", "temp_dew", "temp_wet", "rh", "p_snow", "p_mix", "p_rain"]:
        arr = ds[v].isel(time=ti).values
        finite = np.isfinite(arr)
        print(f"{v:10s} | finite={finite.mean()*100:6.2f}%"
              f" | min={np.nanmin(arr):8.3f} | max={np.nanmax(arr):8.3f} | mean={np.nanmean(arr):8.3f}")

    p_sum = p_snow + p_mix + p_rain
    print("\nProbability sum check at plotted timestep:")
    print(f"min  = {np.nanmin(p_sum):.4f}")
    print(f"max  = {np.nanmax(p_sum):.4f}")
    print(f"mean = {np.nanmean(p_sum):.4f}")
    print(f"std  = {np.nanstd(p_sum):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=["CA", "CO"], default="CO")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; if omitted, auto-picks hour with most MRoS obs")
    parser.add_argument("--base-dir", default=".", help="Project base dir containing outputs/")
    parser.add_argument("--out", default=None, help="Output figure path (PNG)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    region = args.region
    rcfg = REGION_CONFIG[region]

    paths = build_paths(region, base_dir)
    print(f"Region: {region} — {rcfg['label']}")
    print(f"Reading saved data from: {paths['out_dir']}")

    st_hr, mros_hr, ds = load_saved_data(paths)
    ti, t_plot = pick_timestamp(ds, mros_hr, args.date)

    out_path = Path(args.out) if args.out else paths["out_dir"] / f"quicklook_{t_plot:%Y%m%dT%H%M}Z.png"
    make_quicklook_figure(ds, st_hr, mros_hr, ti, t_plot, rcfg["utm_crs"], out_path,
                          dem_path=paths["dem_path"], region=region)


if __name__ == "__main__":
    main()
