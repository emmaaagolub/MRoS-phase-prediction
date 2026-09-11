"""Draw the manuscript figures from saved artifacts.

Reads the outputs of earlier stages only. No model is retrained, no
interpolation is rerun, and no raw data is read.

Sources (all read-only):
  outputs/ML_pipeline/model_artifacts/<REGION>/results_binaryXGB_withKriging_v2/
      trained-model artifacts, predictions, SHAP
  outputs/ML_pipeline/model_artifacts/<REGION>/ablations_v2/
      per-experiment metrics
  outputs/ML_pipeline/model_artifacts/<REGION>/benchmarking_v1/
      benchmark comparison, and .../bootstrap/ for the cluster-bootstrap CIs
  outputs/assimilated/<REGION>/hourly_data/
      station and MRoS hourly tables

Output: outputs/manuscript_figures/<REGION>/<REGION>_<description>[_appendix].png

Naming and labelling conventions
--------------------------------
  - Folder and file names keep the "CA" / "CO" identifiers, because main.tex
    points at figures/CA/CA_*.png. Nothing *drawn inside* a figure says CA or
    CO: the domains are labelled SNM and CRM (see REGION_DISPLAY /
    REGION_FULL_LABEL in the shared body).
  - The ablation configuration using all predictors is labelled "Full
    configuration". The folder name and saved metric keys still read
    baseline_full and *_baseline_minus_config; the translation happens at
    display time via CONFIG_DISPLAY_NAME.
  - The IMERG probability-of-liquid-precipitation predictor is written pLP.
  - The model is labelled XGB-Full.
  - The binary logistic benchmark evaluated with the published Jennings et al.
    (2018) coefficients (binlog_jennings18) is excluded from every figure; only
    the logistic model refit on these domains is shown. See EXCLUDED_METHODS.
  - Titles are short and Title Case, and carry no metric results. Axis labels
    give the quantity and its unit only — the CI method is not repeated there;
    it belongs in the caption.

The study-extent and station-map panels fetch reference boundaries, city
points and basemap tiles over the network. Boundaries are cached under
Data/reference/ after the first successful fetch; if they are unavailable the
affected panel is skipped with a printed message and the rest of the figure is
still written.

Usage
-----
  python manuscript_figures.py
  python manuscript_figures.py --regions CA
  python manuscript_figures.py --figures shap band

KEEP IN SYNC: everything below the "SHARED FIGURE BODY" marker is duplicated
verbatim in Manuscript/produce_manuscript_figures.py. Only this header (paths
and region configuration) differs between the two copies.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

# Needed only by the map panels; those skip themselves if it is missing.
try:
    import geopandas as gpd
    import contextily as cx
    import rasterio as rio
    from rasterio.plot import show as rio_show
    from matplotlib_scalebar.scalebar import ScaleBar
    from shapely.geometry import Polygon
    HAVE_GIS = True
except ImportError:
    HAVE_GIS = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REGIONS as REGION_CONFIG, REGION_IDS  # noqa: E402

warnings.filterwarnings("ignore")

REGIONS = list(REGION_IDS)

# --- Path overrides for the current repository layout -----------------------
# refactored/config.py resolves the pipeline's own outputs under refactored/,
# but in this repository the saved model/evaluation artifacts live under
# outputs/ML_pipeline/model_artifacts/, the compiled hourly tables under
# outputs/assimilated/, and the manuscript figures under
# outputs/manuscript_figures/. Restore the commented-out definitions if the
# pipeline outputs are ever moved back under refactored/.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "outputs" / "ML_pipeline" / "model_artifacts"
_REPO_OUTPUTS = REPO_ROOT / "outputs"

# Cached reference vectors (state boundaries, populated places) for the
# study-extent panel, so it works offline after one successful fetch.
REFERENCE_CACHE_DIR = REPO_ROOT / "Data" / "reference"


def ablations_dir(region: str) -> Path:
    # return OUTPUT_DIR / "evaluation" / region / "ablation"
    return ARTIFACTS_ROOT / region / "ablations_v2"


def benchmarking_dir(region: str) -> Path:
    # return OUTPUT_DIR / "evaluation" / region / "benchmarking"
    return ARTIFACTS_ROOT / region / "benchmarking_v1"


def bootstrap_dir(region: str) -> Path:
    return benchmarking_dir(region) / "bootstrap"


def kriging_dir(region: str) -> Path:
    """The trained model's artifact directory."""
    # return OUTPUT_DIR / "model" / region
    return ARTIFACTS_ROOT / region / "results_binaryXGB_withKriging_v2"


def compiled_dir(region: str) -> Path:
    # return region_paths(region)["compiled_dir"]
    return _REPO_OUTPUTS / "assimilated" / region


def dem_path_for(region: str) -> Path:
    """1-km DEM for a region. The repository spells the folder "elevation";
    config.py spells it "Elevation", which does not resolve on a
    case-sensitive filesystem, so both are tried."""
    name = REGION_CONFIG[region]["dem_1km"]
    for folder in ("elevation", "Elevation"):
        p = REPO_ROOT / "Data" / folder / name
        if p.exists():
            return p
    return REPO_ROOT / "Data" / "elevation" / name


def out_dir(region: str) -> Path:
    # d = OUTPUT_DIR / "figures" / region
    d = _REPO_OUTPUTS / "manuscript_figures" / region
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_phase_diagnostics_inputs(region: str):
    """Read the hourly station and MRoS tables written by the compilation
    stage. Timezone handling, hourly averaging and study-area clipping all
    happened when those files were written; nothing is redone here."""
    hourly = compiled_dir(region) / "hourly_data"
    st_path = hourly / "stations_hourly.parquet"
    mros_path = hourly / "mros_hourly.parquet"
    for label, p in [("stations hourly parquet", st_path),
                     ("MRoS hourly parquet", mros_path)]:
        if not p.exists():
            raise FileNotFoundError(f"missing {label}: {p}")
    return pd.read_parquet(st_path), pd.read_parquet(mros_path)


# ===========================================================================
# SHARED FIGURE BODY — keep byte-identical between
#   refactored/figures/manuscript_figures.py
#   Manuscript/produce_manuscript_figures.py
# Everything above this marker is the per-copy header (imports, region config
# and artifact/output paths). Everything below is shared. If you edit one
# copy, copy this whole block into the other.
# ===========================================================================

SNOW_CODE, RAIN_CODE, MIX_CODE = 0, 1, 2

# ---------------------------------------------------------------------------
# Typography. Everything is set once here so every figure in the manuscript
# has the same, print-legible type size. Individual figures no longer set
# small hard-coded fontsize= values; if a label needs to shrink, shrink it
# relative to these.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 19,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "lines.linewidth": 2.0,
    "savefig.dpi": 200,
})

# Colour-vision-safe phase palette (Wong 2011). Used by every figure that
# colours something by precipitation phase, so snow/rain/mix read the same
# way throughout the manuscript.
PHASE_COLORS = {"snow": "#0072B2", "rain": "#009E73", "mix": "#CC79A7"}
DIAG_PHASE_ORDER = ["snow", "mix", "rain"]
DIAG_PHASE_COLORS = dict(PHASE_COLORS)

SPLIT_COLORS = {"Validation": "#8338ec", "Test": "#ff006e"}

# One-degree wet-bulb bins, matching the model evaluation figures.
TWET_BIN_EDGES = np.arange(-6, 7, 1)

# Air-temperature bins, matching benchmarking.py. Bins holding fewer than
# MIN_BIN_N pure observations are dropped.
TAIR_BIN_WIDTH = 1.0
TAIR_BIN_MIN = -8.0
TAIR_BIN_MAX = 8.0
MIN_BIN_N = 20
MIN_BIN_PHASE_N = 10
TAIR_BIN_EDGES = np.arange(TAIR_BIN_MIN, TAIR_BIN_MAX + TAIR_BIN_WIDTH, TAIR_BIN_WIDTH)
NEARFREEZE_TWET_C = 2.0

MODEL_DISPLAY_NAME = "XGB-Full"
MODEL_NAME_KEY = "xgboost_mros"  # the key used in the saved artifacts

# Benchmarks that are computed and stored in the artifacts but are NOT drawn
# in any manuscript figure. binlog_jennings18 is the binary logistic model
# evaluated with the published Jennings et al. (2018) coefficients; only the
# logistic model refitted on this domain (binlog_fitted) is shown, so the
# comparison is like-for-like. Every place that enumerates methods — from a
# CSV, from parquet column names, or from METHOD_STYLE — filters through
# keep_method().
EXCLUDED_METHODS = {"binlog_jennings18"}


def keep_method(name: str) -> bool:
    """False for benchmarks deliberately left out of the manuscript figures."""
    return str(name) not in EXCLUDED_METHODS


METHOD_LABEL = {
    "ta_1.0": "$T_{a}$ 1.0 °C", "ta_1.5": "$T_{a}$ 1.5 °C",
    "td_0.0": "$T_{d}$ 0.0 °C", "td_0.5": "$T_{d}$ 0.5 °C",
    "tw_0.0": "$T_{w}$ 0.0 °C", "tw_0.5": "$T_{w}$ 0.5 °C", "tw_1.0": "$T_{w}$ 1.0 °C",
    "binlog_fitted": "Bin. logistic (fitted)",
    MODEL_NAME_KEY: MODEL_DISPLAY_NAME,
}
METHOD_STYLE = {
    "ta_1.0": dict(color="#e08214", ls="--", lw=1.8),
    "ta_1.5": dict(color="#b35806", ls="-", lw=1.8),
    "td_0.0": dict(color="#7fbc41", ls="--", lw=1.8),
    "td_0.5": dict(color="#4d9221", ls="-", lw=1.8),
    "tw_0.0": dict(color="#92c5de", ls=":", lw=2.0),
    "tw_0.5": dict(color="#4393c3", ls="--", lw=2.0),
    "tw_1.0": dict(color="#2166ac", ls="-", lw=2.0),
    "binlog_fitted": dict(color="#762a83", ls="-.", lw=2.0),
    MODEL_NAME_KEY: dict(color="black", ls="-", lw=3.0),
}

# Display names for the ablation configurations, used in place of the folder
# names. Short forms match Table \ref{tab:ablation} in the manuscript.
CONFIG_DISPLAY_NAME = {
    "baseline_full": "Full configuration",
    "no_mros_loocv": "No MRoS",
    "no_temp_air": "No air temp.",
    "no_temp_dew": "No dew point",
    "no_temp_wet": "No wet-bulb temp.",
    "no_imerg_plp": "No IMERG pLP",
    "no_elev": "No elevation",
    "thermo_only": "Thermodynamics only",
    "min_core_noplp": "Reduced combination (no pLP)",
    "min_core_wplp": "Reduced combination + pLP",
}

# Display names for model features, matching Table \ref{tab:predictors}.
FEATURE_DISPLAY_NAME = {
    "temp_air": "Air temperature",
    "temp_dew": "Dewpoint temperature",
    "temp_wet": "Wet-bulb temperature",
    "elev": "Elevation",
    "imerg_plp": "IMERG pLP",
    "mros_p_snow_loocv": "MRoS $p_{snow}$",
    "mros_p_rain_loocv": "MRoS $p_{rain}$",
    "mros_p_mix_loocv": "MRoS $p_{mix}$",
}

# Short and long display names for the two domains. The dict keys ("CA",
# "CO") stay as the artifact/folder/filename identifiers — main.tex points at
# figures/CA/CA_*.png — but nothing drawn inside a figure says "CA" or "CO".
REGION_DISPLAY = {"CA": "SNM", "CO": "CRM"}
REGION_FULL_LABEL = {
    "CA": "Sierra Nevada Mountains (SNM)",
    "CO": "Colorado Rocky Mountains (CRM)",
}


def region_label(region: str) -> str:
    """Short display name for a domain (SNM / CRM)."""
    return REGION_DISPLAY.get(region, region)


def region_full_label(region: str) -> str:
    """Long display name for a domain, for figure suptitles."""
    return REGION_FULL_LABEL.get(region, REGION_CONFIG.get(region, {}).get("label", region))


def feature_label(name: str) -> str:
    """Readable name for a model feature."""
    return FEATURE_DISPLAY_NAME.get(name, name.replace("_", " "))


def config_label(name: str) -> str:
    """Readable name for an ablation configuration folder."""
    return CONFIG_DISPLAY_NAME.get(name, name.replace("_", " "))


def method_label(name: str) -> str:
    return METHOD_LABEL.get(name, name)


# ---------------------------------------------------------------------------
# Reference data for the study-extent panel.
#
# The panel needs one thing that is not in this repository: the outline of the
# state each domain sits in. It is fetched once and cached under
# REFERENCE_CACHE_DIR as state_<REGION>.geojson, so every later run — including
# on a machine with no outbound network access — reads the cache.
#
# Two sources are tried in order. The Census TIGER cartographic boundary file
# is the better geometry and is preferred; a GeoJSON mirror on GitHub is the
# fallback, because some networks allow github.com but not www2.census.gov.
#
# City labels are a small built-in table rather than a Natural Earth download.
# The panel only needs a handful of places for orientation, and hard-coding
# them removes a second network dependency (and the naciscdn.org host, which is
# blocked more often than github.com).
# ---------------------------------------------------------------------------
STATE_FULL_NAME = {"CA": "California", "CO": "Colorado"}
_STATE_GEOJSON_SLUG = {"CA": "california", "CO": "colorado"}

_US_STATES_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
)
_STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/glynnbird/usstatesgeojson/master/{slug}.geojson"
)

# (name, lon, lat). Chosen for orientation, not completeness.
CONTEXT_CITIES = {
    "CA": [
        ("Sacramento", -121.4944, 38.5816),
        ("Stockton", -121.2908, 37.9577),
        ("Modesto", -120.9969, 37.6391),
        ("Fresno", -119.7871, 36.7378),
        ("Bakersfield", -119.0187, 35.3733),
        ("San Jose", -121.8863, 37.3382),
        ("Redding", -122.3917, 40.5865),
        ("South Lake Tahoe", -119.9772, 38.9399),
    ],
    "CO": [
        ("Denver", -104.9903, 39.7392),
        ("Colorado Springs", -104.8214, 38.8339),
        ("Fort Collins", -105.0844, 40.5853),
        ("Boulder", -105.2705, 40.0150),
        ("Grand Junction", -108.5506, 39.0639),
        ("Pueblo", -104.6091, 38.2544),
        ("Vail", -106.3742, 39.6403),
    ],
}


@functools.lru_cache(maxsize=4)
def _state_boundary(region: str):
    """One state's outline as a single-row GeoDataFrame in EPSG:4326.

    Reads the cache if present; otherwise tries the Census file, then the
    GitHub GeoJSON mirror, and writes whichever succeeds to the cache. Raises
    the last error if every source fails, which the caller turns into a
    printed skip message.
    """
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = REFERENCE_CACHE_DIR / f"state_{region}.geojson"
    if cached.exists():
        return gpd.read_file(cached)

    errors = []

    # 1. Census TIGER cartographic boundaries (best geometry).
    try:
        states = gpd.read_file(_US_STATES_ZIP_URL)
        sub = states[states["STUSPS"] == region]
        if not sub.empty:
            out = sub[["geometry"]].to_crs("EPSG:4326").reset_index(drop=True)
            out.to_file(cached, driver="GeoJSON")
            return out
        errors.append(f"census: no row with STUSPS == {region!r}")
    except Exception as e:
        errors.append(f"census: {type(e).__name__}: {e}")

    # 2. GeoJSON mirror on GitHub (coarser, but usually reachable).
    slug = _STATE_GEOJSON_SLUG.get(region)
    if slug:
        try:
            g = gpd.read_file(_STATE_GEOJSON_URL.format(slug=slug))
            out = g[["geometry"]].set_crs("EPSG:4326", allow_override=True).reset_index(drop=True)
            out.to_file(cached, driver="GeoJSON")
            return out
        except Exception as e:
            errors.append(f"github: {type(e).__name__}: {e}")

    raise RuntimeError("; ".join(errors))


def gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma):
    """Half-width of the mix band, widening towards 0 °C wet-bulb.

    Matches the definition in model/common.py.
    """
    t = np.asarray(temp_wet, float)
    return np.clip(
        base_half_band + extra_half_band * np.exp(-(t ** 2) / (2.0 * sigma ** 2)),
        0.0,
        0.5,
    )


def safe_savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure: Calibration reliability diagram
# ---------------------------------------------------------------------------

def fig_calibration(region: str):
    """Reliability diagram for the full-configuration ablation
    (beta-calibrated), read from the saved shap_values_all.parquet
    (p_snow_cal / p_snow_raw / split / phase_full). Nothing is recalibrated
    here.

    Brier scores are reported in the manuscript text and table rather than in
    the panel titles.
    """
    from sklearn.calibration import calibration_curve

    pq = ablations_dir(region) / "baseline_full" / "shap_values_all.parquet"
    if not pq.exists():
        print(f"  [calibration:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", "p_snow_cal", "p_snow_raw", "split"])
    df = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])]  # pure-phase only

    band_meta_path = ablations_dir(region) / "baseline_full" / "metrics_summary.json"
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if band_meta_path.exists():
        m = json.loads(band_meta_path.read_text())
        base_hb = m.get("band_base_hb", base_hb)
        extra_hb = m.get("band_extra_hb", extra_hb)
        sigma = m.get("band_sigma", sigma)
    band0 = gaussian_half_band(np.array([0.0]), base_hb, extra_hb, sigma)[0]
    rain_thresh_0, snow_thresh_0 = 0.5 - band0, 0.5 + band0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={"height_ratios": [3, 1]})
    for col, (split_name, split_key) in enumerate([("Validation", "val"), ("Test", "test")]):
        d = df[df["split"] == split_key]
        if d.empty:
            continue
        y_true = (d["phase_full"] == SNOW_CODE).to_numpy(int)
        p_raw = d["p_snow_raw"].to_numpy(float)
        p_cal = d["p_snow_cal"].to_numpy(float)

        ax_rel = axes[0, col]
        frac_pos_raw, mean_pred_raw = calibration_curve(y_true, p_raw, n_bins=15, strategy="quantile")
        frac_pos_cal, mean_pred_cal = calibration_curve(y_true, p_cal, n_bins=15, strategy="quantile")
        ax_rel.plot([0, 1], [0, 1], "--", color="grey", lw=1.4, label="Perfect calibration")
        ax_rel.plot(mean_pred_raw, frac_pos_raw, "o--", color="#999999", ms=7, label="Raw")
        ax_rel.plot(mean_pred_cal, frac_pos_cal, "o-", color=SPLIT_COLORS[split_name], ms=7,
                    label="Calibrated")
        ax_rel.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange")
        ax_rel.set_title(split_name)
        ax_rel.set_xlabel("Mean predicted p(snow)")
        ax_rel.set_ylabel("Observed fraction snow")
        ax_rel.legend(loc="upper left")
        ax_rel.grid(alpha=0.25)

        ax_hist = axes[1, col]
        ax_hist.hist(p_cal, bins=30, color=SPLIT_COLORS[split_name], alpha=0.75)
        ax_hist.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange")
        ax_hist.set_xlabel("Calibrated p(snow)")
        ax_hist.set_ylabel("Count")

    fig.suptitle(f"{region_label(region)}: Calibration Reliability "
                 f"({config_label('baseline_full')})", y=1.01)
    safe_savefig(fig, out_dir(region) / f"{region}_calibration_reliability.png")


# ---------------------------------------------------------------------------
# Figure: Benchmark delta forest plots (already-bootstrapped)
# ---------------------------------------------------------------------------

def fig_benchmark_delta_forest(region: str):
    """Horizontal forest plot of the model-minus-benchmark accuracy
    difference, read from bootstrap_benchmark_deltas_ci.csv (cluster
    bootstrap, computed in benchmarking/bootstrap)."""
    csv = bootstrap_dir(region) / "bootstrap_benchmark_deltas_ci.csv"
    if not csv.exists():
        print(f"  [forest:{region}] missing {csv}, skipping")
        return
    deltas_df = pd.read_csv(csv)
    deltas_df = deltas_df[deltas_df["benchmark"].map(keep_method)]

    for metric, fname, title in [
        ("delta_accuracy", f"{region}_benchmark_delta_forest.png",
         "Overall Accuracy Gain Over Benchmarks"),
        ("delta_nearfreeze_accuracy", f"{region}_benchmark_delta_forest_nearfreeze.png",
         "Near-Freezing Accuracy Gain Over Benchmarks"),
    ]:
        d = deltas_df[deltas_df["metric"] == metric].sort_values("point")
        if d.empty:
            print(f"  [forest:{region}] metric {metric} not found in csv, skipping")
            continue
        # Relabel from METHOD_LABEL rather than trusting the CSV's frozen
        # "label" column, so a renamed benchmark propagates here.
        labels = [method_label(b) for b in d["benchmark"]]
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.62 * len(d))))
        ypos = np.arange(len(d))
        ax.errorbar(
            d["point"] * 100, ypos,
            xerr=[(d["point"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["point"]) * 100],
            fmt="o", color="black", ecolor="#666666", elinewidth=2.0, capsize=5,
            capthick=2.0, ms=9,
        )
        ax.axvline(0, color="#d62728", ls="--", lw=1.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Δ Accuracy (percentage points)")
        ax.set_title(f"{region_label(region)}: {title}")
        ax.grid(axis="x", alpha=0.25)
        safe_savefig(fig, out_dir(region) / fname)


# ---------------------------------------------------------------------------
# Figure: Accuracy by T_air with CI (bootstrap-derived)
# ---------------------------------------------------------------------------

def _cluster_bootstrap_tair_bins(region: str, n_boot: int = 500, block_km: float = 30.0,
                                  seed: int = 42):
    """Per-T_air-bin cluster bootstrap from benchmark_predictions_test.parquet.

    The cluster machinery is copied from bootstrap_cis.py rather than imported,
    so this file has no dependency on that script. Returns per-method
    (n_boot, n_bins) accuracy arrays plus point estimates from the unresampled
    data.
    """
    pq = benchmarking_dir(region) / "benchmark_predictions_test.parquet"
    if not pq.exists():
        print(f"  [tair-bootstrap:{region}] missing {pq}, skipping")
        return None
    df = pd.read_parquet(pq)
    pure = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])].reset_index(drop=True)
    y = pure["phase_full"].to_numpy(int)
    tair = pure["temp_air"].to_numpy(float)

    methods = [c.replace("pred_", "") for c in pure.columns
               if c.startswith("pred_") and not c.startswith(f"pred_{MODEL_NAME_KEY}")]
    methods = [m for m in methods if keep_method(m)]
    all_methods = methods + [MODEL_NAME_KEY]
    preds = {m: pure[f"pred_{m}"].to_numpy(int) for m in methods}
    preds[MODEL_NAME_KEY] = pure[f"pred_{MODEL_NAME_KEY}_bin05"].to_numpy(int)
    correct = {m: (preds[m] == y) for m in all_methods}

    edges = TAIR_BIN_EDGES
    n_bins = len(edges) - 1
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(tair, edges) - 1
    bin_valid = np.array([(bin_idx == b).sum() >= MIN_BIN_N for b in range(n_bins)])

    # --- cluster machinery (ported from bootstrap_cis.py) ---
    def make_cluster_codes(d, block_km):
        bs = block_km * 1000.0
        bx = np.floor(d["x"].to_numpy(float) / bs).astype(np.int64)
        by = np.floor(d["y"].to_numpy(float) / bs).astype(np.int64)
        tt = pd.to_datetime(d["time"]).dt.floor("D")
        key = pd.MultiIndex.from_arrays([bx, by, tt])
        return pd.factorize(key)[0]

    def build_cluster_index(codes):
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
        return np.split(order, boundaries)

    def bootstrap_row_indices(cluster_rows, rng):
        k = len(cluster_rows)
        draws = rng.integers(0, k, size=k)
        return np.concatenate([cluster_rows[d] for d in draws])

    if "x" not in pure.columns or "y" not in pure.columns or "time" not in pure.columns:
        print(f"  [tair-bootstrap:{region}] benchmark_predictions_test.parquet lacks "
              f"x/y/time columns needed for spatial clustering, skipping")
        return None

    clusters = build_cluster_index(make_cluster_codes(pure, block_km))
    rng = np.random.default_rng(seed)

    bin_acc = {m: np.full((n_boot, n_bins), np.nan) for m in all_methods}
    for b in range(n_boot):
        idx = bootstrap_row_indices(clusters, rng)
        bi = bin_idx[idx]
        for m in all_methods:
            c = correct[m][idx]
            for bb in range(n_bins):
                if not bin_valid[bb]:
                    continue
                mm = bi == bb
                if mm.sum() >= 5:
                    bin_acc[m][b, bb] = float(np.mean(c[mm]))

    point_acc = {m: np.full(n_bins, np.nan) for m in all_methods}
    for m in all_methods:
        for bb in range(n_bins):
            if not bin_valid[bb]:
                continue
            mm = bin_idx == bb
            if mm.sum() >= 5:
                point_acc[m][bb] = float(np.mean(correct[m][mm]))

    return dict(bin_mids=bin_mids, bin_valid=bin_valid, bin_acc=bin_acc,
                point_acc=point_acc, methods=methods, all_methods=all_methods)


def fig_benchmark_accuracy_by_tair_ci(region: str):
    """Accuracy by T_air with per-bin bootstrap CI ribbons.

    Bins with fewer than MIN_BIN_N pure observations are dropped."""
    res = _cluster_bootstrap_tair_bins(region)
    if res is None:
        return
    bm, bv = res["bin_mids"], res["bin_valid"]
    best = max(res["methods"], key=lambda m: np.nanmean(res["point_acc"][m]))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plot_set = [(MODEL_NAME_KEY, method_label(MODEL_NAME_KEY)),
                (best, f"{method_label(best)} (best benchmark)")]
    for m, label in plot_set:
        arr = res["bin_acc"][m]
        med = np.nanmedian(arr, axis=0) * 100
        lo = np.nanpercentile(arr, 2.5, axis=0) * 100
        hi = np.nanpercentile(arr, 97.5, axis=0) * 100
        v = bv & ~np.isnan(med)
        style = METHOD_STYLE.get(m, {})
        color = style.get("color", None)
        ax.plot(bm[v], med[v], marker="o", ms=8, lw=3.0, color=color, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axvline(0, color="grey", ls="--", lw=1.4)
    ax.axvspan(0, 4, alpha=0.06, color="orange")
    ax.set(xlabel=f"Air temperature (°C; {TAIR_BIN_WIDTH:g} °C bins, n ≥ {MIN_BIN_N})",
           ylabel="Accuracy (%)", ylim=(0, 102),
           title=f"{region_label(region)}: Accuracy by Air Temperature")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_accuracy_by_tair_ci.png")


def fig_benchmark_relative_improvement_ci(region: str):
    """Model accuracy minus best and average benchmark accuracy by T_air bin,
    with cluster-bootstrap CI ribbons paired within replicate.

    Shares _cluster_bootstrap_tair_bins with fig_benchmark_accuracy_by_tair_ci,
    so both figures use the same bins and bootstrap draws."""
    res = _cluster_bootstrap_tair_bins(region)
    if res is None:
        return
    bm, bv = res["bin_mids"], res["bin_valid"]
    best = max(res["methods"], key=lambda m: np.nanmean(res["point_acc"][m]))
    arr_model = res["bin_acc"][MODEL_NAME_KEY]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for other_arr, color, label in [
        (res["bin_acc"][best], "black", f"vs. best benchmark ({method_label(best)})"),
        (np.nanmean(np.stack([res["bin_acc"][m] for m in res["methods"]]), axis=0),
         "#d62728", "vs. average benchmark"),
    ]:
        d = (arr_model - other_arr) * 100
        med = np.nanmedian(d, axis=0)
        lo = np.nanpercentile(d, 2.5, axis=0)
        hi = np.nanpercentile(d, 97.5, axis=0)
        v = bv & ~np.isnan(med)
        ax.plot(bm[v], med[v], "-o", color=color, lw=2.6, ms=8, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axhline(0, color="grey", lw=1.2)
    ax.axvline(0, color="grey", ls="--", lw=1.4)
    ax.axvspan(0, 4, alpha=0.07, color="orange", label="Benchmark performance-dip zone")
    ax.set(xlabel="Air temperature (°C)",
           ylabel="Δ Accuracy (percentage points)",
           title=f"{region_label(region)}: {MODEL_DISPLAY_NAME} Accuracy Relative to Benchmarks")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_relative_improvement_ci.png")


def fig_benchmark_accuracy_by_tair(region: str):
    """Point-estimate accuracy / snow bias / rain bias by T_air for every
    benchmark plus the model, from the saved profile_by_tair_<method>.csv
    files under benchmarking/.

    Bins with n < MIN_BIN_N pure obs were already dropped when those CSVs were
    written."""
    bdir = benchmarking_dir(region)
    profiles = {}
    for f in bdir.glob("profile_by_tair_*.csv"):
        name = f.stem.replace("profile_by_tair_", "")
        if not keep_method(name):
            continue
        profiles[name] = pd.read_csv(f)
    if not profiles:
        print(f"  [benchmark_accuracy_by_tair:{region}] no profile_by_tair_*.csv found "
              f"under {bdir}, skipping")
        return

    panel_specs = [
        ("accuracy_pct", "Accuracy (%)", (0, 102)),
        ("snow_bias_pct", "Snow bias (%)", (-105, 105)),
        ("rain_bias_pct", "Rain bias (%)", (-105, 105)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 15), sharex=True)
    fig.suptitle(f"{region_label(region)}: Benchmark PPMs vs. {MODEL_DISPLAY_NAME} "
                 f"by Air Temperature")

    order = [n for n in METHOD_STYLE if n in profiles]
    for ax, (col, ylabel, ylim) in zip(axes, panel_specs):
        for name in order:
            prof = profiles[name]
            if prof.empty or col not in prof.columns:
                continue
            valid = prof[col].notna()
            style = METHOD_STYLE[name]
            ax.plot(prof.loc[valid, "t_mid"], np.clip(prof.loc[valid, col], *ylim),
                    marker="o", ms=6 if name == MODEL_NAME_KEY else 4,
                    label=method_label(name),
                    zorder=5 if name == MODEL_NAME_KEY else 2, **style)
        ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
        if "bias" in col:
            ax.axhline(0, color="grey", lw=1.0, alpha=0.7)
        ax.set(ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(f"Air temperature (°C; {TAIR_BIN_WIDTH:g} °C bins, n ≥ {MIN_BIN_N}, "
                        f"pure rain/snow obs, test split)")
    axes[0].legend(fontsize=11, ncol=2, loc="lower left", framealpha=0.9)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_accuracy_by_tair.png")


def fig_overall_accuracy_bars(region: str):
    """Overall and near-freezing accuracy bars, benchmarks vs model, from the
    saved benchmark_comparison.csv."""
    csv = benchmarking_dir(region) / "benchmark_comparison.csv"
    if not csv.exists():
        print(f"  [overall_bars:{region}] missing {csv}, skipping")
        return
    t = pd.read_csv(csv)
    t = t[t["method"].map(keep_method)]
    # benchmark_comparison.csv carries its own "label" column, frozen when the
    # CSV was written (e.g. "XGBoost + MRoS (this study)"). Always recompute
    # from METHOD_LABEL so a renamed model or benchmark is reflected here.
    t["label"] = t["method"].map(method_label)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.62 * len(t))), sharey=True)
    for ax, col, title in [
        (axes[0], "accuracy_pct", "Overall accuracy (%)"),
        (axes[1], "nearfreeze_accuracy_pct",
         f"Near-freezing accuracy (%; |$T_w$| ≤ {NEARFREEZE_TWET_C:g} °C)"),
    ]:
        if col not in t.columns:
            continue
        sub = t.dropna(subset=[col]).sort_values(col)
        colors = ["black" if m == MODEL_NAME_KEY else
                  METHOD_STYLE.get(m, {}).get("color", "grey") for m in sub["method"]]
        ax.barh(sub["label"], sub[col], color=colors, alpha=0.88)
        for _, r in sub.iterrows():
            ax.text(r[col] + 0.6, r["label"], f"{r[col]:.1f}", va="center", fontsize=12)
        ax.set(title=title, xlim=(0, 108))
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{region_label(region)}: Benchmark Comparison")
    safe_savefig(fig, out_dir(region) / f"{region}_overall_accuracy_bars.png")


def fig_ablation_comparison(region: str):
    """Ablation deltas from the full configuration, read from the saved
    ablation_comparison.csv under the ablation folder. Config folder names are
    mapped to readable labels via config_label()."""
    csv = ablations_dir(region) / "ablation_comparison.csv"
    if not csv.exists():
        print(f"  [ablation_comparison:{region}] missing {csv}, skipping")
        return
    comparison_df = pd.read_csv(csv)
    ok = comparison_df[comparison_df["status"] == "ok"].copy()
    if ok.empty:
        print(f"  [ablation_comparison:{region}] no rows with status=='ok', skipping")
        return
    baseline_row = ok[ok["name"] == "baseline_full"]
    if baseline_row.empty:
        print(f"  [ablation_comparison:{region}] no baseline_full row, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(17, max(5, len(ok) * 0.6)))
    fig.suptitle(f"{region_label(region)}: Ablation Deltas from "
                 f"{config_label('baseline_full')}")
    for ax, col, title in [
        (axes[0], "test_macro_f1_binary", "Δ Macro F1 (binary)"),
        (axes[1], "test_roc_auc_cal", "Δ AUROC (calibrated)"),
    ]:
        if col not in ok.columns:
            continue
        baseline_val = float(baseline_row[col].iloc[0])
        ok_delta = ok.copy()
        ok_delta["delta"] = ok_delta[col] - baseline_val
        ok_delta["display_name"] = ok_delta["name"].map(config_label)
        ok_delta = ok_delta.sort_values("delta", ascending=True)
        colors = ["#2dc653" if r["name"] == "baseline_full" else
                  "#d62728" if r["delta"] < 0 else "#3a86ff"
                  for _, r in ok_delta.iterrows()]
        ax.barh(ok_delta["display_name"], ok_delta["delta"], color=colors)
        ax.axvline(0, color="black", lw=1.2)
        ax.set(xlabel=title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        # Pad the x-limits so the value labels drawn at each bar tip sit inside
        # the axes rather than running past the spine into the y tick labels.
        deltas = ok_delta["delta"].to_numpy(float)
        lo = min(0.0, float(np.nanmin(deltas)))
        hi = max(0.0, float(np.nanmax(deltas)))
        span = (hi - lo) or 1e-3
        pad = 0.28 * span
        ax.set_xlim(lo - pad, hi + pad)
        offset = 0.02 * span
        for _, row in ok_delta.iterrows():
            ax.text(row["delta"] + (offset if row["delta"] >= 0 else -offset),
                    row["display_name"], f"{row[col]:.3f}",
                    va="center", ha="left" if row["delta"] >= 0 else "right",
                    fontsize=11, color="dimgrey", clip_on=False)
    safe_savefig(fig, out_dir(region) / f"{region}_ablation_comparison.png")


# ---------------------------------------------------------------------------
# Figure: F1 by temperature (wet-bulb, main; air temperature, counterpart)
# ---------------------------------------------------------------------------

# The two decision rules the manuscript distinguishes (Sections 4.4 / 5.4):
#
#   "binary"     — the calibrated probability is thresholded at 0.5 and EVERY
#                  observation receives a snow or rain call. Nothing is
#                  withheld, so F1 is computed over all pure observations in a
#                  bin. Flag rate and mix capture do not exist under this rule.
#   "selective"  — the Gaussian uncertainty envelope of Equation 1 is applied,
#                  and observations inside it are declined rather than
#                  classified. F1 is then computed only over the observations
#                  the model committed to, which is a different (and easier)
#                  sample; flag rate and mix-capture rate describe the
#                  abstention itself.
#
# The two are not directly comparable, which is exactly why they get separate
# figures rather than separate panels of one figure.
RULE_BINARY = "binary"
RULE_SELECTIVE = "selective"


def _temp_profile(df_f, bin_edges, base_hb, extra_hb, sigma, temp_col, min_n=10,
                  rule=RULE_SELECTIVE):
    """Per-bin F1(snow) / F1(rain), plus flag rate and mix-capture rate under
    the selective rule, binned on any temperature column.

    Under RULE_BINARY the envelope is not applied: every observation is called
    at the 0.5 threshold, F1 is computed over every pure observation in the
    bin, and flag_rate / mix_capture are NaN because they are undefined.

    Under RULE_SELECTIVE the envelope decides, F1 counts only committed
    observations, and both abstention rates are returned. This is the
    behaviour of _twet_profile_binary in the ablation runner.
    """
    from sklearn.metrics import f1_score

    rows = []
    tvals = df_f[temp_col].to_numpy(float)
    p_cal = df_f["p_snow_cal"].to_numpy(float)
    phase = df_f["phase_full"].to_numpy(int)

    if rule == RULE_BINARY:
        pred = np.where(p_cal >= 0.5, SNOW_CODE, RAIN_CODE)
    else:
        hb = gaussian_half_band(tvals, base_hb, extra_hb, sigma)
        pred = np.full(len(p_cal), MIX_CODE, dtype=int)
        pred[p_cal <= 0.5 - hb] = RAIN_CODE
        pred[p_cal >= 0.5 + hb] = SNOW_CODE

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (tvals >= lo) & (tvals < hi)
        n = mask.sum()
        if n < min_n:
            continue
        pure = np.isin(phase[mask], [SNOW_CODE, RAIN_CODE])
        if rule == RULE_BINARY:
            sel = pure
            flag_rate = np.nan
            mix_capture = np.nan
        else:
            committed = pred[mask] != MIX_CODE
            sel = committed & pure
            flag_rate = (~committed).mean()
            mix_mask = phase[mask] == MIX_CODE
            mix_capture = np.nan
            if mix_mask.sum() > 0:
                mix_capture = (pred[mask][mix_mask] == MIX_CODE).mean()

        f1_snow = f1_rain = np.nan
        if sel.sum() >= 2 and len(np.unique(phase[mask][sel])) > 1:
            y_true_sel = phase[mask][sel]
            y_pred_sel = pred[mask][sel]
            f1_snow = f1_score(y_true_sel == SNOW_CODE, y_pred_sel == SNOW_CODE, zero_division=0)
            f1_rain = f1_score(y_true_sel == RAIN_CODE, y_pred_sel == RAIN_CODE, zero_division=0)

        rows.append(dict(bin_mid=(lo + hi) / 2, n=n, f1_snow=f1_snow, f1_rain=f1_rain,
                          flag_rate=flag_rate, mix_capture=mix_capture))
    return pd.DataFrame(rows)


# Kept under its original name so any external caller still works.
def _twet_profile(df_f, bin_edges, base_hb, extra_hb, sigma, min_n=10):
    return _temp_profile(df_f, bin_edges, base_hb, extra_hb, sigma, "temp_wet", min_n=min_n)


def _load_baseline_profiles(region: str, temp_col: str, bin_edges, min_n=10,
                            rule=RULE_SELECTIVE):
    """Validation and test profiles for the full configuration, binned on
    temp_col under the given decision rule. Returns (prof_val, prof_test) or
    None if the artifact is missing."""
    pq = ablations_dir(region) / "baseline_full" / "shap_values_all.parquet"
    meta_path = ablations_dir(region) / "baseline_full" / "metrics_summary.json"
    if not pq.exists():
        print(f"  [profiles:{region}] missing {pq}, skipping")
        return None
    df = pd.read_parquet(pq, columns=["phase_full", temp_col, "p_snow_cal", "split"])
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        base_hb = m.get("band_base_hb", base_hb)
        extra_hb = m.get("band_extra_hb", extra_hb)
        sigma = m.get("band_sigma", sigma)
    return (
        _temp_profile(df[df["split"] == "val"], bin_edges, base_hb, extra_hb, sigma,
                      temp_col, min_n=min_n, rule=rule),
        _temp_profile(df[df["split"] == "test"], bin_edges, base_hb, extra_hb, sigma,
                      temp_col, min_n=min_n, rule=rule),
    )


def _draw_split_panel(ax, prof_val, prof_test, col, title, color, xlabel, xlim=None):
    if prof_val is not None and not prof_val.empty:
        ax.plot(prof_val["bin_mid"], prof_val[col], ls=(0, (6, 2.5)), color=color,
                marker="s", ms=7, lw=2.4, label="Validation")
    if prof_test is not None and not prof_test.empty:
        ax.plot(prof_test["bin_mid"], prof_test[col], ls="-", color=color,
                marker="o", ms=7, lw=2.8, label="Test")
    ax.axvspan(-2, 2, alpha=0.09, color="orange")
    ax.axvline(0, color="grey", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    # Common 0-1 range so panels are visually comparable.
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    # Fixed x-limits from the bin edges, so the same figure for the two domains
    # shares an axis and can be stacked in the manuscript without the reader
    # having to re-read the scale.
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)


F1_PANELS = [("f1_snow", "F1 (snow)", PHASE_COLORS["snow"]),
             ("f1_rain", "F1 (rain)", PHASE_COLORS["rain"])]
SELECTIVE_EXTRA_PANELS = [("flag_rate", "Flag rate", "#D55E00"),
                          ("mix_capture", "Mix capture rate", PHASE_COLORS["mix"])]


def _fig_profile_by_temp(region: str, temp_col: str, bin_edges, temp_label: str,
                         rule: str, panels, fname: str, title: str):
    """Shared body for the four F1/abstention-by-temperature figures.

    One code path for both decision rules and both binning variables, so the
    binary-rule pair and the selective-rule pair differ only in the arguments
    passed here — no chance of the wet-bulb and air-temperature versions of the
    same figure drifting apart.
    """
    profs = _load_baseline_profiles(region, temp_col, bin_edges, rule=rule)
    if profs is None:
        return
    prof_val, prof_test = profs

    xlim = (float(bin_edges[0]), float(bin_edges[-1]))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
    axes = np.atleast_1d(axes)
    for ax, (col, panel_title, color) in zip(axes, panels):
        _draw_split_panel(ax, prof_val, prof_test, col, panel_title, color,
                          temp_label, xlim=xlim)

    fig.suptitle(f"{region_label(region)}: {title} ({config_label('baseline_full')})",
                 y=1.02)
    safe_savefig(fig, out_dir(region) / fname)


def fig_twet_performance(region: str):
    """Figure 3: snow and rain F1 in 1 °C wet-bulb bins under the BINARY rule,
    validation vs test, for the full configuration.

    Every observation receives a call at the 0.5 threshold — the envelope is
    not applied — so F1 counts every pure snow/rain observation in a bin. These
    values are lower than the selective-rule ones in
    fig_selective_by_wetbulb, because the observations the envelope declines
    are disproportionately the ones the model gets wrong; the two rules are not
    directly comparable and any number quoted from this figure in the text must
    come from this figure.

    Matches the f1_snow / f1_rain columns of Manuscript/fig3_binary_f1_by_twet.csv.
    """
    _fig_profile_by_temp(
        region, "temp_wet", TWET_BIN_EDGES, "Wet-bulb temperature (°C)",
        RULE_BINARY, F1_PANELS, f"{region}_f1_by_wetbulb.png",
        "Snow and Rain F1 by Wet-Bulb Temperature",
    )


def fig_f1_by_tair(region: str):
    """Air-temperature counterpart to Figure 3, same BINARY rule.

    Bins use TAIR_BIN_EDGES (-8 to 8 °C, 1 °C wide), the convention used by
    every other air-temperature figure in this script, rather than the
    wet-bulb figure's TWET_BIN_EDGES — the two are physically different
    quantities, so reusing one set of edges for the other would not be
    meaningful. Bins with n < 10 observations are dropped, a lower cutoff than
    the MIN_BIN_N = 20 used for the accuracy-by-T_air figures, following the
    ablation runner's convention.
    """
    _fig_profile_by_temp(
        region, "temp_air", TAIR_BIN_EDGES, "Air temperature (°C)",
        RULE_BINARY, F1_PANELS, f"{region}_f1_by_tair.png",
        "Snow and Rain F1 by Air Temperature",
    )


def fig_selective_by_wetbulb(region: str):
    """Selective-rule counterpart to Figure 3: F1, flag rate and mix-capture
    rate in 1 °C wet-bulb bins, validation vs test.

    The uncertainty envelope of Equation 1 is applied, so observations inside
    it are declined rather than classified. The F1 panels therefore count only
    the observations the model committed to — a different and easier sample
    than the binary-rule figure's — and the flag-rate and mix-capture panels
    describe the abstention itself. Flag rate is the share of all observations
    inside the envelope; mix-capture rate is the share of observer-reported
    mixed events inside it.
    """
    _fig_profile_by_temp(
        region, "temp_wet", TWET_BIN_EDGES, "Wet-bulb temperature (°C)",
        RULE_SELECTIVE, F1_PANELS + SELECTIVE_EXTRA_PANELS,
        f"{region}_selective_by_wetbulb.png",
        "Selective-Rule Performance by Wet-Bulb Temperature",
    )


def fig_selective_by_tair(region: str):
    """Air-temperature counterpart to fig_selective_by_wetbulb.

    Note that the envelope's width is defined on wet-bulb temperature, so the
    flag-rate and mix-capture panels here show how abstention happens to fall
    across air temperature rather than how it is parameterised.
    """
    _fig_profile_by_temp(
        region, "temp_air", TAIR_BIN_EDGES, "Air temperature (°C)",
        RULE_SELECTIVE, F1_PANELS + SELECTIVE_EXTRA_PANELS,
        f"{region}_selective_by_tair.png",
        "Selective-Rule Performance by Air Temperature",
    )


# ---------------------------------------------------------------------------
# Figure: SHAP — wet-bulb heatmap (main) + mean-by-phase bar (appendix)
# ---------------------------------------------------------------------------

def fig_shap(region: str):
    kdir = kriging_dir(region)
    heatmap_csv = kdir / "shap_by_wetbulb_bin.csv"
    phase_csv = kdir / "shap_summary_by_phase.csv"

    if heatmap_csv.exists():
        data = pd.read_csv(heatmap_csv).set_index("feature")
        data = data.loc[:, data.notna().any()]
        data.index = [feature_label(f) for f in data.index]
        figsize = (max(13, len(data.columns) * 1.0), len(data.index) * 0.72 + 2.2)
        if HAVE_SEABORN:
            vmax = data.max().max()
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(data, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, annot=True, fmt=".3f",
                        annot_kws={"size": 11}, linewidths=0.4, linecolor="#cccccc",
                        cbar_kws={"label": "Mean |SHAP| (probability units)"})
            ax.set_title(f"{region_label(region)}: Mean |SHAP| by Wet-Bulb Bin", pad=12)
            ax.set_xlabel("Wet-bulb temperature bin (°C)")
            ax.set_ylabel("Feature")
            ax.tick_params(labelsize=12)
            ax.figure.axes[-1].yaxis.label.set_size(13)
            near_freeze_cols = [i for i, lbl in enumerate(data.columns)
                                if lbl in ["-2–1", "-1–0", "0–1", "1–2",
                                           "-2--1", "-1-0", "0-1", "1-2"]]
            for col_i in near_freeze_cols:
                ax.add_patch(plt.Rectangle((col_i, 0), 1, len(data), fill=True,
                                           color="steelblue", alpha=0.07, zorder=0))
            safe_savefig(fig, out_dir(region) / f"{region}_shap_wetbulb_heatmap.png")
        else:
            print(f"  [shap:{region}] seaborn not available, using imshow fallback")
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(data.columns)))
            ax.set_xticklabels(data.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(data.index)))
            ax.set_yticklabels(data.index)
            fig.colorbar(im, ax=ax, label="Mean |SHAP| (probability units)")
            ax.set_title(f"{region_label(region)}: Mean |SHAP| by Wet-Bulb Bin")
            safe_savefig(fig, out_dir(region) / f"{region}_shap_wetbulb_heatmap.png")
    else:
        print(f"  [shap:{region}] missing {heatmap_csv}, skipping headline SHAP figure")

    # Appendix: mean |SHAP| by phase bar chart
    if phase_csv.exists():
        phase_shap_summary = pd.read_csv(phase_csv)
        FEATURES = phase_shap_summary["feature"].tolist()
        FEATURE_LABELS = [feature_label(f) for f in FEATURES]
        phase_plot_cols = [c for c in ["snow", "rain", "mix"] if c in phase_shap_summary.columns]
        x = np.arange(len(FEATURES))
        width = 0.25
        colors = [PHASE_COLORS.get(c, "grey") for c in phase_plot_cols]
        fig, ax = plt.subplots(figsize=(13, 6.5))
        for i, (phase_col, color) in enumerate(zip(phase_plot_cols, colors)):
            vals = phase_shap_summary.set_index("feature").loc[FEATURES, phase_col].values
            ax.bar(x + i * width, vals, width, label=phase_col, color=color, alpha=0.9)
        ax.set_xticks(x + width)
        ax.set_xticklabels(FEATURE_LABELS, rotation=30, ha="right")
        ax.set_ylabel("Mean |SHAP| (probability units)")
        ax.set_title(f"{region_label(region)}: Feature Importance by Predicted Phase")
        ax.legend(title="Predicted phase")
        ax.grid(axis="y", alpha=0.3)
        safe_savefig(fig, out_dir(region) / f"{region}_shap_mean_by_phase_appendix.png")
    else:
        print(f"  [shap:{region}] missing {phase_csv}, skipping appendix bar chart")


# ---------------------------------------------------------------------------
# Figure: Ablation bootstrap CI forest, per-config deltas
# ---------------------------------------------------------------------------

def fig_ablation_deltas_ci(region: str):
    """Forest plot of bootstrap CI deltas per ablation configuration.

    Reads bootstrap_ablation_deltas_ci.csv (columns: config, metric, point,
    ci_lo, ci_hi, p_delta_le_0, excludes_zero). That file holds AUROC, overall
    accuracy and near-freezing accuracy deltas only; no macro-F1 delta is
    saved, so macro-F1 is not plotted."""
    csv = bootstrap_dir(region) / "bootstrap_ablation_deltas_ci.csv"
    if not csv.exists():
        print(f"  [ablation_ci:{region}] missing {csv}, skipping")
        return
    d = pd.read_csv(csv)

    # The metric KEYS are the literal values stored in the CSV and must keep
    # the on-disk "..._baseline_minus_config" spelling; only the axis labels
    # are relabelled.
    metric_titles = {
        "delta_auc_baseline_minus_config":
            "Δ AUROC (full configuration − ablation)",
        "delta_accuracy_baseline_minus_config":
            "Δ Accuracy (full configuration − ablation)",
        "delta_nearfreeze_acc_baseline_minus_config":
            "Δ Near-freezing accuracy (full configuration − ablation)",
    }
    metrics_present = [m for m in metric_titles if m in d["metric"].unique()]
    if not metrics_present:
        print(f"  [ablation_ci:{region}] no expected metrics found in {csv}, skipping")
        return

    fig, axes = plt.subplots(1, len(metrics_present),
                             figsize=(7.5 * len(metrics_present),
                                      max(5, 0.55 * d["config"].nunique())),
                             sharey=True)
    if len(metrics_present) == 1:
        axes = [axes]
    configs_order = None
    for ax, metric in zip(axes, metrics_present):
        sub = d[d["metric"] == metric].copy()
        if configs_order is None:
            configs_order = sub.sort_values("point")["config"].tolist()
        sub["config"] = pd.Categorical(sub["config"], categories=configs_order, ordered=True)
        sub = sub.sort_values("config")
        ypos = np.arange(len(sub))
        colors = ["#2166ac" if ex else "#999999" for ex in sub["excludes_zero"]]
        for y, (_, row), c in zip(ypos, sub.iterrows(), colors):
            ax.errorbar(row["point"], y,
                        xerr=[[row["point"] - row["ci_lo"]], [row["ci_hi"] - row["point"]]],
                        fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=5,
                        capthick=2.0, ms=9,
                        markeredgecolor="black" if row["excludes_zero"] else c)
        ax.axvline(0, color="#d62728", ls="--", lw=1.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([config_label(str(c)) for c in sub["config"]])
        ax.set_xlabel(metric_titles[metric], fontsize=13)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(f"{region_label(region)}: Ablation Deltas vs. "
                 f"{config_label('baseline_full')}\n"
                 f"(95% cluster-bootstrap CIs; blue = CI excludes zero)", y=1.03)
    safe_savefig(fig, out_dir(region) / f"{region}_ablation_deltas_ci.png")


# ---------------------------------------------------------------------------
# Figure: mix capture by wet-bulb bin + band placement (appendix)
# ---------------------------------------------------------------------------

def _binomial_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def fig_mix_capture_by_wetbulb(region: str):
    """Mix capture rate per T_wet bin, with the band-placement scatter as a
    companion figure.

    Capture rate is the fraction of observed-mix events whose calibrated
    p(snow) falls inside the band. Bins match fig_twet_performance. Read from
    the model's test_full_uncertainty_predictions_combined.parquet, which
    carries no clustering key, so the CI is a per-bin binomial normal
    approximation rather than a cluster bootstrap.
    """
    kdir = kriging_dir(region)
    pq = kdir / "test_full_uncertainty_predictions_combined.parquet"
    meta_path = kdir / "metrics_summary.json"
    if not pq.exists():
        print(f"  [band:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", "temp_wet", "p_snow_cal", "split"])
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        gb = m.get("run_info", {}).get("gaussian_band", {})
        base_hb = gb.get("base_half_band", base_hb)
        extra_hb = gb.get("extra_half_band", extra_hb)
        sigma = gb.get("sigma_degC", sigma)

    df_mix = df[df["phase_full"] == MIX_CODE].copy()
    hb_obs = gaussian_half_band(df_mix["temp_wet"].to_numpy(), base_hb, extra_hb, sigma)
    df_mix["inside_band"] = (
        (df_mix["p_snow_cal"] > 0.5 - hb_obs) & (df_mix["p_snow_cal"] < 0.5 + hb_obs)
    )

    rows = []
    for i in range(len(TWET_BIN_EDGES) - 1):
        lo, hi = TWET_BIN_EDGES[i], TWET_BIN_EDGES[i + 1]
        mask = (df_mix["temp_wet"] >= lo) & (df_mix["temp_wet"] < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(df_mix.loc[mask, "inside_band"].sum())
        ci_lo, ci_hi = _binomial_ci(k, n)
        rows.append(dict(bin_mid=(lo + hi) / 2, n=n, capture_rate=k / n,
                         ci_lo=ci_lo, ci_hi=ci_hi))
    prof = pd.DataFrame(rows)
    if prof.empty:
        print(f"  [band:{region}] no mix events found, skipping")
        return

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()
    ax2.bar(prof["bin_mid"], prof["n"], width=0.8, color="#dddddd", zorder=0,
            label="n (mix events)")
    ax2.set_ylabel("n mix events per bin")
    ax1.errorbar(prof["bin_mid"], prof["capture_rate"] * 100,
                 yerr=[(prof["capture_rate"] - prof["ci_lo"]) * 100,
                       (prof["ci_hi"] - prof["capture_rate"]) * 100],
                 fmt="o-", color=PHASE_COLORS["mix"], ecolor=PHASE_COLORS["mix"],
                 elinewidth=2.0, capsize=5, capthick=2.0, ms=9, lw=2.8, zorder=3,
                 label="Mix capture rate")
    for _, row in prof.iterrows():
        ax1.annotate(f"n={row['n']:.0f}",
                     (row["bin_mid"], min(100, row["capture_rate"] * 100 + 6)),
                     ha="center", fontsize=10, color="#555555")
    ax1.axvspan(-2, 2, alpha=0.09, color="orange")
    ax1.axvline(0, color="grey", lw=1.2)
    ax1.set_xlabel("Wet-bulb temperature (°C)")
    ax1.set_ylabel("Mix capture rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title(f"{region_label(region)}: Mix Capture Rate by Wet-Bulb Bin")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(alpha=0.2)
    safe_savefig(fig, out_dir(region) / f"{region}_mix_capture_by_wetbulb.png")

    # --- Band placement vs observed mix ---
    # The t_grid spans the real data range (with a small pad) so the band
    # boundary and shading always cover every plotted point.
    twet_data_min = float(df_mix["temp_wet"].min())
    twet_data_max = float(df_mix["temp_wet"].max())
    twet_pad = 0.5
    t_grid = np.linspace(twet_data_min - twet_pad, twet_data_max + twet_pad, 300)
    hb_grid = gaussian_half_band(t_grid, base_hb, extra_hb, sigma)
    snow_boundary, rain_boundary = 0.5 + hb_grid, 0.5 - hb_grid
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_between(t_grid, rain_boundary, snow_boundary, alpha=0.12, color="orange")
    ax.plot(t_grid, snow_boundary, color="black", lw=1.8)
    ax.plot(t_grid, rain_boundary, color="black", lw=1.8, ls="--")
    captured = df_mix[df_mix["inside_band"]]
    missed = df_mix[~df_mix["inside_band"]]
    ax.scatter(missed["temp_wet"], missed["p_snow_cal"], color="#cc3311", marker="x",
               s=55, linewidths=2.0, alpha=0.7, label="Missed")
    ax.scatter(captured["temp_wet"], captured["p_snow_cal"], color="#009988", marker="o",
               s=55, alpha=0.7, label="Captured")
    ax.set_xlabel("Wet-bulb temperature (°C)")
    ax.set_ylabel("Calibrated p(snow)")
    ax.set_title(f"{region_label(region)}: Uncertainty Band Placement vs. Observed Mix")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    safe_savefig(fig, out_dir(region) / f"{region}_band_placement_appendix.png")


# ---------------------------------------------------------------------------
# Figure: near-freezing precision and recall, full configuration vs no-MRoS
# ---------------------------------------------------------------------------

def _nf_profile(df_f, bin_edges, temp_col):
    """Per-bin precision/recall for snow and rain at the raw 0.5 threshold (no
    uncertainty band), pure observations only, mix excluded."""
    rows = []
    tvals = df_f[temp_col].to_numpy(float)
    p_cal = df_f["p_snow_cal"].to_numpy(float)
    phase = df_f["phase_full"].to_numpy(int)
    pure = np.isin(phase, [SNOW_CODE, RAIN_CODE])

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (tvals >= lo) & (tvals < hi)
        m = mask & pure
        if m.sum() < 5:
            continue
        yt = phase[m]
        yp = np.where(p_cal[m] >= 0.5, SNOW_CODE, RAIN_CODE)

        def _prf(code):
            tp = np.sum((yt == code) & (yp == code))
            fp = np.sum((yt != code) & (yp == code))
            fn = np.sum((yt == code) & (yp != code))
            pr = tp / (tp + fp) if (tp + fp) else np.nan
            rc = tp / (tp + fn) if (tp + fn) else np.nan
            return pr, rc

        snow_pr, snow_rc = _prf(SNOW_CODE)
        rain_pr, rain_rc = _prf(RAIN_CODE)
        rows.append(dict(t_mid=(lo + hi) / 2, n=int(m.sum()),
                          prec_snow=snow_pr, rec_snow=snow_rc,
                          prec_rain=rain_pr, rec_rain=rain_rc))
    return pd.DataFrame(rows)


NF_BIN_EDGES = np.arange(-4, 5, 1)

CONFIG_COMPARE_STYLE = {
    "baseline_full": dict(color="#2166ac"),
    "no_mros_loocv": dict(color="#d62728"),
}

# Precision is solid with round markers, recall is dashed with square markers.
# Both are spelled out in a dedicated legend (see below) because colour alone
# distinguishes configuration, not quantity.
PRECISION_LINE = dict(ls="-", marker="o", ms=8, lw=2.8)
RECALL_LINE = dict(ls=(0, (6, 2.5)), marker="s", ms=8, lw=2.8)


def _fig_story5_compare(region: str, temp_col: str, fname: str):
    """Full configuration vs no-MRoS overlay of near-freezing precision and
    recall against temperature, test split.

    Both configurations are drawn in the same panel. Each panel carries two
    separate legends so the encoding is unambiguous: one maps colour to
    configuration, the other maps line style and marker to precision vs recall.
    A single combined legend made the dashed/solid distinction hard to read at
    print size, and splitting the two keys across the two panels left each
    panel unreadable on its own — both keys are repeated in both panels.
    """
    from matplotlib.lines import Line2D

    configs = {}
    for cfg in ["baseline_full", "no_mros_loocv"]:
        pq = ablations_dir(region) / cfg / "shap_values_all.parquet"
        if not pq.exists():
            print(f"  [story5:{region}:{fname}] missing {pq} for config '{cfg}', "
                  f"omitting that config from the comparison")
            continue
        d = pd.read_parquet(pq, columns=["phase_full", temp_col, "p_snow_cal", "split"])
        configs[cfg] = d[d["split"] == "test"]
    if not configs:
        print(f"  [story5:{region}:{fname}] no configs available, skipping")
        return

    profiles = {cfg: _nf_profile(d, NF_BIN_EDGES, temp_col) for cfg, d in configs.items()}
    temp_label = "Air temperature" if temp_col == "temp_air" else "Wet-bulb temperature"
    temp_label_title = "Air Temperature" if temp_col == "temp_air" else "Wet-Bulb Temperature"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    panel_specs = [("prec_snow", "rec_snow", "Snow"), ("prec_rain", "rec_rain", "Rain")]
    for ax, (prec_col, rec_col, label) in zip(axes, panel_specs):
        for cfg, prof in profiles.items():
            if prof.empty:
                continue
            color = CONFIG_COMPARE_STYLE.get(cfg, {}).get("color", "grey")
            valid_p = prof[prec_col].notna()
            valid_r = prof[rec_col].notna()
            ax.plot(prof.loc[valid_p, "t_mid"], prof.loc[valid_p, prec_col],
                    color=color, **PRECISION_LINE)
            ax.plot(prof.loc[valid_r, "t_mid"], prof.loc[valid_r, rec_col],
                    color=color, **RECALL_LINE)
        ax.axvspan(-2, 2, alpha=0.09, color="orange", zorder=0)
        ax.axvline(0, color="black", lw=1.0, alpha=0.4, zorder=0)
        ax.set(xlabel=f"{temp_label} (°C)", ylabel=f"{label} precision / recall",
               title=label, xlim=(NF_BIN_EDGES[0], NF_BIN_EDGES[-1]), ylim=(0, 1.08))
        ax.grid(alpha=0.3)

    # Two legends per panel: colour = configuration, line style = quantity.
    # Both are repeated in both panels so each panel reads on its own.
    for ax in axes:
        config_handles = [
            Line2D([0], [0], color=CONFIG_COMPARE_STYLE.get(c, {}).get("color", "grey"),
                   lw=3.2, label=config_label(c))
            for c in profiles
        ]
        quantity_handles = [
            Line2D([0], [0], color="#444444", label="Precision", **PRECISION_LINE),
            Line2D([0], [0], color="#444444", label="Recall", **RECALL_LINE),
        ]
        leg_cfg = ax.legend(handles=config_handles, loc="lower left", fontsize=11,
                            handlelength=2.8, title="Configuration", framealpha=0.92)
        ax.add_artist(leg_cfg)
        ax.legend(handles=quantity_handles, loc="lower right", fontsize=11,
                  handlelength=4.2, title="Quantity", framealpha=0.92)

    fig.suptitle(f"{region_label(region)}: Near-Freezing Precision and Recall by "
                 f"{temp_label_title}\n({config_label('baseline_full')} vs. "
                 f"{config_label('no_mros_loocv')})")
    safe_savefig(fig, out_dir(region) / fname)


def fig_story5_nearfreeze_tair(region: str):
    _fig_story5_compare(region, "temp_air", f"{region}_story5_nearfreeze_tair_comparison.png")


def fig_story5_nearfreeze_twet(region: str):
    _fig_story5_compare(region, "temp_wet", f"{region}_story5_nearfreeze_twet_comparison.png")


# ---------------------------------------------------------------------------
# Figure: 2x2 confusion matrix, XGB-Full at the 0.5 threshold
# ---------------------------------------------------------------------------

def fig_confusion_matrix(region: str):
    """Confusion matrix for XGB-Full on the test split.

    Scope:
      - Pure-phase observations only (phase_full in {snow, rain}); observed
        mix is excluded, as in the other binary-skill figures.
      - The uncertainty band is not applied; this is the raw 0.5-threshold
        decision (pred_xgboost_mros_bin05). Band behaviour is covered by
        fig_mix_capture_by_wetbulb.
      - Read from benchmarking/benchmark_predictions_test.parquet.

    Each cell carries its count and, beneath it, that count as a share of the
    observed class (the row), which is what the colour encodes. No accuracy /
    POD / FAR text is printed anywhere on the figure; those are reported in the
    results table, and repeating them in a panel title duplicated the numbers
    in two places that then had to be kept consistent.
    """
    pq = benchmarking_dir(region) / "benchmark_predictions_test.parquet"
    if not pq.exists():
        print(f"  [confusion:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", f"pred_{MODEL_NAME_KEY}_bin05"])
    pure = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])]
    if pure.empty:
        print(f"  [confusion:{region}] no pure-phase rows, skipping")
        return

    y_true = pure["phase_full"].to_numpy(int)
    y_pred = pure[f"pred_{MODEL_NAME_KEY}_bin05"].to_numpy(int)

    # Rows/cols ordered [snow, rain], the snow-first convention used
    # throughout the manuscript.
    codes = [SNOW_CODE, RAIN_CODE]
    labels = ["Snow", "Rain"]
    cm = np.array([[int(np.sum((y_true == t) & (y_pred == p))) for p in codes]
                   for t in codes])
    row_tot = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_tot, out=np.zeros_like(cm, float), where=row_tot > 0) * 100

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            # White text on dark (high-share) cells, dark text otherwise.
            txt_color = "white" if cm_pct[i, j] > 55 else "#1a1a1a"
            ax.text(j, i - 0.07, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=30, color=txt_color, fontweight="bold")
            ax.text(j, i + 0.16, f"{cm_pct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=16, color=txt_color)

    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel(f"Predicted phase ({MODEL_DISPLAY_NAME}, 0.5 threshold)", labelpad=12)
    ax.set_ylabel("Observed phase", labelpad=12)
    ax.set_title(f"{region_label(region)}: {MODEL_DISPLAY_NAME} Confusion Matrix", pad=14)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05,
                 label="Share of observed class (%)")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    safe_savefig(fig, out_dir(region) / f"{region}_confusion_matrix_appendix.png")


# ---------------------------------------------------------------------------
# Figure: Critical Success Index (CSI) by air temperature
# ---------------------------------------------------------------------------

def _csi_from_counts(hits, misses, false_alarms):
    """CSI = hits / (hits + misses + false alarms). NaN when the denominator
    is zero (no event observed and none forecast in the bin); CSI is undefined
    there rather than 0."""
    denom = hits + misses + false_alarms
    return np.divide(hits, denom, out=np.full(np.shape(hits), np.nan, float),
                     where=denom > 0)


def fig_csi_by_tair(region: str):
    """Critical success index for snow and for rain against air temperature,
    for every benchmark plus XGB-Full.

    CSI (threat score) is hits / (hits + misses + false alarms) for the event
    class, so unlike accuracy it is not inflated by class imbalance within a
    bin. Conventions match the other T_air figures: TAIR_BIN_EDGES bins, pure
    snow/rain observations only, bins with fewer than MIN_BIN_N pure
    observations dropped. XGB-Full uses its 0.5-threshold prediction.
    """
    pq = benchmarking_dir(region) / "benchmark_predictions_test.parquet"
    if not pq.exists():
        print(f"  [csi:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq)
    pure = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])].reset_index(drop=True)
    if pure.empty:
        print(f"  [csi:{region}] no pure-phase rows, skipping")
        return

    y = pure["phase_full"].to_numpy(int)
    tair = pure["temp_air"].to_numpy(float)

    methods = [c.replace("pred_", "") for c in pure.columns
               if c.startswith("pred_") and not c.startswith(f"pred_{MODEL_NAME_KEY}")]
    methods = [m for m in methods if keep_method(m)]
    preds = {m: pure[f"pred_{m}"].to_numpy(int) for m in methods}
    preds[MODEL_NAME_KEY] = pure[f"pred_{MODEL_NAME_KEY}_bin05"].to_numpy(int)
    all_methods = methods + [MODEL_NAME_KEY]

    edges = TAIR_BIN_EDGES
    n_bins = len(edges) - 1
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(tair, edges) - 1
    bin_valid = np.array([(bin_idx == b).sum() >= MIN_BIN_N for b in range(n_bins)])
    if not bin_valid.any():
        print(f"  [csi:{region}] no T_air bin reaches n >= {MIN_BIN_N}, skipping")
        return

    csi = {SNOW_CODE: {}, RAIN_CODE: {}}
    for event in (SNOW_CODE, RAIN_CODE):
        for m in all_methods:
            vals = np.full(n_bins, np.nan)
            for b in range(n_bins):
                if not bin_valid[b]:
                    continue
                sel = bin_idx == b
                yt, yp = y[sel], preds[m][sel]
                hits = int(np.sum((yt == event) & (yp == event)))
                misses = int(np.sum((yt == event) & (yp != event)))
                false_alarms = int(np.sum((yt != event) & (yp == event)))
                vals[b] = _csi_from_counts(hits, misses, false_alarms)
            csi[event][m] = vals

    # Benchmarks first, model last, so the heavy black model line sits on top.
    order = [n for n in METHOD_STYLE if n in all_methods and n != MODEL_NAME_KEY]
    order += [MODEL_NAME_KEY]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, event, event_name in [(axes[0], SNOW_CODE, "Snow"),
                                  (axes[1], RAIN_CODE, "Rain")]:
        for m in order:
            vals = csi[event][m]
            v = bin_valid & ~np.isnan(vals)
            if not v.any():
                continue
            style = METHOD_STYLE.get(m, {})
            ax.plot(bin_mids[v], vals[v], marker="o",
                    ms=6 if m == MODEL_NAME_KEY else 4,
                    zorder=5 if m == MODEL_NAME_KEY else 2,
                    label=method_label(m), **style)
        ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
        ax.axvspan(0, 4, alpha=0.06, color="orange")
        ax.set(xlabel=f"Air temperature (°C; {TAIR_BIN_WIDTH:g} °C bins, n ≥ {MIN_BIN_N})",
               ylim=(0, 1.02), title=f"{event_name} as event class")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Critical success index")
    axes[0].legend(fontsize=11, ncol=2, loc="lower left", framealpha=0.9)

    fig.suptitle(f"{region_label(region)}: Critical Success Index by Air Temperature")
    safe_savefig(fig, out_dir(region) / f"{region}_csi_by_tair_appendix.png")


# ---------------------------------------------------------------------------
# MRoS phase diagnostics — shared panel-drawing helpers.
#
# Each _draw_* function renders one panel into an ax the caller already
# created and returns whatever geometry the combined figure needs, or
# None/False if it skipped (missing dependency, missing saved data, or a
# failed network fetch). The standalone fig_* wrappers create a single-ax
# figure, call the matching _draw_*, and save.
# ---------------------------------------------------------------------------

def _draw_phase_extent_panel(ax, region: str):
    """Regional context: state outline, a few cities, and the AOI overlay.

    The state outline is fetched once and cached (see _state_boundary); the
    city labels are built in. Basemap tiles are drawn if they can be fetched
    and quietly omitted if not — the panel is legible either way, so a blocked
    tile server no longer costs the whole figure.

    Returns {"aoi_proj": ...} on success, None if it skipped.
    """
    if not HAVE_GIS:
        print(f"  [phase_extent:{region}] missing geopandas/contextily/rasterio, skipping")
        return None

    rcfg = REGION_CONFIG[region]
    try:
        state_gdf = _state_boundary(region)
    except Exception as e:
        print(f"  [phase_extent:{region}] state outline unavailable ({e}). It is fetched "
              f"once and cached at {REFERENCE_CACHE_DIR / f'state_{region}.geojson'}; run "
              f"this once on a machine that can reach www2.census.gov or "
              f"raw.githubusercontent.com to populate the cache. Skipping this panel.")
        return None

    aoi_poly = Polygon(rcfg["aoi_lonlat"])
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly], crs="EPSG:4326")
    dem_crs_str = rcfg["utm_crs"]

    state_gdf_proj = state_gdf.to_crs(dem_crs_str)
    aoi_proj = aoi_gdf.to_crs(dem_crs_str)

    minx, miny, maxx, maxy = state_gdf_proj.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    have_tiles = False
    try:
        cx.add_basemap(ax, crs=dem_crs_str,
                       source=cx.providers.CartoDB.PositronNoLabels, zorder=1)
        have_tiles = True
    except Exception as e:
        print(f"  [phase_extent:{region}] basemap tiles unavailable ({type(e).__name__}); "
              f"drawing the state outline on a plain background")

    if not have_tiles:
        # Without tiles an unfilled outline reads as an empty box, so give the
        # state a light fill to stand it off from the page.
        state_gdf_proj.plot(ax=ax, facecolor="#f2f2f0", edgecolor="none", zorder=1)
    state_gdf_proj.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=3)
    aoi_proj.plot(ax=ax, facecolor="red", edgecolor="red", alpha=0.35, linewidth=1.6, zorder=3)

    cities = CONTEXT_CITIES.get(region, [])
    if cities:
        city_gdf = gpd.GeoDataFrame(
            {"name": [c[0] for c in cities]},
            geometry=gpd.points_from_xy([c[1] for c in cities], [c[2] for c in cities]),
            crs="EPSG:4326",
        ).to_crs(dem_crs_str)
        ax.scatter(city_gdf.geometry.x, city_gdf.geometry.y, s=22, color="dimgray",
                   edgecolor="white", linewidth=0.6, zorder=4)
        for name, pt in zip(city_gdf["name"], city_gdf.geometry):
            ax.annotate(name, xy=(pt.x, pt.y), fontsize=9, color="black",
                        xytext=(5, 4), textcoords="offset points", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.8)

    if have_tiles:
        ax.text(0.01, 0.01, "© OpenStreetMap contributors, © CARTO",
                transform=ax.transAxes, fontsize=7, color="dimgray",
                ha="left", va="bottom", zorder=5)

    ax.annotate("N", xy=(0.95, 0.90), xytext=(0.95, 0.78), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=3, headwidth=9, headlength=9),
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=5)

    return {"aoi_proj": aoi_proj}


def _hexbin_modal_phase(ax, x, y, phases, gridsize=42, extent=None,
                        min_alpha=0.35, zorder=3):
    """Draw a hexbin coloured by the most-reported phase in each cell.

    Plotting every MRoS report as its own marker made the map unreadable: in
    the dense parts of both domains the markers overlap several deep, so the
    visible colour was decided by draw order rather than by what was reported.
    Binning fixes that. Each hexagon takes the colour of its modal phase, and
    its opacity scales with log report count, so a hex holding one report
    reads faintly and a hex holding hundreds reads at full strength. Hexagons
    where no phase holds an outright majority (modal share < 50%) are outlined
    in dark grey, so genuinely mixed neighbourhoods are not mistaken for
    uniform ones.

    Counts per phase are obtained by calling hexbin once per phase over the
    *same* point set with C set to that phase's indicator and
    reduce_C_function=np.sum. Passing the full point set each time guarantees
    every call produces the identical hexagon layout, so the returned arrays
    line up cell for cell.

    Returns the drawn collection.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    phases = np.asarray(phases)

    kw = dict(gridsize=gridsize, mincnt=1)
    if extent is not None:
        kw["extent"] = extent

    counts = {}
    for phase in DIAG_PHASE_ORDER:
        indicator = (phases == phase).astype(float)
        hb = ax.hexbin(x, y, C=indicator, reduce_C_function=np.sum, **kw)
        counts[phase] = np.asarray(hb.get_array(), float)
        hb.remove()

    hb = ax.hexbin(x, y, **kw)
    totals = np.asarray(hb.get_array(), float)

    stacked = np.vstack([counts[p] for p in DIAG_PHASE_ORDER])
    modal_idx = np.argmax(stacked, axis=0)
    modal_count = stacked[modal_idx, np.arange(stacked.shape[1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        modal_share = np.where(totals > 0, modal_count / totals, np.nan)

    # log-scaled opacity, so a handful of reports still shows but does not
    # compete visually with a well-sampled cell.
    log_tot = np.log10(np.clip(totals, 1, None))
    span = log_tot.max() if log_tot.max() > 0 else 1.0
    alphas = min_alpha + (1.0 - min_alpha) * (log_tot / span)

    face = np.array([
        mcolors.to_rgba(DIAG_PHASE_COLORS[DIAG_PHASE_ORDER[i]], a)
        for i, a in zip(modal_idx, alphas)
    ])
    no_majority = modal_share < 0.5
    edge = np.array([
        (0.15, 0.15, 0.15, 0.85) if flag else (0, 0, 0, 0) for flag in no_majority
    ])

    hb.set_array(None)
    hb.set_facecolors(face)
    hb.set_edgecolors(edge)
    hb.set_linewidths(np.where(no_majority, 0.7, 0.0))
    hb.set_zorder(zorder)
    return hb


def _draw_phase_coverage_panel(fig, ax, region: str, gridsize: int = 42):
    """Phase reports and station coverage over the grayscale DEM.

    MRoS reports are aggregated into hexagonal bins coloured by modal phase
    (see _hexbin_modal_phase) rather than drawn as individual markers, which
    overplotted badly. Needs `fig` for the elevation colorbar. Returns
    {"bounds": ...} on success, None if it skipped.
    """
    if not HAVE_GIS:
        print(f"  [phase_coverage:{region}] missing geopandas/contextily/rasterio, skipping")
        return None

    try:
        st_hr, mros_hr = load_phase_diagnostics_inputs(region)
    except FileNotFoundError as e:
        print(f"  [phase_coverage:{region}] {e}, skipping")
        return None

    rcfg = REGION_CONFIG[region]
    dpath = dem_path_for(region)
    if not dpath.exists():
        print(f"  [phase_coverage:{region}] missing DEM: {dpath}, skipping")
        return None

    aoi_poly = Polygon(rcfg["aoi_lonlat"])
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly], crs="EPSG:4326")

    with rio.open(dpath) as src:
        dem_crs_str = src.crs.to_string()
        bounds = src.bounds
        xticks = np.linspace(bounds.left, bounds.right, 5)
        yticks = np.linspace(bounds.bottom, bounds.top, 5)

        rio_show(src, ax=ax, cmap="Greys_r", alpha=0.75, zorder=1)
        dem_im = ax.images[-1]  # the AxesImage rio_show just added

        mros_gdf = gpd.GeoDataFrame(
            mros_hr, geometry=gpd.points_from_xy(mros_hr["lon"], mros_hr["lat"]),
            crs="EPSG:4326",
        ).to_crs(dem_crs_str)
        mros_plot = mros_gdf[mros_gdf["phase"].isin(DIAG_PHASE_ORDER)]

        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        _hexbin_modal_phase(ax, mros_plot.geometry.x.to_numpy(),
                            mros_plot.geometry.y.to_numpy(),
                            mros_plot["phase"].to_numpy(),
                            gridsize=gridsize, extent=extent, zorder=3)

        st_gdf = gpd.GeoDataFrame(
            st_hr, geometry=gpd.points_from_xy(st_hr["lon"], st_hr["lat"]), crs="EPSG:4326"
        ).to_crs(dem_crs_str)
        aoi_dem_crs = aoi_gdf.to_crs(dem_crs_str)
        st_gdf_aoi = gpd.sjoin(st_gdf, aoi_dem_crs, how="inner",
                               predicate="within").drop(columns="index_right")
        # One marker per station, not one per station-hour.
        st_unique = st_gdf_aoi.drop_duplicates(subset="id") if "id" in st_gdf_aoi.columns \
            else st_gdf_aoi
        ax.scatter(st_unique.geometry.x, st_unique.geometry.y, marker="^", s=55,
                   facecolor="none", edgecolor="#111111", linewidths=1.3, zorder=4)

        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_aspect("equal")
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x/1000:.0f}" for x in xticks], rotation=30, ha="right")
        ax.set_xlabel("Easting (km)")
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y/1000:.0f}" for y in yticks])
        ax.set_ylabel("Northing (km)")

        patches = [mpatches.Patch(facecolor=DIAG_PHASE_COLORS[p], edgecolor="none",
                                  label=p) for p in DIAG_PHASE_ORDER]
        patches.append(mpatches.Patch(facecolor="#cccccc", edgecolor="#262626",
                                      linewidth=0.9, label="no majority phase"))
        station_marker = plt.Line2D([0], [0], marker="^", color="none",
                                    markerfacecolor="none", markeredgecolor="#111111",
                                    markeredgewidth=1.3, markersize=10, label="Stations")
        leg = ax.legend(handles=patches + [station_marker], loc="lower left",
                        framealpha=0.9, fontsize=11,
                        title="Modal phase per hexagon\n(opacity ∝ log report count)")
        leg.get_title().set_fontsize(11)
        leg.set_zorder(6)

        # Elevation colorbar for the grayscale DEM background.
        cbar = fig.colorbar(dem_im, ax=ax, fraction=0.045, pad=0.03, shrink=0.8)
        cbar.set_label("Elevation (m)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        ax.add_artist(ScaleBar(1, units="m", location="lower right",
                               box_alpha=0.75, font_properties={"size": 11}))

        ax.annotate("N", xy=(0.93, 0.93), xytext=(0.93, 0.80), xycoords="axes fraction",
                    arrowprops=dict(facecolor="black", width=3, headwidth=9, headlength=9),
                    ha="center", va="center", fontsize=13, fontweight="bold", zorder=5)

        return {"bounds": bounds}


def _draw_phase_elevation_kde_panel(ax, region: str) -> bool:
    """Distribution of report elevation by phase.

    The kernel density estimate for each phase is scaled by that phase's
    report count rather than normalised to unit area. A within-phase
    normalised KDE shows P(elevation | phase) and cannot indicate which phase
    is most common at a given elevation, because every curve integrates to one
    regardless of how many reports it represents. Scaling by the count makes
    the area under each curve proportional to that phase's number of reports,
    so the curves are comparable in volume and their crossings mark where the
    dominant phase changes.

    Returns True on success, False if it skipped.
    """
    from scipy.stats import gaussian_kde

    try:
        _, mros_hr = load_phase_diagnostics_inputs(region)
    except FileNotFoundError as e:
        print(f"  [phase_elev_kde:{region}] {e}, skipping")
        return False

    elev_all = mros_hr["elev"].dropna()
    elev_range = np.linspace(elev_all.min(), elev_all.max(), 300)
    for phase in DIAG_PHASE_ORDER:
        sub = mros_hr.loc[mros_hr["phase"] == phase, "elev"].dropna()
        if len(sub) < 2:
            continue
        kde = gaussian_kde(sub, bw_method="scott")
        density = kde(elev_range) * len(sub)
        ax.plot(elev_range, density, color=DIAG_PHASE_COLORS[phase],
                linewidth=3.0, label=f"{phase} (n={len(sub):,})")
        ax.fill_between(elev_range, density, alpha=0.18, color=DIAG_PHASE_COLORS[phase])

    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Reports per metre of elevation")
    ax.legend(framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    return True


def _draw_phase_by_month_panel(ax, region: str) -> bool:
    """Monthly observation counts by phase. Returns True on success, False if
    it skipped."""
    try:
        _, mros_hr = load_phase_diagnostics_inputs(region)
    except FileNotFoundError as e:
        print(f"  [phase_month:{region}] {e}, skipping")
        return False

    tmp = mros_hr.copy()
    tmp["month"] = pd.to_datetime(tmp["hour_utc"]).dt.to_period("M").dt.to_timestamp()
    monthly = (tmp.groupby(["month", "phase"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=DIAG_PHASE_ORDER, fill_value=0)
                  .sort_index())
    months = monthly.index
    x = np.arange(len(months))
    width = 0.28
    for i, phase in enumerate(DIAG_PHASE_ORDER):
        ax.bar(x + i * width, monthly[phase], width=width,
               color=DIAG_PHASE_COLORS[phase], label=phase, alpha=0.9)

    step = max(1, len(months) // 12)
    tick_idx = np.arange(0, len(months), step)
    ax.set_xticks(x[tick_idx] + width)
    ax.set_xticklabels([months[i].strftime("%b %Y") for i in tick_idx],
                       rotation=45, ha="right")
    ax.set_ylabel("Observation count")
    ax.legend(framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    return True


# ---------------------------------------------------------------------------
# Standalone MRoS phase diagnostics figures
# ---------------------------------------------------------------------------

def fig_phase_study_extent(region: str):
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    if _draw_phase_extent_panel(ax, region) is None:
        plt.close(fig)
        return
    ax.set_title(f"Study extent within {STATE_FULL_NAME.get(region, region)}")
    safe_savefig(fig, out_dir(region) / f"{region}_phase_study_extent.png")


def fig_phase_locations_coverage(region: str):
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    if _draw_phase_coverage_panel(fig, ax, region) is None:
        plt.close(fig)
        return
    # Short domain name here: an equal-aspect map plus its colorbar leaves the
    # title little room, and the long form was clipped at the right edge.
    ax.set_title(f"{region_label(region)}: Phase Reports & Station Coverage")
    safe_savefig(fig, out_dir(region) / f"{region}_phase_locations_coverage.png")


def fig_phase_elevation_kde(region: str):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if not _draw_phase_elevation_kde_panel(ax, region):
        plt.close(fig)
        return
    ax.set_title(f"{region_label(region)}: Reports by Elevation and Phase")
    safe_savefig(fig, out_dir(region) / f"{region}_phase_vs_elevation_kde.png")


def fig_phase_by_month(region: str):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if not _draw_phase_by_month_panel(ax, region):
        plt.close(fig)
        return
    ax.set_title(f"{region_label(region)}: Reports by Month and Phase")
    safe_savefig(fig, out_dir(region) / f"{region}_phase_by_month.png")


# ---------------------------------------------------------------------------
# Combined MRoS phase diagnostics figures
# ---------------------------------------------------------------------------

def fig_phase_diagnostics_maps(region: str):
    """Regional context and phase/station coverage.

    Panel A needs reference boundaries and basemap tiles over the network (or
    a populated REFERENCE_CACHE_DIR). Panel B needs only the local DEM and the
    saved hourly tables. If panel A is unavailable the figure is still written
    with panel B alone, rather than being skipped entirely — panel B is the
    one the manuscript's phase-coverage discussion depends on.
    """
    fig = plt.figure(figsize=(16, 8))
    axA = fig.add_subplot(1, 2, 1)
    a_info = _draw_phase_extent_panel(axA, region)

    if a_info is None:
        # Redraw as a single-panel figure rather than leaving an empty axes.
        plt.close(fig)
        fig, axB = plt.subplots(figsize=(9.5, 9.5))
        b_info = _draw_phase_coverage_panel(fig, axB, region)
        if b_info is None:
            print(f"  [phase_maps:{region}] neither panel available, not saving")
            plt.close(fig)
            return
        axB.set_title(f"{region_label(region)}: Phase Reports & Station Coverage")
        print(f"  [phase_maps:{region}] regional-context panel unavailable; "
              f"saved the coverage panel alone")
        safe_savefig(fig, out_dir(region) / f"{region}_phase_maps.png")
        return

    axA.set_title(f"A) Study extent within {STATE_FULL_NAME.get(region, region)}",
                  fontsize=15)
    axB = fig.add_subplot(1, 2, 2)
    b_info = _draw_phase_coverage_panel(fig, axB, region)
    axB.set_title("B) Phase reports & station coverage", fontsize=15)
    fig.suptitle(f"MRoS Phase Reports & Station Coverage — {region_full_label(region)}",
                 fontweight="bold", y=1.02)

    if b_info is not None:
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = a_info["aoi_proj"].total_bounds
        bounds = b_info["bounds"]
        for (yA, yB) in [(aoi_maxy, bounds.top), (aoi_miny, bounds.bottom)]:
            fig.add_artist(ConnectionPatch(
                xyA=(aoi_maxx, yA), coordsA=axA.transData,
                xyB=(bounds.left, yB), coordsB=axB.transData,
                color="red", linewidth=1.2, linestyle="--", alpha=0.7, zorder=10,
            ))
    safe_savefig(fig, out_dir(region) / f"{region}_phase_maps.png")


def fig_phase_diagnostics_distributions(region: str):
    """Elevation distribution and monthly counts, side by side.

    The elevation panel is count-scaled rather than density-normalised; see
    _draw_phase_elevation_kde_panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axC, axD = axes[0], axes[1]

    ok_c = _draw_phase_elevation_kde_panel(axC, region)
    axC.set_title("A) Reports by elevation and phase", fontsize=15)

    ok_d = _draw_phase_by_month_panel(axD, region)
    axD.set_title("B) Reports by month and phase", fontsize=15)

    if not (ok_c or ok_d):
        print(f"  [phase_distributions:{region}] no saved MRoS table, not saving")
        plt.close(fig)
        return

    fig.suptitle(f"MRoS Report Distribution by Elevation & Month — "
                 f"{region_full_label(region)}", fontweight="bold", y=1.02)
    safe_savefig(fig, out_dir(region) / f"{region}_phase_distributions.png")


def fig_phase_diagnostics_combined(region: str):
    """Backwards-compatible wrapper: draws both split figures."""
    fig_phase_diagnostics_maps(region)
    fig_phase_diagnostics_distributions(region)


# ---------------------------------------------------------------------------
# Station data checks
# ---------------------------------------------------------------------------

def _station_summary(st_hr):
    """One row per station: position, elevation, record length, mean conditions."""
    return st_hr.groupby("id", as_index=False).agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        elev=("elev", "first"),
        n_hours=("hour_utc", "count"),
        ta_mean=("temp_air", "mean"),
        tw_mean=("temp_wet", "mean"),
        rh_mean=("rh", "mean"),
    )


def _draw_station_elevation_panel(ax, st_hr, station_summary, id_to_color):
    """Air temperature against elevation, one density contour per station."""
    from scipy.stats import gaussian_kde

    for station_id in station_summary["id"]:
        sub = st_hr[st_hr["id"] == station_id][["temp_air", "elev"]].dropna()
        if len(sub) < 20:
            continue
        try:
            kde = gaussian_kde(np.vstack([sub["temp_air"], sub["elev"]]))
            ta_grid = np.linspace(sub["temp_air"].min(), sub["temp_air"].max(), 60)
            el_grid = np.linspace(sub["elev"].min(), sub["elev"].max(), 60)
            ta_mesh, el_mesh = np.meshgrid(ta_grid, el_grid)
            density = kde(np.vstack([ta_mesh.ravel(), el_mesh.ravel()])).reshape(ta_mesh.shape)
            ax.contour(ta_mesh, el_mesh, density, levels=3,
                       colors=[id_to_color[station_id]], alpha=0.6, linewidths=1.1)
        except Exception:
            # The KDE is singular when a station has too little spread.
            pass

    ax.scatter(st_hr["temp_air"], st_hr["elev"],
               c=[id_to_color.get(sid, "gray") for sid in st_hr["id"]],
               s=1, alpha=0.08, rasterized=True)
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.6, label="0 °C")
    ax.axvline(2, color="gray", lw=1.2, ls=":", alpha=0.6, label="2 °C")
    ax.set(xlabel="Air temperature (°C)", ylabel="Elevation (m)",
           title="Air Temperature vs Elevation")
    ax.legend()
    ax.grid(alpha=0.25)


def _draw_station_distribution_panel(ax, st_hr):
    """Temperature distributions on the bottom axis, humidity on the top."""
    colors = {"temp_air": "#D55E00", "temp_wet": "#0072B2", "rh": "#009E73"}

    for column in ("temp_air", "temp_wet"):
        ax.hist(st_hr[column].dropna(), bins=80, color=colors[column],
                alpha=0.55, density=True)
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set(xlabel="Temperature (°C)", ylabel="Density", title="Observed Distributions")

    ax_rh = ax.twiny()
    ax_rh.hist(st_hr["rh"].dropna(), bins=50, color=colors["rh"], alpha=0.45,
               density=True, histtype="step", linewidth=2.2)
    ax_rh.set_xlabel("Relative humidity (%)", color=colors["rh"])
    ax_rh.tick_params(axis="x", colors=colors["rh"])

    ax.legend(handles=[
        mpatches.Patch(color=colors["temp_air"], alpha=0.7, label="Air temperature"),
        mpatches.Patch(color=colors["temp_wet"], alpha=0.7, label="Wet-bulb temperature"),
        mpatches.Patch(color=colors["rh"], alpha=0.5, label="Relative humidity"),
    ], loc="upper left")
    ax.grid(alpha=0.25)


def _draw_station_map_panel(ax, region, station_summary, id_to_color):
    """Station positions on a relief basemap, sized by length of record."""
    from pyproj import Transformer
    from shapely.geometry import box as shapely_box

    with rio.open(dem_path_for(region)) as src:
        dem_bounds, dem_crs = src.bounds, src.crs

    dem_gdf = gpd.GeoDataFrame({"geometry": [shapely_box(*dem_bounds)]},
                               crs=dem_crs).to_crs("EPSG:3857")
    stations = gpd.GeoDataFrame(
        station_summary,
        geometry=gpd.points_from_xy(station_summary["lon"], station_summary["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    xmin, ymin, xmax, ymax = dem_gdf.total_bounds
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldShadedRelief, zoom=9, alpha=0.85)
    except Exception as e:
        print(f"  [station_map:{region}] relief basemap unavailable ({type(e).__name__}); "
              f"drawing without it")
    dem_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.0, zorder=3)

    sizes = np.interp(station_summary["n_hours"],
                      [station_summary["n_hours"].min(), station_summary["n_hours"].max()],
                      [45, 260])
    ax.scatter(stations.geometry.x, stations.geometry.y,
               c=[id_to_color[sid] for sid in stations["id"]], s=sizes,
               edgecolors="white", linewidths=0.7, zorder=4, alpha=0.9)

    for _, row in stations.iterrows():
        ax.annotate(row["id"], xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4), textcoords="offset points", fontsize=8,
                    color="white", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    lons, _ = to_lonlat.transform(xticks, [0] * len(xticks))
    _, lats = to_lonlat.transform([0] * len(yticks), yticks)
    ax.set_xticks(xticks, [f"{lon:.2f}°" for lon in lons])
    ax.set_yticks(yticks, [f"{lat:.2f}°" for lat in lats])
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Station Locations")

    for hours, size in [(500, 45), (5000, 140),
                        (station_summary["n_hours"].max(), 260)]:
        ax.scatter([], [], c="gray", s=size, alpha=0.8, label=f"{int(hours):,} hrs")
    ax.legend(title="Hours on record", fontsize=10, loc="lower left")


def fig_station_data_checks(region: str):
    """Three-panel check on the compiled station record. Reads only the hourly
    station table written by the compilation stage."""
    st_hr, _ = load_phase_diagnostics_inputs(region)
    station_summary = _station_summary(st_hr)

    cmap = plt.get_cmap("tab20", len(station_summary))
    id_to_color = {sid: cmap(i) for i, sid in enumerate(station_summary["id"])}

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    fig.suptitle(f"{region_full_label(region)} — station record",
                 fontweight="bold", y=1.02)

    _draw_station_elevation_panel(axes[0], st_hr, station_summary, id_to_color)
    _draw_station_distribution_panel(axes[1], st_hr)

    if HAVE_GIS:
        try:
            _draw_station_map_panel(axes[2], region, station_summary, id_to_color)
        except Exception as exc:
            axes[2].set_axis_off()
            axes[2].set_title(f"Station map unavailable\n({exc})", fontsize=11)
    else:
        axes[2].set_axis_off()
        axes[2].set_title("Station map needs geopandas / contextily", fontsize=11)

    safe_savefig(fig, out_dir(region) / f"{region}_station_data_checks.png")

    print(f"  stations {len(station_summary)}, "
          f"elevation {station_summary['elev'].min():.0f}–"
          f"{station_summary['elev'].max():.0f} m")
    for column in ("temp_air", "temp_wet", "rh"):
        print(f"  {column:9s} complete: {(1 - st_hr[column].isna().mean()) * 100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_FUNCS = {
    # Compilation-stage checks, drawn from the hourly observation tables.
    "station_checks": fig_station_data_checks,
    "phase_combined": fig_phase_diagnostics_combined,
    "phase_maps": fig_phase_diagnostics_maps,
    "phase_distributions": fig_phase_diagnostics_distributions,

    # Model and evaluation figures.
    "calibration": fig_calibration,
    "forest": fig_benchmark_delta_forest,
    "tair": fig_benchmark_accuracy_by_tair_ci,
    "f1_twet": fig_twet_performance,
    "shap": fig_shap,
    "ablation_ci": fig_ablation_deltas_ci,
    "band": fig_mix_capture_by_wetbulb,
    "ablation_comparison": fig_ablation_comparison,
    "tair_relimp_ci": fig_benchmark_relative_improvement_ci,
    "tair_accuracy": fig_benchmark_accuracy_by_tair,
    "f1_tair": fig_f1_by_tair,
    "selective_twet": fig_selective_by_wetbulb,
    "selective_tair": fig_selective_by_tair,
    "overall_bars": fig_overall_accuracy_bars,
    "story5_tair": fig_story5_nearfreeze_tair,
    "story5_twet": fig_story5_nearfreeze_twet,
    "confusion": fig_confusion_matrix,
    "csi": fig_csi_by_tair,
    "phase_extent": fig_phase_study_extent,
    "phase_coverage": fig_phase_locations_coverage,
    "phase_elev_kde": fig_phase_elevation_kde,
    "phase_month": fig_phase_by_month,
}


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the manuscript figures from saved artifacts."
    )
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS),
                        help="Regions to draw (default: all).")
    parser.add_argument("--figures", nargs="+", choices=list(FIGURE_FUNCS),
                        default=list(FIGURE_FUNCS),
                        help="Figure keys to draw (default: all).")
    args = parser.parse_args()

    for region in args.regions:
        print(f"=== {region} ({region_label(region)}) ===")
        for fig_key in args.figures:
            try:
                FIGURE_FUNCS[fig_key](region)
            except Exception as exc:
                # A missing artifact skips one figure, not the rest.
                print(f"  [{fig_key}] skipped: {type(exc).__name__}: {exc}")
        print(f"  -> {out_dir(region)}")


if __name__ == "__main__":
    main()
