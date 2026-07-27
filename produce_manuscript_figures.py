"""
produce_manuscript_figures.py
==============================

Regenerates all manuscript figures for the MRoS precipitation-phase
classification project by reading ONLY already-saved output artifacts on
disk. It does NOT retrain any XGBoost model, does NOT rerun kriging /
interpolation, and does NOT re-execute any of the long notebooks.

Canonical run versions this script depends on (read-only):
  - outputs/ML_pipeline/model_artifacts/{CA,CO}/ablations_v2/<config>/...
        (NOT ablations_v1 — v2 has beta calibration)
  - outputs/ML_pipeline/model_artifacts/{CA,CO}/results_binaryXGB_withKriging_v2/...
        (NOT v1 / withIDW — v2 is the beta-calibrated kriging-predictor run)
  - outputs/ML_pipeline/model_artifacts/{CA,CO}/benchmarking_v1/... and
    .../benchmarking_v1/bootstrap/... (cluster-bootstrap CIs; preferred over
    any point-estimate-only equivalent wherever both exist)

Outputs are written to outputs/manuscript_figures/{CA,CO}/ and never
overwrite anything under outputs/ML_pipeline/model_artifacts/.

Usage:
    python produce_manuscript_figures.py                 # all regions, all figures
    python produce_manuscript_figures.py --regions CA     # one region only
    python produce_manuscript_figures.py --figures shap,band  # figure subset

Figure subset keys: calibration, forest, tair, twet, shap, ablation_ci, band,
                    ablation_comparison, tair_relimp_ci, tair_accuracy, f1_tair,
                    overall_bars, story5_tair, story5_twet
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = REPO_ROOT / "outputs" / "ML_pipeline" / "model_artifacts"
OUT_ROOT = REPO_ROOT / "outputs" / "manuscript_figures"

REGIONS = ["CA", "CO"]

SNOW_CODE, RAIN_CODE, MIX_CODE = 0, 1, 2
PHASE_COLORS = {"snow": "#1f77b4", "rain": "#2ca02c", "mix": "#e377c2"}
SPLIT_COLORS = {"Validation": "#8338ec", "Test": "#ff006e"}

TWET_BIN_EDGES = np.arange(-6, 7, 1)  # 1 degC bins, matches story3 in ablation_v2 script

# T_air binning convention, matches ML_XGBoost_binary_uncertainty_benchmarking.py
# constants TAIR_BIN_WIDTH/TAIR_BIN_MIN/TAIR_BIN_MAX/MIN_BIN_N exactly, so all
# T_air figures in this script (accuracy-by-tair, F1-by-tair, bootstrap
# ribbons) use the same bins and the same minimum-n dropout rule.
TAIR_BIN_WIDTH = 1.0
TAIR_BIN_MIN = -8.0
TAIR_BIN_MAX = 8.0
MIN_BIN_N = 20      # bins with fewer than this many pure obs are dropped
MIN_BIN_PHASE_N = 10
TAIR_BIN_EDGES = np.arange(TAIR_BIN_MIN, TAIR_BIN_MAX + TAIR_BIN_WIDTH, TAIR_BIN_WIDTH)
NEARFREEZE_TWET_C = 2.0

# NOTE ON BIN DROPOUT (applies to every T_air-binned figure below): bins with
# fewer than MIN_BIN_N (=20) pure snow/rain observations are dropped rather
# than plotted with a misleadingly noisy estimate. This is why some series
# appear to be "cut off" at the cold/warm tails — it reflects genuine data
# sparsity at temperature extremes, not a leftover filtered time window.

MODEL_DISPLAY_NAME = "MRoS-XGB"
MODEL_NAME_KEY = "xgboost_mros"  # internal key used in saved artifacts/columns

# Human-readable labels for benchmark methods (mirrors METHOD_LABEL in
# ML_XGBoost_binary_uncertainty_benchmarking.py, with the model relabeled).
METHOD_LABEL = {
    "ta_1.0": "$T_{a}$ 1.0 °C", "ta_1.5": "$T_{a}$ 1.5 °C",
    "td_0.0": "$T_{d}$ 0.0 °C", "td_0.5": "$T_{d}$ 0.5 °C",
    "tw_0.0": "$T_{w}$ 0.0 °C", "tw_0.5": "$T_{w}$ 0.5 °C", "tw_1.0": "$T_{w}$ 1.0 °C",
    "binlog_jennings18": "Bin. logistic (Jennings et al. 2018)",
    "binlog_fitted": "Bin. logistic (fitted, this domain)",
    MODEL_NAME_KEY: MODEL_DISPLAY_NAME,
}
METHOD_STYLE = {
    "ta_1.0": dict(color="#e08214", ls="--", lw=1.4),
    "ta_1.5": dict(color="#b35806", ls="-", lw=1.4),
    "td_0.0": dict(color="#7fbc41", ls="--", lw=1.4),
    "td_0.5": dict(color="#4d9221", ls="-", lw=1.4),
    "tw_0.0": dict(color="#92c5de", ls=":", lw=1.6),
    "tw_0.5": dict(color="#4393c3", ls="--", lw=1.6),
    "tw_1.0": dict(color="#2166ac", ls="-", lw=1.6),
    "binlog_jennings18": dict(color="#9970ab", ls="-", lw=1.6),
    "binlog_fitted": dict(color="#762a83", ls="--", lw=1.6),
    MODEL_NAME_KEY: dict(color="black", ls="-", lw=2.6),
}

# Human-readable display names for ablation configs (folder names must never
# leak into a figure title/label/legend — see work item 5). Short forms
# reused verbatim from the manuscript's Table \ref{tab:ablation} (main.tex)
# so figure and table naming stay consistent; two configs not in that table
# (min_core_noplp / single-variable drops) get analogous short labels.
CONFIG_DISPLAY_NAME = {
    "baseline_full": "Baseline (full)",
    "no_mros_loocv": "No MRoS LOOCV",
    "no_temp_air": "No air temp.",
    "no_temp_dew": "No dew point",
    "no_temp_wet": "No wet-bulb temp.",
    "no_imerg_plp": "No IMERG PLP",
    "no_elev": "No elevation",
    "thermo_only": "Thermodynamics only",
    "min_core_noplp": "Minimal core (no PLP)",
    "min_core_wplp": "Minimal core + PLP",
}

# Human-readable display names for raw model feature/column names, used
# wherever a SHAP feature-importance figure would otherwise print internal
# column names (e.g. "mros_p_snow_loocv") verbatim. Names mirror
# Table \ref{tab:predictors} in main.tex.
FEATURE_DISPLAY_NAME = {
    "temp_air": "Air temperature",
    "temp_dew": "Dewpoint temperature",
    "temp_wet": "Wet-bulb temperature",
    "elev": "Elevation",
    "imerg_plp": "IMERG PLP",
    "mros_p_snow_loocv": "MRoS $p_{snow}$ (LOOCV)",
    "mros_p_rain_loocv": "MRoS $p_{rain}$ (LOOCV)",
    "mros_p_mix_loocv": "MRoS $p_{mix}$ (LOOCV)",
}


def feature_label(name: str) -> str:
    """Map a raw model feature/column name to a human-readable display
    label. Used everywhere a raw feature name would otherwise leak into a
    title/label/legend."""
    return FEATURE_DISPLAY_NAME.get(name, name.replace("_", " "))


def config_label(name: str) -> str:
    """Map an ablation config folder name to a human-readable display label.
    Used everywhere a config name would otherwise leak into a title/label/legend."""
    return CONFIG_DISPLAY_NAME.get(name, name.replace("_", " "))


def ablations_dir(region: str) -> Path:
    return ARTIFACTS_ROOT / region / "ablations_v2"


def benchmarking_dir(region: str) -> Path:
    return ARTIFACTS_ROOT / region / "benchmarking_v1"


def bootstrap_dir(region: str) -> Path:
    return benchmarking_dir(region) / "bootstrap"


def kriging_dir(region: str) -> Path:
    return ARTIFACTS_ROOT / region / "results_binaryXGB_withKriging_v2"


def out_dir(region: str) -> Path:
    d = OUT_ROOT / region
    d.mkdir(parents=True, exist_ok=True)
    return d


def gaussian_half_band(temp_wet, base_half_band, extra_half_band, sigma):
    """Reimplementation of gaussian_half_band from
    ML_pipeline/ML_XGBoost_binary_uncertainty_ablation_v2.py (module-level,
    lines ~351-358). Widens the "mix" uncertainty band near 0 degC wet-bulb."""
    t = np.asarray(temp_wet, float)
    return np.clip(
        base_half_band + extra_half_band * np.exp(-(t ** 2) / (2.0 * sigma ** 2)),
        0.0,
        0.5,
    )


def safe_savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1: Calibration reliability diagram
# ---------------------------------------------------------------------------

def fig_calibration(region: str):
    """Reliability diagram for the baseline_full ablations_v2 config
    (beta-calibrated). Reimplements story2_calibration.png logic from
    ML_XGBoost_binary_uncertainty_ablation_v2.py, but reads only the saved
    shap_values_all.parquet (contains p_snow_cal / p_snow_raw / split /
    phase_full) rather than recomputing calibration in-process."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    pq = ablations_dir(region) / "baseline_full" / "shap_values_all.parquet"
    if not pq.exists():
        print(f"  [calibration:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", "p_snow_cal", "p_snow_raw", "split"])
    df = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])]  # pure-phase only, as in original

    band_meta_path = ablations_dir(region) / "baseline_full" / "metrics_summary.json"
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if band_meta_path.exists():
        m = json.loads(band_meta_path.read_text())
        base_hb = m.get("band_base_hb", base_hb)
        extra_hb = m.get("band_extra_hb", extra_hb)
        sigma = m.get("band_sigma", sigma)
    band0 = gaussian_half_band(np.array([0.0]), base_hb, extra_hb, sigma)[0]
    rain_thresh_0, snow_thresh_0 = 0.5 - band0, 0.5 + band0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
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
        ax_rel.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
        ax_rel.plot(mean_pred_raw, frac_pos_raw, "o--", color="#aaaaaa", label="Raw")
        ax_rel.plot(mean_pred_cal, frac_pos_cal, "o-", color=SPLIT_COLORS[split_name], label="Calibrated")
        ax_rel.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange")
        brier_raw = brier_score_loss(y_true, p_raw)
        brier_cal = brier_score_loss(y_true, p_cal)
        ax_rel.set_title(f"{region} {split_name}  (Brier raw={brier_raw:.3f}, cal={brier_cal:.3f})")
        ax_rel.set_xlabel("Mean predicted p(snow)")
        ax_rel.set_ylabel("Observed fraction snow")
        ax_rel.legend(fontsize=8)
        ax_rel.grid(alpha=0.25)

        ax_hist = axes[1, col]
        ax_hist.hist(p_cal, bins=30, color=SPLIT_COLORS[split_name], alpha=0.7)
        ax_hist.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange")
        ax_hist.set_xlabel("Calibrated p(snow)")
        ax_hist.set_ylabel("Count")

    fig.suptitle(f"{region} — calibration reliability ({config_label('baseline_full')})", y=1.02)
    safe_savefig(fig, out_dir(region) / f"{region}_calibration_reliability.png")


# ---------------------------------------------------------------------------
# Figure 2: Benchmark delta forest plots (already-bootstrapped)
# ---------------------------------------------------------------------------

def fig_benchmark_delta_forest(region: str):
    """Reimplements bootstrap_cis.plot_delta_forest, reading directly from
    bootstrap_benchmark_deltas_ci.csv (already computed via cluster
    bootstrap in benchmarking_v1/bootstrap)."""
    csv = bootstrap_dir(region) / "bootstrap_benchmark_deltas_ci.csv"
    if not csv.exists():
        print(f"  [forest:{region}] missing {csv}, skipping")
        return
    deltas_df = pd.read_csv(csv)

    for metric, fname, title in [
        ("delta_accuracy", f"{region}_benchmark_delta_forest.png",
         "Delta overall accuracy (model - benchmark)"),
        ("delta_nearfreeze_accuracy", f"{region}_benchmark_delta_forest_nearfreeze.png",
         "Delta near-freezing accuracy (model - benchmark)"),
    ]:
        d = deltas_df[deltas_df["metric"] == metric].sort_values("point")
        if d.empty:
            print(f"  [forest:{region}] metric {metric} not found in csv, skipping")
            continue
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(d))))
        ypos = np.arange(len(d))
        label_col = "label" if "label" in d.columns else "benchmark"
        ax.errorbar(
            d["point"] * 100, ypos,
            xerr=[(d["point"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["point"]) * 100],
            fmt="o", color="black", ecolor="grey", capsize=3, ms=5,
        )
        ax.axvline(0, color="#d62728", ls="--", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(d[label_col], fontsize=9)
        ax.set_xlabel("Delta Accuracy (percentage points)")
        ax.set_title(f"{region} — {title}\n(95% cluster-bootstrap CIs)")
        ax.grid(axis="x", alpha=0.25)
        safe_savefig(fig, out_dir(region) / fname)


# ---------------------------------------------------------------------------
# Figure 3: Accuracy by T_air with CI (bootstrap-derived)
# ---------------------------------------------------------------------------

def _cluster_bootstrap_tair_bins(region: str, n_boot: int = 500, block_km: float = 30.0,
                                  seed: int = 42):
    """Genuine per-T_air-bin cluster bootstrap, computed directly from
    benchmark_predictions_test.parquet (full per-observation predictions —
    a saved artifact, no retraining). Ports the cluster machinery from
    ML_pipeline/bootstrap_cis.py (make_cluster_codes / build_cluster_index /
    bootstrap_row_indices / pct_ci) rather than importing that script
    directly, so this file stays self-contained.

    THIS FIXES A BUG in the previous fallback version of
    benchmark_accuracy_by_tair_ci.png: that version read
    bootstrap_benchmark_metrics_ci.csv and filtered rows where
    metric.str.contains("accuracy"), which matches BOTH the "accuracy" and
    "nearfreeze_accuracy" metric rows for the same method. Because both
    metrics have similar magnitude for a well-performing method, the two
    rows for a single method plotted as two nearly-coincident points with
    an identical method label in the legend/y-axis — visually indistinguishable
    "duplicates" of the same method. Root cause: no per-T_air-bin data was
    ever saved to disk (bootstrap_cis.py's run_benchmark_bootstrap computes
    per-bin arrays in memory via collect_bins=True but never persists them),
    so the fallback conflated two different scalar metrics instead of giving
    true per-bin ribbons. This function computes real per-bin ribbons
    instead, and each method gets one clearly distinguished color/style
    (from METHOD_STYLE) with exactly one line and one CI band, eliminating
    the ambiguity entirely.

    Returns dict: bin_mids, bin_valid, per-method (n_boot, n_bins) accuracy arrays,
    plus point-estimate accuracy per method per bin (from the unresampled data).
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
    """TRUE per-bin bootstrap ribbons for accuracy by T_air (fixes the
    duplicate-looking bug in the previous fallback — see docstring of
    _cluster_bootstrap_tair_bins for the exact root cause and fix).
    NOTE: bins with n < MIN_BIN_N pure obs are dropped (see module-level
    bin-dropout note above the TAIR_BIN_EDGES constant)."""
    res = _cluster_bootstrap_tair_bins(region)
    if res is None:
        return
    bm, bv = res["bin_mids"], res["bin_valid"]
    best = max(res["methods"], key=lambda m: np.nanmean(res["point_acc"][m]))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    plot_set = [(MODEL_NAME_KEY, METHOD_LABEL[MODEL_NAME_KEY]),
                (best, f"{METHOD_LABEL.get(best, best)} (best benchmark)")]
    for m, label in plot_set:
        arr = res["bin_acc"][m]
        med = np.nanmedian(arr, axis=0) * 100
        lo = np.nanpercentile(arr, 2.5, axis=0) * 100
        hi = np.nanpercentile(arr, 97.5, axis=0) * 100
        v = bv & ~np.isnan(med)
        style = METHOD_STYLE.get(m, {})
        color = style.get("color", None)
        ax.plot(bm[v], med[v], marker="o", ms=5, lw=2.4, color=color, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axvspan(0, 4, alpha=0.05, color="orange")
    ax.set(xlabel="Air temperature (°C)", ylabel="Accuracy (%)", ylim=(0, 102),
           title=f"{region} — accuracy by air temperature, 95% cluster-bootstrap CIs\n"
                 f"({TAIR_BIN_WIDTH:g} °C bins, n≥{MIN_BIN_N})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_accuracy_by_tair_ci.png")


def fig_benchmark_relative_improvement_ci(region: str):
    """Bootstrapped version of plot_fig2_relative_improvement: model accuracy
    minus best/average benchmark accuracy by T_air bin, with cluster-bootstrap
    CI ribbons (paired within replicate). Shares _cluster_bootstrap_tair_bins
    with fig_benchmark_accuracy_by_tair_ci so both figures use identical bins
    and bootstrap draws."""
    res = _cluster_bootstrap_tair_bins(region)
    if res is None:
        return
    bm, bv = res["bin_mids"], res["bin_valid"]
    best = max(res["methods"], key=lambda m: np.nanmean(res["point_acc"][m]))
    arr_model = res["bin_acc"][MODEL_NAME_KEY]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for other_arr, color, label in [
        (res["bin_acc"][best], "black", f"vs. best benchmark ({METHOD_LABEL.get(best, best)})"),
        (np.nanmean(np.stack([res["bin_acc"][m] for m in res["methods"]]), axis=0),
         "#d62728", "vs. average benchmark"),
    ]:
        d = (arr_model - other_arr) * 100
        med = np.nanmedian(d, axis=0)
        lo = np.nanpercentile(d, 2.5, axis=0)
        hi = np.nanpercentile(d, 97.5, axis=0)
        v = bv & ~np.isnan(med)
        ax.plot(bm[v], med[v], "-o", color=color, lw=2, ms=5, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axvspan(0, 4, alpha=0.06, color="orange")
    ax.set(xlabel="Air temperature (°C)", ylabel="Δ Accuracy (percentage points)",
           title=f"{region} — {MODEL_DISPLAY_NAME} accuracy relative to benchmarks\n"
                 f"(positive = model better; 95% cluster-bootstrap CIs; "
                 f"shaded 0–4 °C = benchmark performance-dip zone)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_relative_improvement_ci.png")


def fig_benchmark_accuracy_by_tair(region: str):
    """Point-estimate (non-CI) accuracy/snow-bias/rain-bias by T_air, faithfully
    ported from plot_fig1_by_tair in ML_XGBoost_binary_uncertainty_benchmarking.py.
    Reads the already-saved profile_by_tair_<method>.csv files (one per
    benchmark + model) under benchmarking_v1/ — no recomputation needed.
    NOTE: bins with n < MIN_BIN_N pure obs were already dropped when these
    CSVs were produced (see module-level bin-dropout note)."""
    bdir = benchmarking_dir(region)
    profiles = {}
    for f in bdir.glob("profile_by_tair_*.csv"):
        name = f.stem.replace("profile_by_tair_", "")
        profiles[name] = pd.read_csv(f)
    if not profiles:
        print(f"  [benchmark_accuracy_by_tair:{region}] no profile_by_tair_*.csv found under {bdir}, skipping")
        return

    panel_specs = [
        ("accuracy_pct", "Accuracy (%)", (0, 102)),
        ("snow_bias_pct", "Snow bias (%)", (-105, 105)),
        ("rain_bias_pct", "Rain bias (%)", (-105, 105)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    fig.suptitle(f"{region} test split — benchmark PPMs vs. {MODEL_DISPLAY_NAME}\n"
                 f"(pure rain/snow obs; {TAIR_BIN_WIDTH:g} °C air-temperature bins, "
                 f"bins with n ≥ {MIN_BIN_N})", fontsize=12)

    order = [n for n in METHOD_STYLE if n in profiles]
    for ax, (col, ylabel, ylim) in zip(axes, panel_specs):
        for name in order:
            prof = profiles[name]
            if prof.empty or col not in prof.columns:
                continue
            valid = prof[col].notna()
            style = METHOD_STYLE[name]
            ax.plot(prof.loc[valid, "t_mid"], np.clip(prof.loc[valid, col], *ylim),
                    marker="o", ms=4 if name == MODEL_NAME_KEY else 3,
                    label=METHOD_LABEL.get(name, name),
                    zorder=5 if name == MODEL_NAME_KEY else 2, **style)
        ax.axvline(0, color="grey", ls="--", lw=1.0, alpha=0.7)
        if "bias" in col:
            ax.axhline(0, color="grey", lw=0.8, alpha=0.7)
        ax.set(ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Air temperature (°C)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)
    safe_savefig(fig, out_dir(region) / f"{region}_benchmark_accuracy_by_tair.png")

    # NOTE: "accuracy_bias_by_tair" from the work-item list is the same
    # 3-panel figure (accuracy + snow bias + rain bias vs T_air) as this one;
    # no separate file is produced to avoid a redundant duplicate.


def fig_overall_accuracy_bars(region: str):
    """Faithful port of plot_overall_bars: overall + near-freezing accuracy
    bars, benchmark vs model, reading benchmark_comparison.csv (already saved)."""
    csv = benchmarking_dir(region) / "benchmark_comparison.csv"
    if not csv.exists():
        print(f"  [overall_bars:{region}] missing {csv}, skipping")
        return
    t = pd.read_csv(csv)
    # NOTE: benchmark_comparison.csv on disk carries its own "label" column,
    # frozen at CSV-generation time (e.g. "XGBoost + MRoS (this study)" for
    # the model row) — that stale column previously overrode METHOD_LABEL
    # below since this code only filled "label" in if it was entirely
    # absent. Always recompute the label from METHOD_LABEL/MODEL_DISPLAY_NAME
    # so a renamed model (or any relabeled benchmark) is reflected here
    # regardless of what was saved in the CSV.
    t["label"] = t["method"].map(lambda m: METHOD_LABEL.get(m, m))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * len(t))), sharey=True)
    for ax, col, title in [
        (axes[0], "accuracy_pct", "Overall accuracy (%)"),
        (axes[1], "nearfreeze_accuracy_pct",
         f"Near-freezing accuracy (%)\n(|T_wet| ≤ {NEARFREEZE_TWET_C:g} °C)"),
    ]:
        if col not in t.columns:
            continue
        sub = t.dropna(subset=[col]).sort_values(col)
        colors = ["black" if m == MODEL_NAME_KEY else
                  METHOD_STYLE.get(m, {}).get("color", "grey") for m in sub["method"]]
        ax.barh(sub["label"], sub[col], color=colors, alpha=0.85)
        for _, r in sub.iterrows():
            ax.text(r[col] + 0.3, r["label"], f"{r[col]:.1f}", va="center", fontsize=8)
        ax.set(title=title, xlim=(0, 105))
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{region} test split — benchmark comparison (pure rain/snow obs)", fontsize=12)
    safe_savefig(fig, out_dir(region) / f"{region}_overall_accuracy_bars.png")


def fig_ablation_comparison(region: str):
    """Ports the ablation_comparison.png logic from save_cross_experiment_plots
    in ML_XGBoost_binary_uncertainty_ablation_v2.py, reading the already-saved
    ablation_comparison.csv under ablations_v2/ (produced by load_all_metrics,
    which scans each config subfolder's metrics_summary.json). Config folder
    names are mapped to human-readable labels via config_label()."""
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

    # Wider figure than before (12 -> 14 in) so the (now-shortened, see
    # CONFIG_DISPLAY_NAME) config labels on the y-axis have room and are not
    # clipped/cramped against the plot area.
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ok) * 0.45)))
    fig.suptitle(f"{region} — ablation comparison, delta from {config_label('baseline_full')}", fontsize=13)
    for ax, col, title in [
        (axes[0], "test_macro_f1_binary", "Δ Test macro F1 (binary)"),
        (axes[1], "test_roc_auc_cal", "Δ Test ROC AUC (calibrated)"),
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
        ax.axvline(0, color="black", lw=1.0)
        ax.set(xlabel=title, title=title)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        for _, row in ok_delta.iterrows():
            ax.text(row["delta"] + (0.001 if row["delta"] >= 0 else -0.001),
                    row["display_name"], f"{row[col]:.3f}",
                    va="center", ha="left" if row["delta"] >= 0 else "right",
                    fontsize=8, color="dimgrey")
    safe_savefig(fig, out_dir(region) / f"{region}_ablation_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4: F1 by wet-bulb temperature + mix capture, val vs test
# ---------------------------------------------------------------------------

def _twet_profile(df_f, bin_edges, base_hb, extra_hb, sigma, min_n=10):
    """Reimplementation of _twet_profile_binary from ablation_v2 script:
    per-Twet-bin F1(snow)/F1(rain)/abstain-rate/mix-capture, restricted to
    the baseline_full ablations_v2 config's saved test-set predictions."""
    from sklearn.metrics import f1_score

    rows = []
    tw = df_f["temp_wet"].to_numpy(float)
    p_cal = df_f["p_snow_cal"].to_numpy(float)
    phase = df_f["phase_full"].to_numpy(int)
    hb = gaussian_half_band(tw, base_hb, extra_hb, sigma)
    band_pred = np.full(len(p_cal), MIX_CODE, dtype=int)
    band_pred[p_cal <= 0.5 - hb] = RAIN_CODE
    band_pred[p_cal >= 0.5 + hb] = SNOW_CODE

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (tw >= lo) & (tw < hi)
        n = mask.sum()
        if n < min_n:
            continue
        committed = band_pred[mask] != MIX_CODE
        pure = np.isin(phase[mask], [SNOW_CODE, RAIN_CODE])
        sel = committed & pure
        f1_snow = f1_rain = np.nan
        if sel.sum() >= 2 and len(np.unique(phase[mask][sel])) > 1:
            y_true_sel = phase[mask][sel]
            y_pred_sel = band_pred[mask][sel]
            f1_snow = f1_score(y_true_sel == SNOW_CODE, y_pred_sel == SNOW_CODE, zero_division=0)
            f1_rain = f1_score(y_true_sel == RAIN_CODE, y_pred_sel == RAIN_CODE, zero_division=0)
        abstain_rate = (~committed).mean()
        mix_mask = phase[mask] == MIX_CODE
        mix_capture = np.nan
        if mix_mask.sum() > 0:
            mix_capture = (band_pred[mask][mix_mask] == MIX_CODE).mean()
        rows.append(dict(bin_mid=(lo + hi) / 2, n=n, f1_snow=f1_snow, f1_rain=f1_rain,
                          abstain_rate=abstain_rate, mix_capture=mix_capture))
    return pd.DataFrame(rows)


def fig_twet_performance(region: str):
    pq = ablations_dir(region) / "baseline_full" / "shap_values_all.parquet"
    meta_path = ablations_dir(region) / "baseline_full" / "metrics_summary.json"
    if not pq.exists():
        print(f"  [twet:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", "temp_wet", "p_snow_cal", "split"])
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        base_hb = m.get("band_base_hb", base_hb)
        extra_hb = m.get("band_extra_hb", extra_hb)
        sigma = m.get("band_sigma", sigma)

    prof_val = _twet_profile(df[df["split"] == "val"], TWET_BIN_EDGES, base_hb, extra_hb, sigma)
    prof_test = _twet_profile(df[df["split"] == "test"], TWET_BIN_EDGES, base_hb, extra_hb, sigma)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [("f1_snow", "F1 (snow)", PHASE_COLORS["snow"]),
              ("f1_rain", "F1 (rain)", PHASE_COLORS["rain"]),
              ("abstain_rate", "Abstain rate", "darkorange"),
              ("mix_capture", "Mix capture rate", PHASE_COLORS["mix"])]
    for ax, (col, title, color) in zip(axes, panels):
        if not prof_val.empty:
            ax.plot(prof_val["bin_mid"], prof_val[col], "--", color=color, label="Validation")
        if not prof_test.empty:
            ax.plot(prof_test["bin_mid"], prof_test[col], "-", color=color, label="Test")
        ax.axvspan(-1, 1, alpha=0.08, color="orange")
        ax.axvline(0, color="grey", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Wet-bulb temperature (deg C)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{region} — performance by wet-bulb temperature ({config_label('baseline_full')})", y=1.03)
    safe_savefig(fig, out_dir(region) / f"{region}_f1_by_wetbulb.png")


# ---------------------------------------------------------------------------
# Figure 4b (NEW): F1 by air temperature + mix capture, val vs test
# ---------------------------------------------------------------------------

def _temp_profile(df_f, bin_edges, base_hb, extra_hb, sigma, temp_col, min_n=10):
    """Same logic as _twet_profile but generalized to any temperature column,
    so it can be reused for both wet-bulb and air-temperature binning."""
    from sklearn.metrics import f1_score

    rows = []
    tvals = df_f[temp_col].to_numpy(float)
    p_cal = df_f["p_snow_cal"].to_numpy(float)
    phase = df_f["phase_full"].to_numpy(int)
    hb = gaussian_half_band(tvals, base_hb, extra_hb, sigma)
    band_pred = np.full(len(p_cal), MIX_CODE, dtype=int)
    band_pred[p_cal <= 0.5 - hb] = RAIN_CODE
    band_pred[p_cal >= 0.5 + hb] = SNOW_CODE

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (tvals >= lo) & (tvals < hi)
        n = mask.sum()
        if n < min_n:
            continue
        committed = band_pred[mask] != MIX_CODE
        pure = np.isin(phase[mask], [SNOW_CODE, RAIN_CODE])
        sel = committed & pure
        f1_snow = f1_rain = np.nan
        if sel.sum() >= 2 and len(np.unique(phase[mask][sel])) > 1:
            y_true_sel = phase[mask][sel]
            y_pred_sel = band_pred[mask][sel]
            f1_snow = f1_score(y_true_sel == SNOW_CODE, y_pred_sel == SNOW_CODE, zero_division=0)
            f1_rain = f1_score(y_true_sel == RAIN_CODE, y_pred_sel == RAIN_CODE, zero_division=0)
        abstain_rate = (~committed).mean()
        mix_mask = phase[mask] == MIX_CODE
        mix_capture = np.nan
        if mix_mask.sum() > 0:
            mix_capture = (band_pred[mask][mix_mask] == MIX_CODE).mean()
        rows.append(dict(bin_mid=(lo + hi) / 2, n=n, f1_snow=f1_snow, f1_rain=f1_rain,
                          abstain_rate=abstain_rate, mix_capture=mix_capture))
    return pd.DataFrame(rows)


def fig_f1_by_tair(region: str):
    """T_air analogue of fig_twet_performance / {region}_f1_by_wetbulb.png:
    per-phase F1 + abstain/mix-capture rate, val vs test, binned by air
    temperature instead of wet-bulb temperature. Uses TAIR_BIN_EDGES
    (TAIR_BIN_MIN=-8, TAIR_BIN_MAX=8, width=1 degC) for consistency with the
    other T_air figures in this script, rather than the wet-bulb figure's own
    TWET_BIN_EDGES (-6 to 6 degC) — the two binning variables are physically
    different quantities so reusing the wet-bulb edges would not be
    meaningful; TAIR_BIN_EDGES is the convention already established for
    T_air specifically in ML_XGBoost_binary_uncertainty_benchmarking.py.
    NOTE: bins with n < 10 obs here are dropped (matches the wet-bulb
    figure's own min_n=10 threshold, a different, lower cutoff than the
    stricter MIN_BIN_N=20 used for the accuracy-by-T_air figures — this
    reflects the original ablation_v2 story3/twet code's convention)."""
    pq = ablations_dir(region) / "baseline_full" / "shap_values_all.parquet"
    meta_path = ablations_dir(region) / "baseline_full" / "metrics_summary.json"
    if not pq.exists():
        print(f"  [f1_tair:{region}] missing {pq}, skipping")
        return
    df = pd.read_parquet(pq, columns=["phase_full", "temp_air", "p_snow_cal", "split"])
    base_hb, extra_hb, sigma = 0.2, 0.15, 2.0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        base_hb = m.get("band_base_hb", base_hb)
        extra_hb = m.get("band_extra_hb", extra_hb)
        sigma = m.get("band_sigma", sigma)

    prof_val = _temp_profile(df[df["split"] == "val"], TAIR_BIN_EDGES, base_hb, extra_hb, sigma, "temp_air")
    prof_test = _temp_profile(df[df["split"] == "test"], TAIR_BIN_EDGES, base_hb, extra_hb, sigma, "temp_air")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [("f1_snow", "F1 (snow)", PHASE_COLORS["snow"]),
              ("f1_rain", "F1 (rain)", PHASE_COLORS["rain"]),
              ("abstain_rate", "Abstain rate", "darkorange"),
              ("mix_capture", "Mix capture rate", PHASE_COLORS["mix"])]
    for ax, (col, title, color) in zip(axes, panels):
        if not prof_val.empty:
            ax.plot(prof_val["bin_mid"], prof_val[col], "--", color=color, label="Validation")
        if not prof_test.empty:
            ax.plot(prof_test["bin_mid"], prof_test[col], "-", color=color, label="Test")
        ax.axvspan(-1, 1, alpha=0.08, color="orange")
        ax.axvline(0, color="grey", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Air temperature (deg C)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{region} — performance by air temperature ({config_label('baseline_full')})", y=1.03)
    safe_savefig(fig, out_dir(region) / f"{region}_f1_by_tair.png")


# ---------------------------------------------------------------------------
# Figure 5: SHAP — wet-bulb heatmap (new headline) + mean-by-phase bar (appendix)
# ---------------------------------------------------------------------------

def fig_shap(region: str):
    kdir = kriging_dir(region)
    heatmap_csv = kdir / "shap_by_wetbulb_bin.csv"
    phase_csv = kdir / "shap_summary_by_phase.csv"

    if heatmap_csv.exists():
        data = pd.read_csv(heatmap_csv).set_index("feature")
        data = data.loc[:, data.notna().any()]
        data.index = [feature_label(f) for f in data.index]
        if HAVE_SEABORN:
            vmax = data.max().max()
            fig, ax = plt.subplots(figsize=(max(10, len(data.columns) * 0.7), len(data.index) * 0.55 + 1.5))
            sns.heatmap(data, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, annot=True, fmt=".3f",
                        linewidths=0.4, linecolor="#cccccc",
                        cbar_kws={"label": "Mean |SHAP| (probability units)"})
            ax.set_title(f"{region} — {MODEL_DISPLAY_NAME} mean |SHAP| by wet-bulb bin — all observations",
                         fontsize=11, pad=10)
            ax.set_xlabel("Wet-bulb temperature bin (deg C)")
            ax.set_ylabel("Feature")
            near_freeze_cols = [i for i, lbl in enumerate(data.columns) if lbl in ["-2–1", "-1–0", "0–1", "1–2", "-2--1", "-1-0", "0-1", "1-2"]]
            for col_i in near_freeze_cols:
                ax.add_patch(plt.Rectangle((col_i, 0), 1, len(data), fill=True, color="steelblue", alpha=0.07, zorder=0))
            safe_savefig(fig, out_dir(region) / f"{region}_shap_wetbulb_heatmap.png")
        else:
            print(f"  [shap:{region}] seaborn not available, skipping heatmap (fallback: imshow)")
            fig, ax = plt.subplots(figsize=(max(10, len(data.columns) * 0.7), len(data.index) * 0.55 + 1.5))
            im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(data.columns))); ax.set_xticklabels(data.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(data.index))); ax.set_yticklabels(data.index)
            fig.colorbar(im, ax=ax, label="Mean |SHAP|")
            ax.set_title(f"{region} — {MODEL_DISPLAY_NAME} mean |SHAP| by wet-bulb bin (fallback, no seaborn)")
            safe_savefig(fig, out_dir(region) / f"{region}_shap_wetbulb_heatmap.png")
    else:
        print(f"  [shap:{region}] missing {heatmap_csv}, skipping headline SHAP figure")

    # Appendix: mean |SHAP| by phase bar chart (previous main-text figure)
    if phase_csv.exists():
        phase_shap_summary = pd.read_csv(phase_csv)
        FEATURES = phase_shap_summary["feature"].tolist()
        FEATURE_LABELS = [feature_label(f) for f in FEATURES]
        phase_plot_cols = [c for c in ["snow", "rain", "mix"] if c in phase_shap_summary.columns]
        x = np.arange(len(FEATURES))
        width = 0.25
        colors = [PHASE_COLORS.get(c, "grey") for c in phase_plot_cols]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (phase_col, color) in enumerate(zip(phase_plot_cols, colors)):
            vals = phase_shap_summary.set_index("feature").loc[FEATURES, phase_col].values
            ax.bar(x + i * width, vals, width, label=phase_col, color=color, alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(FEATURE_LABELS, rotation=35, ha="right")
        ax.set_ylabel("Mean |SHAP value| (probability units)")
        ax.set_title(f"{region} — {MODEL_DISPLAY_NAME} feature importance by predicted phase (mean |SHAP|) [appendix]")
        ax.legend(title="Predicted phase")
        ax.grid(axis="y", alpha=0.3)
        safe_savefig(fig, out_dir(region) / f"{region}_shap_mean_by_phase_appendix.png")
    else:
        print(f"  [shap:{region}] missing {phase_csv}, skipping appendix bar chart")


# ---------------------------------------------------------------------------
# Figure 6 (NEW): Ablation bootstrap CI forest, per-config deltas
# ---------------------------------------------------------------------------

def fig_ablation_deltas_ci(region: str):
    """New figure: horizontal dot/whisker forest plot of bootstrap CI deltas
    per ablation config (baseline minus config), for near-freezing accuracy,
    macro-F1-equivalent (accuracy used, see caveat) and AUROC, reading
    bootstrap_ablation_deltas_ci.csv (columns: config, metric, point, ci_lo,
    ci_hi, p_delta_le_0, excludes_zero). No existing plotting function
    reproduces this; designed here in the visual style of
    bootstrap_cis.plot_delta_forest (horizontal errorbars, dashed zero
    line, bold color for CIs excluding zero, grey for CIs including zero).

    CAVEAT: bootstrap_ablation_deltas_ci.csv contains only
    delta_auc_baseline_minus_config, delta_accuracy_baseline_minus_config,
    delta_nearfreeze_acc_baseline_minus_config — no macro-F1 delta is saved,
    so macro-F1 is NOT plotted here (only AUROC, overall accuracy, and
    near-freezing accuracy deltas, which are the three metrics actually on
    disk)."""
    csv = bootstrap_dir(region) / "bootstrap_ablation_deltas_ci.csv"
    if not csv.exists():
        print(f"  [ablation_ci:{region}] missing {csv}, skipping")
        return
    d = pd.read_csv(csv)

    metric_titles = {
        "delta_auc_baseline_minus_config": "Delta AUROC (baseline - config)",
        "delta_accuracy_baseline_minus_config": "Delta accuracy (baseline - config)",
        "delta_nearfreeze_acc_baseline_minus_config": "Delta near-freezing accuracy (baseline - config)",
    }
    metrics_present = [m for m in metric_titles if m in d["metric"].unique()]
    if not metrics_present:
        print(f"  [ablation_ci:{region}] no expected metrics found in {csv}, skipping")
        return

    fig, axes = plt.subplots(1, len(metrics_present), figsize=(6 * len(metrics_present), max(4, 0.4 * d["config"].nunique())), sharey=True)
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
        colors = ["#2166ac" if ex else "#aaaaaa" for ex in sub["excludes_zero"]]
        for y, (_, row), c in zip(ypos, sub.iterrows(), colors):
            ax.errorbar(row["point"], y, xerr=[[row["point"] - row["ci_lo"]], [row["ci_hi"] - row["point"]]],
                        fmt="o", color=c, ecolor=c, capsize=3, ms=6,
                        markeredgecolor="black" if row["excludes_zero"] else c)
        ax.axvline(0, color="#d62728", ls="--", lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["config"], fontsize=9)
        ax.set_xlabel(metric_titles[metric])
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(f"{region} — ablation deltas vs. baseline_full (95% cluster-bootstrap CIs)\n"
                 f"bold/blue = CI excludes zero, grey = CI includes zero", y=1.02)
    safe_savefig(fig, out_dir(region) / f"{region}_ablation_deltas_ci.png")


# ---------------------------------------------------------------------------
# Figure 7 (NEW headline) + appendix: mix capture by wet-bulb bin
# ---------------------------------------------------------------------------

def _binomial_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def fig_mix_capture_by_wetbulb(region: str):
    """New headline figure replacing story4_band_placement.png: mix-capture
    rate (fraction of true-mix events whose calibrated p_snow falls inside
    the Gaussian uncertainty band) per T_wet bin, using the same 1 degC
    bins as fig_twet_performance, for CA/CO from
    results_binaryXGB_withKriging_v2 test_full_uncertainty_predictions_combined.parquet.

    CI METHOD CAVEAT: the parquet has no station_id/cluster_id column
    (confirmed: test_full_uncertainty_predictions_combined.parquet columns
    do not include any clustering key), so a per-bin cluster/block bootstrap
    is not possible from this file alone. We fall back to a simple
    per-bin binomial (normal-approximation, Wald) 95% CI on the capture
    rate — documented here and in the script docstring, not silently
    assumed. Sample size n is annotated above each point.

    Also writes the previous scatter+violin band_placement figure,
    unmodified in spirit, as an appendix output (renamed, not deleted).
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
        rate = k / n
        ci_lo, ci_hi = _binomial_ci(k, n)
        rows.append(dict(bin_mid=(lo + hi) / 2, n=n, capture_rate=rate, ci_lo=ci_lo, ci_hi=ci_hi))
    prof = pd.DataFrame(rows)
    if prof.empty:
        print(f"  [band:{region}] no mix events found, skipping")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    ax2.bar(prof["bin_mid"], prof["n"], width=0.8, color="#dddddd", zorder=0, label="n (mix events)")
    ax2.set_ylabel("n mix events per bin")
    ax1.errorbar(prof["bin_mid"], prof["capture_rate"] * 100,
                 yerr=[(prof["capture_rate"] - prof["ci_lo"]) * 100, (prof["ci_hi"] - prof["capture_rate"]) * 100],
                 fmt="o-", color=PHASE_COLORS["mix"], ecolor=PHASE_COLORS["mix"], capsize=3, ms=6, zorder=3,
                 label="Mix capture rate")
    for _, row in prof.iterrows():
        ax1.annotate(f"n={row['n']:.0f}", (row["bin_mid"], min(100, row["capture_rate"] * 100 + 6)),
                     ha="center", fontsize=7, color="#555555")
    ax1.axvspan(-1, 1, alpha=0.08, color="orange")
    ax1.axvline(0, color="grey", lw=1)
    ax1.set_xlabel("Wet-bulb temperature (deg C)")
    ax1.set_ylabel("Mix capture rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title(f"{region} — mix-capture rate by wet-bulb bin (95% binomial CI)")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)
    safe_savefig(fig, out_dir(region) / f"{region}_mix_capture_by_wetbulb.png")

    # --- Appendix: previous scatter+violin band_placement panel A, renamed ---
    # t_grid previously hardcoded to a fixed +/-6 degC range, which clipped
    # the drawn band boundary/shading short of the actual data extent
    # whenever df_mix["temp_wet"] ranged wider than +/-6 (scatter points would
    # then visibly extend past the shaded region). Instead, size the grid to
    # the real data range (with a small pad) so the curve/shading always
    # spans every plotted point.
    twet_data_min = float(df_mix["temp_wet"].min())
    twet_data_max = float(df_mix["temp_wet"].max())
    twet_pad = 0.5
    t_grid = np.linspace(twet_data_min - twet_pad, twet_data_max + twet_pad, 300)
    hb_grid = gaussian_half_band(t_grid, base_hb, extra_hb, sigma)
    snow_boundary, rain_boundary = 0.5 + hb_grid, 0.5 - hb_grid
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.fill_between(t_grid, rain_boundary, snow_boundary, alpha=0.10, color="orange")
    ax.plot(t_grid, snow_boundary, color="black", lw=1.2)
    ax.plot(t_grid, rain_boundary, color="black", lw=1.2, ls="--")
    captured = df_mix[df_mix["inside_band"]]
    missed = df_mix[~df_mix["inside_band"]]
    ax.scatter(missed["temp_wet"], missed["p_snow_cal"], color="#cc3311", marker="x", s=40, alpha=0.65, label="Missed")
    ax.scatter(captured["temp_wet"], captured["p_snow_cal"], color="#009988", marker="o", s=40, alpha=0.65, label="Captured")
    capture_pct = 100 * df_mix["inside_band"].mean()
    ax.set_xlabel("Wet-bulb temperature (deg C)")
    ax.set_ylabel("Calibrated p(snow)")
    ax.set_title(f"{region} — band placement, mix events (appendix); overall capture={capture_pct:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    safe_savefig(fig, out_dir(region) / f"{region}_band_placement_appendix.png")


# ---------------------------------------------------------------------------
# Figure 8 (NEW): story5 near-freezing comparison, baseline_full vs no_mros_loocv
# ---------------------------------------------------------------------------

def _nf_profile(df_f, bin_edges, temp_col):
    """Ports _nf_profile from ML_XGBoost_binary_uncertainty_ablation_v2.py
    (~line 741): per-bin precision/recall for snow & rain at the raw 0.5
    threshold (no uncertainty band), pure obs only, mix excluded."""
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


NF_BIN_EDGES = np.arange(-4, 5, 1)  # matches story5's zoom window in ablation_v2 script

CONFIG_COMPARE_STYLE = {
    "baseline_full": dict(color="#2166ac"),
    "no_mros_loocv": dict(color="#d62728"),
}


def _fig_story5_compare(region: str, temp_col: str, fname: str):
    """Two-config (baseline_full vs no_mros_loocv) overlay of near-freezing
    precision/recall vs temperature, test split, ported from
    _nf_profile/_plot_nf_profile (story5_nearfreeze_{tair,twet}.png) in
    ML_XGBoost_binary_uncertainty_ablation_v2.py. Both configs are drawn in
    the SAME panel with distinct colors/linestyles so the comparison is
    legible in one figure, rather than as separate uncombined images."""
    configs = {}
    for cfg in ["baseline_full", "no_mros_loocv"]:
        pq = ablations_dir(region) / cfg / "shap_values_all.parquet"
        if not pq.exists():
            print(f"  [story5:{region}:{fname}] missing {pq} for config '{cfg}', "
                  f"omitting that config from the comparison (rerun ablations_v2 "
                  f"for '{cfg}' to fill this in)")
            continue
        d = pd.read_parquet(pq, columns=["phase_full", temp_col, "p_snow_cal", "split"])
        configs[cfg] = d[d["split"] == "test"]
    if not configs:
        print(f"  [story5:{region}:{fname}] no configs available, skipping")
        return

    profiles = {cfg: _nf_profile(d, NF_BIN_EDGES, temp_col) for cfg, d in configs.items()}
    temp_label = "Air temperature" if temp_col == "temp_air" else "Wet-bulb temperature"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel_specs = [("prec_snow", "rec_snow", "Snow", PHASE_COLORS["snow"]),
                   ("prec_rain", "rec_rain", "Rain", PHASE_COLORS["rain"])]
    for ax, (prec_col, rec_col, label, _color) in zip(axes, panel_specs):
        for cfg, prof in profiles.items():
            if prof.empty:
                continue
            style_color = CONFIG_COMPARE_STYLE.get(cfg, {}).get("color", "grey")
            disp = config_label(cfg)
            valid_p = prof[prec_col].notna()
            valid_r = prof[rec_col].notna()
            ax.plot(prof.loc[valid_p, "t_mid"], prof.loc[valid_p, prec_col],
                    "-", color=style_color, lw=2, marker="o", ms=5,
                    label=f"{disp} — precision")
            ax.plot(prof.loc[valid_r, "t_mid"], prof.loc[valid_r, rec_col],
                    "--", color=style_color, lw=2, marker="s", ms=5,
                    label=f"{disp} — recall")
        ax.axvspan(-1, 1, alpha=0.08, color="orange", zorder=0)
        ax.axvline(0, color="black", lw=0.8, alpha=0.4, zorder=0)
        ax.set(xlabel=f"{temp_label} (°C)", ylabel=f"{label} precision / recall",
               title=label, xlim=(NF_BIN_EDGES[0], NF_BIN_EDGES[-1]), ylim=(0, 1.05))
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{region} — near-freezing precision/recall vs {temp_label.lower()}: "
                 f"{config_label('baseline_full')} vs {config_label('no_mros_loocv')}\n"
                 f"(binary 0.5 threshold, observer-reported mix excluded, test split)",
                 fontsize=12)
    safe_savefig(fig, out_dir(region) / fname)


def fig_story5_nearfreeze_tair(region: str):
    _fig_story5_compare(region, "temp_air", f"{region}_story5_nearfreeze_tair_comparison.png")


def fig_story5_nearfreeze_twet(region: str):
    _fig_story5_compare(region, "temp_wet", f"{region}_story5_nearfreeze_twet_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_FUNCS = {
    "calibration": fig_calibration,
    "forest": fig_benchmark_delta_forest,
    "tair": fig_benchmark_accuracy_by_tair_ci,
    "twet": fig_twet_performance,
    "shap": fig_shap,
    "ablation_ci": fig_ablation_deltas_ci,
    "band": fig_mix_capture_by_wetbulb,
    "ablation_comparison": fig_ablation_comparison,
    "tair_relimp_ci": fig_benchmark_relative_improvement_ci,
    "tair_accuracy": fig_benchmark_accuracy_by_tair,
    "f1_tair": fig_f1_by_tair,
    "overall_bars": fig_overall_accuracy_bars,
    "story5_tair": fig_story5_nearfreeze_tair,
    "story5_twet": fig_story5_nearfreeze_twet,
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures from saved artifacts only.")
    parser.add_argument("--regions", default=",".join(REGIONS), help="Comma-separated region list, e.g. CA,CO")
    parser.add_argument("--figures", default=",".join(FIGURE_FUNCS.keys()),
                         help="Comma-separated figure keys: " + ",".join(FIGURE_FUNCS.keys()))
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    figures = [f.strip() for f in args.figures.split(",") if f.strip()]

    for region in regions:
        print(f"=== Region {region} ===")
        for fig_key in figures:
            func = FIGURE_FUNCS.get(fig_key)
            if func is None:
                print(f"  unknown figure key '{fig_key}', skipping")
                continue
            try:
                func(region)
            except Exception as e:
                print(f"  [{fig_key}:{region}] ERROR: {e}")

    print("Done. Outputs under:", OUT_ROOT)


if __name__ == "__main__":
    main()
