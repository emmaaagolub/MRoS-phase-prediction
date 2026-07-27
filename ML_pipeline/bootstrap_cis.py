"""
bootstrap_cis.py — Paired space-time cluster bootstrap confidence intervals
===========================================================================

Computes bootstrap CIs for (1) the benchmark comparison and (2) the ablation
study, entirely from stored per-observation test predictions. No model is
retrained and no gridded data are touched.

Method
------
Observations are assigned to space-time clusters: a square spatial block of
side BLOCK_KM (projected coordinates) crossed with a calendar day. Clusters —
not individual observations — are resampled with replacement (cluster/block
bootstrap), because citizen-science reports from the same storm and
neighbourhood are not independent; an iid bootstrap would understate
uncertainty. Every replicate applies the SAME resampled rows to all methods /
configs, so method differences are paired within replicate and the CI on each
delta is obtained directly (percentile intervals).

Because block size is a judgment call, the analysis is repeated for each size
in BLOCK_KM_GRID (plus an iid bootstrap for reference) and a sensitivity
table is written; headline numbers use BLOCK_KM_PRIMARY.

Caveat to report alongside results: these CIs are conditional on the single
trained model and train/test split — they quantify test-set sampling
uncertainty, not training stochasticity. (Supplement with multi-seed retrains
if needed.)

Inputs
------
  benchmarking:  model_artifacts/{REGION}/benchmarking_v1/benchmark_predictions_test.parquet
  ablations:     model_artifacts/{REGION}/ablations_v2/{config}/shap_values_all.parquet
                 (per-obs calibrated p(snow) + split labels; test rows used)

Outputs (model_artifacts/{REGION}/benchmarking_v1/bootstrap/)
------
  bootstrap_benchmark_metrics_ci.csv    per-method accuracy / near-freeze acc /
                                        biases with CIs (primary block size)
  bootstrap_benchmark_deltas_ci.csv     model-minus-benchmark deltas with CIs
                                        and P(delta <= 0)
  bootstrap_ablation_deltas_ci.csv      config-minus-baseline deltas (AUC,
                                        accuracy, near-freeze accuracy)
  bootstrap_sensitivity_block_size.csv  headline CIs vs. block size + iid
  graphics/fig1_accuracy_by_tair_ci.png     accuracy vs T_air with CI ribbons
  graphics/fig2_relative_improvement_ci.png delta accuracy vs T_air with ribbons
  graphics/benchmark_delta_forest.png       forest plot of overall deltas

Usage
-----
  python bootstrap_cis.py                 # uses REGION below
  python bootstrap_cis.py --region CA
  python bootstrap_cis.py --region CO --n-boot 10000 --block-km 30
"""
# conda activate "C:\Users\EmmaGolub\Desktop\MRoS_local\venv"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# 1.  CONFIG
# =============================================================================

REGION      = "CO"      # default; override with --region
N_BOOT      = 5000
SEED        = 42

BLOCK_KM_GRID    = [15.0, 30.0, 60.0]   # sensitivity analysis
BLOCK_KM_PRIMARY = 30.0                 # headline CIs
TIME_CLUSTER     = "D"                  # calendar day

CI_LO, CI_HI = 2.5, 97.5                # percentile interval

# Phase codes (match pipeline)
SNOW_CODE, RAIN_CODE, MIX_CODE = 0, 1, 2
NEARFREEZE_TWET_C = 2.0

# T_air binning for ribbon figures (match benchmarking script)
TAIR_BIN_WIDTH  = 1.0
TAIR_BIN_MIN    = -8.0
TAIR_BIN_MAX    = 8.0
MIN_BIN_N       = 20

MODEL_NAME = "xgboost_mros"

DATA_DIR = Path(r"C:\Users\EmmaGolub\Desktop\MRoS_local\mros-precipitation-phase-product-prototype\outputs")

METHOD_LABEL = {
    "ta_1.0": "$T_{a}$ 1.0 °C", "ta_1.5": "$T_{a}$ 1.5 °C",
    "td_0.0": "$T_{d}$ 0.0 °C", "td_0.5": "$T_{d}$ 0.5 °C",
    "tw_0.0": "$T_{w}$ 0.0 °C", "tw_0.5": "$T_{w}$ 0.5 °C", "tw_1.0": "$T_{w}$ 1.0 °C",
    "binlog_jennings18": "Bin. logistic (Jennings et al. 2018)",
    "binlog_fitted":     "Bin. logistic (fitted, this domain)",
    MODEL_NAME:          "XGBoost + MRoS (this study)",
}

# =============================================================================
# 2.  CLUSTER CONSTRUCTION AND BOOTSTRAP SETUP
# =============================================================================

def make_cluster_codes(df: pd.DataFrame, block_km: float | None,
                       time_freq: str = TIME_CLUSTER) -> np.ndarray:
    """Integer cluster code per row: spatial block (block_km) x time period.
    block_km=None -> iid bootstrap (each row its own cluster)."""
    if block_km is None:
        return np.arange(len(df))
    bs = block_km * 1000.0  # projected coords are metres
    bx = np.floor(df["x"].to_numpy(float) / bs).astype(np.int64)
    by = np.floor(df["y"].to_numpy(float) / bs).astype(np.int64)
    tt = pd.to_datetime(df["time"]).dt.floor(time_freq)
    key = pd.MultiIndex.from_arrays([bx, by, tt])
    return pd.factorize(key)[0]


def build_cluster_index(cluster_codes: np.ndarray) -> list[np.ndarray]:
    """List of row-index arrays, one per cluster."""
    order = np.argsort(cluster_codes, kind="stable")
    sorted_codes = cluster_codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    return np.split(order, boundaries)


def bootstrap_row_indices(cluster_rows: list[np.ndarray],
                          rng: np.random.Generator) -> np.ndarray:
    """One replicate: draw n_clusters clusters with replacement, concat rows."""
    k = len(cluster_rows)
    draws = rng.integers(0, k, size=k)
    return np.concatenate([cluster_rows[d] for d in draws])


def pct_ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(s, CI_LO)), float(np.percentile(s, CI_HI)))


def fast_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (equivalent to Mann-Whitney U). y_true_bin in {0,1}."""
    n1 = int(y_true_bin.sum()); n0 = len(y_true_bin) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true_bin == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# =============================================================================
# 3.  METRICS (vectorised, per replicate)
# =============================================================================

def accuracy(correct: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean(correct[idx])) if len(idx) else np.nan


def masked_accuracy(correct: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> float:
    m = mask[idx]
    return float(np.mean(correct[idx][m])) if m.sum() >= 5 else np.nan


def phase_bias(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray,
               code: int, min_obs: int = 10) -> float:
    yt = y_true[idx]; yp = y_pred[idx]
    n_obs = int(np.sum(yt == code))
    if n_obs < min_obs:
        return np.nan
    return 100.0 * (int(np.sum(yp == code)) / n_obs - 1.0)


# =============================================================================
# 4.  BENCHMARK BOOTSTRAP
# =============================================================================

def load_benchmark_df(region: str) -> tuple[pd.DataFrame, list[str]]:
    pq = DATA_DIR / f"ML_pipeline/model_artifacts/{region}/benchmarking_v1/benchmark_predictions_test.parquet"
    df = pd.read_parquet(pq)
    methods = [c.replace("pred_", "") for c in df.columns
               if c.startswith("pred_") and not c.startswith(f"pred_{MODEL_NAME}")]
    print(f"Loaded {len(df)} test obs, benchmarks: {methods}")
    return df, methods


def run_benchmark_bootstrap(df: pd.DataFrame, methods: list[str],
                            block_km: float | None, n_boot: int,
                            rng: np.random.Generator,
                            collect_bins: bool = False) -> dict:
    """Paired cluster bootstrap over the pure-phase test subset."""
    pure_df = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])].reset_index(drop=True)
    y = pure_df["phase_full"].to_numpy(int)
    twet = pure_df["temp_wet"].to_numpy(float)
    tair = pure_df["temp_air"].to_numpy(float)
    nf_mask = np.abs(twet) <= NEARFREEZE_TWET_C
    y_bin_snow = (y == SNOW_CODE).astype(int)
    p_model = pure_df[f"p_snow_cal_{MODEL_NAME}"].to_numpy(float)

    all_methods = methods + [MODEL_NAME]
    preds, correct = {}, {}
    for m in methods:
        preds[m] = pure_df[f"pred_{m}"].to_numpy(int)
    preds[MODEL_NAME] = pure_df[f"pred_{MODEL_NAME}_bin05"].to_numpy(int)
    for m in all_methods:
        correct[m] = (preds[m] == y)

    # mix capture uses ALL obs (band prediction), resampled with its own clusters
    mix_df = df.reset_index(drop=True)
    mix_is = (mix_df["phase_full"].to_numpy(int) == MIX_CODE)
    band_pred = mix_df[f"pred_{MODEL_NAME}_band"].to_numpy(int)

    # T_air bin assignment for ribbon curves
    edges = np.arange(TAIR_BIN_MIN, TAIR_BIN_MAX + TAIR_BIN_WIDTH, TAIR_BIN_WIDTH)
    bin_idx = np.digitize(tair, edges) - 1            # -1 / len-1 out of range
    n_bins = len(edges) - 1
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    # bins reported: enough obs in the ORIGINAL sample (match main figures)
    bin_valid = np.array([(bin_idx == b).sum() >= MIN_BIN_N for b in range(n_bins)])

    clusters = build_cluster_index(make_cluster_codes(pure_df, block_km))
    clusters_mix = build_cluster_index(make_cluster_codes(mix_df, block_km))
    print(f"  block_km={block_km}: {len(clusters)} clusters over {len(pure_df)} pure obs "
          f"(mean size {len(pure_df)/len(clusters):.1f})")

    acc_s   = {m: np.empty(n_boot) for m in all_methods}
    nfacc_s = {m: np.empty(n_boot) for m in all_methods}
    sbias_s = {m: np.empty(n_boot) for m in all_methods}
    rbias_s = {m: np.empty(n_boot) for m in all_methods}
    auc_s   = np.empty(n_boot)
    mixcap_s = np.empty(n_boot)
    dacc_s  = {m: np.empty(n_boot) for m in methods}
    dnf_s   = {m: np.empty(n_boot) for m in methods}
    bin_acc_s = {m: np.full((n_boot, n_bins), np.nan) for m in all_methods} if collect_bins else None

    for b in range(n_boot):
        idx = bootstrap_row_indices(clusters, rng)
        for m in all_methods:
            acc_s[m][b]   = accuracy(correct[m], idx)
            nfacc_s[m][b] = masked_accuracy(correct[m], nf_mask, idx)
            sbias_s[m][b] = phase_bias(y, preds[m], idx, SNOW_CODE)
            rbias_s[m][b] = phase_bias(y, preds[m], idx, RAIN_CODE)
        for m in methods:  # paired deltas within the same replicate
            dacc_s[m][b] = acc_s[MODEL_NAME][b] - acc_s[m][b]
            dnf_s[m][b]  = nfacc_s[MODEL_NAME][b] - nfacc_s[m][b]
        auc_s[b] = fast_auc(y_bin_snow[idx], p_model[idx])
        if collect_bins:
            bi = bin_idx[idx]
            for m in all_methods:
                c = correct[m][idx]
                for bb in range(n_bins):
                    if not bin_valid[bb]:
                        continue
                    mm = bi == bb
                    if mm.sum() >= 5:
                        bin_acc_s[m][b, bb] = float(np.mean(c[mm]))
        # mix capture replicate (all-obs clusters)
        idx2 = bootstrap_row_indices(clusters_mix, rng)
        tm = mix_is[idx2]
        mixcap_s[b] = float(np.mean(band_pred[idx2][tm] == MIX_CODE)) if tm.sum() >= 5 else np.nan

    return dict(all_methods=all_methods, methods=methods,
                acc=acc_s, nfacc=nfacc_s, sbias=sbias_s, rbias=rbias_s,
                dacc=dacc_s, dnf=dnf_s, auc=auc_s, mixcap=mixcap_s,
                bin_acc=bin_acc_s, bin_mids=bin_mids, bin_valid=bin_valid,
                point=dict(
                    acc={m: float(np.mean(correct[m])) for m in all_methods},
                    nfacc={m: float(np.mean(correct[m][nf_mask])) for m in all_methods},
                    sbias={m: phase_bias(y, preds[m], np.arange(len(y)), SNOW_CODE) for m in all_methods},
                    rbias={m: phase_bias(y, preds[m], np.arange(len(y)), RAIN_CODE) for m in all_methods},
                    auc=fast_auc(y_bin_snow, p_model),
                    mixcap=(float(np.mean(band_pred[mix_is] == MIX_CODE)) if mix_is.any() else np.nan),
                ))


def benchmark_tables(res: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for m in res["all_methods"]:
        for metric, samp, pt in [
            ("accuracy",            res["acc"][m],   res["point"]["acc"][m]),
            ("nearfreeze_accuracy", res["nfacc"][m], res["point"]["nfacc"][m]),
            ("snow_bias_pct",       res["sbias"][m], res["point"]["sbias"][m]),
            ("rain_bias_pct",       res["rbias"][m], res["point"]["rbias"][m]),
        ]:
            lo, hi = pct_ci(samp)
            rows.append(dict(method=m, label=METHOD_LABEL.get(m, m), metric=metric,
                             point=round(pt, 4) if pt == pt else np.nan,
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    lo, hi = pct_ci(res["auc"])
    rows.append(dict(method=MODEL_NAME, label=METHOD_LABEL[MODEL_NAME], metric="roc_auc",
                     point=round(res["point"]["auc"], 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    lo, hi = pct_ci(res["mixcap"])
    rows.append(dict(method=MODEL_NAME, label=METHOD_LABEL[MODEL_NAME], metric="mix_capture_band",
                     point=round(res["point"]["mixcap"], 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "bootstrap_benchmark_metrics_ci.csv", index=False)

    drows = []
    for m in res["methods"]:
        for metric, samp, pt in [
            ("delta_accuracy",
             res["dacc"][m], res["point"]["acc"][MODEL_NAME] - res["point"]["acc"][m]),
            ("delta_nearfreeze_accuracy",
             res["dnf"][m], res["point"]["nfacc"][MODEL_NAME] - res["point"]["nfacc"][m]),
        ]:
            lo, hi = pct_ci(samp)
            s = samp[~np.isnan(samp)]
            drows.append(dict(benchmark=m, label=METHOD_LABEL.get(m, m), metric=metric,
                              point=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                              p_delta_le_0=round(float(np.mean(s <= 0)), 5),
                              excludes_zero=bool(lo > 0 or hi < 0)))
    deltas_df = pd.DataFrame(drows)
    deltas_df.to_csv(out_dir / "bootstrap_benchmark_deltas_ci.csv", index=False)
    return metrics_df, deltas_df


# =============================================================================
# 5.  ABLATION BOOTSTRAP  (from stored shap_values_all.parquet per config)
# =============================================================================

def load_ablation_test_preds(region: str) -> dict[str, pd.DataFrame]:
    root = DATA_DIR / f"ML_pipeline/model_artifacts/{region}/ablations_v2"
    if not root.exists():
        print(f"No ablation folder at {root} — skipping ablation CIs.")
        return {}
    out = {}
    for cfg_dir in sorted(root.iterdir()):
        pq = cfg_dir / "shap_values_all.parquet"
        if not pq.exists():
            continue
        d = pd.read_parquet(pq, columns=[c for c in
             ["time", "x", "y", "phase_full", "temp_wet", "p_snow_cal", "split"]])
        d = d[d["split"] == "test"].reset_index(drop=True)
        out[cfg_dir.name] = d
    print(f"Ablation configs with stored test predictions: {list(out)}")
    return out


def run_ablation_bootstrap(cfgs: dict[str, pd.DataFrame], block_km: float | None,
                           n_boot: int, rng: np.random.Generator,
                           baseline: str = "baseline_full") -> pd.DataFrame:
    if baseline not in cfgs:
        print("baseline_full not found — skipping ablation CIs.")
        return pd.DataFrame()

    # Align all configs to the baseline's test rows via space-time-phase key.
    def key(d):
        return (pd.to_datetime(d["time"]).astype("int64").astype(str) + "_" +
                d["x"].round(1).astype(str) + "_" + d["y"].round(1).astype(str))

    base = cfgs[baseline].copy()
    base["_k"] = key(base)
    base = base.drop_duplicates("_k").reset_index(drop=True)
    y = base["phase_full"].to_numpy(int)
    pure = np.isin(y, [SNOW_CODE, RAIN_CODE])
    nf = pure & (np.abs(base["temp_wet"].to_numpy(float)) <= NEARFREEZE_TWET_C)
    y_snow = (y == SNOW_CODE).astype(int)

    p_cfg = {baseline: base["p_snow_cal"].to_numpy(float)}
    for name, d in cfgs.items():
        if name == baseline:
            continue
        dd = d.copy(); dd["_k"] = key(dd)
        dd = dd.drop_duplicates("_k")
        merged = base[["_k"]].merge(dd[["_k", "p_snow_cal"]], on="_k", how="left")
        if merged["p_snow_cal"].isna().mean() > 0.02:
            print(f"  WARNING: {name}: {merged['p_snow_cal'].isna().mean():.1%} rows "
                  f"unmatched to baseline — skipping")
            continue
        p_cfg[name] = merged["p_snow_cal"].to_numpy(float)

    correct = {n: (np.where(p >= 0.5, SNOW_CODE, RAIN_CODE) == y) & ~np.isnan(p)
               for n, p in p_cfg.items()}
    clusters = build_cluster_index(make_cluster_codes(base, block_km))
    names = [n for n in p_cfg if n != baseline]

    samp = {n: dict(dauc=np.empty(n_boot), dacc=np.empty(n_boot), dnf=np.empty(n_boot))
            for n in names}
    for b in range(n_boot):
        idx = bootstrap_row_indices(clusters, rng)
        ip = idx[pure[idx]]; inf_ = idx[nf[idx]]
        auc_b = fast_auc(y_snow[ip], p_cfg[baseline][ip])
        acc_b = float(np.mean(correct[baseline][ip]))
        nf_b  = float(np.mean(correct[baseline][inf_])) if len(inf_) >= 5 else np.nan
        for n in names:
            ok = ~np.isnan(p_cfg[n][ip])
            samp[n]["dauc"][b] = auc_b - fast_auc(y_snow[ip][ok], p_cfg[n][ip][ok])
            samp[n]["dacc"][b] = acc_b - float(np.mean(correct[n][ip]))
            okf = ~np.isnan(p_cfg[n][inf_]) if len(inf_) else np.array([], bool)
            samp[n]["dnf"][b] = (nf_b - float(np.mean(correct[n][inf_][okf]))
                                 if len(inf_) >= 5 and okf.sum() >= 5 else np.nan)

    rows = []
    ip_all = np.flatnonzero(pure); inf_all = np.flatnonzero(nf)
    for n in names:
        pts = dict(
            dauc=fast_auc(y_snow[ip_all], p_cfg[baseline][ip_all]) -
                 fast_auc(y_snow[ip_all], p_cfg[n][ip_all]),
            dacc=float(np.mean(correct[baseline][ip_all])) - float(np.mean(correct[n][ip_all])),
            dnf=float(np.mean(correct[baseline][inf_all])) - float(np.mean(correct[n][inf_all])),
        )
        for metric, key_ in [("delta_auc_baseline_minus_config", "dauc"),
                             ("delta_accuracy_baseline_minus_config", "dacc"),
                             ("delta_nearfreeze_acc_baseline_minus_config", "dnf")]:
            lo, hi = pct_ci(samp[n][key_])
            s = samp[n][key_]; s = s[~np.isnan(s)]
            rows.append(dict(config=n, metric=metric, point=round(pts[key_], 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p_delta_le_0=round(float(np.mean(s <= 0)), 5),
                             excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


# =============================================================================
# 6.  FIGURES
# =============================================================================

def plot_ribbons(res: dict, region: str, graphics: Path):
    """Accuracy-by-T_air with CI ribbons: model, best benchmark, benchmark avg."""
    bm = res["bin_mids"]; bv = res["bin_valid"]
    best = max(res["methods"], key=lambda m: res["point"]["acc"][m])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, color, label in [
        (MODEL_NAME, "black",   METHOD_LABEL[MODEL_NAME]),
        (best,       "#2166ac", f"Best benchmark ({METHOD_LABEL.get(best, best)})"),
    ]:
        arr = res["bin_acc"][m]                     # (n_boot, n_bins)
        med = np.nanmedian(arr, axis=0) * 100
        lo  = np.nanpercentile(arr, CI_LO, axis=0) * 100
        hi  = np.nanpercentile(arr, CI_HI, axis=0) * 100
        v = bv & ~np.isnan(med)
        ax.plot(bm[v], med[v], "-o", color=color, lw=2, ms=4, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axvspan(0, 4, alpha=0.05, color="orange")
    ax.set(xlabel="Air temperature (°C)", ylabel="Accuracy (%)", ylim=(0, 102),
           title=f"{region} — accuracy by air temperature with {CI_HI-CI_LO:.0f}% "
                 f"cluster-bootstrap CIs")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphics / "fig1_accuracy_by_tair_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Delta ribbons: model minus best benchmark, model minus average benchmark
    fig, ax = plt.subplots(figsize=(9, 5))
    arr_model = res["bin_acc"][MODEL_NAME]
    for other, color, label in [
        (res["bin_acc"][best], "black", f"vs. best benchmark ({METHOD_LABEL.get(best, best)})"),
        (np.nanmean(np.stack([res["bin_acc"][m] for m in res["methods"]]), axis=0),
         "#d62728", "vs. average benchmark"),
    ]:
        d = (arr_model - other) * 100                # paired within replicate
        med = np.nanmedian(d, axis=0)
        lo  = np.nanpercentile(d, CI_LO, axis=0)
        hi  = np.nanpercentile(d, CI_HI, axis=0)
        v = bv & ~np.isnan(med)
        ax.plot(bm[v], med[v], "-o", color=color, lw=2, ms=4, label=label)
        ax.fill_between(bm[v], lo[v], hi[v], color=color, alpha=0.18)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.axvspan(0, 4, alpha=0.05, color="orange")
    ax.set(xlabel="Air temperature (°C)", ylabel="Δ Accuracy (percentage points)",
           title=f"{region} — model minus benchmark accuracy with {CI_HI-CI_LO:.0f}% "
                 f"paired cluster-bootstrap CIs")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphics / "fig2_relative_improvement_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_forest(deltas_df: pd.DataFrame, region: str, graphics: Path):
    for metric, fname, title in [
        ("delta_accuracy", "benchmark_delta_forest.png",
         "Δ overall accuracy (model − benchmark)"),
        ("delta_nearfreeze_accuracy", "benchmark_delta_forest_nearfreeze.png",
         f"Δ near-freezing accuracy (|T_wet| ≤ {NEARFREEZE_TWET_C:g} °C)"),
    ]:
        d = deltas_df[deltas_df["metric"] == metric].sort_values("point")
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(d))))
        ypos = np.arange(len(d))
        ax.errorbar(d["point"] * 100, ypos,
                    xerr=[(d["point"] - d["ci_lo"]) * 100, (d["ci_hi"] - d["point"]) * 100],
                    fmt="o", color="black", ecolor="grey", capsize=3, ms=5)
        ax.axvline(0, color="#d62728", ls="--", lw=1)
        ax.set_yticks(ypos); ax.set_yticklabels(d["label"], fontsize=9)
        ax.set(xlabel="Δ Accuracy (percentage points)",
               title=f"{region} — {title}\n(95% paired cluster-bootstrap CIs)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(graphics / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 7.  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Paired cluster bootstrap CIs")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--block-km", type=float, default=BLOCK_KM_PRIMARY)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    region = args.region
    out_dir = DATA_DIR / f"ML_pipeline/model_artifacts/{region}/benchmarking_v1/bootstrap"
    graphics = out_dir / "graphics"
    graphics.mkdir(parents=True, exist_ok=True)

    df, methods = load_benchmark_df(region)
    rng = np.random.default_rng(args.seed)

    # ── Primary analysis ──────────────────────────────────────────────────────
    print(f"\nPrimary bootstrap: block_km={args.block_km}, B={args.n_boot}")
    res = run_benchmark_bootstrap(df, methods, args.block_km, args.n_boot, rng,
                                  collect_bins=True)
    metrics_df, deltas_df = benchmark_tables(res, out_dir)
    plot_ribbons(res, region, graphics)
    plot_delta_forest(deltas_df, region, graphics)

    print("\nModel-minus-benchmark deltas (95% CI):")
    print(deltas_df[deltas_df.metric == "delta_accuracy"]
          [["label", "point", "ci_lo", "ci_hi", "excludes_zero"]].to_string(index=False))

    # ── Block-size sensitivity ────────────────────────────────────────────────
    print("\nBlock-size sensitivity (headline delta = model - best benchmark):")
    best = max(methods, key=lambda m: res["point"]["acc"][m])
    sens_rows = []
    for bk in list(dict.fromkeys(BLOCK_KM_GRID + [None])):
        r = (res if bk == args.block_km else
             run_benchmark_bootstrap(df, methods, bk, args.n_boot,
                                     np.random.default_rng(args.seed), collect_bins=False))
        for metric, samp in [("model_accuracy", r["acc"][MODEL_NAME]),
                             ("model_nearfreeze_accuracy", r["nfacc"][MODEL_NAME]),
                             ("delta_accuracy_vs_best", r["dacc"][best]),
                             ("delta_nearfreeze_vs_best", r["dnf"][best])]:
            lo, hi = pct_ci(samp)
            sens_rows.append(dict(block_km=("iid" if bk is None else bk), metric=metric,
                                  ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                  ci_width=round(hi - lo, 4)))
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out_dir / "bootstrap_sensitivity_block_size.csv", index=False)
    print(sens_df.to_string(index=False))

    # ── Ablation deltas ───────────────────────────────────────────────────────
    cfgs = load_ablation_test_preds(region)
    if cfgs:
        abl_df = run_ablation_bootstrap(cfgs, args.block_km, args.n_boot,
                                        np.random.default_rng(args.seed))
        if not abl_df.empty:
            abl_df.to_csv(out_dir / "bootstrap_ablation_deltas_ci.csv", index=False)
            print("\nAblation deltas (baseline − config, 95% CI):")
            print(abl_df[abl_df.metric == "delta_auc_baseline_minus_config"]
                  [["config", "point", "ci_lo", "ci_hi", "excludes_zero"]].to_string(index=False))

    with open(out_dir / "bootstrap_run_config.json", "w") as f:
        json.dump(dict(region=region, n_boot=args.n_boot, block_km=args.block_km,
                       time_cluster=TIME_CLUSTER, seed=args.seed,
                       ci=[CI_LO, CI_HI], block_km_grid=BLOCK_KM_GRID,
                       note=("CIs are conditional on the trained model and split; "
                             "they quantify test-set sampling uncertainty only.")), f, indent=2)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
