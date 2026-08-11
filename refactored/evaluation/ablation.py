"""Ablation study — how much does each predictor actually contribute?

Every experiment retrains the whole pipeline with one predictor (or one group)
removed, so any drop in performance is attributable to that predictor rather
than to a different fit. The gridded cube is sampled to the observation points
once and cached, and every experiment reuses the model's own train/val/test
split, so the numbers are directly comparable with the model's.

Calibration matches the main model: beta calibration, fitted separately for
near-freezing and clear-phase conditions.

Each experiment writes its own subfolder under
  outputs/evaluation/<REGION>/ablation/

Usage
-----
  python ablation.py
  python ablation.py --regions CA
  python ablation.py --configs 0,1,2
  python ablation.py --dry-run
  python ablation.py --replot
"""

from __future__ import annotations

import argparse
import sys
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xarray as xr
import xgboost as xgb
from betacal import BetaCalibration
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score, 
    brier_score_loss, 
    f1_score,
    log_loss, 
    
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_base import (
    ALL_GRID_FEATURES, ALL_LOOCV_FEATURES, BASE_XGB_PARAMS, 
    BINARY_LABEL_MAP, CLEAR_PHASE_TWET_C, EARLY_STOPPING_ROUNDS,
    
    MIX_CODE,
    NF_BIN_LABELS, NUM_BOOST_ROUND, PHASE_COLORS,
    RAIN_CODE, RANDOM_SEED, REGIONS, SCALE_POS_WEIGHT_GRID, 
    SNOW_CODE, TOP_N_SHAP_FEATURES, TRAIN_FRAC, TWET_BIN_EDGES, TWET_BIN_LABELS,
    build_predictor_cube, classify_phase_gaussian_band,
    expected_calibration_error, gaussian_half_band, interp_paths,
    experiment_dir, model_dir, optimize_threshold_params, prep_loocv_table,
    sample_predictor_cube_to_points_batched, sweep_scale_pos_weight,
)

INTERP_TYPE = "kriging"

# Feature set used unless an experiment overrides part of it.
BASE_FEATURE_CONFIG = dict(
    use_temp_air=True,
    use_temp_dew=True,
    use_temp_wet=True,
    use_rh=False,
    use_imerg_plp=True,
    use_elev=True,
    use_mros_loocv=True,
)

# The experiments: one baseline, then drop each predictor in turn, then a few
# combinations that test whether groups of predictors carry each other.
ABLATION_CONFIGS: list[dict] = [
    dict(name="baseline_full"),
    dict(name="no_mros_loocv", use_mros_loocv=False),
    dict(name="no_temp_air", use_temp_air=False),
    dict(name="no_temp_dew", use_temp_dew=False),
    dict(name="no_temp_wet", use_temp_wet=False),
    dict(name="no_imerg_plp", use_imerg_plp=False),
    dict(name="no_elev", use_elev=False),
    dict(name="thermo_only", use_imerg_plp=False, use_elev=False, use_mros_loocv=False),
    dict(name="min_core_noplp", use_temp_air=False, use_temp_dew=False, use_imerg_plp=False),
    dict(name="min_core_wplp", use_temp_air=False, use_temp_dew=False),
]

# Set per region by configure().
REGION = None
PATHS = None
# The model directory, whose split table is shared so this experiment
# trains and tests on exactly the same observations as the model.
SETUP_DIR = None
ABLATION_ROOT = None

def configure(region: str) -> None:
    """Point the module at one region's inputs and outputs."""
    global REGION, PATHS, SETUP_DIR, ABLATION_ROOT
    REGION = region
    PATHS = {INTERP_TYPE: interp_paths(region, INTERP_TYPE),
             "imerg": interp_paths(region, INTERP_TYPE)["imerg"]}
    SETUP_DIR = model_dir(region)
    ABLATION_ROOT = experiment_dir(region, "ablation")
    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)


def plot_per_experiment_stories(
    name, graphics_dir,
    y_val_fit_phase, pred_val_bin05, y_test_fit_phase, pred_test_bin05,
    y_val_full_phase, pred_val_full, y_test_full_phase, pred_test_full,
    y_val_bin, p_val_cal, y_test_bin, p_test_cal,
    p_val_raw, p_test_raw,
    val_full_df, test_full_df,
    base_hb, extra_hb, sigma,
    p_valf_cal, p_testf_cal,
):
    def _norm_cm(ax, y_true, y_pred, labels, display_labels, title, cmap="Blues",
             tick_fontsize=11, cell_fontsize=10, title_fontsize=11):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums,
                            out=np.zeros_like(cm, dtype=float),
                            where=row_sums > 0)
        n = len(labels)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(display_labels, fontsize=tick_fontsize)
        ax.set_yticks(range(n)); ax.set_yticklabels(display_labels, fontsize=tick_fontsize)
        ax.set_xlabel("Predicted", fontsize=tick_fontsize)
        ax.set_ylabel("True", fontsize=tick_fontsize)
        ax.set_title(title, fontsize=title_fontsize); ax.grid(False)
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.0%}\n({cm[i,j]})",
                        ha="center", va="center", fontsize=cell_fontsize, color=color)

    # ── Story 1a: Binary confusion matrices ──────────────────────────────────
    n_mix_val  = int(np.sum(y_val_full_phase  == MIX_CODE))
    n_mix_test = int(np.sum(y_test_full_phase == MIX_CODE))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"{name} — Story 1a: Binary confusion matrices", fontsize=12)

    _norm_cm(axes[0], y_val_fit_phase,  pred_val_bin05,
            [SNOW_CODE, RAIN_CODE], ["snow", "rain"],
            f"Validation (n={len(y_val_fit_phase)})\n"
            f"{n_mix_val} mix-labeled obs. excluded",
            tick_fontsize=11, cell_fontsize=10, title_fontsize=11)

    _norm_cm(axes[1], y_test_fit_phase, pred_test_bin05,
            [SNOW_CODE, RAIN_CODE], ["snow", "rain"],
            f"Test (n={len(y_test_fit_phase)})\n"
            f"{n_mix_test} mix-labeled obs. excluded",
            tick_fontsize=11, cell_fontsize=10, title_fontsize=11)

    plt.tight_layout()
    fig.savefig(graphics_dir / "story1a_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 1b: ROC and PR curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"{name} — Story 1b: ROC and precision-recall curves\n"
                "(pure snow/rain obs. only; mix-labeled excluded)",
                fontsize=12)

    ax_roc = axes[0]
    ax_pr  = axes[1]
    split_colors = {"Validation": "#8338ec", "Test": "#ff006e"}

    for y_true_b, p_cal, split in [
        (y_val_bin,  p_val_cal,  "Validation"),
        (y_test_bin, p_test_cal, "Test"),
    ]:
        color = split_colors[split]
        fpr, tpr, _ = roc_curve(y_true_b, p_cal)
        prec, rec, _ = precision_recall_curve(y_true_b, p_cal)
        auc = roc_auc_score(y_true_b, p_cal)
        ap  = average_precision_score(y_true_b, p_cal)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{split}  AUC={auc:.3f}")
        ax_pr.plot(rec, prec,  color=color, lw=2, label=f"{split}  AP={ap:.3f}")

    ax_roc.plot([0,1],[0,1],"--",color="grey",lw=1,alpha=0.6)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
            title="ROC curve", xlim=(0,1), ylim=(0,1))
    ax_roc.legend(fontsize=10); ax_roc.grid(alpha=0.25)

    ax_pr.set(xlabel="Recall", ylabel="Precision",
            title="Precision-recall curve", xlim=(0,1), ylim=(0,1))
    ax_pr.legend(fontsize=10); ax_pr.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(graphics_dir / "story1b_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

 # ── Story 2 ───────────────────────────────────────────────────────────────
    band_at_zero  = gaussian_half_band(np.array([0.0]), base_hb, extra_hb, sigma)[0]
    rain_thresh_0 = 0.5 - band_at_zero
    snow_thresh_0 = 0.5 + band_at_zero
    split_colors_2 = {"Validation": "#8338ec", "Test": "#ff006e"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{name} — Story 2: Probability calibration", fontsize=12)

    for col, (y_true_b, p_raw, p_cal, split) in enumerate([
        (y_val_bin,  p_val_raw, p_val_cal,  "Validation"),
        (y_test_bin, p_test_raw, p_test_cal, "Test"),
    ]):
        ax_rel  = axes[0, col]
        ax_hist = axes[1, col]
        color   = split_colors_2[split]

        if len(y_true_b) < 10:
            ax_rel.text(0.5, 0.5, "insufficient data",
                        ha="center", va="center", transform=ax_rel.transAxes)
            continue

        frac_raw, mean_raw = calibration_curve(y_true_b, p_raw, n_bins=15, strategy="quantile")
        frac_cal, mean_cal = calibration_curve(y_true_b, p_cal, n_bins=15, strategy="quantile")

        ax_rel.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.10, color="orange",
                       label=f"Band at T_wet=0°C ({rain_thresh_0:.2f}–{snow_thresh_0:.2f})")
        ax_rel.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, alpha=0.7,
                    label="Perfect calibration")
        ax_rel.plot(mean_raw, frac_raw, "o--", color="#aaaaaa", lw=1.5, ms=5,
                    label="Raw XGBoost")
        ax_rel.plot(mean_cal, frac_cal, "o-",  color=color, lw=2, ms=6,
                    label="Calibrated (beta)")
        ax_rel.set(xlim=(0,1), ylim=(0,1),
                   ylabel="Observed snow frequency", title=split)
        ax_rel.legend(fontsize=9); ax_rel.grid(alpha=0.25)

        bs_raw = brier_score_loss(y_true_b, p_raw)
        bs_cal = brier_score_loss(y_true_b, p_cal)
        ax_rel.text(0.03, 0.92, f"Brier  raw={bs_raw:.3f}  cal={bs_cal:.3f}",
                    transform=ax_rel.transAxes, fontsize=9, color="dimgrey")

        ax_hist.hist(p_cal, bins=30, color=color, alpha=0.7, edgecolor="none")
        ax_hist.axvspan(rain_thresh_0, snow_thresh_0, alpha=0.15, color="orange")
        ax_hist.set(xlabel="Predicted p(snow), calibrated", ylabel="Count")
        ax_hist.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(graphics_dir / "story2_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 3 ───────────────────────────────────────────────────────────────
    def _twet_profile_binary(df_f, p_cal_f, y_true_f, bin_edges):
        """
        Binary F1 profile vs T_wet.
        f1_snow / f1_rain: computed on committed events (outside band), pure labels only.
        abstain_rate: fraction of ALL events (any true label) placed in band.
        mix_capture: fraction of observer-reported mix events placed in band (diagnostic only).
        """
        records = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (df_f["temp_wet"].to_numpy() >= lo) & (df_f["temp_wet"].to_numpy() < hi)
            if mask.sum() < 10:
                continue
            yt      = y_true_f[mask]
            p_bin   = p_cal_f[mask]
            tw_bin  = df_f["temp_wet"].to_numpy()[mask]

            # Band prediction for all events in this bin
            yp_band = classify_phase_gaussian_band(p_bin, tw_bin, base_hb, extra_hb, sigma)
            committed = yp_band != MIX_CODE

            # Binary F1: only committed events, only pure true labels
            pure = np.isin(yt, [SNOW_CODE, RAIN_CODE])
            mask_cp = committed & pure
            yt_c = yt[mask_cp]
            yp_c = yp_band[mask_cp]

            def _f1(code):
                tp = np.sum((yt_c == code) & (yp_c == code))
                fp = np.sum((yt_c != code) & (yp_c == code))
                fn = np.sum((yt_c == code) & (yp_c != code))
                pr = tp / (tp + fp) if (tp + fp) else 0.0
                rc = tp / (tp + fn) if (tp + fn) else 0.0
                return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

            # Abstention rate over ALL events in bin (operational flag rate)
            abstain_rate = float(np.mean(yp_band == MIX_CODE))

            # Mix capture: observer-reported mix events landing in band (uncertainty diagnostic)
            tm = yt == MIX_CODE
            mix_cap = float(np.mean(yp_band[tm] == MIX_CODE)) if tm.any() else np.nan

            records.append({
                "t_mid":        (lo + hi) / 2,
                "n":            int(mask.sum()),
                "n_committed":  int(mask_cp.sum()),
                "f1_snow":      _f1(SNOW_CODE),
                "f1_rain":      _f1(RAIN_CODE),
                "abstain_rate": abstain_rate,
                "mix_capture":  mix_cap,
            })
        return pd.DataFrame(records)

    bin_edges = np.arange(-6, 7, 1)
    prof_val  = _twet_profile_binary(val_full_df,  p_valf_cal,  y_val_full_phase,  bin_edges)
    prof_test = _twet_profile_binary(test_full_df, p_testf_cal, y_test_full_phase, bin_edges)

    # Four panels: F1-snow, F1-rain, abstention rate, mix capture (diagnostic)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{name} — Story 3: Binary performance vs T_wet", fontsize=12)

    panel_specs = [
        ("f1_snow",      "F1 — snow (committed events)",           PHASE_COLORS["snow"]),
        ("f1_rain",      "F1 — rain (committed events)",           PHASE_COLORS["rain"]),
        ("abstain_rate", "Abstention rate (fraction in band)",     "darkorange"),
        ("mix_capture",  "Observer-mix in band\n(uncertainty diagnostic)", "#888888"),
    ]
    for ax, (col, ylabel, color) in zip(axes, panel_specs):
        for prof, split, ls in [(prof_val, "Val", "--"), (prof_test, "Test", "-")]:
            if col not in prof.columns or prof.empty:
                continue
            ax.plot(prof["t_mid"], prof[col], ls, color=color, lw=2,
                    label=split, marker="o", ms=4)
        ax.axvspan(-1, 1, alpha=0.08, color="orange")
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        ax.set(xlabel="T_wet (°C)", ylabel=ylabel, title=ylabel,
            xlim=(bin_edges[0], bin_edges[-1]), ylim=(0, 1.05))
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(graphics_dir / "story3_twet_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Story 4 ───────────────────────────────────────────────────────────────
    t_grid = np.linspace(-6, 6, 300)
    hb_grid = gaussian_half_band(t_grid, base_hb, extra_hb, sigma)
    snow_boundary = 0.5 + hb_grid
    rain_boundary = 0.5 - hb_grid

    df_combo = pd.concat([
        val_full_df.assign(split="val"),
        test_full_df.assign(split="test"),
    ], ignore_index=True)
    mix_mask = df_combo["phase_full"] == MIX_CODE
    df_mix   = df_combo[mix_mask].copy()
    if len(df_mix) > 0:
        hb_obs = gaussian_half_band(df_mix["temp_wet"].to_numpy(), base_hb, extra_hb, sigma)
        df_mix["inside_band"] = (
            (df_mix["p_snow_cal"] > 0.5 - hb_obs) &
            (df_mix["p_snow_cal"] < 0.5 + hb_obs)
        )
        capture_pct = 100 * df_mix["inside_band"].mean()
    else:
        capture_pct = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(f"{name} — Story 4: Uncertainty band placement", fontsize=12, y=1.01)

    # Panel A
    ax_a = axes[0]
    ax_a.fill_between(t_grid, rain_boundary, snow_boundary, alpha=0.10, color="orange")
    ax_a.plot(t_grid, snow_boundary, color="black", lw=2, ls="-",  label="Snow threshold")
    ax_a.plot(t_grid, rain_boundary, color="black", lw=2, ls="--", label="Rain threshold")
    if len(df_mix) > 0:
        for flag, label, color, marker in [
            (False, "Missed", "#cc3311", "x"),
            (True,  "Captured", "#009988", "o"),
        ]:
            sub = df_mix[df_mix["inside_band"] == flag]
            ax_a.scatter(sub["temp_wet"], sub["p_snow_cal"], c=color, marker=marker,
                         s=40, alpha=0.65, linewidths=1.0,
                         label=f"{label} (n={len(sub)})", zorder=3+flag)
    ax_a.axvline(0, color="grey", lw=0.8, alpha=0.4)
    ax_a.axhline(0.5, color="grey", lw=0.8, alpha=0.4)
    ax_a.set(xlim=(-6,6), ylim=(-0.04,1.04),
            xlabel="T_wet (°C)", ylabel="Calibrated p(snow)",
            title="A — Observer-reported ambiguous events vs. band\n"
                f"(n={len(df_mix)} mix-labeled obs.; {capture_pct:.0f}% fall inside band)")
    ax_a.legend(fontsize=9)

    # # Panel B — abstention rate by T_wet bin (operational flag diagnostic)
    # ax_b = axes[1]
    # violin_bins   = [(-6,-4),(-4,-2),(-2,0),(0,2),(2,4),(4,6)]
    # bin_centers   = [(lo + hi) / 2 for lo, hi in violin_bins]
    # bin_labels    = [f"{lo}–{hi}" for lo, hi in violin_bins]

    # for df_s, split, color, ls in [
    #     (df_combo[df_combo["split"] == "val"],  "Val",  "#8338ec", "--"),
    #     (df_combo[df_combo["split"] == "test"], "Test", "#ff006e", "-"),
    # ]:
    #     abstain_rates = []
    #     for lo, hi in violin_bins:
    #         mask_bin = (df_s["temp_wet"] >= lo) & (df_s["temp_wet"] < hi)
    #         if mask_bin.sum() == 0:
    #             abstain_rates.append(np.nan)
    #             continue
    #         p_b  = df_s.loc[mask_bin, "p_snow_cal"].to_numpy()
    #         tw_b = df_s.loc[mask_bin, "temp_wet"].to_numpy()
    #         hb   = gaussian_half_band(tw_b, base_hb, extra_hb, sigma)
    #         in_band = (p_b > 0.5 - hb) & (p_b < 0.5 + hb)
    #         abstain_rates.append(float(np.mean(in_band)))
    #     ax_b.plot(bin_centers, abstain_rates, ls, color=color, lw=2,
    #             marker="o", ms=5, label=split)

    # ax_b.axvspan(-1, 1, alpha=0.08, color="orange")
    # ax_b.axvline(0, color="black", lw=0.8, alpha=0.4)
    # ax_b.set_xticks(range(len(bin_labels)))
    # ax_b.set_xticklabels(bin_labels, fontsize=8)
    # ax_b.set(xlabel="T_wet bin (°C)",
    #         ylabel="Fraction of predictions in uncertainty band",
    #         ylim=(0, 1.0),
    #         title="B — Abstention rate by temperature\n"
    #             "(fraction where model declines to commit, val+test)")
    # ax_b.legend(fontsize=9)

    # ── Story 5 ───────────────────────────────────────────────────────────────
    # Near-freezing precision/recall profiles — T_wet and T_air
    # Zoomed to [-4, 4] to focus on the transition zone
    NF_BIN_EDGES = np.arange(-4, 5, 1)  # 1°C bins from -4 to +4

    def _nf_profile(df_f, p_cal_f, y_true_f, bin_edges, temp_col):
        """Same logic as _twet_profile_binary but for any temp column, no band needed —
        pure binary at 0.5 threshold for the near-freeze story."""
        records = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            t_vals = df_f[temp_col].to_numpy()
            mask   = (t_vals >= lo) & (t_vals < hi)
            pure   = np.isin(y_true_f, [SNOW_CODE, RAIN_CODE])
            m      = mask & pure
            if m.sum() < 5:
                continue
            yt = y_true_f[m]
            yp = np.where(p_cal_f[m] >= 0.5, SNOW_CODE, RAIN_CODE)

            def _prf(code):
                tp = np.sum((yt == code) & (yp == code))
                fp = np.sum((yt != code) & (yp == code))
                fn = np.sum((yt == code) & (yp != code))
                pr = tp / (tp + fp) if (tp + fp) else np.nan
                rc = tp / (tp + fn) if (tp + fn) else np.nan
                return pr, rc

            snow_pr, snow_rc = _prf(SNOW_CODE)
            rain_pr, rain_rc = _prf(RAIN_CODE)
            n_mix_in_bin = int(np.sum(mask & ~pure))

            records.append({
                "t_mid":      (lo + hi) / 2,
                "n_pure":     int(m.sum()),
                "n_mix":      n_mix_in_bin,
                "prec_snow":  snow_pr,
                "rec_snow":   snow_rc,
                "prec_rain":  rain_pr,
                "rec_rain":   rain_rc,
            })
        return pd.DataFrame(records)

    def _plot_nf_profile(prof_val, prof_test, temp_label, out_path, name, bin_edges):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{name} — Story 5: Near-freezing precision/recall vs {temp_label}\n"
                    "(binary 0.5 threshold, observer-reported mix excluded)",
                    fontsize=12)

        panel_specs = [
            ("prec_snow", "rec_snow", "Snow", PHASE_COLORS["snow"]),
            ("prec_rain", "rec_rain", "Rain", PHASE_COLORS["rain"]),
        ]

        for ax, (prec_col, rec_col, label, color) in zip(axes, panel_specs):
            for prof, split, prec_ls, rec_ls in [
                (prof_val,  "Val",  "--", ":"),
                (prof_test, "Test", "-",  "-."),
            ]:
                if prof.empty:
                    continue
                valid_p = prof[prec_col].notna()
                valid_r = prof[rec_col].notna()
                ax.plot(prof.loc[valid_p, "t_mid"], prof.loc[valid_p, prec_col],
                        prec_ls, color=color, lw=2, marker="o", ms=5,
                        label=f"{split} precision")
                ax.plot(prof.loc[valid_r, "t_mid"], prof.loc[valid_r, rec_col],
                        rec_ls, color=color, lw=2, marker="s", ms=5,
                        label=f"{split} recall")

            # Sample count bar along bottom (secondary axis)
            ax2 = ax.twinx()
            if not prof_test.empty:
                ax2.bar(prof_test["t_mid"], prof_test["n_pure"],
                        width=0.7, color="lightgrey", alpha=0.4, zorder=0,
                        label="n pure (test)")
                ax2.bar(prof_test["t_mid"], prof_test["n_mix"],
                        width=0.7, bottom=prof_test["n_pure"],
                        color="#e377c2", alpha=0.25, zorder=0,
                        label="n mix excluded (test)")
            ax2.set(ylabel="Event count (test)", ylim=(0, prof_test["n_pure"].max() * 4
                                                        if not prof_test.empty else 100))
            ax2.yaxis.label.set_fontsize(9)
            ax2.tick_params(labelsize=8)

            ax.axvspan(-1, 1, alpha=0.08, color="orange", zorder=1)
            ax.axvline(0, color="black", lw=0.8, alpha=0.4, zorder=1)
            ax.set(xlabel=f"{temp_label} (°C)",
                ylabel=f"{label} precision / recall",
                title=f"{label}",
                xlim=(bin_edges[0], bin_edges[-1]),
                ylim=(0, 1.05))
            ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # T_wet profiles
    nf_val_wet  = _nf_profile(val_full_df,  p_valf_cal,  y_val_full_phase,
                            NF_BIN_EDGES, temp_col="temp_wet")
    nf_test_wet = _nf_profile(test_full_df, p_testf_cal, y_test_full_phase,
                            NF_BIN_EDGES, temp_col="temp_wet")
    _plot_nf_profile(nf_val_wet, nf_test_wet, "T_wet",
                    graphics_dir / "story5_nearfreeze_twet.png", name, NF_BIN_EDGES)

    # T_air profiles
    if "temp_air" in val_full_df.columns:
        nf_val_air  = _nf_profile(val_full_df,  p_valf_cal,  y_val_full_phase,
                                NF_BIN_EDGES, temp_col="temp_air")
        nf_test_air = _nf_profile(test_full_df, p_testf_cal, y_test_full_phase,
                                NF_BIN_EDGES, temp_col="temp_air")
        _plot_nf_profile(nf_val_air, nf_test_air, "T_air",
                        graphics_dir / "story5_nearfreeze_tair.png", name, NF_BIN_EDGES)

    print(f"  Stories saved to: {graphics_dir}")

# =============================================================================
# 4.  ONE-TIME DATA LOAD
# =============================================================================

def load_and_sync_datasets():
    print("\n" + "="*60)
    print("ONE-TIME: Opening and time-syncing datasets")
    print("="*60)
    ds_interp    = xr.open_dataset(PATHS[INTERP_TYPE]["interp_grid"])
    ds_imerg     = xr.open_dataset(PATHS["imerg"])
    df_loocv_raw = pd.read_parquet(PATHS[INTERP_TYPE]["mros_loocv"])

    ds_interp = ds_interp.assign_coords(time=pd.to_datetime(ds_interp.time.values).floor("h"))
    ds_imerg  = ds_imerg.assign_coords(time=pd.to_datetime(ds_imerg.time.values).floor("h"))

    common_times = np.intersect1d(ds_interp.time.values, ds_imerg.time.values)
    print(f"Common timesteps: {len(common_times)}")

    ds_interp = ds_interp.sel(time=common_times)
    ds_imerg  = ds_imerg.sel(time=common_times)
    if ds_imerg.y.values[0] > ds_imerg.y.values[-1]:
        ds_imerg = ds_imerg.isel(y=slice(None, None, -1))

    return ds_interp, ds_imerg, df_loocv_raw, common_times


# =============================================================================
# 5.  ONE-TIME SAMPLING
# =============================================================================

def get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times) -> pd.DataFrame:
    """
    Sample the full-feature predictor cube to MRoS points once and cache to
    parquet.  All ablation runs select columns from this cached table —
    the gridded resampling never runs again after the first call.
    """
    cache_path = SETUP_DIR / "ml_input_points_split.parquet"

    if cache_path.exists():
        print(f"\nONE-TIME SAMPLING: Cache found — loading\n  {cache_path}")
        master_df = pd.read_parquet(cache_path)
        print(f"  Shape: {master_df.shape}")
        return master_df

    print("\nONE-TIME SAMPLING: No cache found — sampling predictor cube now...")
    print("  (Runs once; cached for all subsequent ablation experiments)")

    ds_pred_full = build_predictor_cube(
        ds_interp=ds_interp, ds_imerg=ds_imerg,
        use_temp_air=True, use_temp_dew=True, use_temp_wet=True,
        use_rh=True, use_elev=True, use_imerg_plp=True,
    )
    loocv_df = prep_loocv_table(df_loocv_raw, ds_interp=ds_pred_full)
    loocv_df = loocv_df[loocv_df["time"].isin(pd.to_datetime(common_times))].copy()
    print(f"  LOOCV rows after time intersection: {len(loocv_df)}")

    present_grid = [f for f in ALL_GRID_FEATURES if f in ds_pred_full.data_vars]
    t0 = time.time()
    master_df = sample_predictor_cube_to_points_batched(
        loocv_df, ds_pred=ds_pred_full, predictor_vars=present_grid, verbose=True,
    )
    print(f"  Sampling done in {(time.time()-t0)/60:.1f} min")

    master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
    master_df = master_df.dropna(subset=present_grid + ["phase_full"]).copy()
    master_df["phase_full"] = master_df["phase_full"].astype(int)
    master_df.to_parquet(cache_path, index=False)
    print(f"  Cached to: {cache_path}")
    return master_df


# =============================================================================
# 6.  SHARED TRAIN / VAL / TEST SPLIT
# =============================================================================

def make_split(master_df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 split, computed once and shared across all runs."""
    train_df, valtest_df = train_test_split(
        master_df, test_size=1.0 - TRAIN_FRAC,
        random_state=RANDOM_SEED, stratify=master_df["phase_full"],
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=0.5,
        random_state=RANDOM_SEED, stratify=valtest_df["phase_full"],
    )
    for df, label in [(train_df,"train"), (val_df,"val"), (test_df,"test")]:
        df["split"] = label
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"\nSplit: { {s: int((split_df['split']==s).sum()) for s in ['train','val','test']} }")
    return split_df


# =============================================================================
# 7.  run_experiment()
# =============================================================================

def build_feature_list(cfg: dict) -> list[str]:
    grid = [n for flag, n in [
        (cfg.get("use_temp_air",  BASE_FEATURE_CONFIG["use_temp_air"]),  "temp_air"),
        (cfg.get("use_temp_dew",  BASE_FEATURE_CONFIG["use_temp_dew"]),  "temp_dew"),
        (cfg.get("use_temp_wet",  BASE_FEATURE_CONFIG["use_temp_wet"]),  "temp_wet"),
        (cfg.get("use_rh",        BASE_FEATURE_CONFIG["use_rh"]),        "rh"),
        (cfg.get("use_imerg_plp", BASE_FEATURE_CONFIG["use_imerg_plp"]), "imerg_plp"),
        (cfg.get("use_elev",      BASE_FEATURE_CONFIG["use_elev"]),      "elev"),
    ] if flag]
    if cfg.get("use_mros_loocv", BASE_FEATURE_CONFIG["use_mros_loocv"]):
        grid.extend(ALL_LOOCV_FEATURES)
    return grid


def run_experiment(cfg: dict, split_df: pd.DataFrame, out_dir: Path) -> dict:
    name = cfg.get("name", "unnamed")
    print(f"\n{'='*62}\n  EXPERIMENT: {name}\n{'='*62}")

    out_dir.mkdir(parents=True, exist_ok=True)
    graphics_dir = out_dir / "graphics"
    graphics_dir.mkdir(exist_ok=True)

    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=2)

    FEATURES = build_feature_list(cfg)

    missing = [f for f in FEATURES if f not in split_df.columns]
    if missing:
        print(f"  SKIP — missing features: {missing}")
        return {"name": name, "status": "skipped", "missing_features": missing}

    TARGET_FULL = "phase_full"
    train_full  = split_df[split_df["split"] == "train"].copy()
    val_full    = split_df[split_df["split"] == "val"].copy()
    test_full   = split_df[split_df["split"] == "test"].copy()

    pure = [SNOW_CODE, RAIN_CODE]
    train_fit = train_full[train_full[TARGET_FULL].isin(pure)].copy()
    val_fit   = val_full[val_full[TARGET_FULL].isin(pure)].copy()
    test_fit  = test_full[test_full[TARGET_FULL].isin(pure)].copy()

    X_train     = train_fit[FEATURES]
    X_val       = val_fit[FEATURES]
    X_test      = test_fit[FEATURES]
    X_val_full  = val_full[FEATURES]
    X_test_full = test_full[FEATURES]

    y_train = train_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)
    y_val   = val_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)
    y_test  = test_fit[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)

    y_val_full_phase  = val_full[TARGET_FULL].astype(int).to_numpy()
    y_test_full_phase = test_full[TARGET_FULL].astype(int).to_numpy()
    y_val_bin         = y_val.to_numpy()
    y_test_bin        = y_test.to_numpy()

    # ── scale_pos_weight sweep ────────────────────────────────────────────────
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    best_spw, spw_df = sweep_scale_pos_weight(
        X_train, y_train, X_val, y_val,
        SCALE_POS_WEIGHT_GRID, BASE_XGB_PARAMS,
        probe_rounds=350, early_stopping=30, random_seed=RANDOM_SEED,
    )
    spw_df.to_csv(out_dir / "scale_pos_weight_sweep.csv", index=False)
    print(f"  Best scale_pos_weight: {best_spw}")

    # ── Full model fit ────────────────────────────────────────────────────────
    params   = {**BASE_XGB_PARAMS, "scale_pos_weight": best_spw, "seed": RANDOM_SEED}
    dtrain   = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval_dm  = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURES)
    evals_result = {}
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain,"train"),(dval_dm,"val")], evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=200,
    )
    booster.save_model(out_dir / "xgb_binary_phase_model.bin")
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evals_result["train"]["logloss"], label="train")
    ax.plot(evals_result["val"]["logloss"],   label="val")
    ax.axvline(booster.best_iteration, ls="--", alpha=0.6, label=f"best={booster.best_iteration}")
    ax.set(xlabel="Iteration", ylabel="Logloss", title=f"{name} — training curve")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(graphics_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Beta calibration ──────────────────────────────────────────────────────
    trainval_pure = pd.concat([train_fit, val_fit]).reset_index(drop=True)
    tw_tv         = trainval_pure["temp_wet"].values
    p_raw_tv      = booster.predict(xgb.DMatrix(trainval_pure[FEATURES], feature_names=FEATURES))
    y_tv          = trainval_pure[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)

    clear_mask_tv = np.abs(tw_tv) > CLEAR_PHASE_TWET_C
    cal_clear = BetaCalibration(parameters="abm")
    cal_clear.fit(p_raw_tv[clear_mask_tv].reshape(-1,1), y_tv.values[clear_mask_tv])
    cal_nf = BetaCalibration(parameters="ab")
    cal_nf.fit(p_raw_tv[~clear_mask_tv].reshape(-1,1), y_tv.values[~clear_mask_tv])

    with open(out_dir / "beta_calibration_models.pkl", "wb") as f:
        pickle.dump({"method":"beta_calibration","clear_phase":cal_clear,
                     "near_freezing":cal_nf,"clear_phase_twet_threshold":CLEAR_PHASE_TWET_C}, f)

    def calibrate(p_raw, twet):
        p_raw = np.asarray(p_raw, float); twet = np.asarray(twet, float)
        out   = np.empty_like(p_raw)
        nf    = np.abs(twet) <= CLEAR_PHASE_TWET_C
        if (~nf).any(): out[~nf] = cal_clear.predict(p_raw[~nf].reshape(-1,1))
        if nf.any():    out[nf]  = cal_nf.predict(p_raw[nf].reshape(-1,1))
        return np.clip(out, 1e-4, 1-1e-4)

    # ── Probabilities ─────────────────────────────────────────────────────────
    p_val_raw   = booster.predict(xgb.DMatrix(X_val,       feature_names=FEATURES))
    p_test_raw  = booster.predict(xgb.DMatrix(X_test,      feature_names=FEATURES))
    p_valf_raw  = booster.predict(xgb.DMatrix(X_val_full,  feature_names=FEATURES))
    p_testf_raw = booster.predict(xgb.DMatrix(X_test_full, feature_names=FEATURES))

    p_val_cal   = calibrate(p_val_raw,   val_fit["temp_wet"].to_numpy())
    p_test_cal  = calibrate(p_test_raw,  test_fit["temp_wet"].to_numpy())
    p_valf_cal  = calibrate(p_valf_raw,  val_full["temp_wet"].to_numpy())
    p_testf_cal = calibrate(p_testf_raw, test_full["temp_wet"].to_numpy())

    # ── Gaussian half-band ────────────────────────────────────────────────────
    thr = optimize_threshold_params(p_valf_cal, val_full["temp_wet"].to_numpy(), y_val_full_phase)
    BASE_HB  = thr["base_half_band"]
    EXTRA_HB = thr["extra_half_band"]
    SIGMA    = thr["sigma"]

    def classify(p_snow, twet):
        return classify_phase_gaussian_band(p_snow, twet, BASE_HB, EXTRA_HB, SIGMA)

    pred_val_full  = classify(p_valf_cal,  val_full["temp_wet"].to_numpy())
    pred_test_full = classify(p_testf_cal, test_full["temp_wet"].to_numpy())
    pred_val_bin05  = np.where(p_val_cal  >= 0.5, SNOW_CODE, RAIN_CODE)
    pred_test_bin05 = np.where(p_test_cal >= 0.5, SNOW_CODE, RAIN_CODE)

    # ── Score distribution plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, p_raw, p_cal, split_label in [
        (axes[0], p_val_raw,  p_val_cal,  "Val"),
        (axes[1], p_test_raw, p_test_cal, "Test"),
    ]:
        ax.hist(p_raw, bins=40, alpha=0.55, color="grey",    label="raw",        density=True)
        ax.hist(p_cal, bins=40, alpha=0.55, color="#3a86ff", label="calibrated", density=True)
        ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set(xlabel="p(snow)", ylabel="Density", title=f"{name} — {split_label} score dist.")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graphics_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _safe(fn, *a, **kw):
        try:    return round(float(fn(*a, **kw)), 4)
        except: return None

    metrics = {
        "name": name, "features": FEATURES, "n_features": len(FEATURES),
        "best_iteration": int(booster.best_iteration), "best_spw": float(best_spw),
        "band_base_hb": float(BASE_HB), "band_extra_hb": float(EXTRA_HB), "band_sigma": float(SIGMA),
        # probability
        "val_roc_auc_cal":  _safe(roc_auc_score, y_val_bin,  p_val_cal),
        "test_roc_auc_cal": _safe(roc_auc_score, y_test_bin, p_test_cal),
        "val_brier_cal":    _safe(brier_score_loss, y_val_bin,  p_val_cal),
        "test_brier_cal":   _safe(brier_score_loss, y_test_bin, p_test_cal),
        "val_logloss_cal":  _safe(log_loss, y_val_bin,  p_val_cal, labels=[0,1]),
        "test_logloss_cal": _safe(log_loss, y_test_bin, p_test_cal, labels=[0,1]),
        "val_ece":          expected_calibration_error(y_val_bin,  p_val_cal),
        "test_ece":         expected_calibration_error(y_test_bin, p_test_cal),
        # hard metrics
        "val_macro_f1_binary":      _safe(f1_score, val_fit[TARGET_FULL].to_numpy(), pred_val_bin05,  average="macro", zero_division=0),
        "test_macro_f1_binary":     _safe(f1_score, test_fit[TARGET_FULL].to_numpy(), pred_test_bin05, average="macro", zero_division=0),
        "val_macro_f1_3class":      _safe(f1_score, y_val_full_phase,  pred_val_full,  average="macro", zero_division=0),
        "test_macro_f1_3class":     _safe(f1_score, y_test_full_phase, pred_test_full, average="macro", zero_division=0),
        "val_balanced_acc_3class":  _safe(balanced_accuracy_score, y_val_full_phase,  pred_val_full),
        "test_balanced_acc_3class": _safe(balanced_accuracy_score, y_test_full_phase, pred_test_full),
        # mix capture
        "val_mix_capture":  float(np.mean(pred_val_full[y_val_full_phase   == MIX_CODE] == MIX_CODE)) if (y_val_full_phase  == MIX_CODE).any() else None,
        "test_mix_capture": float(np.mean(pred_test_full[y_test_full_phase == MIX_CODE] == MIX_CODE)) if (y_test_full_phase == MIX_CODE).any() else None,
        # confidence fractions
        "test_frac_pred_snow": round(float(np.mean(pred_test_full == SNOW_CODE)), 4),
        "test_frac_pred_rain": round(float(np.mean(pred_test_full == RAIN_CODE)), 4),
        "test_frac_pred_mix":  round(float(np.mean(pred_test_full == MIX_CODE)),  4),
        # near-freezing regime
        "status": "ok",
    }

    # near-freezing regime metrics
    for split_name, df_full, p_cal_full, y_full, df_fit, p_cal_fit, y_fit_bin in [
        ("val",  val_full,  p_valf_cal,  y_val_full_phase,  val_fit,  p_val_cal,  y_val_bin),
        ("test", test_full, p_testf_cal, y_test_full_phase, test_fit, p_test_cal, y_test_bin),
    ]:
        for regime in ["nearfreeze", "clearphase"]:
            mf_full = (np.abs(df_full["temp_wet"].to_numpy()) <= 2.0) if regime == "nearfreeze" \
                      else (np.abs(df_full["temp_wet"].to_numpy()) > 2.0)
            mf_fit  = (np.abs(df_fit["temp_wet"].to_numpy())  <= 2.0) if regime == "nearfreeze" \
                      else (np.abs(df_fit["temp_wet"].to_numpy())  > 2.0)
            if mf_full.sum() >= 5:
                # Binary near-freeze F1: pure labels only, 0.5 threshold
                pure_nf = mf_full & np.isin(y_full, [SNOW_CODE, RAIN_CODE])
                if pure_nf.sum() >= 5:
                    pred_bin_nf = np.where(p_cal_full[pure_nf] >= 0.5, SNOW_CODE, RAIN_CODE)
                    metrics[f"{split_name}_{regime}_macro_f1_binary"] = _safe(
                        f1_score, y_full[pure_nf], pred_bin_nf,
                        labels=[SNOW_CODE, RAIN_CODE], average="macro", zero_division=0)
                # Keep mix_capture as a separate band diagnostic
                if (y_full[mf_full] == MIX_CODE).any():
                    pred_band_nf = classify(p_cal_full[mf_full], df_full["temp_wet"].to_numpy()[mf_full])
                    metrics[f"{split_name}_{regime}_mix_capture"] = round(
                        float(np.mean(pred_band_nf[y_full[mf_full] == MIX_CODE] == MIX_CODE)), 4)
            if mf_fit.sum() >= 5:
                metrics[f"{split_name}_{regime}_roc_auc"] = _safe(
                    roc_auc_score, y_fit_bin[mf_fit], p_cal_fit[mf_fit])

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Feature importance ────────────────────────────────────────────────────
    imp_gain   = booster.get_score(importance_type="gain")
    imp_weight = booster.get_score(importance_type="weight")
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "gain":    [imp_gain.get(f,   0.0) for f in FEATURES],
        "weight":  [imp_weight.get(f, 0.0) for f in FEATURES],
    }).sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES)*0.5)))
    ax.barh(imp_df["feature"], imp_df["gain"], color="steelblue")
    ax.set(xlabel="Gain", title=f"{name} — XGBoost feature importance")
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(graphics_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("  Computing SHAP values...")

    # Build combined table (all splits, full phase set)
    df_all_full = pd.concat([
        train_full.assign(split="train"),
        val_full.assign(split="val"),
        test_full.assign(split="test"),
    ], ignore_index=True)

    p_all_raw = booster.predict(xgb.DMatrix(df_all_full[FEATURES], feature_names=FEATURES))
    p_all_cal = calibrate(p_all_raw, df_all_full["temp_wet"].to_numpy())
    pred_phase_all = classify(p_all_cal, df_all_full["temp_wet"].to_numpy())
    
    # Stratified background sample from train split
    train_rows = df_all_full[df_all_full["split"] == "train"]
    N_BG = 500
    background = (
        train_rows[FEATURES]
        .groupby(train_rows["phase_full"], group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), max(1, int(N_BG * len(g) / len(train_rows)))),
            random_state=RANDOM_SEED,
        ))
        .reset_index(drop=True)
    )

    explainer   = shap.TreeExplainer(booster, data=background,
                                      feature_perturbation="interventional",
                                      model_output="probability")
    shap_values = explainer.shap_values(df_all_full[FEATURES])

    # Build shap_df
    meta_cols = [c for c in ["time","x","y","phase_full","temp_wet","temp_air","temp_dew","elev"]
                if c in df_all_full.columns]
    shap_df = df_all_full[meta_cols].copy()
    shap_df["p_snow_cal"]                   = p_all_cal
    shap_df["p_snow_raw"]                   = p_all_raw
    shap_df["split"]      = df_all_full["split"].values
    shap_df["prediction_phase_uncertainty"] = pred_phase_all
    shap_df["phase_label"] = shap_df["prediction_phase_uncertainty"].map(
        {SNOW_CODE:"snow", RAIN_CODE:"rain", MIX_CODE:"mix"})
    shap_cols = [f"shap_{f}" for f in FEATURES]
    for i, feat in enumerate(FEATURES):
        shap_df[f"shap_{feat}"] = shap_values[:, i]

    shap_df.to_parquet(out_dir / "shap_values_all.parquet")

    # Run this immediately after run_experiment() returns, while
    low_psnow_mask = (p_all_cal < 0.05) & (df_all_full["phase_full"] == SNOW_CODE)
    low_psnow = df_all_full[low_psnow_mask].copy()
    low_psnow["p_snow_cal"] = p_all_cal[low_psnow_mask]

    print(f"n = {len(low_psnow)}  ({100*len(low_psnow)/len(df_all_full):.1f}% of all obs)")
    print("\nSplit breakdown:")
    print(low_psnow["split"].value_counts())

    print("\nThermodynamic fields:")
    print(low_psnow[["temp_wet", "temp_air", "temp_dew", "elev"]].describe().round(2))

    print("\nLOOCV predictors:")
    print(low_psnow[["mros_p_snow_loocv", "mros_p_rain_loocv", "mros_p_mix_loocv"]].describe().round(3))

    if "imerg_plp" in low_psnow.columns:
        print(f"\nIMERG PLP: mean={low_psnow['imerg_plp'].mean():.3f}, "
            f"median={low_psnow['imerg_plp'].median():.3f}")

    # Compare to the full snow population
    all_snow = df_all_full[df_all_full["phase_full"] == SNOW_CODE]
    print(f"\n--- Contrast: full snow population (n={len(all_snow)}) ---")
    print(all_snow[["temp_wet", "elev", "mros_p_snow_loocv", "mros_p_rain_loocv"]].describe().round(3))
    
    # Are these concentrated at specific stations / locations?
    print("Spatial clustering:")
    print(low_psnow[["x", "y", "elev"]].describe().round(1))

    # Are they concentrated in certain months / storm types?
    low_psnow["month"] = pd.to_datetime(low_psnow["time"]).dt.month
    print("\nMonthly distribution:")
    print(low_psnow["month"].value_counts().sort_index())

    # How does the LOOCV conflict look — is mros_p_rain_loocv consistently near 1?
    print("\nLOOCV conflict severity:")
    print((low_psnow["mros_p_rain_loocv"] > 0.8).sum(), 
        "of 42 have mros_p_rain_loocv > 0.8")
    print((low_psnow["mros_p_rain_loocv"] > 0.9).sum(),
        "of 42 have mros_p_rain_loocv > 0.9")

    # Is there a station_id column that could reveal if it's one or two observers?
    if "station_id" in low_psnow.columns:
        print("\nUnique stations:", low_psnow["station_id"].nunique())
        print(low_psnow["station_id"].value_counts())

    # ── SHAP plot 1: mean |SHAP| by phase (bar chart) ─────────────────────────
    phase_shap = (
        shap_df.groupby("phase_label")[shap_cols]
        .apply(lambda d: d.abs().mean())
        .T
        .rename(index=lambda c: c.replace("shap_", ""))
        .rename_axis("feature")
        .reset_index()
    )
    phase_shap["all"] = pd.DataFrame(np.abs(shap_values), columns=FEATURES).mean(axis=0).values
    phase_shap = phase_shap.sort_values("all", ascending=False)
    phase_shap.to_csv(out_dir / "shap_summary_by_phase.csv", index=False)

    phase_plot_cols = [c for c in ["snow","rain","mix"] if c in phase_shap.columns]
    x = np.arange(len(FEATURES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, color) in enumerate(zip(phase_plot_cols,
                                          [PHASE_COLORS["snow"], PHASE_COLORS["rain"], PHASE_COLORS["mix"]])):
        vals = phase_shap.set_index("feature").reindex(FEATURES)[col].values
        ax.bar(x + i*width, vals, width, label=col, color=color, alpha=0.85)
    ax.set_xticks(x + width); ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set(ylabel="Mean |SHAP|",
           title=f"{name} — feature importance by predicted phase")
    ax.legend(title="Predicted phase"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(graphics_dir / "shap_mean_by_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── SHAP plot 2: top-N features across T_wet bins (line plot) ─────────────
    shap_df["twet_bin"] = pd.cut(shap_df["temp_wet"], bins=TWET_BIN_EDGES,
                                  labels=TWET_BIN_LABELS, right=True)
    shap_by_bin = (
        shap_df.groupby("twet_bin", observed=True)[shap_cols]
        .apply(lambda d: d.abs().mean())
        .T
        .rename(index=lambda c: c.replace("shap_", ""))
        .rename_axis("feature")
    )
    shap_by_bin.to_csv(out_dir / "shap_by_wetbulb_bin.csv")

    # Rank by near-freezing importance
    nf_mask_shap = shap_df["twet_bin"].isin(NF_BIN_LABELS)
    top_feats = (
        shap_df[nf_mask_shap][shap_cols].abs().mean()
        .rename(index=lambda c: c.replace("shap_", ""))
        .nlargest(TOP_N_SHAP_FEATURES).index.tolist()
    )
    valid_bins = [b for b in TWET_BIN_LABELS if b in shap_by_bin.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    for feat, color in zip(top_feats, plt.cm.tab10(np.linspace(0, 0.9, len(top_feats)))):
        if feat not in shap_by_bin.index:
            continue
        ax.plot(valid_bins, shap_by_bin.loc[feat, valid_bins].values.astype(float),
                marker="o", label=feat, color=color, linewidth=1.8)
    nf_idx = [valid_bins.index(b) for b in NF_BIN_LABELS if b in valid_bins]
    if nf_idx:
        ax.axvspan(min(nf_idx)-0.5, max(nf_idx)+0.5, alpha=0.10,
                   color="steelblue", label="|T_wet| ≤ 2°C zone")
    ax.set(xlabel="Wet-bulb temperature bin (°C)", ylabel="Mean |SHAP|",
           title=f"{name} — top {TOP_N_SHAP_FEATURES} features across T_wet bins")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right"); fig.tight_layout()
    fig.savefig(graphics_dir / "shap_wetbulb_lineplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def plot_calibration_detail(y_true_bin, p_cal, name, out_path, n_bins=15):
        """
        Three-panel calibration deep-dive:
        Left:   reliability diagram with bin-level ECE contribution
        Centre: histogram of predicted probabilities (sharpness)
        Right:  ECE contribution per bin (which bins hurt most)
        """
        bins      = np.linspace(0, 1, n_bins + 1)
        bin_accs  = []
        bin_confs = []
        bin_ns    = []
        bin_eces  = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_cal >= lo) & (p_cal < hi)
            n    = mask.sum()
            if n == 0:
                bin_accs.append(np.nan); bin_confs.append((lo+hi)/2)
                bin_ns.append(0);        bin_eces.append(0.0)
                continue
            acc  = float(np.mean((p_cal[mask] >= 0.5).astype(int) == y_true_bin[mask]))
            conf = float(np.mean(p_cal[mask]))
            bin_accs.append(acc); bin_confs.append(conf)
            bin_ns.append(n);     bin_eces.append(n * abs(acc - conf) / len(y_true_bin))

        bin_mids = [(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{name} — calibration deep-dive (test set)", fontsize=12)

        # Panel 1: reliability diagram, points sized by n
        ax = axes[0]
        ax.plot([0,1],[0,1],"--", color="grey", lw=1.2, alpha=0.6, label="Perfect")
        sizes = [max(20, n/max(bin_ns)*400) for n in bin_ns]
        sc = ax.scatter(bin_confs, bin_accs, s=sizes, c=bin_eces,
                        cmap="YlOrRd", vmin=0, vmax=max(bin_eces or [0.01]),
                        zorder=3, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="ECE contribution")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean predicted p(snow)",
            ylabel="Observed fraction snow", title="Reliability diagram\n(dot size = n obs, color = ECE contribution)")
        ax.grid(alpha=0.25)

        # Panel 2: sharpness histogram
        ax = axes[1]
        ax.hist(p_cal, bins=30, color="#3a86ff", alpha=0.7, edgecolor="none")
        ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
        # shade near-freezing band if band params are accessible via closure
        ax.set(xlabel="Calibrated p(snow)", ylabel="Count",
            title="Sharpness (predicted probability distribution)")
        ax.grid(alpha=0.25)

        # Panel 3: ECE contribution by bin
        ax = axes[2]
        bar_colors = plt.cm.YlOrRd(
            np.array(bin_eces) / max(max(bin_eces), 1e-6)
        )
        ax.bar(bin_mids, bin_eces, width=(bins[1]-bins[0])*0.85,
            color=bar_colors, edgecolor="none")
        ax.set(xlabel="Predicted p(snow) bin", ylabel="ECE contribution",
            title="ECE contribution by probability bin\n(which bins drive miscalibration)")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_calibration_detail(
        y_test_bin, p_test_cal, name,
        graphics_dir / "calibration_detail.png",
    )

    # Store top SHAP features in metrics for comparison table
    metrics["shap_top_feature_overall"]    = str(phase_shap.iloc[0]["feature"]) if not phase_shap.empty else ""
    metrics["shap_top_feature_nearfreeze"] = str(top_feats[0]) if top_feats else ""

    print(f"  test ROC AUC (cal):      {metrics['test_roc_auc_cal']}")
    print(f"  test macro F1:           {metrics['test_macro_f1_binary']}")
    print(f"  test mix capture:        {metrics['test_mix_capture']}")
    print(f"  test ECE:                {metrics['test_ece']}")
    print(f"  SHAP top (overall):      {metrics['shap_top_feature_overall']}")
    print(f"  SHAP top (near-freeze):  {metrics['shap_top_feature_nearfreeze']}")
    print(f"  Saved to: {out_dir}")

    # ── Per-experiment story plots ──────────────────────
    # Attach p_snow_cal to full-split DataFrames so story helper is self-contained
    _val_full_story  = val_full.copy();  _val_full_story["p_snow_cal"]  = p_valf_cal
    _test_full_story = test_full.copy(); _test_full_story["p_snow_cal"] = p_testf_cal
    plot_per_experiment_stories(
        name           = name,
        graphics_dir   = graphics_dir,
        y_val_fit_phase   = val_fit[TARGET_FULL].to_numpy(),
        pred_val_bin05    = pred_val_bin05,
        y_test_fit_phase  = test_fit[TARGET_FULL].to_numpy(),
        pred_test_bin05   = pred_test_bin05,
        y_val_full_phase  = y_val_full_phase,
        pred_val_full     = pred_val_full,
        y_test_full_phase = y_test_full_phase,
        pred_test_full    = pred_test_full,
        y_val_bin   = y_val_bin,
        p_val_cal   = p_val_cal,
        y_test_bin  = y_test_bin,
        p_test_cal  = p_test_cal,
        p_val_raw   = p_val_raw,
        p_test_raw  = p_test_raw,
        p_valf_cal   = p_valf_cal, 
        p_testf_cal  = p_testf_cal,
        val_full_df  = _val_full_story,
        test_full_df = _test_full_story,
        base_hb      = BASE_HB,
        extra_hb     = EXTRA_HB,
        sigma        = SIGMA,
    )

    return metrics

# =============================================================================
# 8.  ABLATION LOOP + CROSS-EXPERIMENT PLOTS
# =============================================================================

def load_all_metrics(ablation_root: Path) -> list[dict]:
    """
    Scan all subfolders of ablation_root for metrics_summary.json
    and return a list of metric dicts.  Subfolders without a
    metrics_summary.json are silently skipped.
    """
    all_metrics = []
    for run_dir in sorted(ablation_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        all_metrics.append(m)
        print(f"  Loaded: {run_dir.name}  (status={m.get('status','?')})")
    return all_metrics


def save_cross_experiment_plots(all_metrics: list[dict], ablation_root: Path) -> None:
    """
    Generate and save all cross-experiment comparison outputs.
    Can be called after a fresh run or standalone via --replot.
    """
    comparison_cols = [
    # Primary: binary performance
    "name", "n_features", "status",
    "test_roc_auc_cal", "test_macro_f1_binary",
    "test_nearfreeze_roc_auc", "test_nearfreeze_macro_f1_binary",
    "test_clearphase_macro_f1_binary",
    # Calibration
    "test_ece", "test_brier_cal", "test_logloss_cal",
    # Uncertainty band diagnostics
    "test_mix_capture", "test_frac_pred_mix", "test_nearfreeze_mix_capture",
    # 3-class diagnostic only
    "test_macro_f1_3class", "test_balanced_acc_3class",
    # Validation
    "val_roc_auc_cal", "val_macro_f1_binary",
    "shap_top_feature_overall", "shap_top_feature_nearfreeze",
    ]
    
    comparison_df = pd.DataFrame(all_metrics)
    present_cols  = [c for c in comparison_cols if c in comparison_df.columns]
    comparison_df[present_cols].to_csv(ablation_root / "ablation_comparison.csv", index=False)
    print(f"\nComparison table saved ({len(comparison_df)} runs).")
    print(comparison_df[present_cols].to_string(index=False))

    ok = comparison_df[comparison_df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful runs found — skipping plots.")
        return

    # ── Delta plot: F1 and ROC AUC ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(ok) * 0.45)))
    fig.suptitle("Ablation comparison — delta from baseline_full", fontsize=13)

    baseline_row = ok[ok["name"] == "baseline_full"]

    for ax, col, title in [
        (axes[0], "test_macro_f1_binary", "Δ Test macro F1 (binary)"),
        (axes[1], "test_roc_auc_cal",     "Δ Test ROC AUC (calibrated)"),
    ]:
        if col not in ok.columns or baseline_row.empty:
            continue
        baseline_val = float(baseline_row[col].iloc[0])
        ok_delta = ok.copy()
        ok_delta["delta"] = ok_delta[col] - baseline_val
        ok_delta = ok_delta.sort_values("delta", ascending=True)

        colors = ["#2dc653" if r["name"] == "baseline_full" else
                "#d62728" if r["delta"] < 0 else "#3a86ff"
                for _, r in ok_delta.iterrows()]

        ax.barh(ok_delta["name"], ok_delta["delta"], color=colors)
        ax.axvline(0, color="black", lw=1.0)
        ax.set(xlabel=title, title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Annotate absolute values next to bars
        for _, row in ok_delta.iterrows():
            abs_val = row[col]
            if abs_val is not None:
                ax.text(row["delta"] + (0.001 if row["delta"] >= 0 else -0.001),
                        row["name"], f"{abs_val:.3f}",
                        va="center",
                        ha="left" if row["delta"] >= 0 else "right",
                        fontsize=8, color="dimgrey")

    plt.tight_layout()
    fig.savefig(ablation_root / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ECE bar chart — delta from baseline ──────────────────────────────────
    if "test_ece" in ok.columns:
        baseline_ece = float(ok[ok["name"] == "baseline_full"]["test_ece"].iloc[0]) \
                    if not ok[ok["name"] == "baseline_full"].empty else None

        if baseline_ece is not None:
            ok_ece = ok.copy()
            ok_ece["delta_ece"] = ok_ece["test_ece"] - baseline_ece
            # Sort so largest ECE reduction (most improved) is at top
            ok_ece = ok_ece.sort_values("delta_ece", ascending=False)

            fig, ax = plt.subplots(figsize=(8, max(4, len(ok) * 0.45)))

            colors = ["#2dc653" if r["name"] == "baseline_full" else
                    "#3a86ff" if r["delta_ece"] < 0 else   # better calibration = blue
                    "#d62728"                                # worse calibration = red
                    for _, r in ok_ece.iterrows()]

            ax.barh(ok_ece["name"], ok_ece["delta_ece"], color=colors)
            ax.axvline(0, color="black", lw=1.0)
            ax.set(xlabel="Δ ECE from baseline (negative = better calibration)",
                title="Ablation — test ECE delta from baseline_full\n"
                        "(blue = improved calibration, red = worse)")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

            # Annotate absolute ECE values
            for _, row in ok_ece.iterrows():
                ax.text(row["delta_ece"] + (0.001 if row["delta_ece"] >= 0 else -0.001),
                        row["name"], f"{row['test_ece']:.3f}",
                        va="center",
                        ha="left" if row["delta_ece"] >= 0 else "right",
                        fontsize=8, color="dimgrey")

            fig.tight_layout()
            fig.savefig(ablation_root / "ablation_calibration_ece.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ── Mix capture vs F1 scatter ─────────────────────────────────────────────
    if "test_mix_capture" in ok.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_ok = ok.dropna(subset=["test_mix_capture", "test_macro_f1_binary"])
        color_col  = "test_nearfreeze_roc_auc" if "test_nearfreeze_roc_auc" in scatter_ok.columns else None
        sc = ax.scatter(
            scatter_ok["test_mix_capture"],
            scatter_ok["test_macro_f1_binary"],
            c=scatter_ok[color_col] if color_col else "steelblue",
            cmap="viridis", s=80, zorder=3,
        )
        if color_col:
            plt.colorbar(sc, ax=ax, label="Near-freeze ROC AUC")
        for _, row in scatter_ok.iterrows():
            ax.annotate(row["name"], (row["test_mix_capture"], row["test_macro_f1_binary"]),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.set(xlabel="Observer-mix band capture rate\n(uncertainty diagnostic)",
            ylabel="Test macro F1 (binary)",
            title="Abstention band capture vs. binary F1\n"
                    "(color = near-freeze ROC AUC)")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(ablation_root / "ablation_mix_tradeoff.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Parallel coordinates ──────────────────────────────────────────────────
    radar_cols = [
        "test_roc_auc_cal",
        "test_macro_f1_binary",
        "test_nearfreeze_roc_auc",
        "test_mix_capture",
    ]
    plot_cols = [c for c in radar_cols if c in ok.columns and ok[c].notna().any()]
    if plot_cols:
        ok_norm = ok[["name"] + plot_cols].copy()
        for col in plot_cols:
            col_min = ok_norm[col].min(); col_max = ok_norm[col].max()
            if col_max > col_min:
                ok_norm[col] = (ok_norm[col] - col_min) / (col_max - col_min)
        # Readable axis labels
        label_map = {
            "test_roc_auc_cal":       "ROC AUC (binary)",
            "test_macro_f1_binary":   "Macro F1 (binary)",
            "test_nearfreeze_roc_auc": "Near-freeze ROC AUC",
            "test_mix_capture":       "Observer-mix band capture\n(uncertainty diagnostic)",
        }
        fig, ax = plt.subplots(figsize=(12, 5))
        parallel_coordinates(ok_norm, "name", colormap="tab20", ax=ax, alpha=0.75)
        ax.set_xticklabels([label_map.get(c, c) for c in plot_cols],
                        rotation=20, ha="right", fontsize=9)
        ax.set_title("Ablation — normalized metrics (parallel coordinates)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.25); fig.tight_layout()
        fig.savefig(ablation_root / "ablation_parallel_coords.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll cross-experiment plots saved to: {ablation_root}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--regions", nargs="+", choices=list(REGIONS),
                        default=list(REGIONS), help="Regions to run (default: all)")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated indices of ABLATION_CONFIGS to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the experiments without running them")
    parser.add_argument("--replot", action="store_true",
                        help="Skip training; rebuild the comparison plots from "
                             "the metrics already on disk")
    args = parser.parse_args()

    selected = [int(x) for x in args.configs.split(",") if x.strip().isdigit()]
    configs_to_run = ([ABLATION_CONFIGS[i] for i in selected if i < len(ABLATION_CONFIGS)]
                      if selected else ABLATION_CONFIGS)

    for region in args.regions:
        configure(region)

        if args.replot:
            all_metrics = load_all_metrics(ABLATION_ROOT)
            if not all_metrics:
                print(f"[{region}] no completed runs found in {ABLATION_ROOT}")
                continue
            save_cross_experiment_plots(all_metrics, ABLATION_ROOT)
            continue

        print(f"\n=== {region}: {len(configs_to_run)} experiment(s) -> {ABLATION_ROOT} ===")
        for i, cfg in enumerate(configs_to_run):
            print(f"  [{i:2d}] {cfg.get('name', 'unnamed')}")
        if args.dry_run:
            continue

        ds_interp, ds_imerg, df_loocv_raw, common_times = load_and_sync_datasets()
        master_df = get_master_df(ds_interp, ds_imerg, df_loocv_raw, common_times)
        split_df = make_split(master_df)

        all_metrics = []
        for cfg in configs_to_run:
            full_cfg = {**BASE_FEATURE_CONFIG, **cfg}
            name = full_cfg.get("name", "unnamed")
            try:
                metrics = run_experiment(full_cfg, split_df, ABLATION_ROOT / name)
            except Exception as exc:
                print(f"  ERROR in {name}: {exc}")
                metrics = {"name": name, "status": f"error: {exc}"}
            all_metrics.append(metrics)

        # Include any runs completed earlier so the comparison plots are complete.
        just_run = {m["name"] for m in all_metrics}
        merged = all_metrics + [m for m in load_all_metrics(ABLATION_ROOT)
                                if m["name"] not in just_run]
        save_cross_experiment_plots(merged, ABLATION_ROOT)


if __name__ == "__main__":
    main()
