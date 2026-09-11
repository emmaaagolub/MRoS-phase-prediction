"""SHAP attribution for the trained model.

Computes SHAP values for every observation, then aggregates them two ways:
by the predicted phase, and by wet-bulb temperature bin.

SHAP values are in probability units, which requires the interventional
attribution method and a background dataset. A stratified sample of the
training split is used as the background.

Input:  the model artifacts written by train_model.py
Output: shap_values_all.parquet, shap_summary_by_phase.csv,
        shap_by_wetbulb_bin.csv, shap_nearfreeze_vs_clearphase_ratio.csv
        and figures under graphics/
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES, INTERP_TYPE, MIX_CODE, RAIN_CODE, RANDOM_SEED, REGIONS, SNOW_CODE,
    classify_phase_gaussian_band, model_paths,
)
from train_model import apply_calibration  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import parse_region_args  # noqa: E402

PHASE_COLORS = {"snow": "#1f77b4", "rain": "#2ca02c", "mix": "#e377c2"}
PHASE_LABELS = {SNOW_CODE: "snow", RAIN_CODE: "rain", MIX_CODE: "mix"}

# One-degree wet-bulb bins with open-ended tails.
TWET_BIN_EDGES = [-np.inf, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, np.inf]
TWET_BIN_LABELS = ["<-6", "-6–-5", "-5–-4", "-4–-3", "-3–-2", "-2–-1", "-1–0",
                   "0–1", "1–2", "2–3", "3–4", "4–5", "5–6", ">6"]
NEAR_FREEZE_BINS = ["-2–-1", "-1–0", "0–1", "1–2"]
NEAR_FREEZE_THRESH = 2.0
TOP_N_FEATURES = 5
N_BACKGROUND = 500


def load_model_and_settings(paths):
    booster = xgb.Booster()
    booster.load_model(paths["model_dir"] / "xgb_binary_phase_model.bin")

    with open(paths["model_dir"] / "beta_calibration_models.pkl", "rb") as handle:
        calibrators = pickle.load(handle)
    with open(paths["model_dir"] / "calibration_summary.json") as handle:
        thresholds = json.load(handle)["uncertainty_thresholds"]

    band = (thresholds["base_half_band"], thresholds["extra_half_band"],
            thresholds["sigma"])
    return booster, calibrators, band


def load_all_observations(paths):
    """Train, validation and test rows in one table, with a split label."""
    split_df = pd.read_parquet(paths["split_table"])
    train = split_df[split_df["split"] == "train"].copy()

    val = pd.read_parquet(paths["model_dir"] / "val_full_uncertainty_predictions_combined.parquet")
    test = pd.read_parquet(paths["model_dir"] / "test_full_uncertainty_predictions_combined.parquet")
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)


def compute_shap_values(booster, df_all):
    """SHAP values in probability units, against a stratified training background."""
    train_rows = df_all[df_all["split"] == "train"]
    background = (train_rows[FEATURES]
                  .groupby(train_rows["phase_full"], group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), int(N_BACKGROUND * len(g) / len(train_rows))),
                      random_state=RANDOM_SEED))
                  .reset_index(drop=True))
    print(f"  background sample: {len(background)} rows")

    explainer = shap.TreeExplainer(booster, data=background,
                                   feature_perturbation="interventional",
                                   model_output="probability")
    return explainer.shap_values(df_all[FEATURES])


def build_shap_table(df_all, shap_values, p_cal, pred_phase):
    shap_df = df_all[["time", "x", "y", "phase_full", "temp_wet", "temp_air", "elev"]].copy()
    shap_df["split"] = df_all["split"].values
    shap_df["p_snow_cal"] = p_cal
    shap_df["prediction_phase_uncertainty"] = pred_phase
    shap_df["phase_label"] = shap_df["prediction_phase_uncertainty"].map(PHASE_LABELS)
    for i, feature in enumerate(FEATURES):
        shap_df[f"shap_{feature}"] = shap_values[:, i]
    return shap_df


def mean_abs_shap(frame, shap_cols, group_col):
    """Mean absolute SHAP per feature, grouped by a column, as feature x group."""
    return (frame.groupby(group_col, observed=True)[shap_cols]
                 .apply(lambda g: g.abs().mean())
                 .T
                 .rename(index=lambda c: c.replace("shap_", ""))
                 .rename_axis("feature"))


def plot_mean_shap_by_phase(summary, graphics_dir, interp_type):
    phase_cols = [c for c in ["snow", "rain", "mix"] if c in summary.columns]
    positions = np.arange(len(FEATURES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, phase in enumerate(phase_cols):
        values = summary.set_index("feature").loc[FEATURES, phase].values
        ax.bar(positions + i * width, values, width, label=phase,
               color=PHASE_COLORS[phase], alpha=0.85)

    ax.set_xticks(positions + width)
    ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set_ylabel("Mean |SHAP| (probability units)")
    ax.set_title("Feature importance by predicted phase")
    ax.legend(title="Predicted phase")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(graphics_dir / f"shap_mean_by_phase_{interp_type}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_shap_heatmap(data, title, savepath, vmax=None):
    data = data.loc[:, data.notna().any()]  # drop empty bins
    fig, ax = plt.subplots(
        figsize=(max(10, len(data.columns) * 0.7), len(FEATURES) * 0.55 + 1.5)
    )
    sns.heatmap(data, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax or data.max().max(),
                annot=True, fmt=".3", linewidths=0.4, linecolor="#cccccc",
                cbar_kws={"label": "Mean |SHAP| (probability units)"})
    ax.set(title=title, xlabel="Wet-bulb temperature bin (°C)", ylabel="Feature")

    # Shade the near-freezing columns.
    for col_i, label in enumerate(data.columns):
        if label in NEAR_FREEZE_BINS:
            ax.add_patch(plt.Rectangle((col_i, 0), 1, len(data), fill=True,
                                       color="steelblue", alpha=0.07, zorder=0))
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_features_across_bins(shap_df, shap_by_bin, shap_cols, graphics_dir):
    """Mean |SHAP| across wet-bulb bins for the top features near freezing."""
    near_freeze = shap_df["twet_bin"].isin(NEAR_FREEZE_BINS)
    top_features = (shap_df[near_freeze][shap_cols].abs().mean()
                    .rename(index=lambda c: c.replace("shap_", ""))
                    .nlargest(TOP_N_FEATURES).index.tolist())
    print(f"  top features near freezing: {top_features}")

    valid_bins = [b for b in TWET_BIN_LABELS if b in shap_by_bin.columns]
    colors = plt.cm.tab10(np.linspace(0, 0.9, TOP_N_FEATURES))

    fig, ax = plt.subplots(figsize=(12, 5))
    for feature, color in zip(top_features, colors):
        if feature not in shap_by_bin.index:
            continue
        ax.plot(valid_bins, shap_by_bin.loc[feature, valid_bins].values.astype(float),
                marker="o", label=feature, color=color, linewidth=1.8)

    nf_idx = [valid_bins.index(b) for b in NEAR_FREEZE_BINS if b in valid_bins]
    if nf_idx:
        ax.axvspan(min(nf_idx) - 0.5, max(nf_idx) + 0.5, alpha=0.10,
                   color="steelblue", label="|Twet| ≤ 2°C")

    ax.set(xlabel="Wet-bulb temperature bin (°C)",
           ylabel="Mean |SHAP| (probability units)",
           title=f"Top {TOP_N_FEATURES} features across the transition")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(graphics_dir / "shap_wetbulb_lineplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return top_features


def plot_beeswarm(mask, features_matrix, shap_values, title, savepath, max_display=10):
    if mask.sum() == 0:
        return
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values[mask], features=features_matrix[mask],
                      feature_names=FEATURES, max_display=max_display,
                      show=False, plot_size=None)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_region(region_id, interp_type=INTERP_TYPE):
    paths = model_paths(region_id, interp_type)
    model_dir, graphics_dir = paths["model_dir"], paths["graphics_dir"]
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    booster, calibrators, (base_hb, extra_hb, sigma) = load_model_and_settings(paths)
    df_all = load_all_observations(paths)

    twet = df_all["temp_wet"].to_numpy()
    p_raw = booster.predict(xgb.DMatrix(df_all[FEATURES], feature_names=FEATURES))
    p_cal = apply_calibration(p_raw, twet, calibrators)
    pred_phase = classify_phase_gaussian_band(p_cal, twet, base_hb, extra_hb, sigma)

    print("  computing SHAP values...")
    shap_values = compute_shap_values(booster, df_all)

    shap_df = build_shap_table(df_all, shap_values, p_cal, pred_phase)
    shap_df.to_parquet(model_dir / "shap_values_all.parquet")
    shap_cols = [f"shap_{f}" for f in FEATURES]

    # --- By predicted phase ------------------------------------------------
    summary = mean_abs_shap(shap_df, shap_cols, "phase_label").reset_index()
    summary["all"] = pd.DataFrame(np.abs(shap_values), columns=FEATURES).mean(axis=0).values
    summary = summary.sort_values("all", ascending=False)
    summary.to_csv(model_dir / "shap_summary_by_phase.csv", index=False)
    print(summary.to_string(index=False))
    plot_mean_shap_by_phase(summary, graphics_dir, interp_type)

    # --- By wet-bulb bin ---------------------------------------------------
    shap_df["twet_bin"] = pd.cut(shap_df["temp_wet"], bins=TWET_BIN_EDGES,
                                 labels=TWET_BIN_LABELS, right=True)
    shap_by_bin = mean_abs_shap(shap_df, shap_cols, "twet_bin")
    shap_by_bin.to_csv(model_dir / "shap_by_wetbulb_bin.csv")

    plot_shap_heatmap(shap_by_bin, "Mean |SHAP| by wet-bulb bin — all observations",
                      graphics_dir / "shap_wetbulb_heatmap_all.png")

    # Per-phase heatmaps share one colour scale.
    phase_tables = {}
    for phase in ["snow", "rain", "mix"]:
        subset = shap_df[shap_df["phase_label"] == phase]
        if subset.empty:
            continue
        phase_tables[phase] = mean_abs_shap(subset, shap_cols, "twet_bin")
    shared_vmax = max((t.max().max() for t in phase_tables.values()), default=None)
    for phase, table in phase_tables.items():
        plot_shap_heatmap(table, f"Mean |SHAP| by wet-bulb bin — predicted {phase}",
                          graphics_dir / f"shap_wetbulb_heatmap_{phase}.png",
                          vmax=shared_vmax)

    plot_top_features_across_bins(shap_df, shap_by_bin, shap_cols, graphics_dir)

    # --- Beeswarms by regime ----------------------------------------------
    features_matrix = df_all[FEATURES].values
    near_freeze = np.abs(twet) <= NEAR_FREEZE_THRESH
    clear_phase = ~near_freeze

    plot_beeswarm(near_freeze, features_matrix, shap_values,
                  f"Near-freezing (|Twet| ≤ {NEAR_FREEZE_THRESH}°C)",
                  graphics_dir / "shap_beeswarm_nearfreeze.png")
    plot_beeswarm(clear_phase, features_matrix, shap_values,
                  f"Clear phase (|Twet| > {NEAR_FREEZE_THRESH}°C)",
                  graphics_dir / "shap_beeswarm_clearphase.png")
    for phase in ["snow", "rain", "mix"]:
        mask = near_freeze & (shap_df["phase_label"].values == phase)
        plot_beeswarm(mask, features_matrix, shap_values,
                      f"Near-freezing, predicted {phase}",
                      graphics_dir / f"shap_beeswarm_nearfreeze_{phase}.png")

    # --- Which predictors matter disproportionately near freezing ----------
    mean_nf = pd.DataFrame(np.abs(shap_values[near_freeze]), columns=FEATURES).mean()
    mean_cp = pd.DataFrame(np.abs(shap_values[clear_phase]), columns=FEATURES).mean()
    ratio_df = pd.DataFrame({
        "mean_shap_near_freeze": mean_nf,
        "mean_shap_clear_phase": mean_cp,
        "ratio_nf_over_cp": mean_nf / mean_cp.replace(0, np.nan),
    }).sort_values("ratio_nf_over_cp", ascending=False)
    ratio_df.to_csv(model_dir / "shap_nearfreeze_vs_clearphase_ratio.csv")
    print("\nImportance ratio, near freezing over clear phase:")
    print(ratio_df.to_string())

    print(f"\nSHAP outputs written to {model_dir}")


def main(regions=None, interp_type=INTERP_TYPE):
    for region_id in regions or REGIONS:
        analyze_region(region_id, interp_type)


if __name__ == "__main__":
    main(parse_region_args())
