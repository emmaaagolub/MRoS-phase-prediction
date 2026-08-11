"""Step 10 — Train the precipitation-phase model.

The approach:

1. Split the observations 70 / 15 / 15 into train, validation and test, keeping
   the phase balance the same in each. The split is random rather than by storm.
   The relationship being learned is between an instantaneous atmospheric state
   and the phase that falls out of it, which is physically stable across storms,
   and the dataset is too small and too sparsely sampled for event blocking to
   leave usable pieces. This follows Jennings et al. (2025). The cost is that
   test scores may be mildly optimistic relative to a genuinely new season.

2. Fit XGBoost on the pure-phase observations only, as rain vs snow. Observed
   mix is held out of fitting rather than forced into a third class.

3. Balance the classes by sweeping scale_pos_weight and taking the value that
   brings snow and rain recall closest together.

4. Calibrate the raw probabilities with beta calibration (Kull et al. 2017),
   fitted separately for near-freezing and clear-phase conditions. Beta
   calibration is bounded away from 0 and 1 by construction. Isotonic
   regression was tried first and pushed most test predictions to exact 0 or 1
   whenever a score fell outside the range it was fitted on.

5. Derive the mix class from the calibrated probability using a band around 0.5
   that widens near freezing. The band's shape is chosen on the validation set.

Input:  the table from build_dataset.py
Output: outputs/model/<REGION>/
"""

import itertools
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from betacal import BetaCalibration
from scipy.stats import entropy
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, log_loss, recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASE_HALF_BAND_GRID, BASE_XGB_PARAMS, BINARY_LABEL_MAP, CLEAR_PHASE_TWET_C,
    EARLY_STOPPING_ROUNDS, EXTRA_HALF_BAND_GRID, FEATURES, FULL_CLASS_NAMES,
    INTERP_TYPE, MAX_TOTAL_HALF_BAND, MIN_NF_PURE_COVERAGE, MIN_PURE_COVERAGE,
    MIX_CAPTURE_WEIGHT, MIX_CODE, NUM_BOOST_ROUND, PURE_PHASE_CODES, RAIN_CODE,
    RANDOM_SEED, REGIONS, SCALE_POS_WEIGHT_GRID, SIGMA_GRID, SNOW_CODE,
    TARGET_FULL, TRAIN_FRAC, classify_phase_gaussian_band, gaussian_half_band,
    make_output_dirs, model_paths, transition_score_from_psnow,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import parse_region_args  # noqa: E402

PREDICTION_OUT_COLS = [
    "time", "x", "y", "phase_full", "prediction_binary05",
    "prediction_phase_uncertainty", "near_freezing_air_2C", "near_freezing_wet_2C",
    "p_snow_raw", "p_snow_cal", "p_rain_raw", "p_rain_cal",
    "transition_score", "entropy_binary", "raw_margin", "is_confident",
]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_observations(points_df, graphics_dir, interp_type):
    """Stratified 70 / 15 / 15 split, with a coverage check figure."""
    train_df, valtest_df = train_test_split(
        points_df, test_size=1.0 - TRAIN_FRAC, random_state=RANDOM_SEED,
        stratify=points_df[TARGET_FULL],
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=0.5, random_state=RANDOM_SEED,
        stratify=valtest_df[TARGET_FULL],
    )

    parts = []
    for frame, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        frame = frame.copy()
        frame["split"] = name
        parts.append(frame)

    split_df = pd.concat(parts, ignore_index=True)

    print("\nRows per split:")
    print(split_df["split"].value_counts(normalize=True).round(3).to_string())
    print("\nPhase balance by split:")
    print(split_df.groupby("split")[TARGET_FULL].value_counts(normalize=True)
                  .unstack(fill_value=0.0).round(3).to_string())

    plot_split_diagnostics(split_df, graphics_dir, interp_type)
    return split_df


def plot_split_diagnostics(split_df, graphics_dir, interp_type):
    """Check the splits cover the same phase, temperature and elevation range."""
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    (split_df.groupby("split")[TARGET_FULL].value_counts(normalize=True)
             .unstack(fill_value=0.0)
             .rename(columns={SNOW_CODE: "snow", RAIN_CODE: "rain", MIX_CODE: "mix"})
             .plot(kind="bar", ax=axes[0], color=["#1f77b4", "#2ca02c", "#e377c2"],
                   edgecolor="k", linewidth=0.5))
    axes[0].set(title="Phase balance by split", ylabel="Fraction", xlabel="")
    axes[0].tick_params(axis="x", rotation=0)

    for column, ax, label in [("temp_wet", axes[1], "Wet-bulb temperature (°C)"),
                              ("elev", axes[2], "Elevation (m)")]:
        for split, group in split_df.groupby("split"):
            ax.hist(group[column].dropna(), bins=30 if column == "temp_wet" else 20,
                    alpha=0.5, color=colors[split], label=split, density=True)
        ax.set(xlabel=label, ylabel="Density", title=f"{label} by split")
        ax.legend()

    fig.tight_layout()
    fig.savefig(graphics_dir / f"random_split_diagnostics_{interp_type}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

def sweep_scale_pos_weight(X_train, y_train, X_val, y_val, candidates,
                           probe_rounds=350, early_stopping=30):
    """Short training runs at each candidate weight, scored on recall balance."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

    records = []
    for weight in candidates:
        params = {**BASE_XGB_PARAMS, "scale_pos_weight": weight, "seed": RANDOM_SEED}
        probe = xgb.train(params, dtrain, num_boost_round=probe_rounds,
                          evals=[(dval, "val")], early_stopping_rounds=early_stopping,
                          verbose_eval=False)
        y_pred = (probe.predict(dval) >= 0.5).astype(int)

        snow_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        rain_recall = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        records.append({
            "scale_pos_weight": weight,
            "snow_recall": round(snow_recall, 4),
            "rain_recall": round(rain_recall, 4),
            "recall_gap": round(abs(snow_recall - rain_recall), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_val, y_pred), 4),
            "macro_f1": round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4),
        })

    return pd.DataFrame(records).sort_values("recall_gap").reset_index(drop=True)


def plot_scale_pos_weight_sweep(sweep_df, best_weight, graphics_dir, interp_type):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sweep_df["scale_pos_weight"], sweep_df["snow_recall"],
                 marker="o", label="snow recall")
    axes[0].plot(sweep_df["scale_pos_weight"], sweep_df["rain_recall"],
                 marker="o", label="rain recall")
    axes[0].set(xlabel="scale_pos_weight", ylabel="Recall",
                title="Snow / rain recall vs scale_pos_weight")

    axes[1].plot(sweep_df["scale_pos_weight"], sweep_df["recall_gap"],
                 marker="o", color="tab:red", label="|snow − rain recall|")
    axes[1].set(xlabel="scale_pos_weight", ylabel="|Recall gap|",
                title="Recall gap vs scale_pos_weight")

    for ax in axes:
        ax.axvline(best_weight, ls="--", color="k", alpha=0.6,
                   label=f"selected = {best_weight}")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(graphics_dir / f"scale_pos_weight_sweep_{interp_type}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Uncertainty band
# ---------------------------------------------------------------------------

def optimize_threshold_params(p_snow_cal, temp_wet, y_true_phase):
    """Search the band grids for the best trade-off on the validation set.

    A candidate is only eligible if it still makes a confident call on enough
    of the pure-phase observations, overall and near freezing. Among those, the
    winner maximises a weighted blend of mix capture and accuracy on the pure
    phases it did call.
    """
    p_snow = np.asarray(p_snow_cal, dtype=float)
    t_wet = np.asarray(temp_wet, dtype=float)
    y_true = np.asarray(y_true_phase, dtype=int)

    true_mix = y_true == MIX_CODE
    true_pure = np.isin(y_true, PURE_PHASE_CODES)

    records = []
    for base_hb, extra_hb, sigma in itertools.product(
        BASE_HALF_BAND_GRID, EXTRA_HALF_BAND_GRID, SIGMA_GRID
    ):
        if base_hb + extra_hb >= MAX_TOTAL_HALF_BAND:
            continue

        pred = classify_phase_gaussian_band(p_snow, t_wet, base_hb, extra_hb, sigma)
        pred_mix = pred == MIX_CODE
        pred_confident = pred != MIX_CODE

        mix_capture = pred_mix[true_mix].mean() if true_mix.any() else np.nan
        pure_coverage = pred_confident[true_pure].mean() if true_pure.any() else np.nan
        called_pure = true_pure & pred_confident
        pure_accuracy = ((pred[called_pure] == y_true[called_pure]).mean()
                         if called_pure.any() else np.nan)

        composite = (np.nan if np.isnan(mix_capture) or np.isnan(pure_accuracy) else
                     MIX_CAPTURE_WEIGHT * mix_capture
                     + (1.0 - MIX_CAPTURE_WEIGHT) * pure_accuracy)

        near_freezing_pure = true_pure & (np.abs(t_wet) <= 1.0)
        nf_coverage = (pred_confident[near_freezing_pure].mean()
                       if near_freezing_pure.any() else np.nan)
        feasible = (not np.isnan(pure_coverage) and pure_coverage >= MIN_PURE_COVERAGE
                    and not np.isnan(nf_coverage) and nf_coverage >= MIN_NF_PURE_COVERAGE)

        records.append({
            "base_half_band": base_hb,
            "extra_half_band": extra_hb,
            "sigma": sigma,
            "rain_thresh_0C": round(0.5 - (base_hb + extra_hb), 4),
            "snow_thresh_0C": round(0.5 + (base_hb + extra_hb), 4),
            "rain_thresh_far": round(0.5 - base_hb, 4),
            "snow_thresh_far": round(0.5 + base_hb, 4),
            "mix_capture_rate": mix_capture,
            "pure_coverage": pure_coverage,
            "pure_conf_accuracy": pure_accuracy,
            "composite_score": composite,
            "feasible": feasible,
        })

    results_df = (pd.DataFrame(records)
                    .sort_values(["feasible", "composite_score"], ascending=[False, False])
                    .reset_index(drop=True))
    feasible_df = results_df[results_df["feasible"]]
    if feasible_df.empty:
        warnings.warn("No band met the coverage constraints; using the best unconstrained one.")
    best = feasible_df.iloc[0] if not feasible_df.empty else results_df.iloc[0]

    print("\nSelected uncertainty band:")
    for key in ["base_half_band", "extra_half_band", "sigma", "rain_thresh_0C",
                "snow_thresh_0C", "mix_capture_rate", "pure_coverage",
                "pure_conf_accuracy", "composite_score"]:
        print(f"  {key:22s}: {best[key]:.4f}")

    out = {key: float(best[key]) for key in
           ["base_half_band", "extra_half_band", "sigma", "rain_thresh_0C",
            "snow_thresh_0C", "rain_thresh_far", "snow_thresh_far", "composite_score",
            "mix_capture_rate", "pure_conf_accuracy", "pure_coverage"]}
    out["all_results"] = results_df
    return out


def plot_band_shape(base_hb, extra_hb, sigma, graphics_dir, interp_type):
    twet = np.linspace(-8, 8, 200)
    half_band = gaussian_half_band(twet, base_hb, extra_hb, sigma)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(twet, 0.5 - half_band, 0.5 + half_band, alpha=0.25,
                    color="tab:purple", label="mix / uncertain band")
    ax.plot(twet, 0.5 - half_band, color="tab:purple", lw=1.5)
    ax.plot(twet, 0.5 + half_band, color="tab:purple", lw=1.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axvline(0.0, ls="--", color="k", alpha=0.4, label="Twet = 0°C")
    for sign in (1, -1):
        ax.axvline(sign * CLEAR_PHASE_TWET_C, ls=":", color="gray", alpha=0.6,
                   label=f"calibration regime ±{CLEAR_PHASE_TWET_C}°C" if sign > 0 else None)
    ax.set(xlabel="Wet-bulb temperature (°C)", ylabel="p(snow) threshold", ylim=(0, 1),
           title=f"Uncertainty band  |  base={base_hb:.2f}, extra={extra_hb:.2f}, σ={sigma:.1f}°C")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(graphics_dir / f"gaussian_halfband_shape_{interp_type}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrators(p_raw, y_true, temp_wet):
    """Fit one calibrator per wet-bulb regime.

    Clear-phase uses the full three-parameter beta family; near-freezing uses
    the symmetric two-parameter form, since there is no reason to expect the
    mapping to be lopsided right at the phase boundary.
    """
    near_freezing = np.abs(temp_wet) <= CLEAR_PHASE_TWET_C
    clear_phase = ~near_freezing

    cal_clear = BetaCalibration(parameters="abm")
    cal_clear.fit(p_raw[clear_phase].reshape(-1, 1), y_true[clear_phase])

    cal_nf = BetaCalibration(parameters="ab")
    cal_nf.fit(p_raw[near_freezing].reshape(-1, 1), y_true[near_freezing])

    return {
        "method": "beta_calibration",
        "clear_phase": cal_clear,
        "near_freezing": cal_nf,
        "clear_phase_twet_threshold": CLEAR_PHASE_TWET_C,
        "n_clear_phase": int(clear_phase.sum()),
        "n_near_freezing": int(near_freezing.sum()),
    }


def apply_calibration(p_raw, temp_wet, calibrators):
    """Calibrate raw p(snow), choosing the calibrator by wet-bulb regime."""
    p_raw = np.asarray(p_raw, dtype=float)
    temp_wet = np.asarray(temp_wet, dtype=float)
    out = np.empty_like(p_raw)

    near_freezing = np.abs(temp_wet) <= CLEAR_PHASE_TWET_C
    clear_phase = ~near_freezing
    if clear_phase.any():
        out[clear_phase] = calibrators["clear_phase"].predict(
            p_raw[clear_phase].reshape(-1, 1))
    if near_freezing.any():
        out[near_freezing] = calibrators["near_freezing"].predict(
            p_raw[near_freezing].reshape(-1, 1))

    # Beta calibration cannot reach exactly 0 or 1; this only guards against
    # numerical edge cases.
    return np.clip(out, 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------------------
# Assembling predictions
# ---------------------------------------------------------------------------

def attach_predictions(base_df, p_snow_raw, p_snow_cal, pred_binary, pred_phase, raw_margin):
    df = base_df.copy()
    df["p_snow_raw"] = p_snow_raw
    df["p_snow_cal"] = p_snow_cal
    df["p_rain_raw"] = 1.0 - p_snow_raw
    df["p_rain_cal"] = 1.0 - p_snow_cal
    df["prediction_binary05"] = pred_binary
    df["prediction_phase_uncertainty"] = pred_phase
    df["transition_score"] = transition_score_from_psnow(p_snow_cal)
    df["entropy_binary"] = entropy(np.vstack([1.0 - p_snow_cal, p_snow_cal]), base=2)
    df["raw_margin"] = raw_margin
    df["is_confident"] = pred_phase != MIX_CODE
    df["near_freezing_air_2C"] = df["temp_air"].abs() <= 2.0
    df["near_freezing_wet_2C"] = df["temp_wet"].abs() <= 2.0
    return df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def train_region(region_id, interp_type=INTERP_TYPE):
    paths = model_paths(region_id, interp_type)
    make_output_dirs(paths)
    model_dir, graphics_dir = paths["model_dir"], paths["graphics_dir"]
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    points_df = pd.read_parquet(paths["master_table"])
    split_df = split_observations(points_df, graphics_dir, interp_type)
    split_df.to_parquet(paths["split_table"], index=False)

    # Full splits keep observed mix; fit splits are pure phases only.
    full = {name: split_df[split_df["split"] == name].copy()
            for name in ("train", "val", "test")}
    fit = {name: frame[frame[TARGET_FULL].isin(PURE_PHASE_CODES)].copy()
           for name, frame in full.items()}
    for name, frame in fit.items():
        if frame.empty:
            raise ValueError(f"No pure-phase observations in the {name} split.")

    X = {name: frame[FEATURES].copy() for name, frame in fit.items()}
    y = {name: frame[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int)
         for name, frame in fit.items()}

    print("\nPure-phase counts:")
    for name, frame in fit.items():
        print(f"  {name}: {frame[TARGET_FULL].value_counts().sort_index().to_dict()}")

    # Predictor correlations, for the record.
    corr = fit["train"][FEATURES].corr(numeric_only=True)
    corr.to_csv(model_dir / "predictor_correlation_matrix.csv")
    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().reset_index())
    pairs.columns = ["var1", "var2", "correlation"]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs.sort_values("abs_correlation", ascending=False).to_csv(
        model_dir / "predictor_correlation_pairs.csv", index=False)

    # --- Class weight ------------------------------------------------------
    sweep_df = sweep_scale_pos_weight(X["train"], y["train"], X["val"], y["val"],
                                      SCALE_POS_WEIGHT_GRID)
    best_weight = float(sweep_df.iloc[0]["scale_pos_weight"])
    sweep_df.to_csv(model_dir / "scale_pos_weight_sweep.csv", index=False)
    plot_scale_pos_weight_sweep(sweep_df, best_weight, graphics_dir, interp_type)
    print(f"\nSelected scale_pos_weight = {best_weight} "
          f"(recall gap {sweep_df.iloc[0]['recall_gap']:.4f})")

    # --- Train -------------------------------------------------------------
    params = {**BASE_XGB_PARAMS, "scale_pos_weight": best_weight, "seed": RANDOM_SEED}
    dtrain = xgb.DMatrix(X["train"], label=y["train"], feature_names=FEATURES)
    dval = xgb.DMatrix(X["val"], label=y["val"], feature_names=FEATURES)

    evals_result = {}
    booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                        evals=[(dtrain, "train"), (dval, "val")],
                        evals_result=evals_result,
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose_eval=50)
    print(f"\nBest iteration {booster.best_iteration}, score {booster.best_score:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(evals_result["train"]["logloss"], label="train")
    ax.plot(evals_result["val"]["logloss"], label="val")
    ax.axvline(booster.best_iteration, ls="--", alpha=0.7,
               label=f"best={booster.best_iteration}")
    ax.set(xlabel="Boosting iteration", ylabel="Binary log loss",
           title=f"Rain/snow log loss ({interp_type}, scale_pos_weight={best_weight})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(graphics_dir / f"xgb_binary_logloss_{interp_type}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Calibrate ---------------------------------------------------------
    # Calibrators are fitted on the final booster's own scores over train+val,
    # so they see the same score distribution the model produces at inference.
    trainval = pd.concat([fit["train"], fit["val"]]).reset_index(drop=True)
    y_trainval = trainval[TARGET_FULL].map(BINARY_LABEL_MAP).astype(int).to_numpy()
    twet_trainval = trainval["temp_wet"].to_numpy()
    p_raw_trainval = booster.predict(
        xgb.DMatrix(trainval[FEATURES], feature_names=FEATURES))

    calibrators = fit_calibrators(p_raw_trainval, y_trainval, twet_trainval)
    p_cal_trainval = apply_calibration(p_raw_trainval, twet_trainval, calibrators)
    print(f"\nCalibration pool: {calibrators['n_clear_phase']} clear-phase, "
          f"{calibrators['n_near_freezing']} near-freezing")
    print(f"  log loss raw/cal: {log_loss(y_trainval, p_raw_trainval):.4f} / "
          f"{log_loss(y_trainval, p_cal_trainval):.4f}")

    # --- Predict on every split -------------------------------------------
    preds = {}
    for name, frame, features in [
        ("val_fit", fit["val"], X["val"]),
        ("test_fit", fit["test"], X["test"]),
        ("val_full", full["val"], full["val"][FEATURES]),
        ("test_full", full["test"], full["test"][FEATURES]),
    ]:
        dmat = xgb.DMatrix(features, feature_names=FEATURES)
        p_raw = booster.predict(dmat)
        twet = frame["temp_wet"].to_numpy()
        preds[name] = {
            "frame": frame,
            "p_raw": p_raw,
            "p_cal": apply_calibration(p_raw, twet, calibrators),
            "margin": booster.predict(dmat, output_margin=True),
            "twet": twet,
        }

    # --- Choose the uncertainty band on validation -------------------------
    threshold_result = optimize_threshold_params(
        preds["val_full"]["p_cal"], preds["val_full"]["twet"],
        full["val"][TARGET_FULL].to_numpy(),
    )
    base_hb = threshold_result["base_half_band"]
    extra_hb = threshold_result["extra_half_band"]
    sigma = threshold_result["sigma"]

    threshold_result["all_results"].to_csv(
        model_dir / "threshold_optimisation_all_results.csv", index=False)
    threshold_result["all_results"].head(10).to_csv(
        model_dir / "threshold_optimisation_top10.csv", index=False)
    plot_band_shape(base_hb, extra_hb, sigma, graphics_dir, interp_type)

    # --- Assemble prediction tables ---------------------------------------
    frames = {}
    for name, pred in preds.items():
        pred_phase = classify_phase_gaussian_band(pred["p_cal"], pred["twet"],
                                                  base_hb, extra_hb, sigma)
        pred_binary = np.where(pred["p_cal"] >= 0.5, SNOW_CODE, RAIN_CODE)
        frames[name] = attach_predictions(pred["frame"], pred["p_raw"], pred["p_cal"],
                                          pred_binary, pred_phase, pred["margin"])
    frames["val_fit"]["phase_binary"] = y["val"].values
    frames["test_fit"]["phase_binary"] = y["test"].values

    # --- Quick report ------------------------------------------------------
    y_test_fit_phase = fit["test"][TARGET_FULL].to_numpy()
    print("\nPure-phase test set (0.5 threshold):")
    print(confusion_matrix(y_test_fit_phase, frames["test_fit"]["prediction_binary05"],
                           labels=[SNOW_CODE, RAIN_CODE]))
    print(classification_report(y_test_fit_phase, frames["test_fit"]["prediction_binary05"],
                                labels=[SNOW_CODE, RAIN_CODE],
                                target_names=["snow", "rain"], digits=3, zero_division=0))

    print("Full test set, three classes from the uncertainty band:")
    print(confusion_matrix(full["test"][TARGET_FULL],
                           frames["test_full"]["prediction_phase_uncertainty"],
                           labels=[SNOW_CODE, RAIN_CODE, MIX_CODE]))
    print(classification_report(full["test"][TARGET_FULL],
                                frames["test_full"]["prediction_phase_uncertainty"],
                                labels=[SNOW_CODE, RAIN_CODE, MIX_CODE],
                                target_names=FULL_CLASS_NAMES, digits=3, zero_division=0))
    print("Test calibrated log loss: "
          f"{log_loss(y['test'].to_numpy(), preds['test_fit']['p_cal'], labels=[0, 1]):.4f}"
          " | Brier: "
          f"{brier_score_loss(y['test'].to_numpy(), preds['test_fit']['p_cal']):.4f}")

    # --- Export ------------------------------------------------------------
    export_artifacts(model_dir, booster, calibrators, evals_result, params,
                     best_weight, sweep_df, threshold_result, interp_type,
                     region_id, X, y, frames)
    print(f"\nArtifacts written to {model_dir}")
    return booster, calibrators, threshold_result


def export_artifacts(model_dir, booster, calibrators, evals_result, params,
                     best_weight, sweep_df, threshold_result, interp_type,
                     region_id, X, y, frames):
    booster.save_model(model_dir / "xgb_binary_phase_model.bin")
    with open(model_dir / "beta_calibration_models.pkl", "wb") as handle:
        pickle.dump(calibrators, handle)
    with open(model_dir / "feature_names.json", "w") as handle:
        json.dump(FEATURES, handle, indent=2)
    with open(model_dir / "training_evals_result.json", "w") as handle:
        json.dump(evals_result, handle, indent=2)

    training_summary = {
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "num_boost_round": int(NUM_BOOST_ROUND),
        "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
    }
    with open(model_dir / "training_summary.json", "w") as handle:
        json.dump(training_summary, handle, indent=2)

    calibration_summary = {
        "method": "beta_calibration_regime_stratified",
        "description": (
            "Beta calibration (Kull et al. 2017) of the raw XGBoost probabilities, "
            "fitted separately for near-freezing and clear-phase conditions on the "
            "final booster's scores over the train+val pure-phase pool. Replaces "
            "isotonic regression, which pushed most test predictions to exact 0 or 1 "
            "when scores fell outside its fitted range."
        ),
        "reference": ("Kull, M., Silva Filho, T. M., & Flach, P. (2017). Beta calibration. "
                      "AISTATS 2017, PMLR 54:623-631."),
        "fit_subset": "train+val pure-phase observations",
        "regimes": {
            "clear_phase_threshold_degC": CLEAR_PHASE_TWET_C,
            "clear_phase_model": "BetaCalibration(parameters='abm')",
            "near_freezing_model": "BetaCalibration(parameters='ab')",
            "n_clear_phase": calibrators["n_clear_phase"],
            "n_near_freezing": calibrators["n_near_freezing"],
        },
        "uncertainty_thresholds": {
            "method": "wet_bulb_conditioned_gaussian_half_band",
            "formula": "half_band(Twet) = base + extra * exp(-Twet^2 / (2 * sigma^2))",
            "references": ["Harder & Pomeroy (2013)", "Sims & Liu (2015)",
                           "Jennings et al. (2018, 2023, 2025)"],
            **{k: threshold_result[k] for k in
               ["base_half_band", "extra_half_band", "sigma", "rain_thresh_0C",
                "snow_thresh_0C", "rain_thresh_far", "snow_thresh_far",
                "mix_capture_rate", "pure_conf_accuracy", "pure_coverage"]},
            "min_pure_coverage_constraint": MIN_PURE_COVERAGE,
            "mix_capture_weight": MIX_CAPTURE_WEIGHT,
        },
    }
    with open(model_dir / "calibration_summary.json", "w") as handle:
        json.dump(calibration_summary, handle, indent=2)

    for split, suffix in [("val", "val"), ("test", "test")]:
        X[split].to_parquet(model_dir / f"X_{suffix}_fit.parquet")
        pd.Series(y[split], name="phase_binary").to_frame().to_parquet(
            model_dir / f"y_{suffix}_fit_binary.parquet")

    for name, out_stem in [("val_fit", "val_fit_binary_predictions"),
                           ("test_fit", "test_fit_binary_predictions"),
                           ("val_full", "val_full_uncertainty_predictions"),
                           ("test_full", "test_full_uncertainty_predictions")]:
        frames[name].to_parquet(model_dir / f"{out_stem}_combined.parquet")
        frames[name][PREDICTION_OUT_COLS].rename(columns={"phase_full": "phase"}) \
            .to_parquet(model_dir / f"{out_stem}.parquet")

    metadata = {
        "version": "v7_beta_calibration",
        "region": region_id,
        "classes_full": {"0": "snow", "1": "rain", "2": "mix"},
        "classes_binary_fit": {"0": "rain", "1": "snow"},
        "model_type": ("XGBoost (binary:logistic) with regime-stratified beta "
                       "calibration and a wet-bulb-conditioned uncertainty band"),
        "interpolation_type": interp_type,
        "xgboost_params": params,
        "label_source": "raw MRoS categorical observations",
        "training_target_definition": (
            "Only observed rain and snow are used for supervised fitting; observed "
            "mix is held out and used to evaluate the uncertainty band."
        ),
        "calibration": calibration_summary,
        "weighting": {
            "method": "objective_driven_scale_pos_weight",
            "description": ("scale_pos_weight swept over a grid; the value minimising "
                            "|snow recall - rain recall| on the pure-phase validation "
                            "subset is selected."),
            "scale_pos_weight_grid": SCALE_POS_WEIGHT_GRID,
            "best_scale_pos_weight": best_weight,
            "sweep_results_csv": str(model_dir / "scale_pos_weight_sweep.csv"),
        },
        "mros_predictor_source": ("leave-one-out MRoS support indicators at the raw "
                                  "observation points"),
        "features": FEATURES,
        "training_summary": training_summary,
        "methodology_notes": [
            "No interpolated MRoS surface is used as a label.",
            "Raw MRoS observations provide the supervised target.",
            "Observed mix is not trained as a class.",
            "Leave-one-out MRoS indicators are predictors only.",
            "Mix is derived from calibrated rain/snow probabilities via the "
            "wet-bulb-conditioned band.",
        ],
    }
    with open(model_dir / "model_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    pd.DataFrame({
        "feature": FEATURES,
        "gain": [gain.get(f, 0.0) for f in FEATURES],
        "weight": [weight.get(f, 0.0) for f in FEATURES],
    }).sort_values("gain", ascending=False).to_csv(
        model_dir / "xgb_feature_importance.csv", index=False)


def main(regions=None, interp_type=INTERP_TYPE):
    for region_id in regions or REGIONS:
        train_region(region_id, interp_type)


if __name__ == "__main__":
    main(parse_region_args())
