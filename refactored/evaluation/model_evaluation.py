"""Score the trained model and draw the summary figures.

All metrics are computed once into a single dictionary, written to
metrics_summary.json and reused by the figures. Five figures are produced:

1. story1 - binary discrimination: confusion matrices, ROC and PR curves
2. story2 - reliability diagram, raw and calibrated
3. story3 - per-class F1 and mix capture in 1 degC wet-bulb bins
4. story4 - calibrated p(snow) of observed mix relative to the band
5. story5 - three-class confusion matrices for near-freezing observations

Input:  the prediction tables written by train_model.py
Output: metrics_summary.json and story1..story5 figures under graphics/
"""

import datetime
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from common import (  # noqa: E402
    FEATURES, FULL_CLASS_NAMES, INTERP_TYPE, MIX_CODE, PURE_PHASE_CODES,
    RAIN_CODE, REGIONS, SNOW_CODE, gaussian_half_band, model_paths,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import parse_region_args  # noqa: E402

PHASE_COLORS = {"snow": "#3a86ff", "rain": "#2dc653", "mix": "#e377c2"}
SPLIT_COLORS = {"val": "#8338ec", "test": "#ff006e"}
TWET_BIN_EDGES = np.arange(-6, 7, 1)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(paths):
    model_dir = paths["model_dir"]
    frames = {
        name: pd.read_parquet(model_dir / f"{stem}_combined.parquet")
        for name, stem in [
            ("val_fit", "val_fit_binary_predictions"),
            ("test_fit", "test_fit_binary_predictions"),
            ("val_full", "val_full_uncertainty_predictions"),
            ("test_full", "test_full_uncertainty_predictions"),
        ]
    }
    with open(model_dir / "calibration_summary.json") as handle:
        thresholds = json.load(handle)["uncertainty_thresholds"]
    with open(model_dir / "model_metadata.json") as handle:
        metadata = json.load(handle)
    return frames, thresholds, metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def probability_metrics(y_binary, p_raw, p_cal):
    return {
        "roc_auc_raw": round(float(roc_auc_score(y_binary, p_raw)), 4),
        "roc_auc_cal": round(float(roc_auc_score(y_binary, p_cal)), 4),
        "average_precision": round(float(average_precision_score(y_binary, p_cal)), 4),
        "brier_raw": round(float(brier_score_loss(y_binary, p_raw)), 4),
        "brier_cal": round(float(brier_score_loss(y_binary, p_cal)), 4),
        "log_loss_raw": round(float(log_loss(y_binary, p_raw, labels=[0, 1])), 4),
        "log_loss_cal": round(float(log_loss(y_binary, p_cal, labels=[0, 1])), 4),
    }


def per_class_block(y_true, y_pred, codes, names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=codes, zero_division=0
    )
    return {
        name: {"precision": round(float(precision[i]), 4),
               "recall": round(float(recall[i]), 4),
               "f1": round(float(f1[i]), 4),
               "support": int(support[i])}
        for i, name in enumerate(names)
    }


def hard_metrics(y_true, y_pred, codes, names):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": per_class_block(y_true, y_pred, codes, names),
    }


def mix_metrics(y_true_phase, y_pred_phase):
    """Mix capture rate, pure-phase coverage and accuracy on confident calls."""
    y_true = np.asarray(y_true_phase)
    y_pred = np.asarray(y_pred_phase)

    true_mix = y_true == MIX_CODE
    true_pure = np.isin(y_true, PURE_PHASE_CODES)
    confident = y_pred != MIX_CODE
    called_pure = true_pure & confident

    def rounded(value):
        return round(float(value), 4) if value is not None else None

    return {
        "mix_capture_rate": rounded(np.mean(y_pred[true_mix] == MIX_CODE)
                                    if true_mix.any() else None),
        "pure_phase_confident_coverage": rounded(np.mean(confident[true_pure])
                                                 if true_pure.any() else None),
        "pure_phase_confident_accuracy": rounded(
            np.mean(y_pred[called_pure] == y_true[called_pure]) if called_pure.any() else None),
        "overall_confident_frac": rounded(np.mean(confident)),
    }


def near_freezing_metrics(df_full, flag, codes, names):
    subset = df_full[df_full[flag]]
    if subset.empty:
        return {"n_obs": 0}
    return {
        "n_obs": int(len(subset)),
        "accuracy": round(float(accuracy_score(
            subset["phase_full"], subset["prediction_phase_uncertainty"])), 4),
        "per_class": per_class_block(subset["phase_full"],
                                     subset["prediction_phase_uncertainty"], codes, names),
    }


def twet_performance_profile(df_full, bin_edges):
    """Per-class F1 and mix capture in each wet-bulb bin."""
    records = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        subset = df_full[(df_full["temp_wet"] >= lo) & (df_full["temp_wet"] < hi)]
        if len(subset) < 10:
            continue
        y_true = subset["phase_full"].to_numpy()
        y_pred = subset["prediction_phase_uncertainty"].to_numpy()
        true_mix = y_true == MIX_CODE

        records.append({
            "t_mid": (lo + hi) / 2,
            "n": len(subset),
            "f1_snow": f1_score(y_true, y_pred, labels=[SNOW_CODE], average="micro",
                                zero_division=0),
            "f1_rain": f1_score(y_true, y_pred, labels=[RAIN_CODE], average="micro",
                                zero_division=0),
            "mix_capture": (float(np.mean(y_pred[true_mix] == MIX_CODE))
                            if true_mix.any() else np.nan),
            "accuracy": accuracy_score(y_true, y_pred),
        })
    return pd.DataFrame(records)


def compute_all_metrics(frames, thresholds, metadata, region_id, interp_type):
    val_fit, test_fit = frames["val_fit"], frames["test_fit"]
    val_full, test_full = frames["val_full"], frames["test_full"]

    three_class_codes = [SNOW_CODE, RAIN_CODE, MIX_CODE]
    binary_codes = [SNOW_CODE, RAIN_CODE]

    def split_counts(frame, code):
        return int((frame["phase_full"] == code).sum())

    return {
        "run_info": {
            "region": region_id,
            "interp_type": interp_type,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                                     .isoformat(timespec="seconds"),
            "scale_pos_weight": metadata["weighting"]["best_scale_pos_weight"],
            "calibration": "beta_calibration_regime_stratified",
            "gaussian_band": {
                "base_half_band": thresholds["base_half_band"],
                "extra_half_band": thresholds["extra_half_band"],
                "sigma_degC": thresholds["sigma"],
            },
            "features": FEATURES,
            "n_val_snow": split_counts(val_fit, SNOW_CODE),
            "n_val_rain": split_counts(val_fit, RAIN_CODE),
            "n_test_snow": split_counts(test_fit, SNOW_CODE),
            "n_test_rain": split_counts(test_fit, RAIN_CODE),
        },
        "binary_probability": {
            "val": probability_metrics(val_fit["phase_binary"], val_fit["p_snow_raw"],
                                       val_fit["p_snow_cal"]),
            "test": probability_metrics(test_fit["phase_binary"], test_fit["p_snow_raw"],
                                        test_fit["p_snow_cal"]),
        },
        "binary_hard_05": {
            "val": hard_metrics(val_fit["phase_full"], val_fit["prediction_binary05"],
                                binary_codes, ["snow", "rain"]),
            "test": hard_metrics(test_fit["phase_full"], test_fit["prediction_binary05"],
                                 binary_codes, ["snow", "rain"]),
        },
        "uncertainty_3class": {
            "val": hard_metrics(val_full["phase_full"],
                                val_full["prediction_phase_uncertainty"],
                                three_class_codes, FULL_CLASS_NAMES),
            "test": hard_metrics(test_full["phase_full"],
                                 test_full["prediction_phase_uncertainty"],
                                 three_class_codes, FULL_CLASS_NAMES),
        },
        "mix_behavior": {
            "val": mix_metrics(val_full["phase_full"],
                               val_full["prediction_phase_uncertainty"]),
            "test": mix_metrics(test_full["phase_full"],
                                test_full["prediction_phase_uncertainty"]),
        },
        "near_freezing": {
            split: {
                flag: near_freezing_metrics(frame, flag, three_class_codes, FULL_CLASS_NAMES)
                for flag in ["near_freezing_air_2C", "near_freezing_wet_2C"]
            }
            for split, frame in [("val", val_full), ("test", test_full)]
        },
    }


def print_summary(metrics):
    separator = "=" * 62
    for split in ["val", "test"]:
        label = split.upper()

        pm = metrics["binary_probability"][split]
        print(f"\n{separator}\n  {label} | pure-phase probabilities\n{separator}")
        print(f"  ROC AUC raw/cal      : {pm['roc_auc_raw']:.4f} / {pm['roc_auc_cal']:.4f}")
        print(f"  Average precision    : {pm['average_precision']:.4f}")
        print(f"  Brier raw/cal        : {pm['brier_raw']:.4f} / {pm['brier_cal']:.4f}")
        print(f"  Log loss raw/cal     : {pm['log_loss_raw']:.4f} / {pm['log_loss_cal']:.4f}")

        for key, title in [("binary_hard_05", "pure-phase rain/snow at 0.5"),
                           ("uncertainty_3class", "three classes from the band")]:
            block = metrics[key][split]
            print(f"\n{separator}\n  {label} | {title}\n{separator}")
            print(f"  Accuracy {block['accuracy']:.4f} | "
                  f"balanced {block['balanced_accuracy']:.4f} | "
                  f"macro F1 {block['macro_f1']:.4f}")
            for cls, values in block["per_class"].items():
                print(f"    {cls:5s} P={values['precision']:.3f} R={values['recall']:.3f} "
                      f"F1={values['f1']:.3f} n={values['support']}")

        mb = metrics["mix_behavior"][split]
        print(f"\n{separator}\n  {label} | uncertainty band behaviour\n{separator}")
        for key, text in [("mix_capture_rate", "Observed mix flagged as mix"),
                          ("pure_phase_confident_coverage", "Pure phases called confidently"),
                          ("pure_phase_confident_accuracy", "Accuracy on those calls"),
                          ("overall_confident_frac", "Confident fraction overall")]:
            print(f"  {text:34s}: {mb[key]:.4f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_normalised_cm(ax, y_true, y_pred, labels, display_labels, title, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(labels)

    ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n), display_labels)
    ax.set_yticks(range(n), display_labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i, j] > 0.6 else "black")


def story1_discrimination(frames, metrics, graphics_dir, interp_type):
    """Confusion matrices, ROC and precision-recall curves."""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Story 1 — Telling rain from snow", fontsize=13, y=1.01)

    panels = [
        (frames["val_fit"]["phase_full"], frames["val_fit"]["prediction_binary05"],
         [SNOW_CODE, RAIN_CODE], ["snow", "rain"], "Val — rain/snow at 0.5"),
        (frames["test_fit"]["phase_full"], frames["test_fit"]["prediction_binary05"],
         [SNOW_CODE, RAIN_CODE], ["snow", "rain"], "Test — rain/snow at 0.5"),
        (frames["val_full"]["phase_full"], frames["val_full"]["prediction_phase_uncertainty"],
         [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES, "Val — three classes"),
        (frames["test_full"]["phase_full"], frames["test_full"]["prediction_phase_uncertainty"],
         [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES, "Test — three classes"),
    ]
    for i, (y_true, y_pred, labels, names, title) in enumerate(panels, start=1):
        plot_normalised_cm(fig.add_subplot(2, 4, i), y_true, y_pred, labels, names, title)

    ax_roc = fig.add_subplot(2, 4, (5, 6))
    ax_pr = fig.add_subplot(2, 4, (7, 8))

    for split, frame in [("val", frames["val_fit"]), ("test", frames["test_fit"])]:
        y_binary = frame["phase_binary"].to_numpy()
        p_cal = frame["p_snow_cal"].to_numpy()
        fpr, tpr, _ = roc_curve(y_binary, p_cal)
        precision, recall, _ = precision_recall_curve(y_binary, p_cal)
        pm = metrics["binary_probability"][split]
        ax_roc.plot(fpr, tpr, color=SPLIT_COLORS[split], lw=2,
                    label=f"{split}  AUC={pm['roc_auc_cal']:.3f}")
        ax_pr.plot(recall, precision, color=SPLIT_COLORS[split], lw=2,
                   label=f"{split}  AP={pm['average_precision']:.3f}")

    ax_roc.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.6)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    ax_roc.legend(fontsize=10)
    ax_pr.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(graphics_dir / f"story1_discrimination_{interp_type}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def story2_calibration(frames, metrics, band_at_zero, graphics_dir, interp_type):
    """Reliability diagram, raw and calibrated, with a p(snow) histogram."""
    rain_thresh = 0.5 - band_at_zero
    snow_thresh = 0.5 + band_at_zero

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Story 2 — Are the probabilities honest?", fontsize=13)

    for col, split in enumerate(["val", "test"]):
        frame = frames[f"{split}_fit"]
        y_binary = frame["phase_binary"].to_numpy()
        p_raw = frame["p_snow_raw"].to_numpy()
        p_cal = frame["p_snow_cal"].to_numpy()
        ax_rel, ax_hist = axes[0, col], axes[1, col]

        frac_raw, mean_raw = calibration_curve(y_binary, p_raw, n_bins=15, strategy="quantile")
        frac_cal, mean_cal = calibration_curve(y_binary, p_cal, n_bins=15, strategy="quantile")

        ax_rel.axvspan(rain_thresh, snow_thresh, alpha=0.10, color="orange",
                       label=f"band at Twet=0°C ({rain_thresh:.2f}–{snow_thresh:.2f})")
        ax_rel.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, alpha=0.7,
                    label="perfect calibration")
        ax_rel.plot(mean_raw, frac_raw, "o--", color="#aaaaaa", lw=1.5, ms=5, label="raw")
        ax_rel.plot(mean_cal, frac_cal, "o-", color=SPLIT_COLORS[split], lw=2, ms=6,
                    label="calibrated")
        ax_rel.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed snow frequency", title=split)
        ax_rel.legend(fontsize=9)

        pm = metrics["binary_probability"][split]
        ax_rel.text(0.03, 0.92,
                    f"Brier raw={pm['brier_raw']:.3f} cal={pm['brier_cal']:.3f} | "
                    f"log loss cal={pm['log_loss_cal']:.3f}",
                    transform=ax_rel.transAxes, fontsize=8.5, color="dimgrey")

        ax_hist.hist(p_cal, bins=30, color=SPLIT_COLORS[split], alpha=0.7, edgecolor="none")
        ax_hist.axvspan(rain_thresh, snow_thresh, alpha=0.15, color="orange")
        ax_hist.set(xlabel="Calibrated p(snow)", ylabel="Count")
        ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fig.savefig(graphics_dir / f"story2_calibration_{interp_type}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def story3_twet_performance(profiles, graphics_dir, interp_type):
    """Performance in one-degree wet-bulb bins."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Story 3 — Performance across the transition", fontsize=13)

    for ax, (column, label, color) in zip(axes, [
        ("f1_snow", "F1 — snow", PHASE_COLORS["snow"]),
        ("f1_rain", "F1 — rain", PHASE_COLORS["rain"]),
        ("mix_capture", "Mix capture rate", PHASE_COLORS["mix"]),
    ]):
        for split, linestyle in [("val", "--"), ("test", "-")]:
            profile = profiles[split]
            ax.plot(profile["t_mid"], profile[column], linestyle, color=color, lw=2.2,
                    label=split, marker="o", ms=5)
            ax.scatter(profile["t_mid"], profile[column],
                       s=profile["n"] / profile["n"].max() * 120,
                       color=color, alpha=0.25, edgecolors="none")

        ax.axvspan(-1, 1, alpha=0.08, color="orange", label="|Twet| ≤ 1°C")
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        ax.set(xlabel="Wet-bulb temperature (°C)", ylabel=label, title=label,
               xlim=(TWET_BIN_EDGES[0], TWET_BIN_EDGES[-1]), ylim=(0, 1.05))
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(graphics_dir / f"story3_twet_performance_{interp_type}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def story4_band_placement(frames, band, graphics_dir, interp_type):
    """Observed-mix positions relative to the band, and p(snow) by true phase."""
    base_hb, extra_hb, sigma = band
    t_grid = np.linspace(-6, 6, 300)
    hb_grid = gaussian_half_band(t_grid, base_hb, extra_hb, sigma)
    snow_line, rain_line = 0.5 + hb_grid, 0.5 - hb_grid

    mix_all = pd.concat([
        frames["val_full"][frames["val_full"]["phase_full"] == MIX_CODE],
        frames["test_full"][frames["test_full"]["phase_full"] == MIX_CODE],
    ], ignore_index=True)
    hb_per_obs = gaussian_half_band(mix_all["temp_wet"].to_numpy(), base_hb, extra_hb, sigma)
    p_snow = mix_all["p_snow_cal"].to_numpy()
    mix_all["inside_band"] = (p_snow > 0.5 - hb_per_obs) & (p_snow < 0.5 + hb_per_obs)
    n_inside, n_total = int(mix_all["inside_band"].sum()), len(mix_all)
    capture_pct = 100 * n_inside / n_total if n_total else 0

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Story 4 — Where the uncertainty band sits", fontsize=13, y=1.01)

    ax = axes[0]
    ax.fill_between(t_grid, rain_line, snow_line, alpha=0.10, color="orange", zorder=1)
    ax.plot(t_grid, snow_line, color="black", lw=2.0, zorder=2, label="snow threshold")
    ax.plot(t_grid, rain_line, color="black", lw=2.0, ls="--", zorder=2, label="rain threshold")
    for inside, label, color, marker, z in [
        (False, "missed by band", "#cc3311", "x", 3),
        (True, "captured by band", "#009988", "o", 4),
    ]:
        subset = mix_all[mix_all["inside_band"] == inside]
        ax.scatter(subset["temp_wet"], subset["p_snow_cal"], c=color, marker=marker,
                   s=55, alpha=0.75, linewidths=1.2, zorder=z,
                   label=f"{label} (n={len(subset)})")
    ax.axvline(0, color="grey", lw=0.8, alpha=0.45)
    ax.axhline(0.5, color="grey", lw=0.8, alpha=0.45)
    ax.set(xlim=(-6, 6), ylim=(-0.04, 1.04), xlabel="Wet-bulb temperature (°C)",
           ylabel="Calibrated p(snow)",
           title=f"A — observed mix, val + test (n={n_total}, {capture_pct:.0f}% inside band)")
    ax.legend(fontsize=9, loc="upper right")

    # Panel B: p(snow) spread by true phase in two-degree bins.
    ax = axes[1]
    violin_bins = [(-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6)]
    combined = pd.concat([frames["val_full"], frames["test_full"]], ignore_index=True)
    phase_width = 0.8 / 3

    for bin_idx, (lo, hi) in enumerate(violin_bins):
        bin_rows = combined[(combined["temp_wet"] >= lo) & (combined["temp_wet"] < hi)]
        for phase_idx, (code, name) in enumerate(
            zip([SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES)
        ):
            values = bin_rows.loc[bin_rows["phase_full"] == code, "p_snow_cal"].to_numpy()
            x_pos = bin_idx + (phase_idx - 1) * phase_width
            if len(values) < 4:
                if len(values):
                    ax.plot(x_pos, float(np.median(values)), "_",
                            color=PHASE_COLORS[name], ms=12, mew=2)
                continue
            parts = ax.violinplot(values, positions=[x_pos], widths=phase_width * 0.85,
                                  showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(PHASE_COLORS[name])
                body.set_edgecolor("none")
                body.set_alpha(0.70)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

    ax.axhspan(float(rain_line.min()), float(snow_line.max()), alpha=0.06, color="orange")
    ax.axhline(0.5, color="grey", lw=0.8, alpha=0.45)
    ax.set_xticks(range(len(violin_bins)), [f"{lo}–{hi}" for lo, hi in violin_bins],
                  fontsize=9)
    ax.set(xlabel="Wet-bulb temperature bin (°C)", ylabel="Calibrated p(snow)",
           ylim=(-0.04, 1.04), title="B — p(snow) by observed phase, val + test")
    ax.legend(handles=[Patch(facecolor=PHASE_COLORS[c], label=c, alpha=0.75)
                       for c in FULL_CLASS_NAMES]
                      + [Patch(facecolor="orange", alpha=0.20, label="widest band extent")],
              fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(graphics_dir / f"story4_band_placement_{interp_type}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nObserved mix inside the band: {n_inside} / {n_total} ({capture_pct:.1f}%)")


def story5_near_freezing(frames, graphics_dir, interp_type):
    """Three-class performance restricted to near-freezing observations."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Story 5 — Right at the freezing point", fontsize=13)

    panels = [
        ((0, 0), "val_full", "val", "near_freezing_air_2C"),
        ((0, 1), "val_full", "val", "near_freezing_wet_2C"),
        ((1, 0), "test_full", "test", "near_freezing_air_2C"),
        ((1, 1), "test_full", "test", "near_freezing_wet_2C"),
    ]
    for (row, col), frame_key, split, flag in panels:
        subset = frames[frame_key][frames[frame_key][flag]]
        ax = axes[row, col]
        if subset.empty:
            ax.set_visible(False)
            continue
        flag_label = "|Tair| ≤ 2°C" if "air" in flag else "|Twet| ≤ 2°C"
        plot_normalised_cm(ax, subset["phase_full"],
                           subset["prediction_phase_uncertainty"],
                           [SNOW_CODE, RAIN_CODE, MIX_CODE], FULL_CLASS_NAMES,
                           f"{split.capitalize()} — {flag_label} (n={len(subset)})")

    fig.tight_layout()
    fig.savefig(graphics_dir / f"story5_near_freezing_{interp_type}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def evaluate_region(region_id, interp_type=INTERP_TYPE):
    paths = model_paths(region_id, interp_type)
    graphics_dir = paths["graphics_dir"]
    graphics_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {region_id}: {REGIONS[region_id]['label']} ===")

    frames, thresholds, metadata = load_predictions(paths)
    band = (thresholds["base_half_band"], thresholds["extra_half_band"], thresholds["sigma"])

    metrics = compute_all_metrics(frames, thresholds, metadata, region_id, interp_type)
    print_summary(metrics)

    profiles = {split: twet_performance_profile(frames[f"{split}_full"], TWET_BIN_EDGES)
                for split in ("val", "test")}
    band_at_zero = float(gaussian_half_band(np.array([0.0]), *band)[0])

    story1_discrimination(frames, metrics, graphics_dir, interp_type)
    story2_calibration(frames, metrics, band_at_zero, graphics_dir, interp_type)
    story3_twet_performance(profiles, graphics_dir, interp_type)
    story4_band_placement(frames, band, graphics_dir, interp_type)
    story5_near_freezing(frames, graphics_dir, interp_type)

    out_path = paths["model_dir"] / "metrics_summary.json"
    with open(out_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nWrote {out_path} and five figures to {graphics_dir}")


def main(regions=None, interp_type=INTERP_TYPE):
    for region_id in regions or REGIONS:
        evaluate_region(region_id, interp_type)


if __name__ == "__main__":
    main(parse_region_args())
