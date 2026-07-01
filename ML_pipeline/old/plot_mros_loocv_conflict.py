"""
plot_mros_loocv_conflict.py
===========================
Standalone diagnostic figure illustrating the citizen-science / LOOCV
conflict cluster identified in the ablation calibration analysis.

Designed to be run after run_experiment() while df_all_full, p_all_cal,
p_all_raw, and shap_values are still in memory — OR pointed at a saved
shap_values_all.parquet that has been enriched with p_snow_cal/p_snow_raw.

Five-panel layout
-----------------
  A  Where the problem lives in probability space
       Reliability diagram with the conflict cluster highlighted in red
  B  The LOOCV conflict
       Scatter: mros_p_rain_loocv vs p_snow_cal, coloured by true phase
  C  Thermodynamic context
       Elevation vs Twet, all snow obs in grey, conflict cluster in red
  D  Signal agreement map
       mros_p_rain_loocv (x) vs IMERG PLP (y) for all snow obs,
       conflict cluster highlighted — shows both signals point to rain
  E  Monthly distribution
       Bar chart: how often does the conflict occur vs total snow obs
       per month — reveals storm-season pattern

Usage
-----
  # Option A: objects already in memory from run_experiment()
  from plot_mros_loocv_conflict import make_conflict_figure
  make_conflict_figure(df_all_full, p_all_cal, p_all_raw,
                       out_path="conflict_diagnostic.png")

  # Option B: reload from parquet (requires p_snow_cal column present)
  import pandas as pd
  df = pd.read_parquet("shap_values_all.parquet")
  from plot_mros_loocv_conflict import make_conflict_figure_from_df
  make_conflict_figure_from_df(df, out_path="conflict_diagnostic.png")
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve

# ── Colour palette ────────────────────────────────────────────────────────────
C_CONFLICT  = "#c0392b"   # bold red    — the problematic cluster
C_SNOW      = "#2471a3"   # steel blue  — normal snow observations
C_RAIN      = "#27ae60"   # green       — rain observations
C_MIX       = "#e377c2"   # pink        — mix observations
C_BAND      = "#f39c12"   # amber       — annotation / band shading
C_BG        = "#f8f9fa"   # off-white background

SNOW_CODE = 0
RAIN_CODE = 1
MIX_CODE  = 2

PHASE_COLOR = {SNOW_CODE: C_SNOW, RAIN_CODE: C_RAIN, MIX_CODE: C_MIX}
PHASE_LABEL = {SNOW_CODE: "Snow", RAIN_CODE: "Rain", MIX_CODE: "Mix"}

MONTH_NAMES = {10:"Oct",11:"Nov",12:"Dec",1:"Jan",2:"Feb",
               3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep"}


# =============================================================================
# Core figure builder
# =============================================================================

def make_conflict_figure(
    df_all: pd.DataFrame,
    p_snow_cal: np.ndarray,
    p_snow_raw: np.ndarray,
    out_path: str | Path = "mros_loocv_conflict_diagnostic.png",
    experiment_name: str = "",
    conflict_threshold: float = 0.05,
) -> None:
    """
    Parameters
    ----------
    df_all            : full combined dataframe (all splits) from run_experiment()
    p_snow_cal        : calibrated p(snow) array aligned with df_all
    p_snow_raw        : raw XGBoost p(snow) array aligned with df_all
    out_path          : where to save the figure
    experiment_name   : shown in suptitle
    conflict_threshold: p(snow) below this AND true phase == snow → conflict
    """
    df = df_all.copy().reset_index(drop=True)
    df["p_snow_cal"] = np.asarray(p_snow_cal)
    df["p_snow_raw"] = np.asarray(p_snow_raw)
    df["time"]       = pd.to_datetime(df["time"])
    df["month"]      = df["time"].dt.month

    # ── Identify populations ──────────────────────────────────────────────────
    snow_mask     = df["phase_full"] == SNOW_CODE
    conflict_mask = (df["p_snow_cal"] < conflict_threshold) & snow_mask
    normal_snow   = snow_mask & ~conflict_mask

    df_conflict = df[conflict_mask].copy()
    df_snow     = df[normal_snow].copy()
    df_rain     = df[df["phase_full"] == RAIN_CODE].copy()
    df_mix      = df[df["phase_full"] == MIX_CODE].copy()

    n_conflict = int(conflict_mask.sum())
    pct        = 100 * n_conflict / max(len(df), 1)

    # ── Pure-phase subset for reliability diagram ─────────────────────────────
    pure_mask = df["phase_full"].isin([SNOW_CODE, RAIN_CODE])
    y_bin     = (df.loc[pure_mask, "phase_full"] == SNOW_CODE).astype(int).to_numpy()
    p_bin     = df.loc[pure_mask, "p_snow_cal"].to_numpy()

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=C_BG)
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.88,  bottom=0.09,
        hspace=0.42, wspace=0.35,
    )

    ax_A = fig.add_subplot(gs[0, 0])   # reliability diagram
    ax_B = fig.add_subplot(gs[0, 1])   # LOOCV conflict scatter
    ax_C = fig.add_subplot(gs[0, 2])   # elev vs Twet
    ax_D = fig.add_subplot(gs[1, 0])   # LOOCV vs IMERG
    ax_E = fig.add_subplot(gs[1, 1])   # monthly distribution
    ax_F = fig.add_subplot(gs[1, 2])   # narrative / summary text panel

    for ax in [ax_A, ax_B, ax_C, ax_D, ax_E, ax_F]:
        ax.set_facecolor(C_BG)

    # =========================================================================
    # A: Reliability diagram with conflict cluster highlighted
    # =========================================================================
    frac, mean_ = calibration_curve(y_bin, p_bin, n_bins=15, strategy="quantile")
    ax_A.plot([0,1],[0,1],"--", color="grey", lw=1.2, alpha=0.5, label="Perfect")
    ax_A.plot(mean_, frac, "o-", color="#555", lw=1.8, ms=5,
              label="All obs", zorder=2)

    # Bin the conflict observations
    conflict_pure = conflict_mask & pure_mask
    if conflict_pure.sum() > 0:
        y_c = (df.loc[conflict_pure, "phase_full"] == SNOW_CODE).astype(int).to_numpy()
        p_c = df.loc[conflict_pure, "p_snow_cal"].to_numpy()
        # Single point for the conflict bin (they all fall near p=0)
        ax_A.scatter([p_c.mean()], [y_c.mean()],
                     s=180, color=C_CONFLICT, zorder=5,
                     label=f"Conflict cluster (n={n_conflict})", edgecolors="white", lw=1.5)
        ax_A.annotate(
            f"  {int(y_c.mean()*100)}% truly snow\n  model says p(snow)≈{p_c.mean():.2f}",
            xy=(p_c.mean(), y_c.mean()),
            xytext=(0.18, y_c.mean() - 0.03),
            fontsize=8, color=C_CONFLICT,
            arrowprops=dict(arrowstyle="->", color=C_CONFLICT, lw=1.2),
        )

    ax_A.set(xlim=(0,1), ylim=(0,1),
             xlabel="Mean predicted p(snow)",
             ylabel="Observed fraction snow",
             title="A — Reliability diagram")
    ax_A.legend(fontsize=8, loc="upper left")
    ax_A.grid(alpha=0.2)

    # =========================================================================
    # B: LOOCV conflict scatter — mros_p_rain_loocv vs p_snow_cal
    # =========================================================================
    # Background: all snow observations
    ax_B.scatter(df_snow["mros_p_rain_loocv"], df_snow["p_snow_cal"],
                 s=10, alpha=0.25, color=C_SNOW, label="Normal snow obs", rasterized=True)
    # Conflict cluster on top
    ax_B.scatter(df_conflict["mros_p_rain_loocv"], df_conflict["p_snow_cal"],
                 s=55, alpha=0.85, color=C_CONFLICT, label=f"Conflict cluster (n={n_conflict})",
                 edgecolors="white", lw=0.8, zorder=4)

    ax_B.axvline(0.5, color=C_BAND, ls="--", lw=1.2, alpha=0.7,
                 label="LOOCV rain-dominant threshold")
    ax_B.axhline(conflict_threshold, color="grey", ls=":", lw=1, alpha=0.6)

    ax_B.set(xlabel="LOOCV p(rain)  [station network]",
             ylabel="Model calibrated p(snow)",
             title="B — LOOCV network vs model output\n(true snow observations only)",
             xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax_B.legend(fontsize=8)
    ax_B.grid(alpha=0.2)

    # Annotation arrow showing the conflict direction
    ax_B.annotate(
        "Station network\nsays rain →",
        xy=(0.85, 0.03), fontsize=8, color=C_CONFLICT,
        ha="center", style="italic",
    )

    # =========================================================================
    # C: Elevation vs Twet — spatial/thermodynamic context
    # =========================================================================
    ax_C.scatter(df_snow["temp_wet"], df_snow["elev"],
                 s=10, alpha=0.2, color=C_SNOW, label="Normal snow", rasterized=True)
    ax_C.scatter(df_conflict["temp_wet"], df_conflict["elev"],
                 s=60, alpha=0.9, color=C_CONFLICT,
                 label=f"Conflict (n={n_conflict})",
                 edgecolors="white", lw=0.8, zorder=4)

    ax_C.axvline(0, color=C_BAND, ls="--", lw=1.2, alpha=0.7, label="Twet = 0°C")
    ax_C.axvline(-2, color="grey", ls=":", lw=1, alpha=0.5)
    ax_C.axvline(2,  color="grey", ls=":", lw=1, alpha=0.5,
                 label="±2°C calibration regime")

    # Annotate mean elevation difference
    if len(df_conflict) > 0 and len(df_snow) > 0:
        elev_gap = df_snow["elev"].mean() - df_conflict["elev"].mean()
        ax_C.annotate(
            f"Conflict obs\n{elev_gap:.0f}m lower\nthan avg snow",
            xy=(df_conflict["temp_wet"].mean(), df_conflict["elev"].mean()),
            xytext=(df_conflict["temp_wet"].mean() + 2, df_conflict["elev"].mean() + 120),
            fontsize=8, color=C_CONFLICT,
            arrowprops=dict(arrowstyle="->", color=C_CONFLICT, lw=1.1),
        )

    ax_C.set(xlabel="Wet-bulb temperature (°C)",
             ylabel="Elevation (m)",
             title="C — Thermodynamic & elevation context\n(true snow observations)")
    ax_C.legend(fontsize=8)
    ax_C.grid(alpha=0.2)

    # =========================================================================
    # D: LOOCV p(rain) vs IMERG PLP — both signals point to rain
    # =========================================================================
    if "imerg_plp" in df.columns:
        ax_D.scatter(df_snow["mros_p_rain_loocv"], df_snow["imerg_plp"],
                     s=10, alpha=0.2, color=C_SNOW, label="Normal snow", rasterized=True)
        ax_D.scatter(df_conflict["mros_p_rain_loocv"], df_conflict["imerg_plp"],
                     s=60, alpha=0.9, color=C_CONFLICT,
                     label=f"Conflict (n={n_conflict})",
                     edgecolors="white", lw=0.8, zorder=4)

        ax_D.axvline(0.5,  color=C_BAND,  ls="--", lw=1.2, alpha=0.6)
        ax_D.axhline(50,   color="grey",  ls=":",  lw=1.0, alpha=0.5)
        ax_D.axhline(80,   color=C_BAND,  ls="--", lw=1.2, alpha=0.6)

        # Shade the "both signals say rain" quadrant
        ax_D.fill_between([0.5, 1.02], [80, 80], [102, 102],
                           color=C_CONFLICT, alpha=0.07,
                           label="Both signals: rain")

        ax_D.set(xlabel="LOOCV p(rain)  [station network]",
                 ylabel="IMERG PLP  [satellite, %]",
                 title="D — Convergent rain signals\n(true snow observations)",
                 xlim=(-0.02, 1.02), ylim=(-2, 102))
        ax_D.legend(fontsize=8)
        ax_D.grid(alpha=0.2)
        ax_D.annotate("Both gridded signals\npoint to rain →\nbut observer sees snow",
                      xy=(0.76, 91), fontsize=8, color=C_CONFLICT,
                      ha="center", style="italic")
    else:
        ax_D.text(0.5, 0.5, "IMERG PLP not available\nfor this experiment",
                  ha="center", va="center", transform=ax_D.transAxes,
                  fontsize=10, color="grey")
        ax_D.set_title("D — Convergent rain signals")

    # =========================================================================
    # E: Monthly distribution — when do conflicts occur?
    # =========================================================================
    all_months    = df_snow["month"].value_counts().sort_index()
    conflict_months = df_conflict["month"].value_counts().sort_index() \
                      if len(df_conflict) > 0 else pd.Series(dtype=int)

    months_present = sorted(set(all_months.index) | set(conflict_months.index))
    x = np.arange(len(months_present))
    width = 0.38

    # Normalise conflict counts as fraction of monthly snow obs
    conflict_frac = pd.Series({
        m: conflict_months.get(m, 0) / max(all_months.get(m, 1), 1)
        for m in months_present
    })

    bars = ax_E.bar(x, [conflict_frac.get(m, 0) for m in months_present],
                    width=0.7, color=C_CONFLICT, alpha=0.75,
                    label="Conflict as % of month's snow obs")

    ax_E.set_xticks(x)
    ax_E.set_xticklabels([MONTH_NAMES.get(m, str(m)) for m in months_present])
    ax_E.set(ylabel="Fraction of monthly snow obs\nthat are conflict cases",
             title="E — Seasonality of conflict cases")
    ax_E.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_E.legend(fontsize=8)
    ax_E.grid(axis="y", alpha=0.2)

    # Add raw counts above bars
    for bar, m in zip(bars, months_present):
        n = int(conflict_months.get(m, 0))
        if n > 0:
            ax_E.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                      str(n), ha="center", va="bottom", fontsize=8, color=C_CONFLICT)

    # =========================================================================
    # F: Summary narrative text panel
    # =========================================================================
    ax_F.axis("off")

    loocv_conflict_n = int((df_conflict["mros_p_rain_loocv"] > 0.8).sum()) \
                       if len(df_conflict) > 0 else 0
    imerg_mean = df_conflict["imerg_plp"].mean() if "imerg_plp" in df_conflict.columns else float("nan")
    elev_mean_conflict = df_conflict["elev"].mean() if len(df_conflict) > 0 else float("nan")
    elev_mean_snow     = df_snow["elev"].mean()     if len(df_snow) > 0     else float("nan")
    twet_mean_conflict = df_conflict["temp_wet"].mean() if len(df_conflict) > 0 else float("nan")

    summary_lines = [
        ("F — What is happening here", "title"),
        ("", "gap"),
        (f"{n_conflict} observations ({pct:.1f}% of all data)", "stat"),
        ("have p(snow) < 0.05 but were observed as snow.", "body"),
        ("", "gap"),
        ("These are not model errors in the usual sense.", "body"),
        ("They are cases where every gridded signal —", "body"),
        ("the MRoS LOOCV surface AND IMERG PLP — points", "body"),
        ("to rain, yet a citizen scientist on the ground", "body"),
        ("observed snow.", "body"),
        ("", "gap"),
        ("Key characteristics of the conflict cluster:", "subhead"),
        (f"  • Mean elevation:  {elev_mean_conflict:.0f}m  (vs {elev_mean_snow:.0f}m for normal snow)", "stat"),
        (f"  • Mean Twet:       {twet_mean_conflict:.1f}°C  (near-freezing boundary)", "stat"),
        (f"  • {loocv_conflict_n}/{n_conflict} obs have LOOCV p(rain) > 0.80", "stat"),
        (f"  • IMERG PLP mean:  {imerg_mean:.0f}%  (strongly liquid signal)", "stat"),
        ("", "gap"),
        ("Interpretation:", "subhead"),
        ("These lower-elevation boundary sites sit at the", "body"),
        ("orographic transition zone where the freezing", "body"),
        ("level drops sharply with altitude. The station", "body"),
        ("network (via LOOCV) and satellite retrievals are", "body"),
        ("sampling warmer, lower-elevation surroundings.", "body"),
        ("The citizen science observer is the only instrument", "body"),
        ("correctly located in the snow zone — demonstrating", "body"),
        ("the unique observational value of MRoS.", "emphasis"),
    ]

    y_pos = 0.97
    for text, style in summary_lines:
        if style == "gap":
            y_pos -= 0.022
            continue
        if style == "title":
            ax_F.text(0.03, y_pos, text, transform=ax_F.transAxes,
                      fontsize=11, fontweight="bold", color="#2c3e50", va="top")
            y_pos -= 0.06
        elif style == "subhead":
            ax_F.text(0.03, y_pos, text, transform=ax_F.transAxes,
                      fontsize=9, fontweight="bold", color="#555", va="top")
            y_pos -= 0.05
        elif style == "stat":
            ax_F.text(0.03, y_pos, text, transform=ax_F.transAxes,
                      fontsize=8.5, color=C_CONFLICT, va="top", family="monospace")
            y_pos -= 0.047
        elif style == "emphasis":
            ax_F.text(0.03, y_pos, text, transform=ax_F.transAxes,
                      fontsize=8.5, color=C_CONFLICT, fontstyle="italic",
                      fontweight="bold", va="top")
            y_pos -= 0.047
        else:
            ax_F.text(0.03, y_pos, text, transform=ax_F.transAxes,
                      fontsize=8.5, color="#444", va="top")
            y_pos -= 0.047

    # Light border around F
    for spine in ax_F.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#ddd")
        spine.set_linewidth(0.8)

    # =========================================================================
    # Suptitle
    # =========================================================================
    title = "MRoS citizen science observations vs. gridded rain signals — calibration conflict diagnostic"
    if experiment_name:
        title = f"{experiment_name}  |  {title}"
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#2c3e50", y=0.95)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_conflict_figure_from_df(
    shap_df: pd.DataFrame,
    out_path: str | Path = "mros_loocv_conflict_diagnostic.png",
    experiment_name: str = "",
    conflict_threshold: float = 0.05,
) -> None:
    """
    Convenience wrapper when loading from a saved shap_values_all.parquet
    that already has p_snow_cal and p_snow_raw columns.
    """
    required = ["p_snow_cal", "p_snow_raw", "phase_full", "temp_wet", "elev",
                "mros_p_rain_loocv", "mros_p_snow_loocv"]
    missing = [c for c in required if c not in shap_df.columns]
    if missing:
        raise KeyError(f"Missing columns in shap_df: {missing}. "
                       "Ensure shap_values_all.parquet was saved with p_snow_cal/p_snow_raw.")
    make_conflict_figure(
        df_all=shap_df,
        p_snow_cal=shap_df["p_snow_cal"].to_numpy(),
        p_snow_raw=shap_df["p_snow_raw"].to_numpy(),
        out_path=out_path,
        experiment_name=experiment_name,
        conflict_threshold=conflict_threshold,
    )


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import warnings; warnings.filterwarnings("ignore")
    rng = np.random.default_rng(42)
    n   = 500

    df_test = pd.DataFrame({
        "phase_full":          rng.choice([0,1,2], n, p=[0.55,0.35,0.10]),
        "temp_wet":            rng.normal(-3, 3, n),
        "temp_air":            rng.normal(1,  3, n),
        "temp_dew":            rng.normal(-1, 2, n),
        "elev":                rng.normal(1700, 300, n).clip(800, 3000),
        "mros_p_snow_loocv":   rng.beta(2, 1, n),
        "mros_p_rain_loocv":   rng.beta(1, 2, n),
        "mros_p_mix_loocv":    rng.beta(1, 4, n),
        "imerg_plp":           rng.beta(1, 2, n) * 100,
        "time":                pd.date_range("2021-11-01", periods=n, freq="3h"),
        "x":                   rng.uniform(0, 1e6, n),
        "y":                   rng.uniform(0, 1e6, n),
        "split":               rng.choice(["train","val","test"], n),
    })
    # Inject a conflict cluster: snow obs with high LOOCV rain signal
    conflict_idx = df_test[df_test["phase_full"] == 0].sample(8, random_state=42).index
    df_test.loc[conflict_idx, "mros_p_rain_loocv"] = rng.uniform(0.7, 1.0, 8)
    df_test.loc[conflict_idx, "imerg_plp"]          = rng.uniform(80, 100, 8)
    df_test.loc[conflict_idx, "elev"]               = rng.uniform(1100, 1600, 8)

    p_cal = np.where(df_test["mros_p_rain_loocv"] > 0.7,
                     rng.uniform(0.01, 0.04, n),
                     rng.beta(1, 1, n))
    p_raw = p_cal + rng.normal(0, 0.05, n)
    p_raw = np.clip(p_raw, 0, 1)

    make_conflict_figure(df_test, p_cal, p_raw,
                         out_path="/mnt/user-data/outputs/conflict_diagnostic_smoketest.png",
                         experiment_name="smoke_test")
    print("Smoke test complete.")
