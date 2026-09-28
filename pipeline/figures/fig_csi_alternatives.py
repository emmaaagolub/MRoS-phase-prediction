"""
Two alternative visualization *types* for the CSI-by-air-temperature story,
built from the same underlying benchmark_predictions_test.parquet as
fig_csi_by_tair_simplified.py, for comparison / picking a presentation
direction.

  A) Heatmap: method x temperature-bin grid, colored by CSI. All 9 methods
     visible at once, no overlapping lines, no lines diving to zero -- a
     weak method is just a pale/dark cell, not a plotted cliff.
  B) Advantage chart: CSI(XGB-Full) - CSI(best benchmark at that bin), vs.
     air temperature. Removes the benchmark comparison entirely and just
     shows the one number a presentation audience actually wants: how much
     better is the model, and where does that gap grow. Never dips into a
     "why is this exactly zero" question because it's not plotting any
     individual benchmark's raw skill at all.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import manuscript_figures as mf

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})

SNOW_CODE = mf.SNOW_CODE
RAIN_CODE = mf.RAIN_CODE
MODEL_NAME_KEY = mf.MODEL_NAME_KEY
XLIM = (-5, 5)


def compute_csi_table(region):
    pq = mf.benchmarking_dir(region) / "benchmark_predictions_test.parquet"
    df = pd.read_parquet(pq)
    pure = df[df["phase_full"].isin([SNOW_CODE, RAIN_CODE])].reset_index(drop=True)
    y = pure["phase_full"].to_numpy(int)
    tair = pure["temp_air"].to_numpy(float)

    methods = [c.replace("pred_", "") for c in pure.columns
               if c.startswith("pred_") and not c.startswith(f"pred_{MODEL_NAME_KEY}")]
    methods = [m for m in methods if mf.keep_method(m)]
    preds = {m: pure[f"pred_{m}"].to_numpy(int) for m in methods}
    preds[MODEL_NAME_KEY] = pure[f"pred_{MODEL_NAME_KEY}_bin05"].to_numpy(int)

    edges = mf.TAIR_BIN_EDGES
    n_bins = len(edges) - 1
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(tair, edges) - 1
    bin_valid = np.array([(bin_idx == b).sum() >= mf.MIN_BIN_N for b in range(n_bins)])

    csi = {SNOW_CODE: {}, RAIN_CODE: {}}
    for event in (SNOW_CODE, RAIN_CODE):
        for m in list(preds.keys()):
            vals = np.full(n_bins, np.nan)
            for b in range(n_bins):
                if not bin_valid[b]:
                    continue
                sel = bin_idx == b
                yt, yp = y[sel], preds[m][sel]
                hits = int(np.sum((yt == event) & (yp == event)))
                misses = int(np.sum((yt == event) & (yp != event)))
                fa = int(np.sum((yt != event) & (yp == event)))
                vals[b] = mf._csi_from_counts(hits, misses, fa)
            csi[event][m] = vals
    return bin_mids, bin_valid, csi, methods


# ---------------------------------------------------------------------------
# A) Heatmap
# ---------------------------------------------------------------------------

# High-contrast diverging colormap (red = poor skill, green = good skill),
# with the low end stretched so the 0.3-0.9 range -- where nearly every real
# cell actually falls -- spans most of the color range instead of being
# crammed into one washed-out shade of green.
_stops = [
    (0.00, "#a50026"), (0.15, "#d73027"), (0.30, "#f46d43"),
    (0.45, "#fdae61"), (0.55, "#fee08b"), (0.65, "#d9ef8b"),
    (0.75, "#a6d96a"), (0.85, "#66bd63"), (0.93, "#1a9850"), (1.00, "#006837"),
]
CSI_CMAP = LinearSegmentedColormap.from_list("csi_hc", _stops, N=256)


def make_heatmap(region):
    bin_mids, bin_valid, csi, methods = compute_csi_table(region)
    in_xlim = (bin_mids >= XLIM[0]) & (bin_mids <= XLIM[1])
    mask = bin_valid & in_xlim
    x = bin_mids[mask]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    fig.subplots_adjust(wspace=0.55)
    for ax, event, event_label in [(axes[0], SNOW_CODE, "Snow"), (axes[1], RAIN_CODE, "Rain")]:
        order = sorted(methods, key=lambda m: np.nanmean(csi[event][m][mask]))
        order = order + [MODEL_NAME_KEY]  # XGB-Full on top row
        mat = np.vstack([csi[event][m][mask] for m in order])

        im = ax.imshow(mat, aspect="auto", cmap=CSI_CMAP, vmin=0, vmax=1,
                        extent=[x.min() - 0.5, x.max() + 0.5, -0.5, len(order) - 0.5],
                        origin="lower")
        # Thin white gridlines between cells so neighboring shades stay separable.
        for j in range(len(x) + 1):
            ax.axvline(x[0] - 0.5 + j, color="white", lw=1.0)
        for i in range(len(order) + 1):
            ax.axhline(i - 0.5, color="white", lw=1.0)

        # Numeric value in every cell -- removes any dependence on reading
        # color at all; color becomes the at-a-glance layer, the number is
        # the precise one.
        for i, m in enumerate(order):
            vals = csi[event][m][mask]
            for j, v in enumerate(vals):
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((x[j] - 0.5, i - 0.5), 1, 1,
                                                facecolor="white", edgecolor="#cccccc",
                                                hatch="////", lw=0))
                else:
                    txt_color = "white" if (v < 0.35 or v > 0.75) else "black"
                    ax.text(x[j], i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                            ha="center", va="center", fontsize=8.5, color=txt_color)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([mf.method_label(m) for m in order])
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.axhline(len(order) - 1.5, color="black", lw=2.0)  # separate XGB row
        ax.set_xticks(np.arange(int(np.ceil(x.min())), int(np.floor(x.max())) + 1, 2))
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.set_xlabel("Air temperature (°C)")
        ax.set_title(f"{event_label} as event class")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Critical success index")
    fig.suptitle(f"{mf.region_label(region)}: CSI by method and air temperature", fontsize=15, y=1.02)
    out_d = mf.presentation_dir(region)
    out_d.mkdir(parents=True, exist_ok=True)
    out = str(out_d / f"{region}_csi_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


# ---------------------------------------------------------------------------
# B) Advantage-over-best-benchmark line chart
# ---------------------------------------------------------------------------

def make_advantage_chart(region):
    bin_mids, bin_valid, csi, methods = compute_csi_table(region)
    in_xlim = (bin_mids >= XLIM[0]) & (bin_mids <= XLIM[1])
    mask = bin_valid & in_xlim
    x = bin_mids[mask]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = {SNOW_CODE: "#2166ac", RAIN_CODE: "#b2182b"}
    labels = {SNOW_CODE: "Snow", RAIN_CODE: "Rain"}
    for event in (SNOW_CODE, RAIN_CODE):
        bench_matrix = np.vstack([csi[event][m][mask] for m in methods])
        best_bench = np.nanmax(bench_matrix, axis=0)
        model = csi[event][MODEL_NAME_KEY][mask]
        advantage = model - best_bench
        v = ~np.isnan(advantage)
        ax.plot(x[v], advantage[v], color=colors[event], lw=2.6, marker="o", ms=5,
                label=f"{labels[event]} as event class")
        ax.fill_between(x[v], 0, advantage[v], color=colors[event], alpha=0.12)

    ax.axhline(0, color="grey", lw=1.2)
    ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.set(xlim=XLIM, xlabel="Air temperature (°C)",
           ylabel="XGB-Full CSI minus best benchmark's CSI")
    ax.set_title(f"{mf.region_label(region)}: how much better is XGB-Full, and where", fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_d = mf.presentation_dir(region)
    out_d.mkdir(parents=True, exist_ok=True)
    out = str(out_d / f"{region}_csi_advantage.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    for region in ["CA", "CO"]:
        make_heatmap(region)
        make_advantage_chart(region)
