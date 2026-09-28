"""
CSI-by-air-temperature, simplified to 3 real (not relative) values per panel:
  - XGB-Full (actual CSI)
  - Best single benchmark at that bin (actual CSI)
  - Average across all 8 benchmarks at that bin (actual CSI)

Unlike the advantage chart, every line here is a real, directly-readable CSI
value -- nothing is a difference or a ratio. Unlike the original 9-line
figure, there is no individual line that plunges to an unexplained zero:
the "average" line mixes in whichever benchmarks still have some skill in a
given bin, so it degrades gradually instead of cliff-diving, and no single
labeled curve is the one doing the diving.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def make_panel(ax, bin_mids, bin_valid, csi_event, methods, event_label):
    in_xlim = (bin_mids >= XLIM[0]) & (bin_mids <= XLIM[1])
    mask = bin_valid & in_xlim
    x = bin_mids[mask]

    bench_matrix = np.vstack([csi_event[m][mask] for m in methods])
    avg = np.nanmean(bench_matrix, axis=0)

    # Pick ONE method -- highest mean CSI within +-2C of freezing -- and plot
    # that method's own CSI curve across the whole range. (Previously this
    # plotted np.nanmax across methods per-bin, which silently swapped in
    # whichever method happened to win each individual bin -- e.g. picking
    # up T_a 1.0 C's perfect score in bins where every method ties near 1.0,
    # even though T_a is the method that collapses to zero elsewhere. That
    # produced a composite line that didn't match its own legend label.)
    nf = (bin_mids >= -2) & (bin_mids <= 2) & bin_valid
    means = [np.nanmean(csi_event[m][nf]) for m in methods]
    best_m = methods[int(np.nanargmax(means))]
    best = csi_event[best_m][mask]

    v = ~np.isnan(avg)
    ax.plot(x[v], avg[v], color="#e08214", lw=2.2, ls="--", marker="o", ms=4,
            label="Average across 8 benchmarks")

    v = ~np.isnan(best)
    ax.plot(x[v], best[v], color="#2166ac", lw=2.2, marker="o", ms=4,
            label=f"Best single benchmark ({mf.method_label(best_m)})")

    model_vals = csi_event[MODEL_NAME_KEY][mask]
    v = ~np.isnan(model_vals)
    ax.plot(x[v], model_vals[v], color="black", lw=3.2, marker="o", ms=5.5,
            zorder=5, label="XGB-Full")

    ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.set(xlim=XLIM, ylim=(0, 1.02),
           xlabel="Air temperature (°C)", title=f"{event_label} as event class")
    ax.grid(alpha=0.25)


def make_figure(region):
    bin_mids, bin_valid, csi, methods = compute_csi_table(region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    make_panel(axes[0], bin_mids, bin_valid, csi[SNOW_CODE], methods, "Snow")
    make_panel(axes[1], bin_mids, bin_valid, csi[RAIN_CODE], methods, "Rain")
    axes[0].set_ylabel("Critical success index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle(f"{mf.region_label(region)}: XGB-Full vs. benchmark methods", fontsize=15, y=1.1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_d = mf.presentation_dir(region)
    out_d.mkdir(parents=True, exist_ok=True)
    out = str(out_d / f"{region}_csi_actual_3line.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    for region in ["CA", "CO"]:
        make_figure(region)
