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

    bench_matrix = np.vstack([csi_event[m] for m in methods])
    lo = np.nanmin(bench_matrix, axis=0)
    hi = np.nanmax(bench_matrix, axis=0)

    x = bin_mids[mask]
    lo_m, hi_m = lo[mask], hi[mask]
    valid_band = ~np.isnan(lo_m) & ~np.isnan(hi_m)

    ax.fill_between(x[valid_band], lo_m[valid_band], hi_m[valid_band],
                     color="#9e9e9e", alpha=0.35, lw=0,
                     label="Range across 8 benchmark methods")

    nf = (bin_mids >= -2) & (bin_mids <= 2) & bin_valid
    means = [np.nanmean(csi_event[m][nf]) for m in methods]
    best_m = methods[int(np.nanargmax(means))]
    best_vals = csi_event[best_m][mask]
    v = ~np.isnan(best_vals)
    ax.plot(x[v], best_vals[v], color="#2166ac", lw=2.2, marker="o", ms=4,
            label=f"Best single benchmark ({mf.method_label(best_m)})")

    model_vals = csi_event[MODEL_NAME_KEY][mask]
    v = ~np.isnan(model_vals)
    ax.plot(x[v], model_vals[v], color="black", lw=3.2, marker="o", ms=5,
            zorder=5, label="XGB-Full")

    ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.set(xlim=XLIM, ylim=(0, 1.02),
           xlabel="Air temperature (°C)", title=f"{event_label} as event class")
    ax.grid(alpha=0.25)



def make_figure(region):
    bin_mids, bin_valid, csi, methods = compute_csi_table(region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    make_panel(axes[0], bin_mids, bin_valid, csi[SNOW_CODE], methods, "Snow")
    make_panel(axes[1], bin_mids, bin_valid, csi[RAIN_CODE], methods, "Rain")
    axes[0].set_ylabel("Critical success index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle(f"{mf.region_label(region)}: XGB-Full vs. benchmark methods, near freezing", fontsize=15, y=1.06)
    fig.text(0.5, 0.01,
             "Grey band = range across the 8 fixed-threshold/logistic benchmarks; some individual\n"
             "thresholds have zero skill outside the temperature range they were set for.",
             ha="center", fontsize=10.5, color="#555555")
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out_d = mf.presentation_dir(region)
    out_d.mkdir(parents=True, exist_ok=True)
    out = str(out_d / f"{region}_csi_by_tair_simplified.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


for region in ["CA", "CO"]:
    make_figure(region)
