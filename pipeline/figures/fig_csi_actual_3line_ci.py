"""
CSI-by-air-temperature, 3-line actual-value figure (XGB-Full, best single
benchmark, average across 8 benchmarks) WITH 95% cluster-bootstrap confidence
ribbons.

The bootstrap reuses the same spatial/temporal block-cluster machinery as
manuscript_figures._cluster_bootstrap_tair_bins (30 km x day blocks, resampled
with replacement, seed=42), so these CIs are on the same footing as the
accuracy-by-T_air CI figure already in the manuscript pipeline -- just scored
by CSI per event class instead of accuracy.
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
N_BOOT = 500
BLOCK_KM = 30.0
SEED = 42
MIN_BOOT_BIN_N = 5  # matches _cluster_bootstrap_tair_bins' per-bin bootstrap floor


def _csi_vec(yt, yp, event):
    hits = np.sum((yt == event) & (yp == event))
    misses = np.sum((yt == event) & (yp != event))
    fa = np.sum((yt != event) & (yp == event))
    denom = hits + misses + fa
    return np.nan if denom == 0 else hits / denom


def compute_csi_bootstrap(region):
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
    all_methods = methods + [MODEL_NAME_KEY]

    edges = mf.TAIR_BIN_EDGES
    n_bins = len(edges) - 1
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(tair, edges) - 1
    bin_valid = np.array([(bin_idx == b).sum() >= mf.MIN_BIN_N for b in range(n_bins)])

    # --- same cluster machinery as _cluster_bootstrap_tair_bins ---
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

    clusters = build_cluster_index(make_cluster_codes(pure, BLOCK_KM))
    rng = np.random.default_rng(SEED)

    # Point estimates (unresampled), and bootstrap draws, per event/method/bin.
    point = {SNOW_CODE: {}, RAIN_CODE: {}}
    boot = {SNOW_CODE: {}, RAIN_CODE: {}}
    for event in (SNOW_CODE, RAIN_CODE):
        for m in all_methods:
            point[event][m] = np.full(n_bins, np.nan)
            boot[event][m] = np.full((N_BOOT, n_bins), np.nan)
            for b in range(n_bins):
                if not bin_valid[b]:
                    continue
                sel = bin_idx == b
                point[event][m][b] = _csi_vec(y[sel], preds[m][sel], event)

    for i in range(N_BOOT):
        idx = bootstrap_row_indices(clusters, rng)
        yb = y[idx]
        bib = bin_idx[idx]
        for event in (SNOW_CODE, RAIN_CODE):
            for m in all_methods:
                ypb = preds[m][idx]
                for b in range(n_bins):
                    if not bin_valid[b]:
                        continue
                    sel = bib == b
                    if sel.sum() >= MIN_BOOT_BIN_N:
                        boot[event][m][i, b] = _csi_vec(yb[sel], ypb[sel], event)

    return dict(bin_mids=bin_mids, bin_valid=bin_valid, point=point, boot=boot,
                methods=methods, all_methods=all_methods)


def make_panel(ax, res, event, event_label):
    bin_mids, bin_valid = res["bin_mids"], res["bin_valid"]
    in_xlim = (bin_mids >= XLIM[0]) & (bin_mids <= XLIM[1])
    mask = bin_valid & in_xlim
    x = bin_mids[mask]
    methods = res["methods"]
    point, boot = res["point"][event], res["boot"][event]

    # Best single benchmark, chosen the same way as the non-CI figure:
    # highest mean point-estimate CSI within +-2C of freezing.
    nf = (bin_mids >= -2) & (bin_mids <= 2) & bin_valid
    means = [np.nanmean(point[m][nf]) for m in methods]
    best_m = methods[int(np.nanargmax(means))]

    # Average-across-benchmarks: average the point estimate, and separately
    # average each bootstrap draw across methods before taking percentiles,
    # so the ribbon reflects the same resample each time (not an average of
    # independently-computed per-method ribbons).
    avg_point = np.nanmean(np.vstack([point[m] for m in methods]), axis=0)
    avg_boot = np.nanmean(np.stack([boot[m] for m in methods], axis=0), axis=0)  # (N_BOOT, n_bins)

    series = [
        ("avg", avg_point, avg_boot, "#e08214", "--",
         "Average across 8 benchmarks"),
        ("best", point[best_m], boot[best_m], "#2166ac", "-",
         f"Best single benchmark ({mf.method_label(best_m)})"),
        ("model", point[MODEL_NAME_KEY], boot[MODEL_NAME_KEY], "black", "-",
         "XGB-Full"),
    ]

    for key, pt, bt, color, ls, label in series:
        pt_m = pt[mask]
        lo = np.nanpercentile(bt[:, mask], 2.5, axis=0)
        hi = np.nanpercentile(bt[:, mask], 97.5, axis=0)
        v = ~np.isnan(pt_m)
        lw = 3.2 if key == "model" else 2.2
        ms = 5.5 if key == "model" else 4
        zorder = 5 if key == "model" else 3
        ax.plot(x[v], pt_m[v], color=color, lw=lw, ls=ls, marker="o", ms=ms,
                 zorder=zorder, label=label)
        vv = v & ~np.isnan(lo) & ~np.isnan(hi)
        ax.fill_between(x[vv], lo[vv], hi[vv], color=color, alpha=0.15, lw=0, zorder=zorder - 1)

    ax.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.set(xlim=XLIM, ylim=(0, 1.02),
           xlabel="Air temperature (°C)", title=f"{event_label} as event class")
    ax.grid(alpha=0.25)


def make_figure(region):
    res = compute_csi_bootstrap(region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    make_panel(axes[0], res, SNOW_CODE, "Snow")
    make_panel(axes[1], res, RAIN_CODE, "Rain")
    axes[0].set_ylabel("Critical success index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle(f"{mf.region_label(region)}: XGB-Full vs. benchmark methods (95% CI)",
                 fontsize=15, y=1.1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_d = mf.presentation_dir(region)
    out_d.mkdir(parents=True, exist_ok=True)
    out = str(out_d / f"{region}_csi_actual_3line_ci.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    for region in ["CA", "CO"]:
        make_figure(region)
