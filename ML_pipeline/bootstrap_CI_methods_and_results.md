# Bootstrap confidence intervals — methods and results

*Working notes. From `model_artifacts/{REGION}/benchmarking_v1/bootstrap/` outputs, both regions run with B = 5,000 replicates, seed 42, produced by `ML_pipeline/bootstrap_cis.py`.*

## What we did and why

All benchmark and ablation results are computed on a single held-out test split, so every reported metric carries sampling uncertainty: a different draw of test observations would give somewhat different numbers. To quantify this we used a **paired space–time cluster bootstrap** over the stored per-observation test predictions (`benchmark_predictions_test.parquet` for the benchmark comparison; each ablation config's `shap_values_all.parquet` test rows for ablation deltas). No models were retrained.

**Why clustered, not plain (iid) resampling.** MRoS reports are not independent: multiple observers report the same storm in the same neighbourhood, share the same interpolated fields and IMERG pixel, and inform each other through the LOOCV surface. An iid bootstrap treats every row as independent evidence and produces intervals that are too narrow. We therefore assigned each test observation to a cluster defined by a **30 km spatial block × calendar day**, and resampled whole clusters with replacement. Correlated same-storm/same-place observations enter or leave each pseudo test set together, so the replicate-to-replicate spread honestly reflects "would a different season of events have given a different answer."

**Why paired.** Within each replicate, all methods (model + 9 benchmarks; baseline + all ablation configs) are scored on the identical resampled rows and differenced *within* the replicate. Because all methods share the difficulty of whatever events a replicate draws, differencing cancels that shared variance, giving much tighter (and correct) CIs on the *differences* than comparing two marginal CIs would.

**What is bootstrapped.** Every metric reported in the manuscript now carries an interval. Per replicate we compute, for the model and all nine benchmarks: accuracy, near-freezing accuracy, macro F1, near-freezing macro F1, snow recall, rain recall, and snow/rain bias; plus, for the model only (benchmarks emit no probabilities): ROC AUC, near-freezing ROC AUC, and mix capture. For every ablation config we compute the same probability and hard-decision metrics in both absolute and baseline-minus-config (paired) form. Macro F1 matches `sklearn.metrics.f1_score(labels=[SNOW, RAIN], average='macro', zero_division=0)`; ROC AUC is a rank-based Mann–Whitney computation with average ranks for ties, verified identical to `roc_auc_score`.

**Point estimate vs. interval.** The point estimate is the metric computed **once on the full held-out test set** — it is *not* the median or mean of the replicates. The interval is the 2.5th–97.5th percentile of the replicate distribution around it. Intervals are therefore not necessarily symmetric about the point, and are markedly asymmetric for metrics on small subsets (e.g. CO rain bias, +19.8% with CI 5.6–37.8).

**Details.** 95% percentile intervals from B = 5,000 replicates. Subset metrics (near-freezing = |T_wet| ≤ 2 °C; per-phase biases) are computed within-replicate and return NaN when too thin, so their CIs widen automatically where data are scarce. Block size is a judgment call, so the analysis was repeated at 15/30/60 km and with iid resampling (Table 5) — headline numbers use 30 km × day.

**What these CIs do and do not cover.** They quantify test-set sampling uncertainty, *conditional on* the trained model and the (random, stratified) train/val/test split. They do not capture training stochasticity (could be added via multi-seed retrains) or the train/test same-storm sharing acknowledged in the methodology (the random split matches the deployment condition, where the product always predicts with the live observer network available; blocked splitting was tried in v6b and abandoned because splits became too small for stable regime-stratified calibration). Note the cluster bootstrap costs no training data — it fixes the inference stage, which is why we can afford it even though we could not afford blocked splits.

## Results

Pure rain/snow test observations: CA n = 2,301 (987 clusters at 30 km × day), CO n = 524 (379 clusters). Including observer-reported mix, the full test sets are CA n = 2,621 and CO n = 588; mix capture is resampled on its own clusters over all observations. Asterisk (*) = 95% CI excludes zero.

### Table 1 — Accuracy with 95% CIs

| Method | CA accuracy % (95% CI) | CA near-freeze % | CO accuracy % (95% CI) | CO near-freeze % |
|---|---|---|---|---|
| T_a 1.0 °C | 62.0 (59.2–64.8) | 55.1 (51.0–59.4) | 79.2 (74.6–83.6) | 54.2 (45.9–62.4) |
| T_a 1.5 °C | 64.8 (62.0–67.5) | 57.0 (53.0–61.1) | 81.3 (77.0–85.5) | 58.1 (50.3–66.1) |
| T_d 0.0 °C | 80.1 (77.9–82.2) | 69.0 (65.2–72.8) | 86.3 (82.7–89.7) | 71.5 (64.3–78.7) |
| T_d 0.5 °C | 79.0 (76.8–81.2) | 68.5 (64.6–72.3) | 87.8 (84.7–90.8) | 74.3 (67.2–81.3) |
| T_w 0.0 °C | 75.5 (73.0–77.9) | 62.4 (58.7–66.2) | 86.1 (82.1–89.8) | 69.3 (60.8–77.1) |
| T_w 0.5 °C | 74.8 (72.2–77.3) | 60.5 (56.6–64.4) | 86.5 (82.7–89.9) | 70.4 (62.4–77.7) |
| T_w 1.0 °C | 73.6 (70.9–76.2) | 56.8 (52.7–61.0) | 85.3 (81.5–88.8) | 67.0 (59.1–74.9) |
| Bin. logistic (Jennings et al. 2018) | 72.0 (69.3–74.6) | 59.5 (55.6–63.4) | 80.7 (76.1–85.2) | 57.5 (49.5–65.8) |
| Bin. logistic (fitted, this domain) | 73.8 (71.1–76.4) | 60.0 (56.1–63.9) | 84.0 (80.1–87.7) | 63.7 (55.8–71.4) |
| **XGBoost + MRoS (this study)** | 91.2 (89.9–92.5) | 85.5 (82.8–88.3) | 91.4 (88.4–94.1) | 82.7 (76.4–88.3) |

### Table 2 — Snow / rain bias (%, Jennings et al. definition: 100·(n_pred/n_obs − 1))

| Method | CA snow bias % | CA rain bias % | CO snow bias % | CO rain bias % |
|---|---|---|---|---|
| T_a 1.0 °C | -39.5 (-44.5–-34.4) | 70.0 (57.2–84.7) | -22.2 (-28.0–-16.6) | 99.0 (68.5–139.0) |
| T_a 1.5 °C | -30.3 (-35.4–-25.2) | 53.6 (41.8–67.1) | -15.0 (-20.5–-9.5) | 66.7 (39.7–102.4) |
| T_d 0.0 °C | 4.2 (0.5–8.1) | -7.5 (-13.7–-0.9) | 0.5 (-4.5–5.5) | -2.1 (-23.1–21.9) |
| T_d 0.5 °C | 15.1 (11.0–19.5) | -26.7 (-32.5–-20.7) | 6.1 (1.6–10.9) | -27.1 (-45.2–-7.6) |
| T_w 0.0 °C | 24.2 (19.4–29.6) | -42.9 (-48.5–-37.2) | -3.5 (-8.6–1.4) | 15.6 (-5.5–42.7) |
| T_w 0.5 °C | 30.9 (25.8–36.7) | -54.8 (-59.7–-49.6) | 2.6 (-2.8–7.5) | -11.5 (-31.0–14.1) |
| T_w 1.0 °C | 35.5 (29.9–41.6) | -62.9 (-67.4–-57.9) | 7.2 (1.8–12.7) | -32.3 (-51.4–-9.3) |
| Bin. logistic (Jennings et al. 2018) | -0.6 (-5.5–4.4) | 1.1 (-7.3–10.2) | -15.2 (-21.4–-9.3) | 67.7 (38.9–104.4) |
| Bin. logistic (fitted, this domain) | 11.7 (6.7–17.2) | -20.7 (-28.5–-12.6) | 9.3 (3.7–14.8) | -41.7 (-58.4–-19.1) |
| **XGBoost + MRoS (this study)** | -1.6 (-3.5–0.4) | 2.9 (-0.7–6.4) | -4.4 (-7.8–-1.3) | 19.8 (5.6–37.8) |

### Table 3 — Model minus benchmark deltas (percentage points)

| Benchmark | CA Δ overall (pts) | CA Δ near-freeze (pts) | CO Δ overall (pts) | CO Δ near-freeze (pts) |
|---|---|---|---|---|
| T_a 1.0 °C | +29.2 (26.4–32.2) * | +30.4 (26.0–34.9) * | +12.2 (8.0–16.6) * | +28.5 (18.8–37.6) * |
| T_a 1.5 °C | +26.5 (23.6–29.3) * | +28.5 (24.3–32.9) * | +10.1 (6.2–14.1) * | +24.6 (16.0–33.0) * |
| T_d 0.0 °C | +11.1 (9.0–13.4) * | +16.5 (12.2–21.1) * | +5.1 (1.5–8.6) * | +11.2 (1.8–19.8) * |
| T_d 0.5 °C | +12.2 (10.0–14.5) * | +17.0 (12.7–21.5) * | +3.6 (0.2–6.9) * | +8.4 (-0.6–16.9) |
| T_w 0.0 °C | +15.7 (13.2–18.4) * | +23.2 (18.8–27.7) * | +5.3 (1.8–9.0) * | +13.4 (4.3–22.8) * |
| T_w 0.5 °C | +16.4 (13.8–19.1) * | +25.1 (20.4–29.8) * | +5.0 (1.8–8.3) * | +12.3 (4.1–21.0) * |
| T_w 1.0 °C | +17.6 (14.9–20.5) * | +28.8 (23.7–33.9) * | +6.1 (2.6–9.7) * | +15.6 (6.5–25.1) * |
| Bin. logistic (Jennings et al. 2018) | +19.2 (16.7–22.0) * | +26.1 (22.0–30.4) * | +10.7 (6.6–14.7) * | +25.1 (16.2–33.6) * |
| Bin. logistic (fitted, this domain) | +17.5 (14.8–20.2) * | +25.6 (21.3–30.1) * | +7.4 (3.8–11.2) * | +19.0 (10.0–27.9) * |

### Table 4 — Ablation deltas (baseline_full minus config; positive = feature helps)

| Config (baseline − config) | CA ΔAUC | CA Δ near-freeze acc (pts) | CO ΔAUC | CO Δ near-freeze acc (pts) |
|---|---|---|---|---|
| `no_mros_loocv` | +0.0291 (0.0214–0.0373) * | +9.6 (6.5–12.7) * | +0.0168 (0.0038–0.0328) * | -1.7 (-7.3–3.9) |
| `no_imerg_plp` | +0.0010 (-0.0014–0.0035) | +1.9 (0.5–3.3) * | +0.0055 (-0.0000–0.0117) | +0.0 (-2.8–2.6) |
| `no_elev` | +0.0094 (0.0055–0.0134) * | +2.7 (0.8–4.6) * | +0.0058 (-0.0018–0.0143) | +1.1 (-3.7–5.7) |
| `no_temp_air` | -0.0013 (-0.0026–0.0001) | +1.0 (-0.3–2.4) | -0.0028 (-0.0092–0.0026) | +1.1 (-1.9–4.3) |
| `no_temp_dew` | +0.0010 (-0.0009–0.0029) | +1.7 (0.0–3.3) | +0.0008 (-0.0036–0.0049) | +0.0 (-2.2–2.2) |
| `no_temp_wet` | +0.0002 (-0.0015–0.0018) | +1.7 (0.4–3.0) * | -0.0013 (-0.0066–0.0033) | +1.7 (-1.7–5.1) |
| `min_core_wplp` | -0.0004 (-0.0024–0.0015) | +2.4 (0.8–4.2) * | +0.0016 (-0.0054–0.0081) | +1.1 (-2.4–4.7) |
| `min_core_noplp` | +0.0016 (-0.0018–0.0050) | +3.6 (1.7–5.5) * | +0.0099 (0.0013–0.0197) * | +2.2 (-1.3–6.0) |
| `thermo_only` | +0.0582 (0.0462–0.0703) * | +15.7 (12.2–19.4) * | +0.0395 (0.0173–0.0625) * | +11.2 (3.1–19.5) * |

### Table 1b — Macro F1 with 95% CIs (new)

| Method | CA macro F1 | CA near-freeze macro F1 | CO macro F1 | CO near-freeze macro F1 |
|---|---|---|---|---|
| T_a 1.0 °C | 0.620 (0.592–0.647) | 0.469 (0.432–0.508) | 0.739 (0.687–0.788) | 0.537 (0.453–0.618) |
| T_a 1.5 °C | 0.645 (0.618–0.672) | 0.514 (0.475–0.552) | 0.747 (0.695–0.796) | 0.581 (0.500–0.660) |
| T_d 0.0 °C | 0.780 (0.757–0.802) | 0.689 (0.651–0.726) | 0.769 (0.719–0.816) | 0.684 (0.610–0.755) |
| T_d 0.5 °C | 0.756 (0.732–0.779) | 0.676 (0.637–0.713) | 0.771 (0.717–0.818) | 0.692 (0.614–0.760) |
| T_w 0.0 °C | 0.698 (0.671–0.725) | 0.618 (0.580–0.655) | 0.780 (0.726–0.832) | 0.673 (0.586–0.754) |
| T_w 0.5 °C | 0.674 (0.648–0.702) | 0.577 (0.538–0.616) | 0.763 (0.706–0.816) | 0.650 (0.563–0.732) |
| T_w 1.0 °C | 0.645 (0.617–0.674) | 0.510 (0.472–0.550) | 0.718 (0.658–0.774) | 0.559 (0.473–0.646) |
| Bin. logistic (Jennings et al. 2018) | 0.697 (0.670–0.723) | 0.581 (0.541–0.620) | 0.740 (0.690–0.790) | 0.575 (0.492–0.654) |
| Bin. logistic (fitted, this domain) | 0.700 (0.672–0.727) | 0.598 (0.558–0.636) | 0.677 (0.618–0.737) | 0.514 (0.439–0.593) |
| **MRoS-XGB (this study)** | 0.905 (0.891–0.920) | 0.855 (0.828–0.882) | 0.867 (0.821–0.906) | 0.820 (0.753–0.877) |

### Table 1c — Per-phase recall with 95% CIs (new)

| Method | CA snow recall | CA rain recall | CO snow recall | CO rain recall |
|---|---|---|---|---|
| T_a 1.0 °C | 0.505 (0.465–0.545) | 0.823 (0.777–0.867) | 0.762 (0.705–0.816) | 0.927 (0.872–0.976) |
| T_a 1.5 °C | 0.573 (0.532–0.613) | 0.779 (0.732–0.825) | 0.811 (0.759–0.861) | 0.823 (0.736–0.902) |
| T_d 0.0 °C | 0.865 (0.840–0.890) | 0.687 (0.646–0.725) | 0.918 (0.880–0.954) | 0.615 (0.504–0.717) |
| T_d 0.5 °C | 0.912 (0.891–0.930) | 0.576 (0.532–0.619) | 0.956 (0.928–0.981) | 0.531 (0.412–0.640) |
| T_w 0.0 °C | 0.929 (0.913–0.945) | 0.446 (0.403–0.490) | 0.897 (0.854–0.935) | 0.698 (0.595–0.797) |
| T_w 0.5 °C | 0.958 (0.945–0.970) | 0.377 (0.336–0.420) | 0.930 (0.888–0.963) | 0.573 (0.462–0.681) |
| T_w 1.0 °C | 0.971 (0.960–0.981) | 0.319 (0.280–0.361) | 0.946 (0.910–0.975) | 0.438 (0.326–0.551) |
| Bin. logistic (Jennings et al. 2018) | 0.778 (0.747–0.808) | 0.617 (0.569–0.665) | 0.806 (0.749–0.860) | 0.812 (0.725–0.894) |
| Bin. logistic (fitted, this domain) | 0.853 (0.825–0.880) | 0.532 (0.484–0.580) | 0.949 (0.910–0.978) | 0.354 (0.261–0.459) |
| **MRoS-XGB (this study)** | 0.923 (0.908–0.938) | 0.893 (0.868–0.916) | 0.925 (0.894–0.953) | 0.865 (0.789–0.930) |

### Table 3b — Model minus benchmark, macro F1 (new)

| Benchmark | CA Δ macro F1 | CA Δ NF macro F1 | CO Δ macro F1 | CO Δ NF macro F1 |
|---|---|---|---|---|
| T_a 1.0 °C | +0.286 (+0.257–+0.315) * | +0.385 (+0.341–+0.428) * | +0.128 (+0.079–+0.177) * | +0.283 (+0.188–+0.373) * |
| T_a 1.5 °C | +0.260 (+0.232–+0.289) * | +0.341 (+0.296–+0.386) * | +0.120 (+0.072–+0.168) * | +0.239 (+0.156–+0.319) * |
| T_d 0.0 °C | +0.125 (+0.102–+0.149) * | +0.166 (+0.124–+0.211) * | +0.098 (+0.045–+0.148) * | +0.135 (+0.043–+0.220) * |
| T_d 0.5 °C | +0.149 (+0.125–+0.175) * | +0.179 (+0.138–+0.222) * | +0.096 (+0.042–+0.152) * | +0.128 (+0.039–+0.215) * |
| T_w 0.0 °C | +0.207 (+0.180–+0.237) * | +0.237 (+0.193–+0.282) * | +0.086 (+0.036–+0.138) * | +0.146 (+0.056–+0.240) * |
| T_w 0.5 °C | +0.231 (+0.202–+0.261) * | +0.278 (+0.234–+0.323) * | +0.104 (+0.051–+0.158) * | +0.169 (+0.082–+0.259) * |
| T_w 1.0 °C | +0.260 (+0.231–+0.291) * | +0.344 (+0.297–+0.391) * | +0.149 (+0.087–+0.212) * | +0.261 (+0.162–+0.359) * |
| Bin. logistic (Jennings et al. 2018) | +0.209 (+0.182–+0.237) * | +0.274 (+0.232–+0.321) * | +0.127 (+0.079–+0.172) * | +0.245 (+0.158–+0.329) * |
| Bin. logistic (fitted, this domain) | +0.205 (+0.177–+0.235) * | +0.257 (+0.215–+0.303) * | +0.190 (+0.123–+0.253) * | +0.305 (+0.214–+0.393) * |

### Table 4b — Ablation absolute metrics with 95% CIs (new)

| Config | CA ROC AUC | CA macro F1 | CA NF ROC AUC | CA mix capture |
|---|---|---|---|---|
| `baseline_full` | 0.965 (0.957–0.973) | 0.905 (0.891–0.919) | 0.922 (0.902–0.941) | 0.30 (0.24–0.36) |
| `no_mros_loocv` | 0.936 (0.925–0.947) | 0.850 (0.832–0.867) | 0.861 (0.835–0.887) | 0.34 (0.28–0.40) |
| `no_imerg_plp` | 0.964 (0.956–0.972) | 0.899 (0.885–0.913) | 0.916 (0.894–0.935) | 0.29 (0.24–0.35) |
| `no_elev` | 0.956 (0.947–0.964) | 0.887 (0.873–0.902) | 0.905 (0.882–0.926) | 0.33 (0.28–0.38) |
| `no_temp_air` | 0.966 (0.959–0.974) | 0.902 (0.887–0.915) | 0.924 (0.904–0.943) | 0.29 (0.24–0.35) |
| `no_temp_dew` | 0.964 (0.956–0.972) | 0.900 (0.885–0.914) | 0.921 (0.900–0.940) | 0.33 (0.27–0.38) |
| `no_temp_wet` | 0.965 (0.957–0.972) | 0.900 (0.886–0.914) | 0.922 (0.902–0.941) | 0.32 (0.26–0.38) |
| `min_core_wplp` | 0.966 (0.958–0.973) | 0.900 (0.885–0.914) | 0.922 (0.902–0.941) | 0.32 (0.26–0.37) |
| `min_core_noplp` | 0.963 (0.955–0.971) | 0.895 (0.880–0.909) | 0.917 (0.895–0.937) | 0.33 (0.28–0.39) |
| `thermo_only` | 0.907 (0.892–0.921) | 0.817 (0.797–0.836) | 0.796 (0.761–0.830) | 0.49 (0.43–0.55) |

| Config | CO ROC AUC | CO macro F1 | CO NF ROC AUC | CO mix capture |
|---|---|---|---|---|
| `baseline_full` | 0.965 (0.941–0.982) | 0.867 (0.820–0.906) | 0.919 (0.864–0.962) | 0.33 (0.21–0.45) |
| `no_mros_loocv` | 0.948 (0.916–0.973) | 0.863 (0.820–0.903) | 0.896 (0.836–0.946) | 0.31 (0.20–0.43) |
| `no_imerg_plp` | 0.959 (0.935–0.978) | 0.871 (0.825–0.910) | 0.903 (0.841–0.951) | 0.31 (0.20–0.43) |
| `no_elev` | 0.959 (0.936–0.977) | 0.862 (0.818–0.901) | 0.912 (0.857–0.956) | 0.34 (0.22–0.47) |
| `no_temp_air` | 0.968 (0.948–0.983) | 0.862 (0.817–0.902) | 0.929 (0.880–0.966) | 0.36 (0.24–0.49) |
| `no_temp_dew` | 0.964 (0.941–0.981) | 0.865 (0.821–0.904) | 0.921 (0.867–0.963) | 0.31 (0.20–0.44) |
| `no_temp_wet` | 0.966 (0.946–0.982) | 0.864 (0.822–0.901) | 0.919 (0.865–0.962) | 0.33 (0.21–0.46) |
| `min_core_wplp` | 0.963 (0.940–0.981) | 0.862 (0.816–0.903) | 0.924 (0.873–0.965) | 0.39 (0.26–0.52) |
| `min_core_noplp` | 0.955 (0.930–0.974) | 0.853 (0.803–0.896) | 0.901 (0.842–0.948) | 0.41 (0.29–0.53) |
| `thermo_only` | 0.925 (0.896–0.951) | 0.798 (0.746–0.846) | 0.791 (0.715–0.862) | 0.53 (0.40–0.65) |

### Table 5 — Sensitivity of the headline delta CI to resampling unit

| Resampling unit | CA Δacc vs best benchmark, CI | CA CI width | CO Δacc vs best benchmark, CI | CO CI width |
|---|---|---|---|---|
| 15 km × day blocks | 9.0–13.2 pts | 4.2 pts | 0.4–7.0 pts | 6.6 pts |
| 30 km × day blocks | 9.0–13.4 pts | 4.4 pts | 0.2–6.9 pts | 6.7 pts |
| 60 km × day blocks | 9.0–13.4 pts | 4.4 pts | 0.2–7.1 pts | 6.9 pts |
| iid (no clustering) | 9.4–12.9 pts | 3.4 pts | 0.8–6.7 pts | 5.9 pts |

CA model ROC AUC: 0.965 (0.957–0.972); CO: 0.965 (0.942–0.982). Mix capture (Gaussian band, all test obs): CA 0.30 (0.25–0.36), CO 0.33 (0.21–0.45) — vs. 0% mix capture for all binary benchmarks and for the ML methods in Jennings et al. (2025).

## Key takeaways

1. **CA is decisive on every metric.** All 36 paired model-minus-benchmark differences (9 benchmarks × {overall, near-freezing} × {accuracy, macro F1}) exclude zero, and no replicate out of 5,000 favoured a benchmark (p < 0.0002). Vs. the best benchmark (T_d 0.0 °C): Δ overall accuracy = +11.1 pts (9.0–13.4); even the CI floor is ~9 pts. Near-freezing advantage: +16.5 pts (12.2–21.1).
2. **CO is metric-dependent, and macro F1 is the honest metric there.** 35 of 36 paired differences exclude zero. On overall accuracy the gain vs. the best benchmark (T_d 0.5 °C) is +3.6 pts (0.2–6.9, p = 0.021) and the near-freezing accuracy gain, +8.4 pts (−0.6 to 16.9), is the *single* comparison that includes zero. On macro F1 every CO difference excludes zero (p ≤ 0.0002), e.g. +0.096 (0.042–0.152) vs. T_d 0.5 °C. Accuracy understates the model in CO because 82% of observations are snow; macro F1 exposes the benchmarks' rain-recall collapse (T_d 0.5 °C rain recall 0.531 (0.412–0.640) vs. the model's 0.865 (0.789–0.930)). **Lead with macro F1 for CO, not accuracy.**
3. **Ranking degrades less than deciding.** Near-freezing ROC AUC is 0.922 (0.903–0.941) CA and 0.920 (0.865–0.962) CO, against 0.965 overall in both — the model still orders snow above rain well in the regime where committing to a call is unreliable.
4. **Bias:** in CA the model is the only method with both bias CIs including zero. In CO it shows a small but significant snow underprediction (−4.4%, CI −7.8 to −1.3) and rain overprediction (+19.8%, CI 5.6–37.8); the latter rests on only 96 rain obs but excludes zero and should be acknowledged.
5. **Ablations:** in CA, removing the MRoS LOOCV surfaces produces the largest significant drops (ΔAUC +0.029, Δ macro F1 +0.056, Δ near-freeze acc +9.6 pts, Δ near-freeze AUC +0.061, all excluding zero); `thermo_only` is worse still on every metric; individual thermodynamic variables are interchangeable (CIs straddle zero, except `no_temp_wet` near-freezing accuracy); elevation and IMERG matter mainly near freezing. In CO only `no_mros_loocv` (AUC) and `thermo_only` (all metrics) are separable from zero, plus `no_imerg_plp` on near-freezing AUC (+0.017, 0.001–0.036) — small-sample effect, same direction as CA.
6. **Mix capture responds to feature content.** `thermo_only` *raises* mix capture (CA 0.49 (0.43–0.55), CO 0.53 (0.40–0.65) vs. baseline 0.30 and 0.33), with the baseline-minus-config delta excluding zero in both domains. A weaker model is less confident, so its envelope is wider and catches more mix — which is why mix capture must be read jointly with pure-phase accuracy and not maximised alone.
7. **Sensitivity:** CIs are essentially unchanged across 15/30/60 km blocks and ~13–28% wider than iid. The clustering correction matters, but no conclusion depends on the block-size choice.

## Reproduce

```
python ML_pipeline/bootstrap_cis.py --region CA
python ML_pipeline/bootstrap_cis.py --region CO
# options: --n-boot 10000 --block-km 30 --seed 42 --data-dir /path/to/outputs
#   (--data-dir, or the MROS_OUTPUTS env var, overrides the hard-coded Windows path)
```

Outputs per region in `benchmarking_v1/bootstrap/`: `bootstrap_benchmark_metrics_ci.csv`, `bootstrap_benchmark_deltas_ci.csv`, `bootstrap_ablation_metrics_ci.csv` (new), `bootstrap_ablation_deltas_ci.csv`, `bootstrap_sensitivity_block_size.csv`, `bootstrap_run_config.json`.

Figures (per region, in `benchmarking_v1/bootstrap/graphics/`): accuracy-by-T_air with CI ribbons, model-minus-benchmark delta ribbons, and delta forest plots (overall + near-freezing).
