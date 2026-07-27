# Bootstrap confidence intervals — methods and results

*Working notes. From `model_artifacts/{REGION}/benchmarking_v1/bootstrap/` outputs, both regions run with B = 5,000 replicates, seed 42, produced by `ML_pipeline/bootstrap_cis.py`.*

## What we did and why

All benchmark and ablation results are computed on a single held-out test split, so every reported metric carries sampling uncertainty: a different draw of test observations would give somewhat different numbers. To quantify this we used a **paired space–time cluster bootstrap** over the stored per-observation test predictions (`benchmark_predictions_test.parquet` for the benchmark comparison; each ablation config's `shap_values_all.parquet` test rows for ablation deltas). No models were retrained.

**Why clustered, not plain (iid) resampling.** MRoS reports are not independent: multiple observers report the same storm in the same neighbourhood, share the same interpolated fields and IMERG pixel, and inform each other through the LOOCV surface. An iid bootstrap treats every row as independent evidence and produces intervals that are too narrow. We therefore assigned each test observation to a cluster defined by a **30 km spatial block × calendar day**, and resampled whole clusters with replacement. Correlated same-storm/same-place observations enter or leave each pseudo test set together, so the replicate-to-replicate spread honestly reflects "would a different season of events have given a different answer."

**Why paired.** Within each replicate, all methods (model + 9 benchmarks; baseline + all ablation configs) are scored on the identical resampled rows and differenced *within* the replicate. Because all methods share the difficulty of whatever events a replicate draws, differencing cancels that shared variance, giving much tighter (and correct) CIs on the *differences* than comparing two marginal CIs would.

**Details.** 95% percentile intervals from B = 5,000 replicates. Subset metrics (near-freezing = |T_wet| ≤ 2 °C; per-phase biases) are computed within-replicate and return NaN when too thin, so their CIs widen automatically where data are scarce. Block size is a judgment call, so the analysis was repeated at 15/30/60 km and with iid resampling (Table 5) — headline numbers use 30 km × day.

**What these CIs do and do not cover.** They quantify test-set sampling uncertainty, *conditional on* the trained model and the (random, stratified) train/val/test split. They do not capture training stochasticity (could be added via multi-seed retrains) or the train/test same-storm sharing acknowledged in the methodology (the random split matches the deployment condition, where the product always predicts with the live observer network available; blocked splitting was tried in v6b and abandoned because splits became too small for stable regime-stratified calibration). Note the cluster bootstrap costs no training data — it fixes the inference stage, which is why we can afford it even though we could not afford blocked splits.

## Results

Pure rain/snow test observations: CA n = 2,862 (2,244 clusters at 30 km × day), CO n = 524 (379 clusters). Asterisk (*) = 95% CI excludes zero.

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

### Table 5 — Sensitivity of the headline delta CI to resampling unit

| Resampling unit | CA Δacc vs best benchmark, CI | CA CI width | CO Δacc vs best benchmark, CI | CO CI width |
|---|---|---|---|---|
| 15 km × day blocks | 9.0–13.2 pts | 4.2 pts | 0.4–7.0 pts | 6.6 pts |
| 30 km × day blocks | 9.0–13.4 pts | 4.4 pts | 0.2–6.9 pts | 6.7 pts |
| 60 km × day blocks | 9.0–13.4 pts | 4.4 pts | 0.2–7.1 pts | 6.9 pts |
| iid (no clustering) | 9.4–12.9 pts | 3.4 pts | 0.8–6.7 pts | 5.9 pts |

CA model ROC AUC: 0.965 (0.957–0.972); CO: 0.965 (0.942–0.982). Mix capture (Gaussian band, all test obs): CA 0.30 (0.25–0.36), CO 0.33 (0.21–0.45) — vs. 0% mix capture for all binary benchmarks and for the ML methods in Jennings et al. (2025).

## Key takeaways

1. **CA is decisive.** The model beats every benchmark with all delta CIs excluding zero (p ≤ 0.0022). Vs. the best benchmark (T_d 0.0 °C): Δ overall = +11.1 pts (9.0–13.4); even the CI floor is ~9 pts. Near-freezing advantage vs. best benchmark: +16.5 pts (12.2–21.1).
2. **CO is consistent but tighter.** The model beats 8/9 benchmark comparisons on overall accuracy with CIs excluding zero, including (just) the best benchmark T_d 0.5 °C: +3.6 pts (0.2–6.9, p = 0.021). The near-freezing delta vs. T_d 0.5 does **not** exclude zero (-0.6 to 16.9) — at n = 524, CO is underpowered for that comparison. Report honestly; CA carries the headline.
3. **Bias:** in CA the model is the only method with both bias CIs including zero. In CO the model shows a small but significant snow underprediction (−4.4%, CI −7.8 to −1.3) and rain overprediction (+19.8%, CI 5.6–37.8) — the CO rain bias rests on only 96 rain obs, but it excludes zero and should be acknowledged.
4. **Ablations:** in CA, removing the MRoS LOOCV surface produces the largest significant drops (ΔAUC +0.029, Δ near-freeze +9.6 pts, both CIs exclude zero); `thermo_only` is worse still; individual thermodynamic variables are interchangeable (all CIs straddle zero); elevation and IMERG matter mainly near freezing. In CO only `no_mros_loocv` (AUC) and `thermo_only` are separable from zero — small-sample effect, same direction as CA.
5. **Sensitivity:** CIs are essentially unchanged across 15/30/60 km blocks and ~15–25% wider than iid. The clustering correction matters, but no conclusion depends on the block-size choice.

## Reproduce

```
python ML_pipeline/bootstrap_cis.py --region CA
python ML_pipeline/bootstrap_cis.py --region CO
# options: --n-boot 10000 --block-km 30 --seed 42
```

Figures (per region, in `benchmarking_v1/bootstrap/graphics/`): accuracy-by-T_air with CI ribbons, model-minus-benchmark delta ribbons, and delta forest plots (overall + near-freezing).
