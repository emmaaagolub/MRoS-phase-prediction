# Methodology Summary

---

## 1. Input Data

### Observation-level (MRoS)
- Hourly MRoS phase reports filtered to valid observations within the study domain and time period
- Phase classes (snow / mixed / rain) converted to one-hot binary indicators to preserve class structure and avoid imposing an artificial ordinal scale
- Station data includes: hourly air temperature, dewpoint, wet-bulb temperature, relative humidity

### Gridded
- Common 1-km DEM-derived projected grid (EPSG:26911); grid-cell coordinates and elevations extracted from DEM
- Study period: October 2024 – May 2025 (5,160 common hourly timesteps across interpolation, IMERG, and PRISM grids)
- Grid dimensions: 142 × 251 cells at 1-km spacing

---

## 2. Spatial Interpolation — Two Parallel Frameworks

Both frameworks process station meteorological data and MRoS phase observations separately, producing identical output types. IDW was used in v7.

### A. Station-variable interpolation

**IDW approach:**
- Temperature variables detrended against station elevation to estimate hourly lapse rate
- Residuals interpolated via inverse-distance weighting, then retreneded using DEM elevations
- Relative humidity interpolated directly without lapse-rate adjustment

**Kriging approach:**
- Same elevation-detrend / residual / retrend procedure for temperature variables
- Relative humidity uses terrain-aware kriging with elevation as optional drift term
- Variogram parameters estimated from pooled hourly residuals for stability during sparse observation periods

### B. MRoS phase interpolation

**IDW approach:**
- Each one-hot class indicator (snow, mixed, rain) interpolated independently via IDW
- Elevation optionally included as a third spatial coordinate to reflect topographic similarity
- Resulting class-support fields clipped to non-negative values and normalized to sum to 1

**Indicator kriging approach:**
- Each one-hot indicator treated as a binary spatial field and kriged independently
- Outputs renormalized to ensure closure across three classes

**Final products:** hourly gridded probability surfaces — p_snow, p_mix, p_rain — for each method

---

## 3. Leakage Prevention via LOOCV

This step is critical: MRoS observations serve as both potential predictors *and* training labels, so naive use of full-grid interpolated surfaces as predictors would be circular.

**Full-grid interpolation** (all MRoS observations per hour) → domain-wide phase probability surfaces for mapping and gridded prediction only

**Pointwise LOOCV interpolation** (for ML training):
- For each MRoS observation, that observation is withheld from its hour's interpolation dataset
- Remaining observations used to interpolate phase probabilities back to the held-out location
- Produces a leakage-safe (p_snow, p_mix, p_rain) triplet at each observation point
- These LOOCV predictions stored in a separate table for use as ML predictors only

---

## 4. Output Products

For each interpolation method (IDW and kriging):
- **Hourly gridded NetCDF:** continuous station-derived predictor fields + three MRoS probability surfaces
- **Tabular LOOCV file:** one row per MRoS observation, with observed phase class and leakage-safe interpolated probabilities at that location and hour
- **Cross-validation metrics:** summary statistics comparing IDW vs. kriging interpolation performance

---

## 5. ML Framework Overview

Two model architectures were developed and compared.

### 5A. Multiclass XGBoost (snow / rain / mixed)

**Predictor assembly:**
- Gridded predictors sampled at each MRoS observation location (nearest-neighbor from hourly NetCDF stacks): air temperature, dewpoint, wet-bulb temperature, relative humidity, elevation
- External gridded predictors: IMERG precipitation-phase likelihood; selected PRISM temperature variables
- LOOCV MRoS phase probabilities joined as additional point-level predictors

**Training target:** raw MRoS phase observations used directly as class labels — not derived from an interpolated surface

**Spatial blocking:**
- Domain divided into regular spatial blocks in projected coordinates
- All observations within a block assigned together to train/val/test to reduce spatial autocorrelation
- Approximate split: 70% train / 15% val / 15% test

**Model:** XGBoost in multiclass probabilistic mode, class weighting applied inversely to class frequency, additional sample weighting near the rain–snow transition zone (0–4°C wet-bulb)

### 5B. Binary XGBoost (snow vs. rain, with derived mixed)

The binary approach reframes classification around the two physically unambiguous endmembers (pure snow and pure rain), treating mixed phase as a derived transition state inferred from model uncertainty rather than a directly learned class. Motivated by the observation that mixed precipitation often reflects thermodynamic transitions, subgrid heterogeneity, or observational ambiguity, and may not form a stable, separable cluster in feature space.

**Training target:** only pure snow and pure rain observations used to fit the binary classifier; mixed observations withheld from supervised fitting

**Feature set (10 predictors):**
- `temp_air`, `prism_tair`, `temp_dew`, `temp_wet`, `rh`
- `imerg_plp`
- `elev`
- `mros_p_snow_loocv`, `mros_p_mix_loocv`, `mros_p_rain_loocv`

**Dataset (v7):**
- 6,150 total observations (48% snow, 40% rain, 12% mix)
- Pure-phase subset for binary fitting: 3,788 train, 812 val/test combined (snow/rain only)
- Split sizes: train 4,304 | val 923 | test 923

---

## 6. Binary Model Components

### 6a. Data Splitting

**v6 (spatial blocking):**
- Domain divided into regular spatial blocks; all observations in a block assigned together to the same split
- 14 spatial blocks; approximate 70/15/15 split

**v7 (random stratified split):**
- Switched to randomized stratified split (`sklearn.train_test_split`, stratified on `phase_full`)
- Two-step: train vs. (val+test), then val vs. test (50/50 of the held-out 30%)
- Fixed seed: 42
- *Scientific justification:* the model learns the relationship between instantaneous meteorological state variables and phase, which is physically stable across storm events — a given Twet predicts snow with approximately the same probability regardless of synoptic system. Randomized split preserves the overall distribution of meteorological states across splits, which is the primary requirement for unbiased generalization evaluation of a physics-based feature set. Consistent with Jennings et al. (2025).
- *Known limitation:* observations from the same storm event may appear in both train and test, so test performance may be mildly optimistic relative to deployment on a genuinely novel future season. More complex event-based blocking would risk producing splits too small to support stable fitting and calibration given the dataset size (6,150 obs across 82 active days).

**Split diagnostics (v7):**
| Split | N rows | Twet mean | Elev mean |
|---|---|---|---|
| Train | 4,304 | +0.12°C | 1,648 m |
| Val | 923 | −0.01°C | 1,650 m |
| Test | 923 | +0.03°C | 1,659 m |

Class balance is well-preserved across splits (snow ~48%, rain ~40%, mix ~12% in all three).

### 6b. Class Weighting — `scale_pos_weight` Sweep

**Problem:** MRoS network is geographically skewed toward higher-elevation, snowier sites → more snow than rain observations → snow recall substantially higher than rain recall without correction.

**Approach:** sweep `scale_pos_weight` over candidates {0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00}
- For each value, train a lightweight probe model (200 rounds, same hyperparameters)
- Evaluate snow recall, rain recall, and |snow recall − rain recall| on the validation pure-phase subset
- Select value minimizing recall gap — targets recall symmetry directly rather than class frequency

**v6 result:** selected 0.75 (mild snow downweight)
**v7 result:** selected 0.50 (stronger downweight) — sweep showed 0.50 achieves near-equal recall (snow 0.890, rain 0.900, gap 0.010) while higher values progressively trade rain recall for snow recall

**XGBoost hyperparameters:**
- `objective: binary:logistic`, `eval_metric: logloss`
- `max_depth: 6`, `eta: 0.05`, `subsample: 0.8`, `colsample_bytree: 0.8`
- `min_child_weight: 5`, `lambda: 1.0`, `tree_method: hist`
- `num_boost_round: 2000`, `early_stopping_rounds: 50` (monitored on val logloss)
- Final model: best iteration 145, best val logloss 0.259

### 6c. Probability Calibration

Raw XGBoost scores reflect ranking skill but are not calibrated probabilities.

**v6 (LOBO Platt scaling):**
- Leave-one-block-out cross-fitting across all train+val spatial blocks
- Fitted a Platt (logistic regression) calibration model on remaining blocks' raw margins, applied to held-out block
- Final Platt model applied forward to test set
- Limitation: spatial distribution shift between val and test subsets caused calibration to generalize imperfectly

**v7 (K-fold isotonic regression, regime-stratified):**
- With a randomized split, no block structure to cross-fit over; switched to 10-fold cross-fitting within the train+val pure-phase pool
- **Regime stratification:** two separate calibration models fitted by wet-bulb temperature regime:
  - *Clear-phase*: |Twet| > 2°C (n=2,025 train+val obs) — well-separated scores
  - *Near-freezing*: |Twet| ≤ 2°C (n=2,575 train+val obs) — scores compressed near 0.5
  - Regime boundary of 2°C consistent with Sims & Liu (2015) empirical uncertainty range
- Isotonic regression (non-parametric) instead of Platt scaling — more flexible, makes no distributional assumptions
- Cross-fitting procedure: pool divided into 10 folds; for each fold, calibration models fitted on remaining 9 folds and applied to held-out fold → every observation gets a calibrated probability from independent data
- Final calibration models (one per regime) re-fitted on full train+val pool and applied to test set
- XGBoost booster not re-trained at any point

### 6d. 3-Class Derivation — Wet-Bulb-Conditioned Gaussian Half-Band

Mixed phase is not a directly learned class. It is derived post-hoc from the calibrated binary probability using a temperature-adaptive uncertainty band.

**Formulation:**
```
half_band(Twet) = base_half_band + extra_half_band × exp(−Twet² / 2σ²)

rain_thresh  = 0.5 − half_band(Twet)
snow_thresh  = 0.5 + half_band(Twet)
```

- p(snow) ≥ snow_thresh → snow label
- p(snow) ≤ rain_thresh → rain label
- rain_thresh < p(snow) < snow_thresh → mix/uncertain label

The band widens near 0°C wet-bulb (where phase is genuinely ambiguous) and narrows away from freezing (where phase is more deterministic). Physically motivated by Harder & Pomeroy (2013), Sims & Liu (2015), Jennings et al. (2018, 2023, 2025).

**Parameter optimization:**
- Grid search over `base_half_band` × `extra_half_band` × `sigma`
- Optimized on full validation set (including observed mixed-phase observations)
- Objective: weighted composite of mix capture rate and pure-phase confident accuracy
  - `composite = 0.5 × mix_capture_rate + 0.5 × pure_conf_accuracy`
- Feasibility constraints (prevent degenerate solutions):
  - Global pure-phase coverage ≥ 50%
  - Near-freezing (|Twet| ≤ 1°C) pure-phase coverage ≥ 35%
  - `base + extra ≤ 0.40` (prevents band from spanning full probability range at 0°C)
- The near-freezing and max-width constraints were added after v6a found a degenerate solution (base=0.20, extra=0.30, σ=2.0°C → everything labeled mix at Twet ≈ 0°C)

**v7 selected parameters** (same grid/constraints as v6b):

| Parameter | Value |
|---|---|
| `base_half_band` | 0.20 |
| `extra_half_band` | 0.15 |
| `sigma` | 2.0°C |
| rain_thresh at Twet=0°C | 0.15 |
| snow_thresh at Twet=0°C | 0.85 |
| rain_thresh far from freezing | 0.30 |
| snow_thresh far from freezing | 0.70 |
| Mix capture rate (val) | 0.279 |
| Pure-phase coverage (val) | 0.889 |
| Pure-phase confident accuracy (val) | 0.934 |

*Note: v7 uses σ=2.0°C vs. v6b's σ=4.0°C. This is because sigma search grid in v7 was bounded to (1.0, 1.5, 2.0)°C following Sims & Liu (2015) empirical range — v6b's grid allowed larger sigma values.*

---

## 7. Evaluation Protocol

### Pure-phase binary evaluation
- Calibrated probability thresholded at 0.5 for hard binary predictions
- Metrics: confusion matrix, precision, recall, F1, balanced accuracy, ROC AUC, average precision, Brier score, log loss

### Full-dataset 3-class evaluation
- Uncertainty-derived labels (snow / mix / rain) applied to full dataset including observed mixed-phase observations
- Metrics: same as above plus mix-specific: true-mix capture rate, pure-phase confident coverage and accuracy, overall confident fraction

### Near-freezing stress tests
- |Tair| ≤ 1°C subset
- |Twet| ≤ 1°C subset *(tighter and more physically meaningful)*
- Evaluated separately because phase discrimination is hardest and most consequential in the transition zone

### Calibration evaluation
- Brier score and log loss: probability reliability
- ROC AUC and average precision: class-ranking skill (unaffected by calibration)
- Confidence-binning analysis: accuracy within probability bins

### Saved outputs
- Trained binary XGBoost booster, regime-stratified isotonic calibration models
- Predictor metadata, split tables, `scale_pos_weight` sweep results (CSV + plot)
- Test-set predictions with raw and calibrated probabilities, hard binary labels, derived 3-class assignments, per-sample transition scores, binary entropy
- Threshold optimization results and Gaussian band shape plot
- Model metadata JSON documenting all parameter choices, calibration coefficients, threshold values, and methodology notes
- SHAP handoff files: `xgb_binary_phase_model.bin`, `shap_handoff_features.parquet`, `shap_handoff_metadata.parquet` (includes `is_mix_uncertainty` boolean index), `shap_handoff_readme.txt`

---

## 8. SHAP Analysis (v7b)

**Approach:**
- SHAP values computed on the full dataset using `shap.TreeExplainer(booster)`
- Binary model outputs one SHAP value per feature per observation (explaining p(snow) vs. p(rain)); no separate mix output head
- 3-class phase labels assigned via Gaussian band thresholds → observations filtered by label to get SHAP profiles per phase
- Mix SHAP profile = SHAP values for observations that fell inside the uncertainty band (not a separate computation)
- Interpretive framing: what features pushed p(snow) into the ambiguous middle range vs. what drove confident pure-phase calls?

**Output table columns:** observation metadata (time, station, elevation, Twet), all feature values, raw p(snow), calibrated p(snow), 3-class label, one SHAP column per feature

---

## Version Comparison Summary

| Aspect | v6 | v7 |
|---|---|---|
| **Data splitting** | Spatial block assignment (LOBO-style) | Random stratified split (sklearn, seed=42) |
| **Calibration method** | LOBO Platt scaling (logistic regression on margins) | 10-fold K-fold isotonic regression |
| **Calibration stratification** | None | Regime-stratified by |Twet| (clear vs. near-freezing, boundary 2°C) |
| **scale_pos_weight** | 0.75 | 0.50 |
| **Gaussian band sigma** | 4.0°C (v6b) | 2.0°C (bounded grid per Sims & Liu 2015) |
| **Selected band params** | base=0.20, extra=0.15, σ=4.0°C | base=0.20, extra=0.15, σ=2.0°C |
| **Transition weight** | Applied (1.0 + α for 0–4°C Twet zone) | Present in code but not used in v7 final |
| **SHAP** | Not run | Run on full dataset, phased-labeled stratification |
