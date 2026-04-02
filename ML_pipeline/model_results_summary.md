# Model Results Summary

---

## Metric Glossary

### Pure-Phase Binary Discrimination
| Metric | What it measures |
|---|---|
| ROC AUC | Ranking quality across all thresholds — how well the model separates snow from rain |
| Average precision | Same, but weighted toward high-confidence predictions |
| Brier score | Numerical accuracy of probabilities; penalizes overconfident wrong calls |
| Log loss | Like Brier score but punishes confident errors more severely |

### Hard Classification (post-threshold)
| Metric | What it measures |
|---|---|
| Accuracy | Fraction of all predictions correct |
| Balanced accuracy | Accuracy averaged equally across classes (controls for class imbalance) |
| Macro F1 | Per-class F1 averaged equally — rare classes (mix) count as much as common ones |
| Precision (per class) | When model predicts this phase, how often it's right |
| Recall (per class) | Of all true events of this phase, how many the model caught |
| F1 (per class) | Harmonic mean of precision and recall |
| Recall gap | Asymmetry between snow and rain recall; ideally near zero |

### Uncertainty Product (3-class derived output)
| Metric | What it measures |
|---|---|
| True-mix capture rate | Of all observed mixed events, how many were flagged as uncertain |
| Pure-phase confident coverage | Of all true pure-phase events, how many got a confident snow/rain label |
| Pure-phase confident accuracy | Of confident pure-phase calls, how often correct |
| Overall confident fraction | Share of all observations receiving any confident label |

### Near-Freezing Subsets
- **\|Tair\| ≤ 1°C** — performance where air temp is within 1°C of freezing
- **\|Twet\| ≤ 1°C** — performance where wet-bulb temp is within 1°C of freezing *(tighter and more physically meaningful)*

---

## Multiclass XGBoost

### Setup
- Direct 3-class classification (snow / rain / mix)
- Hard labels assigned by argmax of predicted probabilities

### Key Results
- **Overall accuracy: 0.76**
- Snow: precision 0.84, recall 0.87, **F1 0.86** — best-performing class
- Rain: precision 0.87, recall 0.72, **F1 0.79** — reasonable
- Mix: precision 0.23, recall 0.33, **F1 0.27** — substantially weaker

### Key Takeaways
- Model captures dominant rain-vs-snow structure reasonably well
- Mix class fails badly — model cannot learn a clean decision boundary around a physically diffuse category
- Mixed precipitation reflects thermodynamic transitions, spatial heterogeneity, and labeling ambiguity — not a crisp separable class
- These results motivated the pivot to a binary + uncertainty-derivation framework

---

## Binary v5

### Setup
- Binary classifier: snow vs. rain only
- Mixed phase **derived** from uncertainty in probabilistic output (probability band around 0.5), not directly learned

### Pure-Phase Performance (test set)
| Metric | Value |
|---|---|
| Accuracy | 0.881 |
| Balanced accuracy | 0.863 |
| Macro F1 | 0.873 |
| ROC AUC | 0.962 |
| Average precision | 0.973 |

- Snow recall especially high — strong snow identification
- Rain recall somewhat lower — some overcalling of snow in borderline cases

### Calibration
- Post-hoc calibration improved Brier score / log loss on **validation** set
- On **test** set, calibration slightly *worsened* probability metrics (ROC AUC unchanged)
- Likely mild overfitting of calibration mapping; treat calibrated probabilities cautiously

### Uncertainty-Derived 3-Class (test set)
| Metric | Value |
|---|---|
| Accuracy | 0.768 |
| Balanced accuracy | 0.612 |
| Macro F1 | 0.618 |

- Snow and rain F1 remain reasonable; mix recall low (conservative band, under-identifies mix)
- Confident pure-phase predictions were highly accurate — uncertainty framework shows promise as a confidence-aware output

### Near-Freezing (\|Twet\| ≤ 1°C, test set)
| Phase | F1 |
|---|---|
| Snow | 0.726 |
| Rain | 0.493 |
| Mix | 0.262 |
| Overall accuracy | 0.586 |

### Key Takeaways
- Big improvement over multiclass for pure-phase discrimination
- Calibration generalization is a concern
- Mix derivation conservative — only captures ~1/3 of true mix events
- Near-freezing performance degrades substantially, especially for rain

---

## Binary v6a

### Setup
- Same binary framework as v5
- Added Gaussian uncertainty band centered at Twet = 0°C to improve mix flagging
- Band width parameterized as `base_half_band + extra_half_band × exp(−Twet²/2σ²)`

### 3-Class Output (test set) vs. v5
| Metric | v5 | v6a | Change |
|---|---|---|---|
| Accuracy | 0.768 | 0.671 | ↓ |
| Balanced accuracy | 0.612 | 0.652 | ↑ |
| Mix F1 | 0.27 | 0.306 | ↑ |
| Mix recall | 0.33 | 0.636 | ↑↑ |

- Accuracy drop is **expected** — the wider band is correctly pushing borderline cases into mix instead of forcing them into pure-phase labels
- When confident (`is_confident = True`): **97.1% accuracy on test set** — high-confidence calls are extremely reliable

### Near-Freezing Results
**\|Tair\| ≤ 1°C:**
- Snow recall: 0.831 ✓
- Mix recall: 0.528 ✓
- Rain recall: 0.000 ✗ — rain events near 0°C Tair are being correctly pushed into mix band (psychrometric constraint), but registers as failure on hard-label eval

**\|Twet\| ≤ 1°C — degenerate solution:**
- Mix recall: 0.925 — but mix precision only 0.199 (~80% of mix predictions are actually pure-phase swept up by the band)
- Snow recall: 0.307 / Rain recall: 0.052 — essentially everything flagged as mix
- Root cause: optimizer found `base + extra = 0.50`, meaning the half-band spans the full [0,1] probability range at Twet = 0°C → every observation labeled mix regardless of p(snow)

### Key Takeaways
- Gaussian band concept is correct but optimizer hit a degenerate edge case
- No constraint prevented `base + extra` from reaching 0.50
- Need guardrail: `base_half_band + extra_half_band < 0.45`
- High-confidence accuracy (97.1%) is the most important positive result from v6a

---

## Binary v6b

### Setup
- Added constraint: `base_half_band + extra_half_band < 0.45` to prevent degenerate solution
- Constrained optimizer landed on: `base=0.20, extra=0.15, sigma=4.0°C`
  - Large sigma (4°C) = broad uncertainty plateau, consistent with empirical literature (Sims & Liu 2015: ~±2–3°C range)
  - Narrower `extra` (0.15 vs. 0.30 in v6a) = band reaches 0.35 half-width at 0°C anchor

### 3-Class Output (test set) vs. v5 and v6a
| Metric | v5 | v6a | v6b |
|---|---|---|---|
| Accuracy | 0.768 | 0.671 | 0.767 |
| Balanced accuracy | 0.612 | 0.652 | 0.656 |
| Macro F1 | 0.618 | — | 0.653 |
| Mix F1 | — | 0.306 | 0.312 |

- Accuracy essentially fully recovered vs. v5
- Balanced accuracy and macro F1 both improved vs. v5
- Snow and rain F1 at or above v5 levels
- **Key achievement**: constraints prevented 3-class degradation while still improving mix behavior

### Near-Freezing Results
**\|Twet\| ≤ 1°C** — major improvement vs. v6a:
| Metric | v6a | v6b |
|---|---|---|
| Snow recall | 0.307 | 0.827 |
| Rain recall | 0.052 | 0.338 |
| Mix recall | 0.925 | 0.493 |
| Overall accuracy | — | 0.594 |

- v6b mix recall (0.493) is a more honest number — flagging ~half of true near-freezing mix events while still making reliable snow calls

**Persistent problem — \|Tair\| ≤ 1°C rain recall:**
- Rain recall ≈ 0.000–0.026 in both v6a and v6b
- Structural, not a tuning issue: rain events near 0°C Tair are psychrometrically forced to near-0°C Twet, so they fall inside the uncertainty band regardless of parameterization
- Only 39 such events on test set — not numerically dominant, but worth flagging as expected limitation

### Key Takeaways
- Best 3-class output so far — better than v5 on almost every metric simultaneously
- Broad sigma (4°C) is physically well-motivated
- Near-freezing performance substantially recovered
- Warm-side rain near 0°C Tair is an irreducible ambiguity with surface met alone

---

## Binary v7

### Setup
- Further development of binary framework (specific changes not detailed in notes)
- Includes preliminary SHAP analysis

### Pure-Phase Performance (test set)
| Metric | Value |
|---|---|
| ROC AUC | 0.97 |
| Average precision | 0.97 |
| Accuracy | 0.919 |
| Snow balanced recall | 0.921 |
| Rain balanced recall | 0.916 |

- Strong performance on pure-phase (~55/45 snow/rain class balance)
- Val slightly lower (~89%) but consistent

### Calibration
- ~83–87% of pure-phase predictions fall in the 0.9–1.0 confidence bin → accuracy 94.8–95.5% in that bin ✓
- Moderate-probability bins (0.5–0.8) show overconfidence (e.g., 0.7–0.8 bin: ~57% accuracy vs. ~74% stated confidence)
  - These bins are small (17–28 observations), so estimates noisy — classic XGBoost pre-calibration behavior
- Brier scores: 0.066–0.081 — decent but not exceptional
- Worth checking reliability diagram if not already plotted

### 3-Class Output (test set)
- Snow F1: 0.85–0.86; Rain F1: 0.81–0.84 — solid
- Mix F1: 0.27–0.29; true-mix capture rate: 28–30% — weak, as expected from indirect derivation
- Near-freezing air regime: rain recall collapses to 19–21% (uncertainty band swallowing ambiguous rain)
- Near-freezing wet-bulb regime: rain recall recovers to 47–51% — Twet is a more appropriate thermodynamic threshold

### Mix Signal Assessment
- **Mix precision 0.26–0.28** → wrong ~3 out of 4 times when flagging mix; barely above baseline for a 12%-prevalence class
- **The honest framing**: mix signal is more useful as a *confidence flag* ("exercise caution here") than as a positive mixed-phase detector
- Within the uncertainty envelope, snow/rain calls remain reasonably precise (0.75–0.88)
- Mixed-phase ground truth labels from MRoS may themselves be noisy — some low recall may be label noise, not model failure

### Preliminary SHAP Results (v7b)
**Elevation:**
- Highest mean |SHAP| for mix (0.101) > snow (0.085) > rain (0.056)
- Physically sensible: elevation proxies for the thermal environment sustaining mixed-phase conditions
- Elevation asymmetry (more important for snow than rain) also expected

**Temperature features:**
- `temp_air` and `temp_wet` are *less* important for mix than for confident pure-phase predictions
- Mix: SHAP ~0.052–0.055 vs. ~0.084–0.096 for rain/snow
- Exactly the right behavior: model is uncertain precisely when temperature is less diagnostic

**LOOCV MRoS predictors:**
- `mros_p_snow_loocv` and `mros_p_rain_loocv` are among top features across all phases
- `mros_p_rain_loocv` is highest-ranked feature overall (0.087), elevated even for mix (0.079) — surrounding rain signal is useful even under uncertainty
- `mros_p_mix_loocv` near bottom for all phases (0.004–0.012) — good sign; uncertainty product isn't just recycling the LOOCV mix indicator, it's deriving uncertainty from temperature/elevation structure

**One flag:**
- `imerg_plp` matters more for snow (0.050) than rain (0.028) — somewhat counterintuitive; worth watching whether this is a real signal or training data artifact

**Defensible mix interpretation from SHAP:**
- Mix has a *distinct feature importance profile* (elevated elevation, suppressed temperature) vs. snow and rain
- Not noise — the model is expressing a physically coherent story about where mixed phase occurs; usable argument for peer discussion

### Key Takeaways
- Best pure-phase binary performance yet
- Calibration behaves as expected (XGBoost overconfidence at moderate probabilities); warrants reliability diagram
- Mix remains hard; honest product framing = confidence flag, not phase classifier
- SHAP tells a physically coherent story and supports the mix interpretability argument

---

## Summary Table — Key Metrics by Model and Phase

### 3-Class Hard-Label Performance

| Model | Accuracy | Balanced Acc | Macro F1 | Snow F1 | Rain F1 | Mix F1 |
|---|---|---|---|---|---|---|
| Multiclass | 0.760 | — | — | 0.86 | 0.79 | 0.27 |
| Binary v5 | 0.768 | 0.612 | 0.618 | — | — | — |
| Binary v6a | 0.671 | 0.652 | — | — | — | 0.306 |
| Binary v6b | 0.767 | 0.656 | 0.653 | — | — | 0.312 |
| Binary v7 | — | — | — | 0.85–0.86 | 0.81–0.84 | 0.27–0.29 |

### Pure-Phase Binary Discrimination

| Model | Accuracy | Balanced Acc | ROC AUC | Avg Precision | Brier Score |
|---|---|---|---|---|---|
| Binary v5 | 0.881 | 0.863 | 0.962 | 0.973 | — |
| Binary v7 | 0.919 | — | 0.970 | 0.970 | 0.066–0.081 |

### Near-Freezing \|Twet\| ≤ 1°C Performance

| Model | Snow F1 | Rain F1 | Mix F1 | Overall Acc |
|---|---|---|---|---|
| Binary v5 | 0.726 | 0.493 | 0.262 | 0.586 |
| Binary v6a | — | — | — | — *(degenerate — see notes)* |
| Binary v6b | — | — | — | 0.594 |
| Binary v7 (recall) | — | 0.47–0.51 | — | — |

### Near-Freezing \|Twet\| ≤ 1°C Recall Detail

| Model | Snow recall | Rain recall | Mix recall |
|---|---|---|---|
| Binary v6a | 0.307 | 0.052 | 0.925 *(degenerate)* |
| Binary v6b | 0.827 | 0.338 | 0.493 |
| Binary v7 | — | 0.47–0.51 | — |

### Near-Freezing \|Tair\| ≤ 1°C Recall Detail

| Model | Snow recall | Rain recall | Mix recall |
|---|---|---|---|
| Binary v6a | 0.831 | 0.000 | 0.528 |
| Binary v6b | — | 0.026 | — |
| Binary v7 | 0.94–0.98 | 0.19–0.21 | — |

*Note: Rain recall near 0°C Tair is a structural limitation — rain events with near-freezing air temperature are psychrometrically constrained to near-freezing wet-bulb, so they fall inside the uncertainty band regardless of parameterization. Only ~39 test events; not numerically dominant.*
