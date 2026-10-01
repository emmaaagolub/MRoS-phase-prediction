# Mountain Rain or Snow — gridded precipitation phase

Code for a 1 km hourly precipitation-phase product over two mountain regions:
the Sierra Nevada Mountains (SNM; folder and file key `CA`) and the Colorado
Rocky Mountains (CRM; key `CO`), October 2022 – May 2026.

Crowdsourced reports from the Mountain Rain or Snow (MRoS) project provide the
labels. Weather stations, GPM IMERG probability of liquid precipitation (pLP)
and elevation provide the predictors, interpolated onto an hourly 1 km grid.
A calibrated XGBoost model (XGB-Full) gives p(snow) for every observation,
which is turned into a phase by one of two decision rules (see
[Decision rules](#decision-rules-binary-and-selective)).

## Quick start

```bash
git clone https://github.com/emmaaagolub/MRoS-phase-prediction.git
cd MRoS-phase-prediction
pip install -r pipeline/requirements.txt      # Python 3.10+

python pipeline/get_data.py                   # step 1: input data from Zenodo (~2 GB)
python pipeline/run_pipeline.py               # step 2: every stage, both regions
```

Run `get_data.py` first; `run_pipeline.py` stops with a reminder if the
data are missing. The R data-collection scripts are not needed: the raw data
come from Zenodo.

### The faster route: start from the manuscript's grids

Kriging (stage 5) takes many hours per region. To skip it, download the
manuscript's intermediate products and start at `build_dataset`:

```bash
python pipeline/get_data.py --include interim            # + ~33 GB
python pipeline/run_pipeline.py --from build_dataset      # ~2 h on a laptop, mostly SHAP
```

### What it needs

| | Full run | `--from build_dataset` |
| --- | --- | --- |
| Download | ~2 GB | ~35 GB |
| New output | ~33 GB (IMERG grids ~6 GB, kriged surfaces ~26 GB) | < 1 GB |
| Memory | 8 GB is enough; 16 GB is comfortable | 4 GB |
| Time | kriging: many hours per region; everything else ~2–3 h | ~2 h |

The two newest runs of the large grids are kept (`keep_last` in
`project_paths.yaml`), so repeated full runs can hold ~65 GB. Both scripts
estimate the size, check free space, and ask before anything over 5 GB
(`--yes` skips the question). `--list` on either script shows sizes without
doing anything.

### Running parts of it

```bash
python pipeline/run_pipeline.py --list                  # the 13 stages
python pipeline/run_pipeline.py --regions CA            # one region
python pipeline/run_pipeline.py --from build_dataset    # resume from a stage
python pipeline/run_pipeline.py --only manuscript_figures
```

Every script also runs on its own, e.g. `python pipeline/model/train_model.py
--regions CO`. See [pipeline/README.md](pipeline/README.md) for what each
stage reads and writes.

### Checking the kriging without running it in full

```bash
python pipeline/preprocessing/kriging_interpolation.py --test --regions CO   # < 1 min
```

`--test` interpolates the two days with the most MRoS reports over a 50 km
box: variogram fitting, lapse-rate detrending with kriged residuals, kriging of
humidity with elevation as drift, indicator kriging of the three MRoS phases,
and the leave-one-out MRoS table all run. If the full kriging output is
present (the interim download), the test kriges with its variograms and
prints the largest cell-by-cell difference from it, which should be at
floating-point level. `--start/--end/--bbox` choose your own window. Test runs
go to `data/interim/kriging_test/`, which no later stage reads.

## What the pipeline writes

Every run of a stage writes a **new** folder named by date and time, so
re-running never overwrites earlier results, and later stages read the newest
completed run of the stage before them. Each folder holds a
`run_manifest.json` (when it ran, git commit, which input runs it read).

```
data/interim/                         intermediate grids
  dem_1km/                              1 km DEMs (define the model grid)
  hourly_compiled/<R>/<run>/            stations, IMERG, MRoS on an hourly grid (parquet)
  resampled_1km/<R>/<run>/              IMERG pLP on the 1 km grid (netCDF)
  kriging/<R>/<run>/                    kriged predictor surfaces (netCDF) and the
                                        leave-one-out MRoS table
results/pipeline/
  model/<R>/<run>/                      point table, trained model, calibrators,
                                        val/test predictions, SHAP (diagnostic fit)
  benchmarking/<R>/<run>/               XGB-Full vs the benchmark methods
    benchmark_comparison.csv              point estimates for every method
    benchmark_predictions_test.parquet    every test observation, every method,
                                          both decision rules
    bootstrap/                            95% cluster-bootstrap CIs:
      xgbfull_decision_rules_ci.csv         XGB-Full, binary and selective rule
                                            (the paper's two XGB-Full tables)
      bootstrap_benchmark_metrics_ci.csv    every method (benchmark table)
      bootstrap_benchmark_deltas_ci.csv     XGB-Full minus each benchmark
      bootstrap_ablation_metrics_ci.csv     every ablation configuration (ablation table)
      bootstrap_ablation_deltas_ci.csv      full minus each configuration
  ablations/<R>/<run>/                  one subfolder per predictor configuration
    ablation_comparison.csv               all configurations side by side
  figures/<run>/                        every manuscript figure (PDF)
```

`<R>` is `CA` or `CO`. Folder names in `data/` and `results/` say CA/CO; the
figures say SNM/CRM.

All test-set numbers in the manuscript come
from `benchmarking/` and `ablations/` (the same XGB-Full model appears in
both, as `xgboost_mros` and `baseline_full`). The `model/` run is a separate
fit of the same configuration on a different random split, kept for its
diagnostics; see "EVALUATION SPLIT" in `pipeline/evaluation/experiment_base.py`.

## Decision rules: binary and selective

The model's output is a calibrated probability, p(snow). Two rules turn it
into a phase. Neither changes training, so both are computed from the same
predictions in every run.

| Rule | What it does | Where |
| --- | --- | --- |
| **Binary** | snow if p(snow) ≥ 0.5, else rain; every observation gets a call | columns `pred_xgboost_mros_bin05` / `prediction_binary05`; benchmark and ablation results |
| **Selective** (optional) | applies the uncertainty envelope (Eq. 1); inside it the model abstains and the observation is flagged | columns `pred_xgboost_mros_band` / `prediction_phase_uncertainty` (flag = code 2); `rule == "selective"` rows |

`bootstrap/xgbfull_decision_rules_ci.csv` holds both side by side. Under the
selective rule, accuracy, macro F1 and recall are computed on the rain and
snow observations the model committed to; coverage is the share of rain and
snow observations it committed to; flag rate is the share of all
observations (mixed included) inside the envelope; mix capture is the share
of observer-reported mixed events inside it. AUROC and Brier score do not
depend on the rule.

## Reproducing the published numbers

* **Benchmarks, ablation inputs and the evaluation split** reproduce exactly
  from the Zenodo grids.
* **XGBoost re-training** can differ slightly between machines even with the
  pinned version (XGBoost 3.1.2): the row/column subsampling draws differ
  between operating systems. In our tests on Linux, with identical inputs and
  split, XGB-Full test accuracy came out 90.7% (SNM) and 92.0% (CRM) against
  the published 91.2% and 91.4%, well inside the published intervals; the
  benchmark methods reproduced to the decimal.
* **MRoS locations** on Zenodo are rounded to protect observers (lat/lon to 4
  decimals, ~10 m; projected x/y to 10 m), in the raw reports and in the
  interim tables (`mros_hourly.parquet`, `mros_processed.parquet`,
  `mros_loocv_point_predictions_kriging.*`). The gridded products were
  computed from full-precision locations, so a few reports may fall in a
  neighbouring 1 km cell. Full-precision locations can be shared on request,
  subject to approval by the Mountain Rain or Snow project team.

## Configuration files

Two YAML files sit at the repository root.

* `project_paths.yaml` — details where every input and output lives, how many old
  runs to keep, and figure options (`figures.stage_figures: true` also writes
  each stage's diagnostic PNGs; `figures.copy_to_manuscript` copies the paper
  figures into a LaTeX folder and is used by the authors only).
* `pinned_runs.yaml` — the run folders the manuscript was made from. Its
  interim entries match the folders `get_data.py --include interim` creates;
  its model/benchmarking/ablation entries are the authors' local runs and are
  skipped when absent, i.e., when newest runs are used.

To read a specific run instead of the newest, set an environment variable,
e.g. `MROS_RUN_KRIGING=data/interim/kriging/{region}/20260806_kriging`.

## Layout

```
pipeline/              the code (tracked in repository)
  get_data.py            downloads the data from Zenodo
  run_pipeline.py        runs the stages in order
  config.py              regions, study period, run-folder handling
  preprocessing/  model/  evaluation/  figures/  tools/
project_paths.yaml     where every input and output lives
pinned_runs.yaml       the runs the manuscript figures are drawn from
data/                  NOT in git; filled by get_data.py (see data/README.md)
results/               NOT in git; created when you run the pipeline
```

## Requirements

**Python 3.10+** — `pip install -r pipeline/requirements.txt`.

**R** (only to re-collect the raw data with `--with-collection`) — terra, sf,
tidyverse, lubridate, purrr, readr, httr, jsonlite, glue, furrr, arrow, curl,
devtools, geosphere, and the
[rainOrSnowTools](https://github.com/LynkerIntel/rainOrSnowTools) package
checked out alongside this repository. IMERG downloads need a free NASA
Earthdata account; put `NASA_DATA_USER` and `NASA_DATA_PASSWORD` in `.Renviron`.
