# pipeline/

Run everything with `python pipeline/run_pipeline.py` (see the top-level
README). Every script also runs on its own, e.g.
`python pipeline/model/train_model.py --regions CO`.

| # | Stage | Script | Reads | Writes (a new `<run_id>/` each time) |
| --- | --- | --- | --- | --- |
| 1 | process_dem | `preprocessing/process_dem.py` | `data/raw/dem/` (skipped if only the 1 km DEMs were downloaded) | `data/interim/dem_1km/` (overwritten) |
| 2 | compile | `preprocessing/compile_observations.py` | `data/raw/*` | `data/interim/hourly_compiled/<R>/` |
| 3 | compilation_figures | `figures/manuscript_figures.py` | hourly_compiled | `results/pipeline/figures/` |
| 4 | resample_gridded | `preprocessing/resample_gridded.py` | hourly_compiled | `data/interim/resampled_1km/<R>/` |
| 5 | kriging_interpolation | `preprocessing/kriging_interpolation.py` | hourly_compiled | `data/interim/kriging/<R>/` |
| 6 | build_dataset | `model/build_dataset.py` | kriging, resampled_1km | `results/pipeline/model/<R>/` |
| 7 | train_model | `model/train_model.py` | model run | same model run |
| 8 | shap_analysis | `model/shap_analysis.py` | model run | same model run |
| 9 | model_evaluation | `evaluation/model_evaluation.py` | model run | same model run |
| 10 | benchmarking | `evaluation/benchmarking.py` | model, kriging, resampled_1km | `results/pipeline/benchmarking/<R>/` |
| 11 | ablation | `evaluation/ablation.py` | model, kriging, resampled_1km | `results/pipeline/ablations/<R>/` |
| 12 | bootstrap_cis | `evaluation/bootstrap_cis.py` | benchmarking, ablations | `<benchmarking run>/bootstrap/` |
| 13 | manuscript_figures | `figures/manuscript_figures.py` | pinned or newest runs | `results/pipeline/figures/` (+ manuscript folder) |

"Reads" always means the newest completed run of that stage, printed at the
start of each script. To read a different run, set an environment variable,
e.g. `MROS_RUN_KRIGING=data/interim/kriging/{region}/20260806_kriging`, or
`MROS_INPUTS=pinned` to use the runs in `pinned_runs.yaml`.

## What each stage does (in the manuscript's terms)

* **compile** — station records to UTC hourly means with four quality
  filters; missing dewpoint and humidity filled by lapse-rate-adjusted IDW
  from neighbouring stations; wet-bulb temperature (Stull 2011, iterative
  fallback). IMERG half-hourly pLP averaged to hourly. MRoS: last report per
  observer-hour. Everything clipped to the study-area polygon.
* **resample_gridded** — hourly IMERG pLP, bilinear onto the 1 km DEM grid.
* **kriging_interpolation** — temperatures: hourly lapse-rate fit against
  elevation, ordinary kriging of the residuals, trend restored from the DEM.
  Relative humidity: universal kriging with elevation as an external drift.
  MRoS: snow/mix/rain indicators kriged separately, clipped and rescaled to sum
  to one. Plus the leave-one-out MRoS table (each report predicted from the
  others in the same hour), which supplies the MRoS predictors. `--test` runs
  it on two days over a small box (see the top-level README).
* **build_dataset** — samples the gridded predictors at every MRoS report.
* **train_model** — 70/15/15 split stratified on phase; XGBoost on rain and
  snow only; class weight chosen to balance snow and rain recall on
  validation; beta calibration fitted separately for |Tw| ≤ 2 °C and
  |Tw| > 2 °C; envelope parameters chosen on validation.
* **benchmarking** — retrains XGB-Full on the evaluation split (see below) and
  scores it against the benchmark methods on the same test observations: air,
  dewpoint and wet-bulb thresholds and the Jennings et al. (2018) logistic
  regression refit on each domain (the eight in the paper). The logistic
  regression with its published coefficients is also computed
  (`binlog_jennings18`) but left out of the paper's tables and figures.
* **ablation** — retrains with predictors withheld (`ABLATION_CONFIGS`).
* **bootstrap_cis** — 95% intervals from 5,000 resamples of 30 km × 1 day
  clusters; paired differences between methods and configurations; XGB-Full
  under both decision rules (`xgbfull_decision_rules_ci.csv`).

### Evaluation split

benchmarking.py and ablation.py start from the model run's
`ml_input_points_split.parquet` and redraw the same stratified 70/15/15 split
from it. Because its rows are ordered train/val/test, this gives a different
random partition from train_model's. Every test-set number in the manuscript
comes from this evaluation split; the model run is kept for its diagnostics.
The note "EVALUATION SPLIT" in `evaluation/experiment_base.py` explains why it
was kept this way.

### Decision rules

Both rules are applied to the same calibrated p(snow) in every stage: the
binary rule (`*_bin05`, `prediction_binary05`) and the selective rule
(`*_band`, `prediction_phase_uncertainty`, where code 2 = flagged). See the
"Decision rules" block in `model/common.py` and the top-level README.

## Data download and packaging

`get_data.py` downloads the input data from Zenodo; `tools/package_for_zenodo.py`
builds the archives that get uploaded there (each run adds its groups to
`zenodo_upload/zenodo_manifest.json`), using the location-coarsened MRoS file
made by `tools/make_public_mros.py`.

The R scripts in `preprocessing/` (`get_elevation.R`, `download_station_data.R`,
`download_imerg.R`) re-collect the raw data from the source services. They
are only needed to rebuild `data/raw/` from scratch
(`run_pipeline.py --with-collection`).

## Figures

`manuscript_figures.py` reads saved outputs only. By default it draws the
figures that appear in the manuscript; `--all` adds the extras (SHAP, station
checks, single-panel variants); `--figures` draws chosen keys:

```bash
python pipeline/figures/manuscript_figures.py --figures calibration forest
```

| Manuscript file | Key | Main / appendix |
| --- | --- | --- |
| study-domains-and-reports.pdf | `phase_maps` | main |
| f1-by-wetbulb-binary-rule.pdf | `f1_twet` | main |
| accuracy-vs-benchmarks.pdf | `overall_bars` | main |
| accuracy-by-air-temperature.pdf | `tair` | main |
| csi-by-air-temperature.pdf | `csi` | main |
| ablation-change-in-skill.pdf | `ablation_comparison` | main |
| mros-effect-near-freezing-wetbulb.pdf | `story5_twet` | main |
| selective-rule-by-wetbulb.pdf | `selective_twet` | main |
| mixed-events-vs-envelope.pdf | `band` | main |
| report-elevation-and-monthly-counts.pdf | `phase_distributions` | appendix |
| kriged-surfaces-{sierra-nevada,colorado}.png | `kriged_surfaces` (needs the kriging grids) | appendix |
| confusion-matrices.pdf | `confusion` | appendix |
| calibration-reliability-diagrams.pdf | `calibration` | appendix |
| f1-by-air-temperature-binary-rule.pdf | `f1_tair` | appendix |
| selective-rule-by-air-temperature.pdf | `selective_tair` | appendix |
| mros-effect-near-freezing-air-temperature.pdf | `story5_tair` | appendix |
| accuracy-difference-vs-benchmarks.pdf | `forest` | appendix |
| ablation-differences-with-intervals.pdf | `ablation_ci` | appendix |

The framework flowchart is drawn by hand. In the run folder each figure is
saved under its key-based name (e.g. `f1_by_wetbulb.pdf`); the copy in the
manuscript folder uses the name in the first column.

The stages themselves write no PNGs unless `figures.stage_figures: true` in
`project_paths.yaml`; those per-stage diagnostics (training curves,
class-weight sweep, per-configuration confusion matrices, ...) then go to
`graphics/` inside each run folder.
