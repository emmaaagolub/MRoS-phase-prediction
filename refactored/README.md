# Mountain Rain or Snow — gridded precipitation phase

Code for a 1 km hourly precipitation-phase product over two mountain regions:
the Sierra Nevada / Lake Tahoe area of California (CA) and the Colorado
Mountains (CO).

Crowdsourced observations from the Mountain Rain or Snow project provide the
labels. Weather stations, satellite and gridded climate data provide the
predictors. A gradient-boosted model predicts rain versus snow, and an
uncertainty band around the resulting probability produces a third "mix" class
where the answer is genuinely ambiguous.

## Running it

```bash
cd refactored
pip install -r requirements.txt

python run_pipeline.py                  # everything, both regions
python run_pipeline.py --skip-download  # data already downloaded
python run_pipeline.py --list           # see the stages
```

Both regions are handled automatically by every stage. Use `--regions CA` to
restrict to one. If a stage fails, fix the problem and resume with
`python run_pipeline.py --from <stage>`.

Every script also runs on its own:

```bash
python preprocessing/compile_observations.py --regions CO
python figures/manuscript_figures.py --figures shap band
Rscript preprocessing/download_prism.R
```

Source data is read from the repository's shared `Data/` folder. Everything the
pipeline writes goes to `refactored/outputs/`, so a run here never disturbs
results produced by the earlier notebook version.

## Layout

```
refactored/
  config.py            region definitions, paths and the study period
  run_pipeline.py      runs every stage in order
  requirements.txt

  preprocessing/
    regions.R                 shared region definitions for the R scripts
    get_elevation.R           download raw elevation models from the USGS
    download_station_data.R   HADS, LCD and SNOTEL station observations
    download_imerg.R          GPM IMERG liquid-precipitation probability
    download_prism.R          daily PRISM climate rasters
    process_dem.py            reproject and coarsen elevation to 1 km
    compile_observations.py   stations, satellite and reports onto one hourly grid
    resample_gridded.py       PRISM and IMERG onto the 1 km hourly grid
    kriging_interpolation.py  interpolate observations onto the grid

  model/
    common.py            settings shared by the model and evaluation code
    build_dataset.py     assemble the point table the model trains on
    train_model.py       fit, calibrate, choose the uncertainty band, export
    shap_analysis.py     which predictors drive which predictions

  evaluation/
    experiment_base.py   machinery shared by the experiment runners
    model_evaluation.py  score the model and draw the summary figures
    benchmarking.py      compare against standard published methods
    ablation.py          retrain with each predictor removed in turn
    bootstrap_cis.py     cluster bootstrap confidence intervals

  figures/
    manuscript_figures.py  redraw every figure from the saved artifacts
```

## What each stage produces

| Stage | Output |
| --- | --- |
| `get_elevation.R` | `Data/Elevation/<region>_DEM_AOI_TNM_10m.tif` |
| `download_station_data.R` | `Data/Stations/<REGION>/*.csv` |
| `download_imerg.R` | `Data/IMERG/<REGION>/gpm_imerg_<date>.parquet` |
| `download_prism.R` | `Data/PRISM/<REGION>/combined_prism_*.parquet` |
| `process_dem.py` | `Data/Elevation/<REGION>_DEM_AOI_1km.tif` |
| `compile_observations.py` | `outputs/compiled/<REGION>/hourly_data/*.parquet` |
| `resample_gridded.py` | `outputs/resampled_grids/<REGION>/*.nc` |
| `kriging_interpolation.py` | `outputs/interpolated/<REGION>/indicator_kriging/` |
| `build_dataset.py`, `train_model.py`, `shap_analysis.py` | `outputs/model/<REGION>/` |
| `benchmarking.py`, `ablation.py`, `bootstrap_cis.py` | `outputs/evaluation/<REGION>/` |
| `manuscript_figures.py` | `outputs/figures/<REGION>/` |

## Figures

Two sets of figures are produced.

Diagnostic figures live beside the artifacts they describe, under
`outputs/model/<REGION>/graphics/` and `outputs/evaluation/<REGION>/.../graphics/`.
They are written by the stage that computes them — training curves, the class
weight sweep, per-experiment confusion matrices and so on.

Presentation figures go to `outputs/figures/<REGION>/`, all drawn by
`figures/manuscript_figures.py`, which reads saved artifacts only and retrains
nothing. Filenames follow `<REGION>_<description>.png`, with `_appendix` on
figures meant for the supplement. Draw a subset with `--figures`:

```bash
python figures/manuscript_figures.py --figures calibration forest shap
```

Available keys: `station_checks`, `phase_combined`, `phase_extent`,
`phase_coverage`, `phase_elev_kde`, `phase_month`, `calibration`, `forest`,
`tair`, `tair_accuracy`, `tair_relimp_ci`, `f1_twet`, `f1_tair`, `shap`,
`ablation_ci`, `ablation_comparison`, `band`, `overall_bars`, `story5_tair`,
`story5_twet`, `confusion`, `csi`.

`station_checks` and `phase_combined` depend only on the compiled observations,
so the pipeline draws them right after that stage as an early sanity check.

## Requirements

**Python** — see `requirements.txt`.

**R** — terra, sf, tidyverse, lubridate, purrr, readr, httr, jsonlite, glue,
furrr, arrow, curl, devtools, geosphere. Station downloads also need the
[rainOrSnowTools](https://github.com/LynkerIntel/rainOrSnowTools) package
checked out alongside this repository.

**Credentials** — IMERG downloads need a free NASA Earthdata account with the
"NASA GESDISC DATA ARCHIVE" application approved. Put `NASA_DATA_USER` and
`NASA_DATA_PASSWORD` in your `.Renviron`.

## Notes on the approach

The model is trained only on observations reported as pure rain or pure snow.
Reports of mixed precipitation are held back rather than trained as a third
class, because "mix" is as much a statement about uncertainty as it is a
distinct physical state. Predicted probabilities are calibrated separately for
near-freezing and clear-phase conditions, and mix is then defined as the region
where the calibrated probability sits close enough to 0.5 — with "close enough"
widening as wet-bulb temperature approaches freezing.

The MRoS-derived predictors are computed leave-one-out: each observation's
predictor value comes from the other observations in that hour, never from
itself, so the labels cannot leak into the features.
