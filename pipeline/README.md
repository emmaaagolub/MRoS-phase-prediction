# pipeline/

Run everything with `python pipeline/run_pipeline.py` (see the top-level
README). Every script also runs on its own, e.g.
`python pipeline/model/train_model.py --regions CO`.

| # | Stage | Script | Reads | Writes (a new `<run_id>/` each time) |
| --- | --- | --- | --- | --- |
| 1 | process_dem | `preprocessing/process_dem.py` | `data/raw/dem/` | `data/interim/dem_1km/` (overwritten) |
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

`get_data.py` downloads the input data from Zenodo; `tools/package_for_zenodo.py`
builds the archives that get uploaded there.

The R scripts in `preprocessing/` (`get_elevation.R`, `download_station_data.R`,
`download_imerg.R`) re-collect the raw data from the source services. They
are only needed to rebuild `data/raw/` from scratch
(`run_pipeline.py --with-collection`).

## Figures

`manuscript_figures.py` reads saved outputs only; draw a subset with
`--figures`:

```bash
python pipeline/figures/manuscript_figures.py --figures calibration forest shap
```

Keys: `station_checks`, `phase_combined`, `phase_extent`, `phase_coverage`,
`phase_elev_kde`, `phase_month`, `calibration`, `forest`, `tair`,
`tair_accuracy`, `tair_relimp_ci`, `f1_twet`, `f1_tair`, `shap`, `ablation_ci`,
`ablation_comparison`, `band`, `overall_bars`, `selective_twet`,
`selective_tair`, `story5_tair`, `story5_twet`, `confusion`,
`confusion_percent`, `csi`.

Diagnostic figures (training curves, class-weight sweep, per-experiment
confusion matrices) are written by the stage that computes them, into
`graphics/` inside that stage's run folder.
