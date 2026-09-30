# Mountain Rain or Snow — gridded precipitation phase

Code for a 1 km hourly precipitation-phase product over two mountain regions:
the Sierra Nevada / Lake Tahoe area of California (CA) and the Colorado
Mountains (CO).

Crowdsourced observations from the Mountain Rain or Snow project provide the
labels. Weather stations and satellite (GPM IMERG) data provide the
predictors. A gradient-boosted model predicts rain versus snow, and an
uncertainty band around the resulting probability produces a third "mix" class
where the answer is genuinely ambiguous.

## Quick start

```bash
git clone https://github.com/emmaaagolub/MRoS-phase-prediction.git
cd MRoS-phase-prediction
pip install -r pipeline/requirements.txt

python pipeline/get_data.py          # step 1: download the input data from Zenodo (~2 GB)
python pipeline/run_pipeline.py      # step 2: run every stage for both regions
```

Always run `get_data.py` first; `run_pipeline.py` stops with a reminder if the
data are missing. Everything the pipeline writes lands in `data/interim/` and
`results/pipeline/`, one dated folder per run.

> **Disk space.** A full pipeline run writes about **33 GB** of gridded files
> (1 km IMERG grids ~6 GB, kriged surfaces ~26 GB), and the two newest runs of
> each are kept, so repeated runs can hold ~65 GB. Downloading the
> manuscript's intermediate grids (`--include interim`) is also **~33 GB**.
> Both scripts show the size, check free space, and ask before anything over
> 5 GB (`--yes` skips the question). `python pipeline/get_data.py --list` and
> `python pipeline/run_pipeline.py --list` show sizes without doing anything.

Run parts of it:

```bash
python pipeline/run_pipeline.py --list                  # the 13 stages
python pipeline/run_pipeline.py --regions CA            # one region
python pipeline/run_pipeline.py --from build_dataset    # resume from a stage
python pipeline/run_pipeline.py --only manuscript_figures
```

Kriging (stage 5) is slow; it can take many hours per region. The
manuscript's intermediate products (compiled station, IMERG and MRoS tables,
the 1 km IMERG grid, the kriged predictor surfaces and the leave-one-out MRoS
table) can be downloaded instead:

```bash
python pipeline/get_data.py --include interim           # adds ~33 GB (check free space first)
python pipeline/run_pipeline.py --from build_dataset    # train from those grids
```

### MRoS observation locations

To protect observers, every MRoS table on Zenodo has its locations rounded:
latitude/longitude to 4 decimal places (~10 m) and projected x/y to 10 m. This
applies to the raw reports and to the tables in the interim bundles
(`mros_hourly.parquet`, `mros_processed.parquet`,
`mros_loocv_point_predictions_kriging.*`). The gridded products were computed
from the full-precision locations, so results you compute from the Zenodo data,
whether from `build_dataset` or from the start of the pipeline, will differ
slightly from the published ones: a few reports fall in a neighbouring 1 km
grid cell.

Full-precision MRoS observation locations can be shared on request, subject to
approval by the Mountain Rain or Snow project team.

## Layout

```
pipeline/              the code (this is what the repository tracks)
  run_pipeline.py        runs the stages in order
  get_data.py            downloads the data from Zenodo
  config.py              regions, study period, run-folder handling
  preprocessing/  model/  evaluation/  figures/
project_paths.yaml     where every input and output lives
pinned_runs.yaml       the runs the manuscript figures are drawn from
data/                  NOT in git; filled by get_data.py (see data/README.md)
  raw/                   stations, IMERG, MRoS observations, DEMs, boundaries
  interim/               derived grids, one folder per run
results/               NOT in git; created when you run the pipeline
  pipeline/              model, benchmarking, ablations, figures (one folder per run)
```

## How runs are stored

Every run of a stage writes a **new** folder named by date and time, e.g.
`results/pipeline/model/CA/20261002-1415/`, so re-running never overwrites
earlier results. Each folder holds a `run_manifest.json` recording when it
ran, the git commit, and which input runs it read. Later stages read the
newest completed run of the stage before them.

The kriging and resampled-IMERG stages are large (3–14 GB per run), so only
the newest two runs are kept (`keep_last` in `project_paths.yaml`); runs
listed in `pinned_runs.yaml` are never removed.

`manuscript_figures.py` draws from the runs in `pinned_runs.yaml` by default
and copies each PDF into the manuscript folder (turn that off with
`figures.copy_to_manuscript: false`). `--inputs latest` draws from your newest
runs instead.

## Requirements

**Python 3.10+** — `pip install -r pipeline/requirements.txt`.

**R** (only to re-collect the raw data with `--with-collection`) — terra, sf,
tidyverse, lubridate, purrr, readr, httr, jsonlite, glue, furrr, arrow, curl,
devtools, geosphere, and the
[rainOrSnowTools](https://github.com/LynkerIntel/rainOrSnowTools) package
checked out alongside this repository. IMERG downloads need a free NASA
Earthdata account; put `NASA_DATA_USER` and `NASA_DATA_PASSWORD` in `.Renviron`.

## Notes on the approach

The model is fitted on observations reported as pure rain or pure snow.
Reports of mixed precipitation are excluded from fitting and retained for
evaluation. Predicted probabilities are calibrated separately for near-freezing
and clear-phase observations. Mix is then assigned where the calibrated
probability falls inside a band around 0.5 whose width increases as wet-bulb
temperature approaches freezing.

The MRoS-derived predictors are computed leave-one-out: each observation's
predictor value is interpolated from the other observations in that hour, not
from itself.
