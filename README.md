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

python pipeline/get_data.py          # download the input data from Zenodo into data/
python pipeline/run_pipeline.py      # run every stage for both regions
```

That's it. Everything the pipeline writes lands in `data/interim/` and
`results/pipeline/`, one dated folder per run.

Run part of it:

```bash
python pipeline/run_pipeline.py --list                  # the 13 stages
python pipeline/run_pipeline.py --regions CA            # one region
python pipeline/run_pipeline.py --from build_dataset    # resume from a stage
python pipeline/run_pipeline.py --only manuscript_figures
```

Kriging (stage 5) is the slow step: it takes many hours per region. To skip it
and start from the manuscript's interpolated grids, download them too:

```bash
python pipeline/get_data.py --include interim           # adds ~30 GB
python pipeline/run_pipeline.py --from build_dataset
```

To redraw the manuscript figures from the exact runs used in the paper:

```bash
python pipeline/get_data.py --include interim results
python pipeline/figures/manuscript_figures.py
```

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
