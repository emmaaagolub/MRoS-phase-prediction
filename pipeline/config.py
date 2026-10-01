"""Shared settings for the precipitation-phase pipeline.

Region definitions and the study period are defined here. Every file location
comes from project_paths.yaml at the repository root, and the runs used for
the manuscript come from pinned_runs.yaml.

Versioned runs
--------------
Every run of a stage writes a new folder, <stage root>/<region>/<run_id>/,
with run_id = YYYYMMDD-HHMM. Nothing is overwritten. A run_manifest.json in
each folder records when it ran, the git commit, and the input runs it read.

  open_run(stage, region, step)    create the output folder for this run
  finish_step(run, step)           mark the step complete (and prune old runs)
  append_run(stage, region, after) reopen the newest run to add a later step
                                   (train_model, shap and evaluation add to the
                                   model run that build_dataset created)
  input_run(stage, region)         the run a later stage reads from: the newest
                                   completed run, or a pinned run

Set MROS_RUN_<STAGE>=<path> (e.g. MROS_RUN_KRIGING) to read a specific run,
or MROS_INPUTS=pinned to read the pinned runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Found relative to this file so the scripts work from any working directory.
PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent

PATHS_FILE = REPO_ROOT / "project_paths.yaml"
PINS_FILE = REPO_ROOT / "pinned_runs.yaml"

with open(PATHS_FILE) as _fh:
    SETTINGS = yaml.safe_load(_fh)
PINS = {}
if PINS_FILE.exists():
    with open(PINS_FILE) as _fh:
        PINS = yaml.safe_load(_fh) or {}

# Study period, shared by every stage.
WY_START = "2022-10-01T00:00:00Z"
WY_END = "2026-05-01T23:59:59Z"

# Suffix on the station files written by download_station_data.R.
STATION_DATE_SUFFIX = "20221001_20260501"

CRS_WGS84 = "EPSG:4326"

# The two study areas. Polygon vertices are (lon, lat) in WGS84 and are the same
# ones used by get_elevation.R to cut the DEMs.
REGIONS = {
    "CA": {
        "label": "California: Sierra Nevada / Lake Tahoe",
        "utm_crs": "EPSG:26911",
        "dem_10m": "california_DEM_AOI_TNM_10m.tif",
        "dem_1km": "CA_DEM_AOI_1km.tif",
        # Months for which MRoS surfaces are interpolated.
        "mros_active_months": (10, 11, 12, 1, 2, 3, 4, 5),
        "aoi_lonlat": [
            (-119.45505750721992, 39.65343608043361),
            (-121.27878797084242, 39.66189413918429),
            (-119.11448133630248, 36.726935737063016),
            (-118.49924696303225, 37.235952484988736),
            (-119.46604383531404, 38.37304030164334),
        ],
    },
    "CO": {
        "label": "Colorado Mountains",
        "utm_crs": "EPSG:32613",
        "dem_10m": "colorado_DEM_AOI_TNM_10m.tif",
        "dem_1km": "CO_DEM_AOI_1km.tif",
        "mros_active_months": (9, 10, 11, 12, 1, 2, 3, 4, 5, 6),
        "aoi_lonlat": [
            (-105.19885928678391, 40.62046076499234),
            (-106.88927700287375, 40.555465783967925),
            (-107.66078646416787, 38.79540171139857),
            (-104.87856373593310, 38.77382201116306),
        ],
    },
}

REGION_IDS = tuple(REGIONS)


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

def repo_path(rel) -> Path:
    """A path from project_paths.yaml / pinned_runs.yaml, made absolute."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def raw_path(key, region=None) -> Path:
    """A raw input location from the `raw:` block of project_paths.yaml."""
    fmt = {"region": region or ""}
    if region:
        fmt["dem_10m"] = REGIONS[region]["dem_10m"]
    return repo_path(SETTINGS["raw"][key].format(**fmt))


def dem_1km_path(region) -> Path:
    return repo_path(SETTINGS["dem_1km"].format(dem_1km=REGIONS[region]["dem_1km"]))


def region_paths(region_id):
    """Raw input locations for one region (plus the derived 1 km DEM)."""
    return {
        "dem_10m": raw_path("dem_10m", region_id),
        "dem_1km": dem_1km_path(region_id),
        "station_dir": raw_path("stations", region_id),
        "station_meta": raw_path("station_metadata", region_id),
        "imerg_dir": raw_path("imerg", region_id),
        "mros_parquet": raw_path("mros"),
    }


# Per-stage PNG figures (diagnostics drawn by benchmarking, ablation,
# bootstrap_cis, model_evaluation and shap_analysis). manuscript_figures.py
# redraws every figure the paper uses from the CSV/parquet outputs, so these
# are off unless figures.stage_figures is true in project_paths.yaml.
STAGE_FIGURES = bool(SETTINGS.get("figures", {}).get("stage_figures", False))


def save_stage_figure(fig, path, **kwargs):
    """fig.savefig(path) when per-stage figures are switched on; otherwise skip.
    The caller still closes the figure."""
    if STAGE_FIGURES:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **kwargs)


def manuscript_figures_dir():
    """The LaTeX figures folder, or None when copying is switched off."""
    fig = SETTINGS.get("figures", {})
    if not fig.get("copy_to_manuscript", False):
        return None
    return repo_path(fig["manuscript_dir"])


# ---------------------------------------------------------------------------
# Versioned runs
# ---------------------------------------------------------------------------

MANIFEST = "run_manifest.json"

# The step whose completion makes a run usable by later stages.
FINAL_STEP = {
    "hourly_compiled": "compile",
    "resampled_1km": "resample_gridded",
    "kriging": "kriging_interpolation",
    "kriging_test": "kriging_interpolation",
    "model": "train_model",
    "benchmarking": "benchmarking",
    "ablations": "ablation",
    "figures": "figures",
}

_OPEN_RUNS = {}


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                "--", "pipeline"], capture_output=True, text=True, timeout=20)
        commit = out.stdout.strip() or None
        if commit and dirty.stdout.strip():
            commit += " (pipeline/ has uncommitted changes)"
        return commit
    except Exception:
        return None


def stage_root(stage, region=None) -> Path:
    cfg = SETTINGS["stages"][stage]
    root = repo_path(cfg["root"])
    if cfg.get("per_region", True) and region:
        root = root / region
    return root


def read_manifest(run_dir):
    f = Path(run_dir) / MANIFEST
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def _write_manifest(run_dir, manifest):
    tmp = Path(run_dir) / (MANIFEST + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, Path(run_dir) / MANIFEST)


def is_complete(run_dir, stage, step=None):
    """A run is usable once its final step (or `step`) has finished. Folders
    made before runs were versioned have no manifest and count as complete."""
    m = read_manifest(run_dir)
    if m is None or m.get("imported"):
        return True
    step = step or FINAL_STEP.get(stage)
    return m.get("steps", {}).get(step, {}).get("status") == "complete"


def _sort_key(run_dir):
    """Chronological order from the run_id: YYYYMMDD-HHMM[...] or YYYYMMDD_label."""
    name = run_dir.name
    hhmm = name[9:13] if len(name) >= 13 and name[8] == "-" else "0000"
    return (name[:8], hhmm, name)


def list_runs(stage, region=None, complete_only=True, step=None):
    """Run folders of a stage, oldest first."""
    root = stage_root(stage, region)
    if not root.exists():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and d.name[:8].isdigit()]
    if complete_only:
        runs = [d for d in runs if is_complete(d, stage, step)]
    return sorted(runs, key=_sort_key)


def pinned_run(stage, region=None):
    """The run pinned for the manuscript in pinned_runs.yaml, if it exists."""
    rel = (PINS.get(region) or {}).get(stage) if region else PINS.get(stage)
    if rel and repo_path(rel).exists():
        return repo_path(rel)
    return None


def input_run(stage, region=None, step=None, prefer_pinned=None, quiet=False) -> Path:
    """The run a later stage reads from.

    Order: the MROS_RUN_<STAGE> environment variable; the pinned run when
    prefer_pinned (or MROS_INPUTS=pinned); the newest completed run; and
    finally the pinned run if the stage has never been run by the pipeline.
    """
    env = os.environ.get(f"MROS_RUN_{stage.upper()}")
    if env:
        run = repo_path(env.format(region=region or ""))
        if not run.exists():
            raise FileNotFoundError(f"MROS_RUN_{stage.upper()} points to a missing folder: {run}")
        chosen, why = run, f"MROS_RUN_{stage.upper()}"
    else:
        if prefer_pinned is None:
            prefer_pinned = os.environ.get("MROS_INPUTS", "latest") == "pinned"
        chosen = why = None
        if prefer_pinned:
            chosen, why = pinned_run(stage, region), "pinned"
        if chosen is None:
            runs = list_runs(stage, region, step=step)
            if runs:
                chosen, why = runs[-1], "newest completed"
        if chosen is None:
            chosen, why = pinned_run(stage, region), "pinned (no pipeline run yet)"
        if chosen is None:
            where = stage_root(stage, region)
            raise FileNotFoundError(
                f"No completed '{stage}' run for {region or 'any region'} in {where}.\n"
                f"Run that stage first (python pipeline/run_pipeline.py --list shows the "
                f"stages), or pin a run in pinned_runs.yaml."
            )
    if not quiet:
        print(f"  reading {stage} [{region or '-'}]: {_rel(chosen)}  ({why})")
    return chosen


def open_run(stage, region=None, step=None, inputs=None, resume=False, label=None) -> Path:
    """Create the output folder for a new run of `stage` and record `step`.

    Called once per stage and region per process; later calls return the same
    folder. With resume=True the newest unfinished run is reopened instead
    (used by kriging, which can pick up where an interrupted run stopped).
    """
    key = (stage, region)
    run = _OPEN_RUNS.get(key)
    if run is None:
        root = stage_root(stage, region)
        root.mkdir(parents=True, exist_ok=True)
        if resume:
            unfinished = [d for d in list_runs(stage, region, complete_only=False)
                          if not is_complete(d, stage)]
            if unfinished:
                run = unfinished[-1]
                print(f"  resuming unfinished run {_rel(run)}")
        if run is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M") + (f"_{label}" if label else "")
            run, n = root / run_id, 2
            while run.exists():
                run, n = root / f"{run_id}-{n}", n + 1
            run.mkdir(parents=True)
            _write_manifest(run, {
                "stage": stage, "region": region, "run_id": run.name,
                "created": _now(), "git_commit": _git_commit(),
                "python": sys.version.split()[0], "steps": {},
            })
            print(f"  writing {stage} [{region or '-'}]: {_rel(run)}")
        _OPEN_RUNS[key] = run
    if step:
        start_step(run, step, inputs)
    return run


def append_run(stage, region, after, step=None, inputs=None) -> Path:
    """Reopen the newest pipeline run of `stage` whose step `after` is complete,
    to add a later step to it. Never returns a pinned or legacy run."""
    key = (stage, region)
    run = _OPEN_RUNS.get(key)
    if run is None:
        runs = list_runs(stage, region, step=after)
        runs = [r for r in runs if read_manifest(r) is not None]
        if not runs:
            raise FileNotFoundError(
                f"No '{stage}' run for {region} has finished '{after}' yet "
                f"(looked in {stage_root(stage, region)}). Run {after} first."
            )
        run = runs[-1]
        _OPEN_RUNS[key] = run
        print(f"  adding to {stage} [{region or '-'}]: {_rel(run)}")
    if step:
        start_step(run, step, inputs)
    return run


def start_step(run, step, inputs=None):
    m = read_manifest(run) or {"steps": {}}
    m.setdefault("steps", {})[step] = {
        "status": "running", "started": _now(),
        "inputs": {k: _rel(v) for k, v in (inputs or {}).items()},
    }
    _write_manifest(run, m)


def finish_step(run, step, **info):
    """Mark `step` complete. When it is the stage's final step, older runs
    beyond keep_last are pruned (see project_paths.yaml)."""
    m = read_manifest(run)
    entry = m.setdefault("steps", {}).setdefault(step, {})
    entry.update(status="complete", finished=_now(), **info)
    _write_manifest(run, m)
    stage = m.get("stage")
    if stage and FINAL_STEP.get(stage) == step:
        prune_runs(stage, m.get("region"))


def _size_gb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / 1024 ** 3


def prune_runs(stage, region=None):
    """Remove completed runs older than the newest keep_last, sparing pinned runs."""
    keep = SETTINGS["stages"][stage].get("keep_last")
    if not keep:
        return
    runs = list_runs(stage, region)
    pinned = {p.resolve() for r in REGION_IDS if (p := pinned_run(stage, r))}
    old = [d for d in runs[:-keep] if d.resolve() not in pinned]
    if not old:
        return
    mode = SETTINGS.get("prune_older_runs", "ask")
    print(f"\n  {stage} [{region}]: keeping the newest {keep} run(s) plus pinned runs.")
    for d in old:
        print(f"    older run: {_rel(d)}  ({_size_gb(d):.1f} GB)")
    if mode == "off":
        print("    not removed (prune_older_runs: off)")
        return
    if mode == "ask":
        if not sys.stdin or not sys.stdin.isatty():
            print("    not removed: no terminal to ask. Set prune_older_runs: auto "
                  "in project_paths.yaml to remove without asking.")
            return
        answer = input("    Delete these older runs? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("    kept")
            return
    for d in old:
        shutil.rmtree(d)
        print(f"    removed {_rel(d)}")


def parse_region_args(argv=None):
    """Shared --regions flag. Defaults to running every region."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGION_IDS,
        default=list(REGION_IDS),
        help="Regions to process (default: all).",
    )
    return parser.parse_args(argv).regions
