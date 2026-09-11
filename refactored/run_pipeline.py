"""Run the pipeline stages in order.

Every stage processes both regions, so nothing needs editing between a
California run and a Colorado run.

  Preprocessing
    1  get_elevation.R          download the raw elevation models
    2  download_station_data.R  download station observations
    3  download_imerg.R         download IMERG phase probability
    4  download_prism.R         download daily PRISM rasters
    5  process_dem.py           reproject and resample elevation to 1 km
    6  compile_observations.py  observations onto a common hourly grid
    7  compilation_figures      figures from the compiled observations
    8  resample_gridded.py      PRISM and IMERG onto the 1 km hourly grid
    9  kriging_interpolation.py interpolate observations onto the grid

  Model
   10  build_dataset.py         assemble the point table for model fitting
   11  train_model.py           fit, calibrate and export the model
   12  shap_analysis.py         SHAP attribution

  Evaluation
   13  model_evaluation.py      score the model and draw the summary figures
   14  benchmarking.py          compare against established methods
   15  ablation.py              retrain with predictors removed
   16  bootstrap_cis.py         bootstrap confidence intervals

  Figures
   17  manuscript_figures.py    redraw the figures from saved artifacts

Usage
-----
  python run_pipeline.py                          run every stage
  python run_pipeline.py --regions CA             one region only
  python run_pipeline.py --from process_dem       resume from a stage
  python run_pipeline.py --only compile kriging_interpolation
  python run_pipeline.py --skip-download          skip the four download stages
  python run_pipeline.py --list                   list the stages and exit
  python run_pipeline.py --dry-run                print the commands only

The download stages skip files already present on disk.

All pipeline output is written under refactored/outputs/.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import REGION_IDS

HERE = Path(__file__).resolve().parent

# (name, folder, script, is_download, extra arguments)
STAGES = [
    ("get_elevation", "preprocessing", "get_elevation.R", True, []),
    ("download_station_data", "preprocessing", "download_station_data.R", True, []),
    ("download_imerg", "preprocessing", "download_imerg.R", True, []),
    ("download_prism", "preprocessing", "download_prism.R", True, []),
    ("process_dem", "preprocessing", "process_dem.py", False, []),
    ("compile", "preprocessing", "compile_observations.py", False, []),
    # The two figures that depend only on the compiled observations.
    ("compilation_figures", "figures", "manuscript_figures.py", False,
     ["--figures", "station_checks", "phase_combined"]),
    ("resample_gridded", "preprocessing", "resample_gridded.py", False, []),
    ("kriging_interpolation", "preprocessing", "kriging_interpolation.py", False, []),
    ("build_dataset", "model", "build_dataset.py", False, []),
    ("train_model", "model", "train_model.py", False, []),
    ("shap_analysis", "model", "shap_analysis.py", False, []),
    ("model_evaluation", "evaluation", "model_evaluation.py", False, []),
    ("benchmarking", "evaluation", "benchmarking.py", False, []),
    ("ablation", "evaluation", "ablation.py", False, []),
    ("bootstrap_cis", "evaluation", "bootstrap_cis.py", False, []),
    ("manuscript_figures", "figures", "manuscript_figures.py", False, []),
]

STAGE_NAMES = [stage[0] for stage in STAGES]


def build_command(stage, regions):
    _, folder, script, _, extra = stage
    path = HERE / folder / script

    # The R scripts loop over both regions themselves and take no arguments.
    if script.endswith(".R"):
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(
                f"Rscript not found on PATH, needed for {script}. Install R, or "
                "use --skip-download if the data are already downloaded."
            )
        return [rscript, str(path)]

    return [sys.executable, str(path), "--regions", *regions, *extra]


def select_stages(args):
    stages = STAGES

    if args.only:
        unknown = set(args.only) - set(STAGE_NAMES)
        if unknown:
            raise SystemExit(f"Unknown stage(s): {sorted(unknown)}")
        return [s for s in stages if s[0] in args.only]

    if args.from_stage:
        if args.from_stage not in STAGE_NAMES:
            raise SystemExit(f"Unknown stage: {args.from_stage}")
        stages = stages[STAGE_NAMES.index(args.from_stage):]

    if args.skip_download:
        stages = [s for s in stages if not s[3]]

    return stages


def run(stages, regions, dry_run):
    print(f"Regions: {', '.join(regions)}")
    print(f"Stages : {', '.join(s[0] for s in stages)}\n")

    for i, stage in enumerate(stages, start=1):
        name = stage[0]
        command = build_command(stage, regions)

        print("=" * 70)
        print(f"[{i}/{len(stages)}] {name}")
        print("=" * 70)
        print("  " + " ".join(command))

        if dry_run:
            continue

        started = time.time()
        result = subprocess.run(command, cwd=HERE / stage[1])
        elapsed = (time.time() - started) / 60

        if result.returncode != 0:
            raise SystemExit(
                f"\n{name} failed with exit code {result.returncode} after "
                f"{elapsed:.1f} min. Nothing after it has run; fix the problem "
                f"and resume with:\n  python run_pipeline.py --from {name}"
            )
        print(f"  done in {elapsed:.1f} min\n")

    if not dry_run:
        print("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the precipitation-phase pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--regions", nargs="+", choices=list(REGION_IDS),
                        default=list(REGION_IDS),
                        help="Regions to process (default: all).")
    parser.add_argument("--from", dest="from_stage", metavar="STAGE",
                        help="Start at this stage and run everything after it.")
    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help="Run only these stages.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the four data download stages.")
    parser.add_argument("--list", action="store_true",
                        help="List the stages and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    args = parser.parse_args()

    if args.list:
        for i, (name, folder, script, is_download, _) in enumerate(STAGES, start=1):
            tag = "  (download)" if is_download else ""
            print(f"  {i:2d}  {name:22s} {folder}/{script}{tag}")
        return

    run(select_stages(args), args.regions, args.dry_run)


if __name__ == "__main__":
    main()
