"""Run the pipeline, start to finish or a few stages at a time.

Every stage processes both regions (CA and CO), and every run of a stage
writes a new dated folder, so nothing is overwritten (see project_paths.yaml).
Later stages read the newest completed run of the stage before them.

  Preprocessing
    1  process_dem              10 m DEM -> 1 km grid           data/interim/dem_1km/
    2  compile                  stations, IMERG, MRoS -> hourly data/interim/hourly_compiled/
    3  compilation_figures      checks drawn from the compiled tables
    4  resample_gridded         IMERG onto the 1 km hourly grid data/interim/resampled_1km/
    5  kriging_interpolation    observations onto the grid      data/interim/kriging/

  Model                                                         results/pipeline/model/
    6  build_dataset            point table for model fitting
    7  train_model              fit, calibrate, choose the band
    8  shap_analysis            SHAP attribution

  Evaluation
    9  model_evaluation         scores and summary figures      (added to the model run)
   10  benchmarking             compare with established methods results/pipeline/benchmarking/
   11  ablation                 retrain with predictors removed  results/pipeline/ablations/
   12  bootstrap_cis            cluster-bootstrap CIs            (added to the benchmarking run)

  Figures
   13  manuscript_figures       every manuscript figure          results/pipeline/figures/

  Data collection (optional, --with-collection; normally the data come from
  Zenodo via `python pipeline/get_data.py`)
    get_elevation.R, download_station_data.R, download_imerg.R  -> data/raw/

Usage
-----
  python pipeline/run_pipeline.py                        every stage, both regions
  python pipeline/run_pipeline.py --regions CA           one region only
  python pipeline/run_pipeline.py --from build_dataset   resume from a stage
  python pipeline/run_pipeline.py --only compile kriging_interpolation
  python pipeline/run_pipeline.py --list                 list the stages and exit
  python pipeline/run_pipeline.py --dry-run              print the commands only
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import REGION_IDS, raw_path

HERE = Path(__file__).resolve().parent

# (name, folder, script, extra arguments)
COLLECTION_STAGES = [
    ("get_elevation", "preprocessing", "get_elevation.R", []),
    ("download_station_data", "preprocessing", "download_station_data.R", []),
    ("download_imerg", "preprocessing", "download_imerg.R", []),
]

STAGES = [
    ("process_dem", "preprocessing", "process_dem.py", []),
    ("compile", "preprocessing", "compile_observations.py", []),
    # The figures that depend only on the compiled observations.
    ("compilation_figures", "figures", "manuscript_figures.py",
     ["--figures", "station_checks", "phase_combined", "--inputs", "latest"]),
    ("resample_gridded", "preprocessing", "resample_gridded.py", []),
    ("kriging_interpolation", "preprocessing", "kriging_interpolation.py", []),
    ("build_dataset", "model", "build_dataset.py", []),
    ("train_model", "model", "train_model.py", []),
    ("shap_analysis", "model", "shap_analysis.py", []),
    ("model_evaluation", "evaluation", "model_evaluation.py", []),
    ("benchmarking", "evaluation", "benchmarking.py", []),
    ("ablation", "evaluation", "ablation.py", []),
    ("bootstrap_cis", "evaluation", "bootstrap_cis.py", []),
    ("manuscript_figures", "figures", "manuscript_figures.py", ["--inputs", "latest"]),
]

ALL_STAGES = COLLECTION_STAGES + STAGES
STAGE_NAMES = [stage[0] for stage in ALL_STAGES]


def build_command(stage, regions):
    _, folder, script, extra = stage
    path = HERE / folder / script

    # The R scripts loop over both regions themselves and take no arguments.
    if script.endswith(".R"):
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(f"Rscript not found on PATH, needed for {script}.")
        return [rscript, str(path)]

    return [sys.executable, str(path), "--regions", *regions, *extra]


def select_stages(args):
    stages = ALL_STAGES if args.with_collection else STAGES

    if args.only:
        unknown = set(args.only) - set(STAGE_NAMES)
        if unknown:
            raise SystemExit(f"Unknown stage(s): {sorted(unknown)}")
        return [s for s in ALL_STAGES if s[0] in args.only]

    if args.from_stage:
        names = [s[0] for s in stages]
        if args.from_stage not in names:
            raise SystemExit(f"Unknown stage: {args.from_stage}")
        stages = stages[names.index(args.from_stage):]

    return stages


def check_raw_data(stages, regions):
    """Stop early, with a pointer to get_data.py, if the raw data are missing."""
    needs_raw = {"process_dem", "compile"}
    if not needs_raw & {s[0] for s in stages}:
        return
    missing = [raw_path("mros")] if not raw_path("mros").exists() else []
    for r in regions:
        for key in ("stations", "imerg"):
            p = raw_path(key, r)
            if not p.exists() or not any(p.iterdir()):
                missing.append(p)
    if missing:
        listing = "\n".join(f"  {p}" for p in missing)
        raise SystemExit(
            f"Raw input data not found:\n{listing}\n\n"
            "Download it first:\n  python pipeline/get_data.py\n"
        )


def run(stages, regions, dry_run):
    print(f"Regions: {', '.join(regions)}")
    print(f"Stages : {', '.join(s[0] for s in stages)}\n")
    if not dry_run:
        check_raw_data(stages, regions)

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
                f"and resume with:\n  python pipeline/run_pipeline.py --from {name}"
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
    parser.add_argument("--with-collection", action="store_true",
                        help="Also run the R data-collection scripts first.")
    parser.add_argument("--skip-download", action="store_true",
                        help=argparse.SUPPRESS)  # old flag; collection is now off by default
    parser.add_argument("--list", action="store_true",
                        help="List the stages and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    args = parser.parse_args()

    if args.list:
        print("Data collection (only with --with-collection):")
        for name, folder, script, _ in COLLECTION_STAGES:
            print(f"      {name:22s} {folder}/{script}")
        print("\nPipeline:")
        for i, (name, folder, script, _) in enumerate(STAGES, start=1):
            print(f"  {i:2d}  {name:22s} {folder}/{script}")
        return

    run(select_stages(args), args.regions, args.dry_run)


if __name__ == "__main__":
    main()
