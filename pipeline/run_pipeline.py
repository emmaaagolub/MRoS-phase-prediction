"""STEP 2 of 2 — run the pipeline, start to finish or a few stages at a time.

Run `python pipeline/get_data.py` FIRST (step 1) to download the input data
from Zenodo into data/. This script stops with a reminder if the data it
needs are missing.

CAUTION — disk space. The preprocessing stages write large gridded files:
about 6 GB for the 1 km IMERG grids (resample_gridded) and about 26 GB for
the kriged surfaces (kriging_interpolation), ~33 GB per full run for both
regions. The two newest runs of each are kept (keep_last in
project_paths.yaml), so repeated runs can hold ~65 GB. Before starting, the
script estimates what the selected stages will write, checks free space, and
asks for confirmation above 5 GB (skip the question with --yes). Kriging might
also take many hours per region.

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
  python pipeline/run_pipeline.py --yes                  no confirmation prompt
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import REGION_IDS, REPO_ROOT, list_runs, pinned_run, raw_path

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


# Approximate size (GB) of what one run of a stage writes, per region, taken
# from the manuscript's runs. Stages not listed write well under 1 GB.
OUTPUT_GB = {
    "resample_gridded": {"CA": 3.7, "CO": 2.5},
    "kriging_interpolation": {"CA": 11.7, "CO": 14.5},
    "compile": {"CA": 0.1, "CO": 0.1},
}
SMALL_STAGE_GB = 0.1
CONFIRM_ABOVE_GB = 5


def estimate_output_gb(stages, regions):
    return sum(OUTPUT_GB.get(s[0], {}).get(r, SMALL_STAGE_GB)
               for s in stages for r in regions)


def check_disk_space(stages, regions, assume_yes):
    """Warn about large outputs, stop if the disk is too full, ask above 5 GB."""
    need = estimate_output_gb(stages, regions)
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    print(f"Estimated new output: ~{need:.0f} GB   free on this disk: {free:.0f} GB")
    big = [s[0] for s in stages if s[0] in ("resample_gridded", "kriging_interpolation")]
    if big:
        print(f"  ({', '.join(big)} write large gridded files; see the note at the top "
              "of this script)")
    if free < need * 1.1:
        raise SystemExit(
            f"\nNot enough free disk space (~{need:.0f} GB needed). Free up space, or "
            "start later in the pipeline, e.g. download the manuscript's grids with\n"
            "  python pipeline/get_data.py --include interim\n"
            "and run\n  python pipeline/run_pipeline.py --from build_dataset")
    if need > CONFIRM_ABOVE_GB and not assume_yes:
        if not sys.stdin or not sys.stdin.isatty():
            raise SystemExit(f"\nThese stages will write ~{need:.0f} GB. "
                             "Re-run with --yes to confirm.")
        answer = input(f"These stages will write ~{need:.0f} GB. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            raise SystemExit("Cancelled.")
    print()


def check_interim_data(stages, regions):
    """Starting after kriging needs kriging and IMERG-grid runs to read."""
    names = {s[0] for s in stages}
    if not names & {"build_dataset", "benchmarking", "ablation"}:
        return
    missing = [f"{stage} [{r}]"
               for r in regions
               for stage, producer in (("kriging", "kriging_interpolation"),
                                       ("resampled_1km", "resample_gridded"))
               if producer not in names
               and not list_runs(stage, r) and pinned_run(stage, r) is None]
    if missing:
        raise SystemExit(
            f"No {', '.join(missing)} output to start from.\n\n"
            "Either download the manuscript's grids (~33 GB) first:\n"
            "  python pipeline/get_data.py --include interim\n"
            "or run the preprocessing stages too:\n"
            "  python pipeline/run_pipeline.py\n")


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
            "Step 1 is to download it from Zenodo (~2 GB):\n  python pipeline/get_data.py\n"
        )


def run(stages, regions, dry_run, assume_yes=False):
    print(f"Regions: {', '.join(regions)}")
    print(f"Stages : {', '.join(s[0] for s in stages)}\n")
    if not dry_run:
        check_raw_data(stages, regions)
        check_interim_data(stages, regions)
        check_disk_space(stages, regions, assume_yes)

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
        description="Step 2: run the precipitation-phase pipeline. "
                    "Run `python pipeline/get_data.py` first (step 1).",
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
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Do not ask for confirmation before writing large outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    args = parser.parse_args()

    if args.list:
        print("Data collection (only with --with-collection):")
        for name, folder, script, _ in COLLECTION_STAGES:
            print(f"      {name:22s} {folder}/{script}")
        print("\nPipeline (writes, both regions):")
        for i, (name, folder, script, _) in enumerate(STAGES, start=1):
            gb = sum(OUTPUT_GB.get(name, {}).get(r, 0) for r in REGION_IDS)
            size = f"~{gb:.0f} GB" if gb >= 1 else "< 1 GB"
            print(f"  {i:2d}  {name:22s} {folder + '/' + script:40s} {size}")
        print("\nStep 1, if not done yet: python pipeline/get_data.py")
        return

    run(select_stages(args), args.regions, args.dry_run, args.yes)


if __name__ == "__main__":
    main()
