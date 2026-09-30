"""Package data/ (and optionally the pinned runs) for upload to Zenodo.

Writes zip archives plus zenodo_manifest.json into zenodo_upload/ (ignored by
git). Upload every file in that folder to one Zenodo record, then put the
record id in project_paths.yaml (zenodo.record_id) so get_data.py can find it.

  python pipeline/tools/package_for_zenodo.py                  core inputs
  python pipeline/tools/package_for_zenodo.py --groups core dem_10m interim results
  python pipeline/tools/package_for_zenodo.py --dry-run        sizes only

Archive paths are relative to the repository root, so get_data.py unpacks
them straight into place. Zenodo allows 50 GB per record (100 files).

MRoS locations: the core bundle contains the PUBLIC MRoS file made by
make_public_mros.py (coordinates rounded to 4 decimals, etc.), stored under
the raw.mros path so the pipeline finds it. The original file is never
packaged. In the interim group, the tables that list individual MRoS reports
(MROS_POINT_FILES below) are packaged as copies with coordinates rounded the
same way as the public file (lat/lon to 4 decimals, x/y to 10 m), with a
note in each bundle; everything else is packaged as is. The results group still holds
full-precision observer locations (lat/lon and UTM x/y in the point tables),
so it is refused unless --allow-precise-locations is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REGION_IDS, REPO_ROOT, dem_1km_path, pinned_run, raw_path  # noqa: E402
from make_public_mros import (  # noqa: E402
    ACCESS_NOTE, DECIMALS, XY_ROUND_M, coarsen_point_table, public_mros_path,
)

OUT_DIR = REPO_ROOT / "zenodo_upload"
SKIP_NAMES = {"run_manifest.json.tmp"}
SKIP_DIRS = {"__pycache__", "_download_logs", "hourly_chunks", "loocv_chunks"}

# Interim files holding one row per MRoS report with full-precision lat/lon
# (and UTM x/y). The interim bundles get copies with rounded coordinates
# (make_public_mros.coarsen_point_table), never the originals.
MROS_POINT_FILES = {
    "mros_hourly.parquet",                            # hourly_compiled
    "mros_processed.parquet",                         # kriging/processed_inputs
    "mros_loocv_point_predictions_kriging.parquet",   # kriging
    "mros_loocv_point_predictions_kriging.csv",       # kriging
}


def md5sum(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def files_under(folder, recursive=True):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return [(p, arc(p)) for p in sorted(
        p for p in it if p.is_file() and p.name not in SKIP_NAMES
        and not SKIP_DIRS & set(p.relative_to(folder).parts[:-1]))]


def interim_note(rounded_files):
    names = "\n".join(f"  {Path(a).name}" for a in rounded_files)
    return f"""MRoS observation locations in this folder

To protect observers' locations, these tables list MRoS reports with rounded
coordinates, the same rounding as the public MRoS file
(data/raw/mros/README_public_version.txt): latitude/longitude to {DECIMALS} decimal
places (~10 m), projected x/y to the nearest {XY_ROUND_M} m.
{names}

The gridded products in this folder were computed from the full-precision
locations. Training the model from these tables (starting at build_dataset)
therefore gives results that differ slightly from the published ones: a few
reports fall in a neighbouring 1 km grid cell. Re-running the whole pipeline
on the public MRoS file is self-consistent but also differs slightly.

{ACCESS_NOTE}
"""


def arc(path):
    """Archive name for a file: its path relative to the repository root."""
    return Path(path).relative_to(REPO_ROOT).as_posix()


def packages(groups):
    """(archive name, group, [(file, name inside archive)], unzip, destination)."""
    out = []
    if "core" in groups:
        public = public_mros_path()
        if not public.exists():
            raise SystemExit(f"{public} not found. Make it first:\n"
                             "  python pipeline/tools/make_public_mros.py")
        mros = [(public, arc(raw_path("mros"))),
                (public.parent / "README_public_version.txt",
                 arc(raw_path("mros").parent / "README_public_version.txt"))]
        out.append(("mros_observations.zip", "core",
                    mros + files_under(raw_path("reference")), True, "data/raw/"))
        dems = [dem_1km_path(r) for r in REGION_IDS]
        dems += [p for d in dems if (p := Path(str(d) + ".aux.xml")).exists()]
        out.append(("dem_1km.zip", "core", [(p, arc(p)) for p in dems],
                    True, "data/interim/dem_1km/"))
        for r in REGION_IDS:
            out.append((f"stations_{r}.zip", "core", files_under(raw_path("stations", r)),
                        True, f"data/raw/stations/{r}/"))
            out.append((f"imerg_{r}.zip", "core", files_under(raw_path("imerg", r)),
                        True, f"data/raw/imerg/{r}/"))
    if "dem_10m" in groups:
        for r in REGION_IDS:
            p = raw_path("dem_10m", r)
            out.append((p.name, "dem_10m", [(p, arc(p))], False, f"data/raw/dem/{p.name}"))
    for group, stages in (("interim", ("hourly_compiled", "resampled_1km", "kriging")),
                          ("results", ("model", "benchmarking", "ablations"))):
        if group not in groups:
            continue
        for r in REGION_IDS:
            for stage in stages:
                run = pinned_run(stage, r)
                if run is None:
                    print(f"  no pinned {stage} run for {r}; skipped")
                    continue
                rel = run.relative_to(REPO_ROOT).as_posix()
                files = files_under(run)
                if group == "interim":
                    rounded = [a for p, a in files if p.name in MROS_POINT_FILES]
                    # A third element marks a file to be packaged as a rounded copy.
                    files = [(p, a, "coarsen") if p.name in MROS_POINT_FILES else (p, a)
                             for p, a in files]
                    for a in rounded:
                        print(f"  rounded copy (MRoS report locations): {a}")
                    if rounded:
                        files.append((None, f"{rel}/README_mros_locations.txt",
                                      interim_note(rounded)))
                out.append((f"{group}_{stage}_{r}.zip", group, files, True, rel + "/"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["core"],
                    choices=["core", "dem_10m", "interim", "results"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-precise-locations", action="store_true",
                    help="Package the results group even though its point tables hold "
                         "full-precision MRoS observer locations.")
    args = ap.parse_args()

    if "results" in args.groups and not args.allow_precise_locations:
        raise SystemExit(
            "The results group contains full-precision MRoS observer locations "
            "(lat/lon and UTM x/y in the point tables). Rebuild it from the public MRoS "
            "file first, or pass --allow-precise-locations if you intend to publish it."
        )

    pkgs = packages(set(args.groups))
    grand = 0
    for name, group, files, _, dest in pkgs:
        size = sum(item[0].stat().st_size for item in files if item[0] is not None)
        grand += size
        print(f"  [{group:7s}] {name:36s} {len(files):5d} files {size / 1e9:7.2f} GB -> {dest}")
    print(f"  total before compression: {grand / 1e9:.1f} GB")
    if args.dry_run:
        return

    OUT_DIR.mkdir(exist_ok=True)
    manifest = {"description": "Mountain Rain or Snow gridded precipitation-phase data. "
                               "Unpack with pipeline/get_data.py. MRoS observation "
                               "locations are rounded to ~10 m; " + ACCESS_NOTE,
                "files": []}
    for name, group, files, unzip, dest in pkgs:
        target = OUT_DIR / name
        print(f"writing {target.name} ...")
        if unzip:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf, \
                    tempfile.TemporaryDirectory() as tmp:
                for f, name_in_zip, *extra in files:
                    if f is None:                      # generated text (the note)
                        zf.writestr(name_in_zip, extra[0])
                    elif extra == ["coarsen"]:         # rounded copy, never the original
                        zf.write(coarsen_point_table(f, Path(tmp) / f.name), name_in_zip)
                    else:
                        zf.write(f, name_in_zip)
        else:
            shutil.copy2(files[0][0], target)
        manifest["files"].append({"name": name, "group": group, "dest": dest,
                                  "unzip": unzip, "size": target.stat().st_size,
                                  "md5": md5sum(target)})
    (OUT_DIR / "zenodo_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nUpload everything in {OUT_DIR} (including zenodo_manifest.json) to one "
          "Zenodo record, then set zenodo.record_id in project_paths.yaml.")


if __name__ == "__main__":
    main()
