"""Download the project data from Zenodo into data/.

  python pipeline/get_data.py                         core inputs (~2 GB)
  python pipeline/get_data.py --include dem_10m       also the raw 10 m DEMs (~6 GB)
  python pipeline/get_data.py --include interim results
                                                      also the manuscript's
                                                      intermediate grids and model
                                                      runs, to redraw the figures
                                                      without re-running kriging
  python pipeline/get_data.py --list                  show what the record holds
  python pipeline/get_data.py --from-dir ~/Downloads  use files already downloaded
                                                      by hand from the Zenodo page

The Zenodo record id is set in project_paths.yaml (zenodo.record_id) or given
with --record. The record holds zenodo_manifest.json, which says where each
file goes; every file is checked against its MD5 checksum. Files already in
place are skipped, so the script can be re-run after an interruption.

Groups
  core     station tables, IMERG, MRoS observations, state boundaries, 1 km DEMs
  dem_10m  raw 10 m DEMs (only needed to re-run process_dem)
  interim  pinned compiled tables, resampled IMERG grid and kriging output
  results  pinned model, benchmarking and ablation runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO_ROOT, SETTINGS  # noqa: E402

API = "https://zenodo.org/api/records/{record}"
FILE_URL = "https://zenodo.org/api/records/{record}/files/{key}/content"
MANIFEST_NAME = "zenodo_manifest.json"
DOWNLOAD_DIR = REPO_ROOT / "data" / "_downloads"


def md5sum(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def download(url, dest, size=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        total = size or int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            block = resp.read(8 * 1024 * 1024)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.replace(dest)


def resolve_source(args):
    """Return (manifest, fetch) where fetch(name, dest) puts a record file at dest."""
    if args.from_dir:
        src = Path(args.from_dir).expanduser()
        manifest = json.loads((src / MANIFEST_NAME).read_text())

        def fetch(name, dest, size=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dest)
        return manifest, fetch

    record = args.record or (SETTINGS.get("zenodo") or {}).get("record_id")
    if not record:
        raise SystemExit(
            "No Zenodo record id. Set zenodo.record_id in project_paths.yaml, pass "
            "--record <id>, or download the files by hand and use --from-dir <folder>."
        )
    meta = fetch_json(API.format(record=record))
    sizes = {f["key"]: f.get("size") for f in meta.get("files", [])}
    if MANIFEST_NAME not in sizes:
        raise SystemExit(f"Record {record} has no {MANIFEST_NAME}.")
    manifest = fetch_json(FILE_URL.format(record=record, key=MANIFEST_NAME))

    def fetch(name, dest, size=None):
        download(FILE_URL.format(record=record, key=name), dest, size or sizes.get(name))
    return manifest, fetch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include", nargs="*", default=[],
                    choices=["dem_10m", "interim", "results"],
                    help="Optional groups to download as well as the core inputs.")
    ap.add_argument("--record", help="Zenodo record id (overrides project_paths.yaml).")
    ap.add_argument("--from-dir", help="Folder holding files already downloaded from Zenodo.")
    ap.add_argument("--keep-archives", action="store_true",
                    help="Keep the downloaded .zip files after unpacking.")
    ap.add_argument("--list", action="store_true", help="List the record's files and exit.")
    args = ap.parse_args()

    manifest, fetch = resolve_source(args)
    groups = {"core", *args.include}
    files = manifest["files"]

    if args.list:
        for f in files:
            print(f"  [{f['group']:7s}] {f['name']:40s} {f['size'] / 1e9:7.2f} GB  -> {f['dest']}")
        return

    wanted = [f for f in files if f["group"] in groups]
    total = sum(f["size"] for f in wanted) / 1e9
    print(f"Downloading {len(wanted)} file(s), {total:.1f} GB, groups: {', '.join(sorted(groups))}")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for f in wanted:
        name, dest = f["name"], REPO_ROOT / f["dest"]
        marker = DOWNLOAD_DIR / f"{name}.done"
        if marker.exists() and marker.read_text().strip() == f["md5"]:
            print(f"  {name}: already in place")
            continue

        print(f"  {name} ({f['size'] / 1e9:.2f} GB)")
        target = DOWNLOAD_DIR / name if f["unzip"] else dest
        if not (target.exists() and md5sum(target) == f["md5"]):
            fetch(name, target, f["size"])
            got = md5sum(target)
            if got != f["md5"]:
                raise SystemExit(f"    checksum mismatch for {name}: {got} != {f['md5']}")

        if f["unzip"]:
            # Archive paths are relative to the repository root.
            with zipfile.ZipFile(target) as zf:
                for member in zf.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise SystemExit(f"    unsafe path in {name}: {member}")
                zf.extractall(REPO_ROOT)
            print(f"    unpacked into {f['dest']}")
            if not args.keep_archives:
                target.unlink()
        marker.write_text(f["md5"])

    print("\nDone. Next:\n  python pipeline/run_pipeline.py")


if __name__ == "__main__":
    main()
