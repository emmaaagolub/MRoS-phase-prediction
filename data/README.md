# data/

The data are too large for GitHub and are archived on Zenodo. Download them
into this folder with

```bash
python pipeline/get_data.py                     # core inputs, ~2 GB
python pipeline/get_data.py --include dem_10m   # + raw 10 m DEMs (~6 GB)
python pipeline/get_data.py --include interim   # + the manuscript's intermediate and processed products (~33 GB)
```

(Zenodo DOI: 10.5281/zenodo.23022221)

```
data/
  raw/                              inputs, never modified by the pipeline
    stations/{CA,CO}/                 hourly station records (HADS, LCD, ASOS, SNOTEL)
                                      + station_metadata_20221001_20260501.csv
    imerg/{CA,CO}/                    GPM IMERG liquid-precipitation probability, one file per day
    mros/                             Mountain Rain or Snow observations, both regions. The Zenodo
                                      copy is the public version: locations rounded to 4 decimals
                                      (~10 m), comments removed; see README_public_version.txt
    dem/                              10 m DEMs (optional; only process_dem uses them)
    reference/                        state boundaries used on the maps
  interim/                          written by the pipeline
    dem_1km/                          1 km DEMs that define the model grid
    hourly_compiled/{CA,CO}/<run>/    stations, IMERG and MRoS on a common hourly grid
    resampled_1km/{CA,CO}/<run>/      IMERG on the 1 km hourly grid
    kriging/{CA,CO}/<run>/            kriged predictor surfaces + leave-one-out MRoS table
    kriging_test/{CA,CO}/<run>/       small test runs (kriging_interpolation.py --test);
                                      never read by later stages
```

All MRoS locations on Zenodo are rounded to ~10 m, so results computed from
them differ slightly from the published ones. Full-precision MRoS observation
locations can be shared on request, subject to approval by the Mountain Rain
or Snow project team.

Study period: 1 October 2022 – 1 May 2026. Every location above is set in
`project_paths.yaml` at the repository root.
