GPM IMERG data:
- Spatial resolution = 0.1° x 0.1° (~10km x 10km)
- Temporal resolution = 30 minute

Files:
- gpm_aoi.tiff: the gridded output for the AOI (Sierra Nevada bounding box provided by Zeed)
- /imerg_data/~: IMERG plp (probabilityLiquidPrecipitation) values for each grid in table format.
-- data format: x, y, timestep... (48 in total for 1 day)

More info:
The "probabilityLiquidPrecipitation" field is a probability from 0 to 100, 0% = MOST likely snow and 100% = MOST likely rain.

Here are common precip types (https://gpm.nasa.gov/data/imerg#imergfrequentlyaskedquestions): 
- Rain:  Ordinary falling liquid typically happens for Tw>0°C, so pLP is high.
- Freezing Rain:  Liquid that freezes upon contact with the Earth's surface typically falls in Tw<0°C, so pLP is low.
- Snow, ice pellets, snow pellets:  These frozen hydrometeors occur around or below Tw<0°C, so pLP varies from around 50% to very low.
- Sleet:  Frozen droplets (U.S. definition) typically fall in Tw<0°C, so pLP is usually below 50%.
- Mixed snow and rain; falling slush:  The mixed category is likely to occur around the pLP=50% mark.  If one uses 50% as a liquid/solid threshold, that implies that mixed cases will end up in both categories, depending on the details.
- Hail:  Hail typically occurs when the surface air temperature is well above freezing (i.e., on summer afternoons).  Thus, pLP is very high.  But, hail is even rarer than mixed and unlikely to be correctly specified in this scheme, and anyway, in such conditions it rapidly melts and so is properly lumped into "liquid".
- Dew and frost:  These phenomena are not forms of precipitation.  They are liquid or solid water that condenses directly at the Earth's surface.  For this reason, any amount of surface accumulation due to dew or frost is not included in the IMERG precip estimate.


