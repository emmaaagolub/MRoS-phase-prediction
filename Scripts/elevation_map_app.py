# Run using: python elevation_map_app.py
# Don't forget to download the dependencies: pip install flask rasterio numpy pillow matplotlib folium

from flask import Flask, jsonify, send_from_directory, render_template_string, request
import rasterio as rio
from rasterio.warp import transform
import numpy as np
import os

# ---- CONFIG ----
TIF_PATH = "DEM_1km.tif"   #  GeoTIFF file
HOST = "127.0.0.1"
PORT = 5000

app = Flask(__name__, static_folder="static")

# Load once at startup
src = rio.open(TIF_PATH)
NODATA = src.nodata

# Precompute bounds in WGS84 for overlay/map fit
left, bottom, right, top = src.bounds
src_crs = src.crs
(wgs_left, wgs_bottom), (wgs_right, wgs_top) = list(zip(*transform(
    src_crs, "EPSG:4326",
    [left, right], [bottom, top]
)))

# A lightweight downsampled PNG overlay so you can see the DEM extent
import io
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from PIL import Image
import matplotlib.pyplot as plt

MAX_DIM = 800
band = src.read(1).astype("float32")
if NODATA is not None:
    band = np.where(band == NODATA, np.nan, band)
h, w = band.shape
if w >= h:
    new_w = min(w, MAX_DIM)
    new_h = max(1, int(h * (new_w / w)))
else:
    new_h = min(h, MAX_DIM)
    new_w = max(1, int(w * (new_h / h)))

dem_wgs = np.empty((new_h, new_w), dtype="float32")
dst_transform = from_bounds(min(wgs_left, wgs_right), min(wgs_bottom, wgs_top),
                            max(wgs_left, wgs_right), max(wgs_bottom, wgs_top),
                            new_w, new_h)

reproject(
    source=band,
    destination=dem_wgs,
    src_transform=src.transform,
    src_crs=src.crs,
    dst_transform=dst_transform,
    dst_crs="EPSG:4326",
    resampling=Resampling.bilinear,
    src_nodata=np.nan,
    dst_nodata=np.nan,
)

finite = np.isfinite(dem_wgs)
if finite.any():
    vmin, vmax = np.nanpercentile(dem_wgs, [2, 98])
    if vmin == vmax:
        vmax = vmin + 1.0
else:
    vmin, vmax = 0.0, 1.0

cmap = plt.get_cmap("terrain")
norm = np.clip((dem_wgs - vmin) / (vmax - vmin), 0, 1)
rgba = cmap(norm, bytes=True)
rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
img = Image.fromarray(rgba, mode="RGBA")
buf = io.BytesIO()
img.save(buf, format="PNG")
OVERLAY_BYTES = buf.getvalue()

@app.route("/")
def index():
    # Inline HTML (Leaflet) – click map → fetch elevation from /elev
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DEM Elevation Click</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .info {{
      position: absolute; z-index: 1000; background: #fff; padding: 8px 10px;
      border: 1px solid #ccc; border-radius: 6px; top: 10px; left: 10px;
      font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      box-shadow: 0 2px 6px rgba(0,0,0,.15);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="info">Click anywhere to see <b>Lat, Lon, Elevation (m)</b>.</div>
  <script>
    const map = L.map('map').setView([{(wgs_bottom+wgs_top)/2}, {(wgs_left+wgs_right)/2}], 6);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 18, attribution: "&copy; OpenStreetMap"
    }}).addTo(map);

    // DEM overlay (downsampled PNG)
    const imgBounds = [[{min(wgs_bottom, wgs_top)}, {min(wgs_left, wgs_right)}],
                       [{max(wgs_bottom, wgs_top)}, {max(wgs_left, wgs_right)}]];
    L.imageOverlay("/overlay.png", imgBounds, {{opacity: 0.7}}).addTo(map);
    map.fitBounds(imgBounds);

    map.on('click', async (e) => {{
      try {{
        const url = `/elev?lat=${{e.latlng.lat}}&lon=${{e.latlng.lng}}`;
        const res = await fetch(url);
        const j = await res.json();
        const elev = (j.ok && j.elev !== null) ? j.elev.toFixed(2) + " m" : "No data";
        const html = `Lat: ${{e.latlng.lat.toFixed(5)}}<br>Lon: ${{e.latlng.lng.toFixed(5)}}<br><b>Elevation:</b> ${{elev}}`;
        L.popup().setLatLng(e.latlng).setContent(html).openOn(map);
      }} catch (err) {{
        console.error(err);
      }}
    }});
  </script>
</body>
</html>
"""
    return render_template_string(html)

@app.route("/overlay.png")
def overlay_png():
    return app.response_class(OVERLAY_BYTES, mimetype="image/png")

@app.route("/elev")
def elev():
    """
    Query exact elevation at given lat/lon from the full-resolution GeoTIFF.
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify(ok=False, error="Missing lat/lon"), 400

    # Transform click (WGS84) to the DEM CRS
    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
    x, y = xs[0], ys[0]
    # Convert world coords -> row/col
    row, col = src.index(x, y)
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        return jsonify(ok=True, elev=None)  # outside raster

    val = src.read(1, window=((row, row+1), (col, col+1))).squeeze()
    if (NODATA is not None and val == NODATA) or not np.isfinite(val):
        return jsonify(ok=True, elev=None)
    return jsonify(ok=True, elev=float(val))

if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT} in your browser")
    app.run(host=HOST, port=PORT, debug=True)
