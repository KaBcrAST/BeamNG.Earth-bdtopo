"""Roof shapes over France from the IGN LiDAR HD height model (the user, 2026-10-26: what would make the
buildings really realistic). Overture and OpenStreetMap give a roof shape to almost nobody (central Lille: 2.8 %
of the cell), so buildings.plan() picks every pitched roof from the footprint alone, and a French town's
rooflines, the thing you read first from a car, are invented.

The IGN publishes LiDAR HD as point clouds, which no streaming server can use, but it also serves the derived
MNH (modele numerique de hauteur: metres above the ground) as a plain raster over the same Geoplateforme and the
same Licence Ouverte 2.0 as BD TOPO (bdtopo.py):

    https://data.geopf.fr/wms-r/wms   IGNF_LIDAR-HD_MNH_..., GeoTIFF float, EPSG:4326

One request covers a map cell. Each Overture footprint is clipped out of it and read: the eaves, the ridge, and
where the roof is near its top along the footprint's own two axes, which tells a ridge line from a hip from a
point. Values the mapped data already carries are left alone, nothing outside metropolitan France is asked for,
and BNE_LIDAR=off turns the source off. The raster itself is never cached: only the few values per building it
yields are, in the buildings layer.

Measured against the live service over Bailleul on 2026-10-26 (cell z14/8316/5504, 3151 buildings), which is also
what the thresholds below are set from:
  * 2965 of the 3151 footprints (94 %) got a roof at 0.50 m per pixel.
  * The shapes track what the building is, which nothing in this module knows: 97 % of the houses came out
    pitched, 75 % of the sheds flat, and the flat ones' median footprint is 20 m2 against 85 m2 for the pitched.
  * The ridge heights agree with BD TOPO's own "hauteur" to within a metre on 61 % of them: two independent IGN
    products, and this module never sees the other one.
  * The ridge bearings cluster on two perpendicular street-aligned peaks (chi2 253 against 83 a bin if they were
    random), so the bearing is signal, not noise. It is measured but not stored: see join().
  * What plan() draws today matches the real roof on only 43 % of them, and worst of all it draws 54 % of the
    cell flat where the LiDAR finds 29 %: about 900 buildings of one cell get a flat roof over a real pitch.
  * METRES_PER_PX is 1.0 rather than 0.5 for the bytes: 9.6 MB a cell instead of 38, against the 6 MB the cell's
    Overture footprints already cost. It keeps 99 % of the houses and 100 % of the flats, and loses the sheds,
    which plan() already draws flat and the LiDAR agrees are flat 75 % of the time.
"""
import io
import logging
import os
import urllib.parse

import numpy as np

import bdtopo

log = logging.getLogger("earthd.roofs")

WMS = "https://data.geopf.fr/wms-r/wms"
LAYER = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.WGS84G"
METRES_PER_PX = 1.0             # of the fetched raster (the trade-off in the header)
MAX_PX = 5010                   # the service's own GetMap limit, from its capabilities
SHRINK = 0.88                   # the footprint pulled in before reading, so a neighbour's wall cannot leak in
MIN_PIX = 12                    # pixels of roof needed to say anything (at 1 m: a 12 m2 building)
FLAT_RELIEF = 1.0               # m, a roof flatter than this is flat
RIDGE_FRAC = 0.70               # a pixel this far up the relief counts as near the top
LONG_RIDGE = 0.55               # the top runs this much of an axis -> a ridge line along it
SHORT_RIDGE = 0.38              # ... this little -> a hip or a point
POINT_RIDGE = 0.22              # ... this little on both -> a pyramid
MIN_BUILDING_M = 2.0            # m, below it the LiDAR saw no building here (a flight older than the house)
HEIGHT_SHRINK = 0.70            # the footprint pulled in harder for the roof HEIGHT than for the shape ...
HEIGHT_LO, HEIGHT_HI = 35, 98   # ... and read between these percentiles (calibrated in height_of)
HEIGHT_MIN_PIX = 20             # pixels needed before a height is believed at all
MIN_ROOF_M, MAX_ROOF_M = 0.8, 25.0      # a roof height outside this is not one

PROVENANCE = {
    "id": "ign-lidarhd",
    "name": "IGN LiDAR HD MNH (roof shapes)",
    "attribution": "Roof shapes: IGN LiDAR HD",
    "licence": "Licence Ouverte 2.0 (https://www.etalab.gouv.fr/licence-ouverte-open-licence/)",
    "url": "https://geoservices.ign.fr/lidarhd",
}
CREDIT_LINE = "Roof shapes: IGN LiDAR HD (Licence Ouverte 2.0)"


def enabled():
    """False when the player set BNE_LIDAR=off (the roofs keep buildings.plan()'s guesses)."""
    return os.environ.get("BNE_LIDAR", "on").strip().lower() not in ("0", "off", "no", "false")


def covers(bounds):
    """BD TOPO's extent is LiDAR HD's here: metropolitan France and Corsica (bdtopo.FRANCE)."""
    return bdtopo.covers(bounds)


def map_url(bounds, w, h):
    """The WMS GetMap URL for the cell's height raster.

    WMS 1.3.0 with EPSG:4326 takes the BBOX as lat,lon, the other way round from the WFS of bdtopo.py (measured
    on the live service, 2026-10-26): the two must not be copied from one another."""
    west, south, east, north = bounds
    q = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": LAYER, "STYLES": "",
        "CRS": "EPSG:4326", "BBOX": f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}",
        "WIDTH": str(int(w)), "HEIGHT": str(int(h)), "FORMAT": "image/geotiff",
    }
    return f"{WMS}?{urllib.parse.urlencode(q)}"


def _size(bounds):
    """(width, height) in pixels for METRES_PER_PX over the cell, within the service's limit."""
    west, south, east, north = bounds
    kx = 111320.0 * np.cos(np.radians((south + north) * 0.5))
    w = (east - west) * kx / METRES_PER_PX
    h = (north - south) * 110540.0 / METRES_PER_PX
    k = min(1.0, MAX_PX / max(w, h, 1.0))
    return max(16, int(w * k)), max(16, int(h * k))


def fetch(fetcher, bounds, kind="demand"):
    """The cell's height-above-ground raster as a (h, w) float array, or None outside France / when turned off.

    Raises whatever netfetch raises: the caller decides whether the cell is built without the roofs."""
    if not (enabled() and covers(bounds)):
        return None
    w, h = _size(bounds)
    res = fetcher.get(map_url(bounds, w, h), kind=kind)
    from PIL import Image
    with Image.open(io.BytesIO(res.body)) as im:
        a = np.asarray(im, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 8 or a.shape[1] < 8:
        raise ValueError(f"roofs: {a.shape} is not a height raster")
    return a


def _axes(ring):
    """(u, v, metres per degree of longitude): the footprint's long axis and its normal, in metres."""
    lat0 = float(ring[:, 1].mean())
    kx = 111320.0 * np.cos(np.radians(lat0))
    p = np.stack([ring[:, 0] * kx, ring[:, 1] * 110540.0], axis=1)
    p = p - p.mean(axis=0)
    _vals, vec = np.linalg.eigh(np.cov(p.T) + np.eye(2) * 1e-9)
    u = vec[:, -1] / np.linalg.norm(vec[:, -1])
    return u, np.array([-u[1], u[0]]), kx


def _pixels(a, ring, bounds, shrink=None):
    """(heights, pixel x, pixel y) of the footprint, pulled in by shrink (default SHRINK); (None, None, None)
    when too small."""
    west, south, east, north = bounds
    h, w = a.shape
    c = ring.mean(axis=0)
    inner = c + (ring - c) * (SHRINK if shrink is None else shrink)
    xs = (inner[:, 0] - west) / (east - west) * w
    ys = (north - inner[:, 1]) / (north - south) * h
    x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max())) + 1
    y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max())) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None, None
    from PIL import Image, ImageDraw
    m = Image.new("1", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).polygon([(float(x) - x0, float(y) - y0) for x, y in zip(xs, ys)], fill=1)
    mask = np.array(m, dtype=bool)
    if int(mask.sum()) < MIN_PIX:
        return None, None, None
    yy, xx = np.nonzero(mask)
    return a[y0:y1, x0:x1][mask], xx + x0, yy + y0


def shape_of(hgt, px, py, ring, bounds, shape_w, shape_h):
    """(shape, ridge bearing in degrees or None) of one roof, or (None, None) when the raster has no building.

    The bearing is the ridge's, 0-180 degrees clockwise from north. The relief measured here is only used to
    tell the shapes apart and is NOT the roof height: see height_of, which needs its own, tighter footprint."""
    hi = float(np.percentile(hgt, 98))
    lo = float(np.percentile(hgt, 12))
    if hi < MIN_BUILDING_M:
        return None, None                       # no building here in the flight: leave plan() its guess
    relief = hi - lo
    if relief < FLAT_RELIEF:
        return "flat", None
    west, south, east, north = bounds
    u, v, kx = _axes(ring)
    lon = west + (px + 0.5) / shape_w * (east - west)
    lat = north - (py + 0.5) / shape_h * (north - south)
    p = np.stack([lon * kx, lat * 110540.0], axis=1)
    p = p - p.mean(axis=0)
    pu, pv = p @ u, p @ v
    top = hgt >= lo + RIDGE_FRAC * relief
    if int(top.sum()) < 6:
        return "flat", None

    def span(proj):
        full = float(proj.max() - proj.min())
        if full <= 0.5:
            return 0.0
        q = proj[top]
        return float(np.percentile(q, 95) - np.percentile(q, 5)) / full

    def bearing(axis):
        return float(np.degrees(np.arctan2(axis[0], axis[1])) % 180.0)

    su, sv = span(pu), span(pv)
    if su >= LONG_RIDGE and sv <= SHORT_RIDGE:
        axis = u
    elif sv >= LONG_RIDGE and su <= SHORT_RIDGE:
        axis = v
    elif su <= SHORT_RIDGE and sv <= SHORT_RIDGE:
        # the top is a patch, not a line: a hip, or a point when it is tight on both axes
        if max(su, sv) < POINT_RIDGE:
            return "pyramidal", None            # a point has no direction
        return "hipped", bearing(u if su >= sv else v)
    else:
        # neither a clean line nor a point: a hip is the safe middle, laid on whichever axis the top runs
        # further along (a hipped roof still has a short ridge, and it still faces a way)
        return "hipped", bearing(u if su >= sv else v)
    return "gabled", bearing(axis)


def height_of(raster, ring, bounds):
    """The eaves-to-ridge height (m) of one footprint's roof, or None when it cannot be read.

    Its own measurement, not shape_of's relief. At 1 m per pixel a footprint's edge pixels mix roof and
    ground, so a percentile taken over the lightly shrunk footprint the classifier uses reads far too low
    at the eaves and the relief comes out too big: against BD TOPO's own surveyed roof heights over Bailleul
    (1368 buildings, 2026-10-28), shape_of's 0.88 / 12-98 reads +2.02 m too high at the median and lands
    within a metre of the survey on 26 % of them, while this 0.70 / 35-98 reads -0.05 m at the median and
    within a metre on 54 %. The pitch that comes out of it is closer than the region's constant on 84 % of
    the roofs (median error 6.6 deg against 16.9 deg), which is why the second rasterisation is paid for."""
    hgt, _px, _py = _pixels(raster, ring, bounds, HEIGHT_SHRINK)
    if hgt is None or len(hgt) < HEIGHT_MIN_PIX:
        return None
    h = float(np.percentile(hgt, HEIGHT_HI) - np.percentile(hgt, HEIGHT_LO))
    return h if MIN_ROOF_M <= h <= MAX_ROOF_M else None


def join(feats, raster, bounds):
    """Give every building of feats with no mapped roof shape the shape, ridge bearing and roof height its
    footprint's LiDAR says.

    Returns (shapes named, ridges named, heights named); feats is changed in place.

    The bearing goes in as ridge_direction, our own field rather than Overture's roof_direction, because
    OpenStreetMap's roof:direction is the way the roof FACES and this is the ridge's own line; and the
    height as roof_height, which plan() already reads where the mapped data carries one. Both matter now
    and neither did when this module was written: the mesh builder laid every ridge on the footprint's long
    side and every pitch at the region's constant, and over Bailleul that is the wrong way round for 88 %
    of the row houses (buildings._ridge_across) and off by more than 5 degrees on 77 % of the pitches."""
    n = len(feats)
    if not n or raster is None:
        return 0, 0, 0
    h, w = raster.shape
    shapes = bdtopo._text(feats, "roof_shape", n)
    # numbers, not text: plan() reads roof_height with np.isfinite, and bdtopo writes the same field as floats
    ridges = bdtopo._column(feats, "ridge_direction", n)
    heights = bdtopo._column(feats, "roof_height", n)
    named = ridged = raised = 0
    for i in range(n):
        if shapes[i] and ridges[i] is not None and heights[i] is not None:
            continue                            # Overture or OSM already knows all of it
        ring = bdtopo._outer(feats, i)
        if ring is None:
            continue
        hgt, px, py = _pixels(raster, ring, bounds)
        if hgt is None:
            continue
        s, bearing = shape_of(hgt, px, py, ring, bounds, w, h)
        if s is None:
            continue
        if not shapes[i]:
            shapes[i] = s
            named += 1
        if ridges[i] is None and bearing is not None:
            ridges[i] = bearing
            ridged += 1
        # only for a pitched roof: a flat one's relief is its parapets and its plant, not a roof to raise
        if heights[i] is None and s in ("gabled", "hipped"):
            rh = height_of(raster, ring, bounds)
            if rh is not None:
                heights[i] = rh
                raised += 1
    if named:
        feats.attrs["roof_shape"] = np.array([v or "" for v in shapes])
    if ridged:
        feats.attrs["ridge_direction"] = bdtopo._floats(ridges)
    if raised:
        feats.attrs["roof_height"] = bdtopo._floats(heights)
    return named, ridged, raised
