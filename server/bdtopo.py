"""IGN BD TOPO building heights for France (the user, 2026-10-26: in France the buildings look nothing like the
real thing). Overture and OpenStreetMap almost never carry a height for a French building, so buildings.plan()
guesses one from the footprint and whole French towns come out the same height.

BD TOPO(R) is the IGN's national topographic database. Its "batiment" layer carries a height, a floor count,
ground / roof altitudes and the wall and roof materials of every building in France under the Licence Ouverte 2.0
(Etalab), which allows reuse and redistribution with attribution. It is read live, one request set per map-data
cell, from the Geoplateforme:

    https://data.geopf.fr/wfs/ows     BDTOPO_V3:batiment, GeoJSON, EPSG:4326

Each Overture building WITHOUT a height takes the values of the BD TOPO building its footprint centre falls in,
when the two footprints are within AREA_FACTOR of each other (France's Overture footprints come from the same
survey, so they line up; the area test drops the cases where one source holds a whole block and the other its
separate houses). The same join names the facade material where Overture has none, which is what makes a brick
town like Lille come out of brick instead of plaster. Values Overture or OSM already carry are left alone, and
BNE_BDTOPO=off turns the source off.

Checked against the live service over Lille, 2026-10-26 (cell z14/8331/5511, 5799 buildings), because none of
this is in the WFS capabilities:
  * BBOX is minlon,minlat,maxlon,maxlat and MUST carry a CRS token. Without one the service answers an empty
    FeatureCollection with HTTP 200 rather than an error, so the join would silently do nothing.
  * The GeoJSON is always lon,lat whatever SRSNAME asks for, always MultiPolygon, and always THREE values per
    position: the third is an altitude whose no-data value is -1000. A two-value unpack would raise on every
    feature, so _rings() takes the first two values of a position and drops the rest.
  * COUNT is capped at PAGE = 5000 server-side (COUNT=20000 still answers 5000), so a dense cell needs
    STARTINDEX paging; numberMatched gives the total up front and consecutive pages do not overlap.
  * "hauteur" is 0, not null, when the height is unknown, so 0 is read as missing there (an altitude of 0 is a
    real altitude and is kept). In that cell 95 % of the buildings had a height, 86 % both altitudes and 64 % a
    floor count.
  * gzip and PROPERTYNAME together take a page from 6.8 MB to 0.47 MB. netfetch sends no Accept-Encoding of its
    own, so the header goes per request and the body is unzipped here.
"""
import gzip
import json
import logging
import os
import urllib.parse

import numpy as np

import polygon

log = logging.getLogger("earthd.bdtopo")

WFS = "https://data.geopf.fr/wfs/ows"
LAYER = "BDTOPO_V3:batiment"
PAGE = 5000                     # the service's own cap, measured (a COUNT above it is clamped)
MAX_PAGES = 8                   # 40k buildings a cell, far above the densest (central Paris is ~7k)
# only what the join needs: the service sends every attribute otherwise, five times the bytes
PROPS = ("geometrie", "hauteur", "nombre_d_etages",
         "altitude_minimale_sol", "altitude_minimale_toit", "altitude_maximale_toit",
         "materiaux_des_murs", "materiaux_de_la_toiture")
# metropolitan France and Corsica, as imagery_regional's "france" row: the overseas departements have their own
# BD TOPO extents and are left for later
FRANCE = (-5.20, 41.30, 9.60, 51.10)        # west, south, east, north
AREA_FACTOR = 3.0               # the two footprints must be within this factor to be the same building
MIN_ROOF = 0.8                  # m, below it the roof is flat enough to leave to buildings.plan()
HEIGHT_RANGE = (2.0, 300.0)     # m, outside it the record is a data error, not a building
Z_NODATA = -1000.0              # the altitude the service sends for a position that has none
SOURCE = "IGN BD TOPO"          # height_src, alongside Overture's own dataset names

# BD TOPO's wall and roof materials come straight from the DGFiP land-registry files (the fichiers fonciers
# variables dmatgm and dmatto), matched onto the buildings once, in 2021. The field is TWO CHARACTERS, one digit a
# material, and a building with a single material is written X0 or 0X (Lille is 43 % "40", brick alone, which is
# what the city is built of). The digit lists are the Cerema's, read 2026-10-26:
#   https://doc-datafoncier.cerema.fr/doc/ff/pb40_pevprincipale/dmatgm
#   https://doc-datafoncier.cerema.fr/doc/ff/pb40_pevprincipale/dmatto
# They are turned into the OpenStreetMap building:material words Overture carries and buildings.plan() already
# knows (STONE_FACADES, "brick", "glass"), so nothing downstream needs a new vocabulary.
WALL_CODES = {"1": "stone", "2": "stone",       # pierre; meuliere is a stone too (the Paris basin's)
              "3": "concrete", "4": "brick", "5": "concrete_block", "6": "wood"}
ROOF_CODES = {"1": "roof_tiles", "2": "slate", "3": "metal", "4": "concrete"}
# Which of two materials is the one you see: the field says nothing about that (the Cerema's labels are in their
# own order, not the digits'), so the facade takes the outer skin a French building of that pair normally shows.
WALL_RANK = ("brick", "stone", "wood", "concrete", "concrete_block")
ROOF_RANK = ("slate", "roof_tiles", "metal", "concrete")

PROVENANCE = {
    "id": "ign-bdtopo",
    "name": "IGN BD TOPO(R) batiment (building heights)",
    "attribution": "Building heights: IGN BD TOPO(R)",
    "licence": "Licence Ouverte 2.0 (https://www.etalab.gouv.fr/licence-ouverte-open-licence/)",
    "url": "https://geoservices.ign.fr/bdtopo",
}
CREDIT_LINE = "Building heights: IGN BD TOPO (Licence Ouverte 2.0)"


def enabled():
    """False when the player set BNE_BDTOPO=off (the buildings keep buildings.plan()'s guesses)."""
    return os.environ.get("BNE_BDTOPO", "on").strip().lower() not in ("0", "off", "no", "false")


def covers(bounds):
    """Does the cell (lon0, lat0, lon1, lat1) meet BD TOPO's extent? No request ever leaves France."""
    w, s, e, n = bounds
    fw, fs, fe, fn = FRANCE
    return w < fe and e > fw and s < fn and n > fs


def page_url(bounds, start=0):
    """The WFS GetFeature URL for one page of the cell's buildings."""
    w, s, e, n = bounds
    q = {
        "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
        "TYPENAMES": LAYER, "OUTPUTFORMAT": "application/json", "SRSNAME": "EPSG:4326",
        # the CRS token is not optional: without it the service answers no features at all
        "BBOX": f"{w:.7f},{s:.7f},{e:.7f},{n:.7f},EPSG:4326",
        "PROPERTYNAME": ",".join(PROPS), "COUNT": str(PAGE), "STARTINDEX": str(int(start)),
    }
    return f"{WFS}?{urllib.parse.urlencode(q)}"


def _doc(res):
    """The decoded JSON of a netfetch Response (gzipped when the service took the Accept-Encoding)."""
    raw = res.body or b""
    enc = (res.headers or {}).get("content-encoding", "").lower()
    if "gzip" in enc or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def fetch(fetcher, bounds, kind="demand"):
    """Every BD TOPO building of the cell, as (ring, values) records; [] outside France or when turned off.

    Raises whatever netfetch raises: the caller decides whether the cell is built without the heights."""
    if not (enabled() and covers(bounds)):
        return []
    out, start, total = [], 0, None
    for _ in range(MAX_PAGES):
        res = fetcher.get(page_url(bounds, start), kind=kind, headers={"Accept-Encoding": "gzip"})
        doc = _doc(res)
        feats = doc.get("features") or []
        if total is None:
            got = doc.get("numberMatched")
            total = got if isinstance(got, int) else None       # it may answer the string "unknown"
        out.extend(_records(feats, bounds))
        start += len(feats)
        if len(feats) < PAGE or (total is not None and start >= total):
            break
    else:
        log.warning("bdtopo: more than %d buildings in %s: the rest keep their guessed heights",
                    MAX_PAGES * PAGE, bounds)
    return out


def _records(feats, bounds):
    """(ring, values) per part of each BD TOPO feature that carries something worth joining."""
    out = []
    for f in feats:
        vals = values((f or {}).get("properties") or {})
        if vals is None:
            continue                            # neither a height nor altitudes: nothing to give
        for ring in _rings(f.get("geometry") or {}, bounds):
            out.append((ring, vals))
    return out


def _rings(geom, bounds):
    """The outer ring of each part of a (Multi)Polygon, as (k, 2) lon/lat arrays.

    The service sends three values a position (lon, lat, altitude, the altitude Z_NODATA where it has none), so
    only the first two are taken. A ring whose coordinates look swapped is put back the right way round: the
    live service answers lon,lat, but a change there must not silently misplace every building in France."""
    polys = geom.get("coordinates") or []
    if geom.get("type") == "Polygon":
        polys = [polys]
    elif geom.get("type") != "MultiPolygon":
        return []
    w, s, e, n = bounds
    out = []
    for part in polys:
        if not part:
            continue
        pts = np.asarray([p[:2] for p in part[0] if len(p) >= 2], np.float64)
        if len(pts) < 4 or not np.isfinite(pts).all():
            continue
        lon, lat = pts[:, 0].mean(), pts[:, 1].mean()
        if not (w <= lon <= e and s <= lat <= n) and (w <= lat <= e and s <= lon <= n):
            pts = pts[:, ::-1]                  # the service answered lat,lon
        out.append(pts)
    return out


def material(code, codes, rank):
    """The material word of a two-character BD TOPO material code, or None.

    "40" and "04" are both one material; "14" names two, and the one that shows on the outside wins (rank)."""
    s = str(code or "").strip()
    if not s:
        return None
    got = {codes[c] for c in s if c in codes}
    for want in rank:
        if want in got:
            return want
    return None


def values(props):
    """(height, roof_height, num_floors, facade_material, roof_material) of one BD TOPO record, or None when it
    has no usable height.

    height is the top of the roof above the lowest ground the building stands on, which is what buildings.plan()
    draws; "hauteur" stands in where the altitudes are missing."""
    sol = _num(props.get("altitude_minimale_sol"))
    toit_min = _num(props.get("altitude_minimale_toit"))
    toit_max = _num(props.get("altitude_maximale_toit"))
    lo, hi = HEIGHT_RANGE
    height = toit_max - sol if (sol is not None and toit_max is not None) else None
    if height is None or not (lo <= height <= hi):
        h = _num(props.get("hauteur"), zero_missing=True)
        height = h if h is not None and lo <= h <= hi else None
    if height is None:
        return None
    roof = None
    if toit_min is not None and toit_max is not None:
        r = toit_max - toit_min
        if MIN_ROOF < r < height:
            roof = r
    floors = _num(props.get("nombre_d_etages"), zero_missing=True)
    if floors is not None and not (1.0 <= floors <= 150.0):
        floors = None
    wall = material(props.get("materiaux_des_murs"), WALL_CODES, WALL_RANK)
    roof_mat = material(props.get("materiaux_de_la_toiture"), ROOF_CODES, ROOF_RANK)
    return height, roof, floors, wall, roof_mat


def _num(v, zero_missing=False):
    """A finite float from a BD TOPO field, or None. Its unknowns are null and "", plus 0 for the fields where
    zero is not a value ("hauteur", "nombre_d_etages"); an altitude of 0 m is a real altitude."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f == Z_NODATA or (zero_missing and f == 0.0):
        return None
    return f


def _ring_area_m2(ring):
    """The ring's area in m2, on a local equal-area approximation (a few m2 out on a building: the join only
    compares it with another footprint's)."""
    lat0 = float(ring[:, 1].mean())
    kx = 111320.0 * np.cos(np.radians(lat0))
    x, y = ring[:, 0] * kx, ring[:, 1] * 110540.0
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


class _Grid:
    """The cell's BD TOPO footprints bucketed by a ~25 m lon/lat grid: a point tests the few rings of its own
    bucket instead of all 5800 of a dense cell."""

    STEP = 0.00035              # deg, ~25-39 m: a building falls in one or two buckets

    def __init__(self, records):
        self.records = records
        self.boxes = np.array([[r[0][:, 0].min(), r[0][:, 1].min(), r[0][:, 0].max(), r[0][:, 1].max()]
                               for r in records], np.float64) if records else np.zeros((0, 4))
        self.areas = np.array([_ring_area_m2(r[0]) for r in records], np.float64)
        self.buckets = {}
        for i, (w, s, e, n) in enumerate(self.boxes):
            for gx in range(int(np.floor(w / self.STEP)), int(np.floor(e / self.STEP)) + 1):
                for gy in range(int(np.floor(s / self.STEP)), int(np.floor(n / self.STEP)) + 1):
                    self.buckets.setdefault((gx, gy), []).append(i)

    def find(self, lon, lat, area):
        """The values of the footprint that holds (lon, lat) and is within AREA_FACTOR of area, or None."""
        cand = self.buckets.get((int(np.floor(lon / self.STEP)), int(np.floor(lat / self.STEP))))
        if not cand:
            return None
        pt = np.array([[lon, lat]], np.float64)
        for i in cand:
            w, s, e, n = self.boxes[i]
            if not (w <= lon <= e and s <= lat <= n):
                continue
            a = self.areas[i]
            if not (area > 0.0 and a > 0.0 and 1.0 / AREA_FACTOR <= a / area <= AREA_FACTOR):
                continue
            if bool(polygon.contains_points(self.records[i][0], pt)[0]):
                return self.records[i][1]
        return None


def join(feats, records):
    """Give every building of feats that has no height (or no facade material) the values of the BD TOPO
    footprint its own footprint centre falls in.

    Returns (heights filled in, facades named); feats is changed in place."""
    n = len(feats)
    if not n or not records:
        return 0, 0
    grid = _Grid(records)
    height = _column(feats, "height", n)
    roof = _column(feats, "roof_height", n)
    floors = _column(feats, "num_floors", n)
    src = _text(feats, "height_src", n)
    facade = _text(feats, "facade_material", n)
    roof_mat = _text(feats, "roof_material", n)
    filled, dressed = 0, 0
    for i in range(n):
        ring = _outer(feats, i)
        if ring is None:
            continue
        known = height[i] is not None          # Overture or OSM already knows this one's height
        if known and facade[i] and roof_mat[i]:
            continue                            # nothing left to give it
        vals = grid.find(float(ring[:, 0].mean()), float(ring[:, 1].mean()), _ring_area_m2(ring))
        if vals is None:
            continue
        h, rh, nf, wall, rmat = vals
        if not known:
            height[i] = h
            if rh is not None and roof[i] is None:
                roof[i] = rh
            if nf is not None and floors[i] is None:
                floors[i] = nf
            src[i] = SOURCE
            filled += 1
        # the facade and the roof skin: only where the mapped data has none (buildings.plan() reads
        # facade_material already, so a brick town stops coming out as plaster)
        if wall and not facade[i]:
            facade[i] = wall
            dressed += 1
        if rmat and not roof_mat[i]:
            roof_mat[i] = rmat
    if filled:
        feats.attrs["height"] = _floats(height)
        feats.attrs["roof_height"] = _floats(roof)
        feats.attrs["num_floors"] = _floats(floors)
        feats.attrs["height_src"] = np.array([v or "" for v in src])
    if dressed:
        feats.attrs["facade_material"] = np.array([v or "" for v in facade])
        feats.attrs["roof_material"] = np.array([v or "" for v in roof_mat])
    return filled, dressed


def _outer(feats, i):
    """The first outer ring of building i as a (k, 2) lon/lat array, or None."""
    parts = feats.parts(i)
    for rings in parts:
        if rings is not None and len(rings) and len(rings[0]) >= 4:
            return np.asarray(rings[0], np.float64)
    return None


def _column(feats, name, n):
    """A per-feature list of the attribute, None where it is absent or NaN (the arrays hold NaN for 'unknown')."""
    a = feats.attrs.get(name)
    if a is None:
        return [None] * n
    a = np.asarray(a)
    out = []
    for v in a[:n]:
        f = _f(v)
        out.append(None if f is None or not np.isfinite(f) else f)
    return out + [None] * (n - len(out))


def _text(feats, name, n):
    """A per-feature list of a text attribute, "" where it is absent (Overture writes "" for null)."""
    a = feats.attrs.get(name)
    if a is None:
        return [""] * n
    out = [("" if v is None else str(v)) for v in np.asarray(a)[:n]]
    return out + [""] * (n - len(out))


def _floats(col):
    return np.array([np.nan if v is None else float(v) for v in col], np.float64)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
