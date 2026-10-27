"""roofs: roof shapes from the IGN LiDAR HD height raster.

The fake service answers synthetic height rasters whose shapes are known, so the classifier is checked against
roofs whose truth is not in doubt. The thresholds themselves come from the live service over Bailleul on
2026-10-26 (cell z14/8316/5504); the live service is never contacted here.
"""
import io
import sys
import unittest
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mapcells  # noqa: E402
import mapgeom  # noqa: E402
import roofs  # noqa: E402

BAILLEUL = mapcells.CellKey(14, 8316, 5504)
BOUNDS = BAILLEUL.bounds()


def tif(a):
    """A float GeoTIFF of the array, as the service returns one."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(a, np.float32), mode="F").save(buf, format="TIFF")
    return buf.getvalue()


class Res:
    def __init__(self, body):
        self.body, self.headers = body, {"content-type": "image/geotiff"}


class FakeWms:
    def __init__(self, a):
        self.a, self.urls = a, []

    def get(self, url, kind="demand", headers=None):
        self.urls.append(url)
        return Res(tif(self.a))


def ring_of(bounds, fx0, fy0, fx1, fy1):
    """A rectangle given as fractions of the cell, as a lon/lat ring."""
    w, s, e, n = bounds
    lo0, lo1 = w + (e - w) * fx0, w + (e - w) * fx1
    la0, la1 = s + (n - s) * fy0, s + (n - s) * fy1
    return np.array([[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1]], np.float64)


def buildings(rings, shapes=None):
    coords, coord_off = [], [0]
    for r in rings:
        coords.extend(map(tuple, np.asarray(r, np.float64)))
        coord_off.append(len(coords))
    k = len(rings)
    attrs = {}
    if shapes is not None:
        attrs["roof_shape"] = np.array(shapes)
    return mapgeom.Features(layer="buildings", ids=np.array([f"b{i:03d}" for i in range(k)]),
                            kind=np.full(k, mapgeom.POLYGON, np.uint8),
                            part_off=np.arange(k + 1, dtype=np.int64), ring_off=np.arange(k + 1, dtype=np.int64),
                            coord_off=np.array(coord_off, np.int64), lonlat=np.array(coords, np.float64),
                            attrs=attrs)


def canvas(px=400):
    return np.zeros((px, px), np.float32)


def put_flat(a, y0, y1, x0, x1, h):
    a[y0:y1, x0:x1] = h


def put_gable(a, y0, y1, x0, x1, eave, ridge, along_x=True):
    """A tent: the ridge runs along x (a row) or along y (a column)."""
    if along_x:
        k = np.arange(y1 - y0)
        prof = eave + (ridge - eave) * (1 - np.abs(k - (len(k) - 1) / 2) / ((len(k) - 1) / 2))
        a[y0:y1, x0:x1] = prof[:, None]
    else:
        k = np.arange(x1 - x0)
        prof = eave + (ridge - eave) * (1 - np.abs(k - (len(k) - 1) / 2) / ((len(k) - 1) / 2))
        a[y0:y1, x0:x1] = prof[None, :]


def put_hip(a, y0, y1, x0, x1, eave, ridge):
    """A pyramid-ish hip: falls away on all four sides."""
    yy = np.arange(y1 - y0)[:, None]
    xx = np.arange(x1 - x0)[None, :]
    fy = 1 - np.abs(yy - (y1 - y0 - 1) / 2) / ((y1 - y0 - 1) / 2)
    fx = 1 - np.abs(xx - (x1 - x0 - 1) / 2) / ((x1 - x0 - 1) / 2)
    a[y0:y1, x0:x1] = eave + (ridge - eave) * np.minimum(fy, fx)


class RequestTest(unittest.TestCase):
    def test_bbox_is_lat_lon_unlike_the_wfs(self):
        """WMS 1.3.0 with EPSG:4326 takes lat,lon; bdtopo's WFS takes lon,lat. Copying one into the other would
        ask for a cell somewhere off the coast of Somalia."""
        w, s, e, n = BOUNDS
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(roofs.map_url(BOUNDS, 512, 512)).query)
        parts = [float(x) for x in q["BBOX"][0].split(",")]
        self.assertEqual(len(parts), 4)
        self.assertAlmostEqual(parts[0], s, places=5)
        self.assertAlmostEqual(parts[1], w, places=5)
        self.assertAlmostEqual(parts[2], n, places=5)
        self.assertAlmostEqual(parts[3], e, places=5)
        self.assertEqual(q["FORMAT"], ["image/geotiff"])
        self.assertEqual(q["LAYERS"], [roofs.LAYER])

    def test_size_follows_the_resolution_and_the_service_limit(self):
        w, h = roofs._size(BOUNDS)
        self.assertLessEqual(max(w, h), roofs.MAX_PX)
        west, south, east, north = BOUNDS
        metres = (east - west) * 111320 * np.cos(np.radians(50.73))
        self.assertAlmostEqual(metres / w, roofs.METRES_PER_PX, delta=0.05)
        # a whole country's bbox would be clamped rather than asked for at 1 m
        big = roofs._size((-5.2, 41.3, 9.6, 51.1))
        self.assertLessEqual(max(big), roofs.MAX_PX)

    def test_no_request_leaves_france(self):
        for cell in (mapcells.CellKey(14, 4000, 6000), mapcells.CellKey(14, 12000, 7000)):
            if roofs.covers(cell.bounds()):
                continue
            wms = FakeWms(canvas(64))
            self.assertIsNone(roofs.fetch(wms, cell.bounds()))
            self.assertEqual(wms.urls, [], f"{cell} is outside France: no request")

    def test_turned_off(self):
        import os
        wms = FakeWms(canvas(64))
        old = os.environ.get("BNE_LIDAR")
        os.environ["BNE_LIDAR"] = "off"
        try:
            self.assertIsNone(roofs.fetch(wms, BOUNDS))
            self.assertEqual(wms.urls, [])
        finally:
            os.environ.pop("BNE_LIDAR") if old is None else os.environ.__setitem__("BNE_LIDAR", old)

    def test_a_float_geotiff_comes_back_as_an_array(self):
        a = canvas(64)
        a[10:20, 10:20] = 7.5
        got = roofs.fetch(FakeWms(a), BOUNDS)
        self.assertEqual(got.shape, (64, 64))
        self.assertAlmostEqual(float(got[15, 15]), 7.5, places=3)

    def test_a_body_that_is_not_a_raster_raises(self):
        class Bad:
            body, headers = b"<html>maintenance</html>", {}

        class S:
            def get(self, url, kind="demand", headers=None):
                return Bad()
        with self.assertRaises(Exception):
            roofs.fetch(S(), BOUNDS)


class ShapeTest(unittest.TestCase):
    """The classifier against roofs whose shape is known by construction."""

    def shape(self, draw, fx0=0.30, fy0=0.30, fx1=0.50, fy1=0.42, px=400):
        a = canvas(px)
        # the footprint's pixel box (row 0 = north, so the y fractions flip)
        x0, x1 = int(fx0 * px), int(fx1 * px)
        y0, y1 = int((1 - fy1) * px), int((1 - fy0) * px)
        draw(a, y0, y1, x0, x1)
        ring = ring_of(BOUNDS, fx0, fy0, fx1, fy1)
        hgt, qx, qy = roofs._pixels(a, ring, BOUNDS)
        self.assertIsNotNone(hgt, "the footprint should cover enough pixels")
        return roofs.shape_of(hgt, qx, qy, ring, BOUNDS, px, px)

    def test_flat(self):
        s, bearing = self.shape(lambda a, y0, y1, x0, x1: put_flat(a, y0, y1, x0, x1, 6.0))
        self.assertEqual(s, "flat")
        self.assertIsNone(bearing)

    def test_a_nearly_flat_roof_is_flat(self):
        def draw(a, y0, y1, x0, x1):
            put_gable(a, y0, y1, x0, x1, 6.0, 6.0 + roofs.FLAT_RELIEF * 0.5)
        self.assertEqual(self.shape(draw)[0], "flat")

    def test_gable_along_the_long_axis(self):
        """A wide shallow building whose ridge runs east-west: the ridge is a line along the long axis."""
        def draw(a, y0, y1, x0, x1):
            put_gable(a, y0, y1, x0, x1, 3.0, 9.0, along_x=True)
        s, bearing = self.shape(draw, 0.30, 0.30, 0.52, 0.40)
        self.assertEqual(s, "gabled")
        self.assertIsNotNone(bearing)
        self.assertTrue(0.0 <= bearing < 180.0)

    def test_gable_across_the_long_axis(self):
        """The same building with its ridge the other way: still a gable, and the bearing differs by ~90 deg from
        the case above (this is the case plan()'s long-axis guess gets wrong)."""
        def draw(a, y0, y1, x0, x1):
            put_gable(a, y0, y1, x0, x1, 3.0, 9.0, along_x=False)
        s_along, b_along = self.shape(lambda a, y0, y1, x0, x1: put_gable(a, y0, y1, x0, x1, 3.0, 9.0, True),
                                      0.30, 0.30, 0.52, 0.40)
        s_across, b_across = self.shape(draw, 0.30, 0.30, 0.52, 0.40)
        self.assertEqual((s_along, s_across), ("gabled", "gabled"))
        d = abs(b_along - b_across) % 180.0
        self.assertGreater(min(d, 180.0 - d), 60.0, "the two ridges are not the same direction")

    def test_hip(self):
        def draw(a, y0, y1, x0, x1):
            put_hip(a, y0, y1, x0, x1, 3.0, 9.0)
        s, bearing = self.shape(draw, 0.30, 0.30, 0.46, 0.42)
        self.assertIn(s, ("hipped", "pyramidal"))
        self.assertIsNone(bearing, "a hip has no ridge line to point along")

    def test_no_building_in_the_flight(self):
        """The LiDAR is older than the house: bare ground, so plan() keeps its guess rather than getting 'flat'."""
        s, _b = self.shape(lambda a, y0, y1, x0, x1: put_flat(a, y0, y1, x0, x1, 0.3))
        self.assertIsNone(s)

    def test_a_footprint_too_small_to_read(self):
        a = canvas(400)
        put_gable(a, 100, 104, 100, 104, 3.0, 8.0)
        tiny = ring_of(BOUNDS, 0.30, 0.30, 0.3004, 0.3004)
        self.assertEqual(roofs._pixels(a, tiny, BOUNDS)[0], None)


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.px = 400
        self.a = canvas(self.px)
        # one gable and one flat roof, side by side
        put_gable(self.a, int(0.58 * self.px), int(0.70 * self.px), int(0.30 * self.px), int(0.50 * self.px),
                  3.0, 9.0, along_x=True)
        put_flat(self.a, int(0.58 * self.px), int(0.70 * self.px), int(0.55 * self.px), int(0.72 * self.px), 6.0)
        self.gable = ring_of(BOUNDS, 0.30, 0.30, 0.50, 0.42)
        self.flat = ring_of(BOUNDS, 0.55, 0.30, 0.72, 0.42)

    def test_names_the_shapes(self):
        f = buildings([self.gable, self.flat])
        self.assertEqual(roofs.join(f, self.a, BOUNDS), 2)
        self.assertEqual(str(f.attrs["roof_shape"][0]), "gabled")
        self.assertEqual(str(f.attrs["roof_shape"][1]), "flat")

    def test_a_mapped_shape_is_kept(self):
        f = buildings([self.gable, self.flat], shapes=["dome", ""])
        self.assertEqual(roofs.join(f, self.a, BOUNDS), 1)
        self.assertEqual(str(f.attrs["roof_shape"][0]), "dome", "Overture's own shape is the more precise")
        self.assertEqual(str(f.attrs["roof_shape"][1]), "flat")

    def test_no_ridge_direction_is_written(self):
        """It is measured but nothing reads it yet (the mesh lays a gable on the footprint's long axis), and a
        field written for nobody is how roof_direction got into the download in the first place."""
        f = buildings([self.gable])
        roofs.join(f, self.a, BOUNDS)
        self.assertNotIn("roof_direction", f.attrs)

    def test_nothing_to_join(self):
        self.assertEqual(roofs.join(buildings([self.gable]), None, BOUNDS), 0)
        self.assertEqual(roofs.join(buildings([]), self.a, BOUNDS), 0)

    def test_the_shapes_are_words_plan_accepts(self):
        """buildings.plan() takes a mapped roof_shape straight through for these three."""
        import buildings as bld
        f = buildings([self.gable, self.flat])
        roofs.join(f, self.a, BOUNDS)
        ring = np.array([(0, 0), (12, 0), (12, 8), (0, 8)], np.float64)
        for s in set(str(v) for v in f.attrs["roof_shape"]) | {"hipped", "pyramidal"}:
            spec = bld.plan(ring, {"class": "house", "height": 9.0, "roof_shape": s}, lat=50.73, lon=2.735)
            self.assertIn(spec.roof, ("flat", "gabled", "hipped"), s)


class ProvenanceTest(unittest.TestCase):
    def test_licence_is_named(self):
        self.assertEqual(roofs.PROVENANCE["id"], "ign-lidarhd")
        self.assertIn("Licence Ouverte 2.0", roofs.PROVENANCE["licence"])
        self.assertIn("IGN LiDAR HD", roofs.CREDIT_LINE)
        import mapdata
        self.assertEqual(mapdata.CREDIT_LINES["ign-lidarhd"], roofs.CREDIT_LINE)


if __name__ == "__main__":
    unittest.main()
