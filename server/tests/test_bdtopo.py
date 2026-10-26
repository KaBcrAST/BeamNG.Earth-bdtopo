"""bdtopo: the IGN BD TOPO heights joined onto France's Overture buildings.

Every fact the fake service reproduces here was read off the live Geoplateforme WFS over Lille on 2026-10-26
(cell z14/8331/5511): lon,lat with a mandatory CRS token in the BBOX, MultiPolygon geometry with three values a
position, a 5000-feature server cap, and "hauteur" 0 for unknown. The live service is never contacted.
"""
import gzip
import json
import sys
import unittest
import urllib.parse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bdtopo  # noqa: E402
import mapcells  # noqa: E402
import mapgeom  # noqa: E402

LILLE = mapcells.CellKey(14, 8331, 5511)        # the cell the live checks were run against


def ring(lon, lat, w_deg, h_deg):
    """A closed rectangle, lon/lat, as the service sends it (three values a position, the third an altitude)."""
    return [[lon, lat, -1000.0], [lon + w_deg, lat, -1000.0], [lon + w_deg, lat + h_deg, -1000.0],
            [lon, lat + h_deg, -1000.0], [lon, lat, -1000.0]]


def feature(lon, lat, w_deg=2e-4, h_deg=1.5e-4, **props):
    full = {"hauteur": 0, "nombre_d_etages": None, "altitude_minimale_sol": None,
            "altitude_minimale_toit": None, "altitude_maximale_toit": None}
    full.update(props)
    return {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[ring(lon, lat, w_deg, h_deg)]]},
            "properties": full}


class Res:
    def __init__(self, doc, gz=False):
        raw = json.dumps(doc).encode()
        self.headers = {}
        if gz:
            raw, self.headers = gzip.compress(raw), {"content-encoding": "gzip"}
        self.body = raw


class FakeWfs:
    """A GetFeature service that answers from a list of features, capped at PAGE like the real one."""

    def __init__(self, feats, gz=False, cap=bdtopo.PAGE):
        self.feats, self.gz, self.cap, self.urls = feats, gz, cap, []

    def get(self, url, kind="demand", headers=None):
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        # the live service answers an EMPTY collection, with HTTP 200, when the BBOX carries no CRS token
        if len(q["BBOX"][0].split(",")) != 5:
            return Res({"type": "FeatureCollection", "features": [], "numberMatched": 0}, self.gz)
        start = int(q.get("STARTINDEX", ["0"])[0])
        count = min(int(q.get("COUNT", [self.cap])[0]), self.cap)
        page = self.feats[start:start + count]
        return Res({"type": "FeatureCollection", "features": page, "numberMatched": len(self.feats)}, self.gz)


def buildings(rings, heights=None):
    """An Overture-like buildings layer from (k, 2) lon/lat rings."""
    coords, coord_off = [], [0]
    for r in rings:
        coords.extend(map(tuple, np.asarray(r, np.float64)))
        coord_off.append(len(coords))
    k = len(rings)
    attrs = {"height": np.full(k, np.nan) if heights is None else np.asarray(heights, np.float64)}
    return mapgeom.Features(layer="buildings", ids=np.array([f"b{i:03d}" for i in range(k)]),
                            kind=np.full(k, mapgeom.POLYGON, np.uint8),
                            part_off=np.arange(k + 1, dtype=np.int64), ring_off=np.arange(k + 1, dtype=np.int64),
                            coord_off=np.array(coord_off, np.int64), lonlat=np.array(coords, np.float64),
                            attrs=attrs)


def box(lon, lat, w_deg, h_deg, shrink=1.0):
    """A rectangle as an Overture footprint (open ring), optionally shrunk about its centre."""
    r = np.array([[lon, lat], [lon + w_deg, lat], [lon + w_deg, lat + h_deg], [lon, lat + h_deg]], np.float64)
    c = r.mean(axis=0)
    return c + (r - c) * shrink


class ValuesTest(unittest.TestCase):
    """What one BD TOPO record gives, including the service's ways of saying "unknown"."""

    def test_height_from_altitudes(self):
        h, roof, floors, _w, _r = bdtopo.values({"altitude_minimale_sol": 20.0, "altitude_minimale_toit": 38.0,
                                         "altitude_maximale_toit": 42.0, "nombre_d_etages": 5, "hauteur": 19.0})
        self.assertAlmostEqual(h, 22.0)         # the roof's top above the lowest ground, not "hauteur"
        self.assertAlmostEqual(roof, 4.0)
        self.assertEqual(floors, 5.0)

    def test_hauteur_when_no_altitudes(self):
        h, roof, floors, _w, _r = bdtopo.values({"hauteur": 12.5})
        self.assertAlmostEqual(h, 12.5)
        self.assertIsNone(roof)
        self.assertIsNone(floors)

    def test_zero_hauteur_is_unknown(self):
        """The live service sends 0, not null, for a building whose height it does not have."""
        self.assertIsNone(bdtopo.values({"hauteur": 0}))
        self.assertIsNone(bdtopo.values({"hauteur": 0, "nombre_d_etages": 0}))

    def test_altitude_zero_is_a_real_altitude(self):
        """0 means "unknown" for hauteur only: a building at sea level has altitude_minimale_sol 0."""
        h, _roof, _f, _w, _r = bdtopo.values({"altitude_minimale_sol": 0.0, "altitude_maximale_toit": 9.0})
        self.assertAlmostEqual(h, 9.0)

    def test_flat_roof_left_to_plan(self):
        _h, roof, _f, _w, _r = bdtopo.values({"altitude_minimale_sol": 0.0, "altitude_minimale_toit": 11.8,
                                      "altitude_maximale_toit": 12.0})
        self.assertIsNone(roof, "0.2 m is flat: buildings.plan() decides the shape")

    def test_nonsense_is_dropped(self):
        self.assertIsNone(bdtopo.values({"hauteur": 1200.0}))
        self.assertIsNone(bdtopo.values({"hauteur": 0.3}))
        self.assertIsNone(bdtopo.values({"altitude_minimale_sol": 90.0, "altitude_maximale_toit": 20.0}))
        self.assertIsNone(bdtopo.values({"hauteur": "", "nombre_d_etages": ""}))
        self.assertIsNone(bdtopo.values({"hauteur": None}))

    def test_floors_of_a_record_without_height(self):
        """A floor count alone is not joined: height is what the join is for."""
        self.assertIsNone(bdtopo.values({"nombre_d_etages": 4}))


class GeometryTest(unittest.TestCase):
    def test_three_values_a_position(self):
        """The live service sends (lon, lat, altitude); a two-value unpack would raise on every feature."""
        g = {"type": "MultiPolygon", "coordinates": [[ring(3.06, 50.63, 2e-4, 2e-4)]]}
        rings = bdtopo._rings(g, LILLE.bounds())
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].shape[1], 2)
        self.assertAlmostEqual(rings[0][0, 0], 3.06)
        self.assertAlmostEqual(rings[0][0, 1], 50.63)

    def test_plain_polygon_and_multipart(self):
        one = {"type": "Polygon", "coordinates": [ring(3.06, 50.63, 2e-4, 2e-4)]}
        self.assertEqual(len(bdtopo._rings(one, LILLE.bounds())), 1)
        two = {"type": "MultiPolygon", "coordinates": [[ring(3.06, 50.63, 2e-4, 2e-4)],
                                                       [ring(3.07, 50.634, 2e-4, 2e-4)]]}
        self.assertEqual(len(bdtopo._rings(two, LILLE.bounds())), 2)

    def test_swapped_axes_are_put_back(self):
        """The live service answers lon,lat; a change there must not misplace every building in France."""
        swapped = {"type": "MultiPolygon",
                   "coordinates": [[[[50.63, 3.06, -1000.0], [50.6301, 3.06, -1000.0],
                                     [50.6301, 3.0602, -1000.0], [50.63, 3.0602, -1000.0], [50.63, 3.06, -1000.0]]]]}
        r = bdtopo._rings(swapped, LILLE.bounds())[0]
        self.assertAlmostEqual(r[0, 0], 3.06, places=3)
        self.assertAlmostEqual(r[0, 1], 50.63, places=3)

    def test_unusable_geometry_is_skipped(self):
        self.assertEqual(bdtopo._rings({"type": "Point", "coordinates": [3.06, 50.63]}, LILLE.bounds()), [])
        self.assertEqual(bdtopo._rings({}, LILLE.bounds()), [])
        short = {"type": "MultiPolygon", "coordinates": [[[[3.06, 50.63, 0.0], [3.0601, 50.63, 0.0]]]]}
        self.assertEqual(bdtopo._rings(short, LILLE.bounds()), [])


class RequestTest(unittest.TestCase):
    def test_bbox_is_lon_lat_with_a_crs_token(self):
        """Measured on the live service: lon first, and without the CRS token it answers no features at all."""
        w, s, e, n = LILLE.bounds()
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(bdtopo.page_url(LILLE.bounds())).query)
        parts = q["BBOX"][0].split(",")
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[4], "EPSG:4326")
        self.assertAlmostEqual(float(parts[0]), w, places=5)
        self.assertAlmostEqual(float(parts[1]), s, places=5)
        self.assertAlmostEqual(float(parts[2]), e, places=5)
        self.assertAlmostEqual(float(parts[3]), n, places=5)
        self.assertEqual(q["TYPENAMES"], [bdtopo.LAYER])
        self.assertEqual(q["COUNT"], [str(bdtopo.PAGE)])
        self.assertIn("geometrie", q["PROPERTYNAME"][0])

    def test_no_request_leaves_france(self):
        for cell in (mapcells.CellKey(14, 8192, 5000),         # Greenwich / the North Sea is partly in the box
                     mapcells.CellKey(14, 4000, 6000),         # the Atlantic, west of Ireland
                     mapcells.CellKey(14, 12000, 7000)):       # central Asia
            if bdtopo.covers(cell.bounds()):
                continue
            wfs = FakeWfs([feature(3.06, 50.63, hauteur=10.0)])
            self.assertEqual(bdtopo.fetch(wfs, cell.bounds()), [])
            self.assertEqual(wfs.urls, [], f"{cell} is outside France: no request")

    def test_bordeaux_and_corsica_are_inside(self):
        self.assertTrue(bdtopo.covers(mapcells.CellKey(14, 8104, 5687).bounds()))       # Bordeaux
        self.assertTrue(bdtopo.covers((8.73, 41.91, 8.75, 41.93)))                      # Ajaccio
        self.assertFalse(bdtopo.covers((-74.1, 40.6, -74.0, 40.7)))                     # New York

    def test_turned_off(self):
        import os
        wfs = FakeWfs([feature(3.06, 50.63, hauteur=10.0)])
        old = os.environ.get("BNE_BDTOPO")
        os.environ["BNE_BDTOPO"] = "off"
        try:
            self.assertEqual(bdtopo.fetch(wfs, LILLE.bounds()), [])
            self.assertEqual(wfs.urls, [])
        finally:
            os.environ.pop("BNE_BDTOPO") if old is None else os.environ.__setitem__("BNE_BDTOPO", old)


class PagingTest(unittest.TestCase):
    def test_pages_until_the_total_is_reached(self):
        """A dense French cell holds more than PAGE buildings (central Lille: 5799), and the service caps a page
        at PAGE whatever COUNT asks for."""
        lon0, lat0, _e, _n = LILLE.bounds()
        feats = [feature(lon0 + 1e-4 + (i % 70) * 2.5e-4, lat0 + 1e-4 + (i // 70) * 2e-4, hauteur=9.0)
                 for i in range(bdtopo.PAGE + 123)]
        wfs = FakeWfs(feats)
        got = bdtopo.fetch(wfs, LILLE.bounds())
        self.assertEqual(len(got), bdtopo.PAGE + 123)
        self.assertEqual(len(wfs.urls), 2)
        starts = [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query)["STARTINDEX"][0] for u in wfs.urls]
        self.assertEqual(starts, ["0", str(bdtopo.PAGE)])

    def test_one_page_is_one_request(self):
        wfs = FakeWfs([feature(3.06, 50.63, hauteur=10.0)])
        self.assertEqual(len(bdtopo.fetch(wfs, LILLE.bounds())), 1)
        self.assertEqual(len(wfs.urls), 1, "a short page is the last page")

    def test_gzip_body(self):
        """netfetch sends no Accept-Encoding of its own, so bdtopo asks and unzips the answer itself."""
        wfs = FakeWfs([feature(3.06, 50.63, hauteur=10.0)], gz=True)
        self.assertEqual(len(bdtopo.fetch(wfs, LILLE.bounds())), 1)


class JoinTest(unittest.TestCase):
    """The join itself: only buildings without a height, only footprints that are the same building."""

    def setUp(self):
        self.lon, self.lat = 3.06, 50.63
        self.w, self.h = 2.4e-4, 1.6e-4

    def records(self, **props):
        wfs = FakeWfs([feature(self.lon, self.lat, self.w, self.h, **props)])
        return bdtopo.fetch(wfs, LILLE.bounds())

    def test_fills_a_building_without_a_height(self):
        recs = self.records(altitude_minimale_sol=20.0, altitude_minimale_toit=35.0,
                            altitude_maximale_toit=38.0, nombre_d_etages=4)
        f = buildings([box(self.lon, self.lat, self.w, self.h, shrink=0.94)])
        self.assertEqual(bdtopo.join(f, recs)[0], 1)
        self.assertAlmostEqual(float(f.attrs["height"][0]), 18.0)
        self.assertAlmostEqual(float(f.attrs["roof_height"][0]), 3.0)
        self.assertEqual(float(f.attrs["num_floors"][0]), 4.0)
        self.assertEqual(str(f.attrs["height_src"][0]), bdtopo.SOURCE)

    def test_keeps_a_height_overture_already_has(self):
        recs = self.records(hauteur=30.0)
        f = buildings([box(self.lon, self.lat, self.w, self.h, shrink=0.94)], heights=[7.5])
        self.assertEqual(bdtopo.join(f, recs)[0], 0)
        self.assertAlmostEqual(float(f.attrs["height"][0]), 7.5)
        self.assertNotIn("height_src", f.attrs)

    def test_a_footprint_of_another_size_is_not_the_same_building(self):
        """One source has the whole block, the other its separate houses: AREA_FACTOR drops the pair."""
        recs = self.records(hauteur=30.0)
        f = buildings([box(self.lon + self.w * 0.3, self.lat + self.h * 0.3, self.w * 0.2, self.h * 0.2)])
        self.assertEqual(bdtopo.join(f, recs)[0], 0)
        self.assertFalse(np.isfinite(f.attrs["height"][0]))

    def test_a_building_outside_every_footprint(self):
        recs = self.records(hauteur=30.0)
        f = buildings([box(self.lon + 0.002, self.lat + 0.002, self.w, self.h)])
        self.assertEqual(bdtopo.join(f, recs)[0], 0)
        self.assertFalse(np.isfinite(f.attrs["height"][0]))

    def test_u_shaped_footprint_whose_centre_falls_outside(self):
        """A U: its centroid is in the courtyard, so point-in-polygon finds nothing and the height stays guessed
        rather than taking a neighbour's."""
        u = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [3, 3], [3, 4], [0, 4]]
        scale = np.array([self.w / 3.0, self.h / 4.0])
        poly = np.array(u, np.float64) * scale + np.array([self.lon, self.lat])
        geo = {"type": "MultiPolygon",
               "coordinates": [[[[float(x), float(y), -1000.0] for x, y in np.vstack([poly, poly[:1]])]]]}
        wfs = FakeWfs([{"type": "Feature", "geometry": geo, "properties": {"hauteur": 24.0}}])
        recs = bdtopo.fetch(wfs, LILLE.bounds())
        self.assertEqual(len(recs), 1)
        f = buildings([poly])
        bdtopo.join(f, recs)        # whatever it decides, it must not raise and must not invent a height
        if np.isfinite(f.attrs["height"][0]):
            self.assertAlmostEqual(float(f.attrs["height"][0]), 24.0)

    def test_many_buildings_take_the_right_footprints(self):
        lon0, lat0, _e, _n = LILLE.bounds()
        feats, boxes, want = [], [], []
        for i in range(60):
            lo = lon0 + 2e-4 + (i % 10) * 4e-4
            la = lat0 + 2e-4 + (i // 10) * 3e-4
            feats.append(feature(lo, la, self.w, self.h, hauteur=float(6 + i % 17)))
            boxes.append(box(lo, la, self.w, self.h, shrink=0.93))
            want.append(float(6 + i % 17))
        recs = bdtopo.fetch(FakeWfs(feats), LILLE.bounds())
        f = buildings(boxes)
        self.assertEqual(bdtopo.join(f, recs)[0], 60)
        np.testing.assert_allclose(f.attrs["height"], want)

    def test_nothing_to_join(self):
        f = buildings([box(self.lon, self.lat, self.w, self.h)])
        self.assertEqual(bdtopo.join(f, []), (0, 0))
        self.assertEqual(bdtopo.join(buildings([]), self.records(hauteur=12.0)), (0, 0))


class MaterialTest(unittest.TestCase):
    """The two-character DGFiP codes (dmatgm / dmatto) behind materiaux_des_murs and materiaux_de_la_toiture,
    turned into the OpenStreetMap words buildings.plan() already knows."""

    def wall(self, code):
        return bdtopo.material(code, bdtopo.WALL_CODES, bdtopo.WALL_RANK)

    def roof(self, code):
        return bdtopo.material(code, bdtopo.ROOF_CODES, bdtopo.ROOF_RANK)

    def test_one_material_either_way_round(self):
        """"40" and "04" both mean brick alone: the field holds up to two materials, in no fixed order."""
        self.assertEqual(self.wall("40"), "brick")
        self.assertEqual(self.wall("04"), "brick")
        self.assertEqual(self.wall("44"), "brick")

    def test_the_lille_codes(self):
        """The codes the live service actually sent for central Lille, by frequency: 40 is 43 % of the cell, and
        Lille is a brick city."""
        self.assertEqual(self.wall("40"), "brick")          # BRIQUES
        self.assertEqual(self.wall("10"), "stone")          # PIERRE
        self.assertEqual(self.wall("20"), "stone")          # MEULIERE
        self.assertEqual(self.wall("30"), "concrete")       # BETON
        self.assertEqual(self.wall("50"), "concrete_block")  # AGGLOMERE
        self.assertEqual(self.wall("46"), "brick")          # BOIS - BRIQUES: the brick is the skin
        self.assertEqual(self.wall("34"), "brick")          # BETON - BRIQUES
        self.assertEqual(self.wall("43"), "brick")          # BETON - BRIQUES (the other way round)
        self.assertEqual(self.wall("41"), "brick")          # BRIQUES - PIERRE
        self.assertEqual(self.wall("14"), "brick")          # BRIQUES - PIERRE (the other way round)
        self.assertEqual(self.roof("10"), "roof_tiles")     # TUILES
        self.assertEqual(self.roof("20"), "slate")          # ARDOISES
        self.assertEqual(self.roof("30"), "metal")          # ZINC ALUMINIUM
        self.assertEqual(self.roof("40"), "concrete")       # BETON
        self.assertEqual(self.roof("13"), "roof_tiles")     # TUILES - ZINC ALUMINIUM
        self.assertEqual(self.roof("12"), "slate")          # ARDOISES - TUILES

    def test_unknown_codes(self):
        """The service's unknowns: null, "", "00" (indetermine) and "90" (autres) say nothing about the look."""
        for code in (None, "", "00", "90", "09", "99", "TL", "  "):
            self.assertIsNone(self.wall(code), code)
        for code in (None, "", "00", "90"):
            self.assertIsNone(self.roof(code), code)

    def test_facade_is_named_only_where_overture_has_none(self):
        lon, lat, w, h = 3.06, 50.63, 2.4e-4, 1.6e-4
        wfs = FakeWfs([feature(lon, lat, w, h, hauteur=11.0, materiaux_des_murs="40",
                               materiaux_de_la_toiture="20")])
        recs = bdtopo.fetch(wfs, LILLE.bounds())
        f = buildings([box(lon, lat, w, h, shrink=0.94)])
        filled, dressed = bdtopo.join(f, recs)
        self.assertEqual((filled, dressed), (1, 1))
        self.assertEqual(str(f.attrs["facade_material"][0]), "brick")
        self.assertEqual(str(f.attrs["roof_material"][0]), "slate")

        # Overture's own facade_material wins
        g = buildings([box(lon, lat, w, h, shrink=0.94)])
        g.attrs["facade_material"] = np.array(["glass"])
        _filled, dressed2 = bdtopo.join(g, recs)
        self.assertEqual(dressed2, 0)
        self.assertEqual(str(g.attrs["facade_material"][0]), "glass")

    def test_a_known_height_still_gets_its_facade(self):
        """A building Overture has a height for but no material: the join still dresses it."""
        lon, lat, w, h = 3.06, 50.63, 2.4e-4, 1.6e-4
        recs = bdtopo.fetch(FakeWfs([feature(lon, lat, w, h, hauteur=11.0, materiaux_des_murs="10")]),
                            LILLE.bounds())
        f = buildings([box(lon, lat, w, h, shrink=0.94)], heights=[8.0])
        filled, dressed = bdtopo.join(f, recs)
        self.assertEqual((filled, dressed), (0, 1))
        self.assertAlmostEqual(float(f.attrs["height"][0]), 8.0, msg="its own height is untouched")
        self.assertEqual(str(f.attrs["facade_material"][0]), "stone")

    def test_words_buildings_py_understands(self):
        """plan() reads facade_material: "brick" and the stone words must be the ones it matches."""
        import buildings as bld
        self.assertIn("stone", bld.STONE_FACADES)
        self.assertIn("brick", bld.MASONRY_MATERIALS)
        for word in bdtopo.WALL_CODES.values():
            self.assertEqual(word, word.lower(), "plan() lower-cases what it compares")


class ProvenanceTest(unittest.TestCase):
    def test_licence_is_named(self):
        """Licence Ouverte 2.0 asks for the source to be named: CREDITS.md and the in-game overlay."""
        self.assertEqual(bdtopo.PROVENANCE["id"], "ign-bdtopo")
        self.assertIn("Licence Ouverte 2.0", bdtopo.PROVENANCE["licence"])
        self.assertIn("IGN BD TOPO", bdtopo.CREDIT_LINE)
        import mapdata
        self.assertEqual(mapdata.CREDIT_LINES["ign-bdtopo"], bdtopo.CREDIT_LINE)


if __name__ == "__main__":
    unittest.main()
