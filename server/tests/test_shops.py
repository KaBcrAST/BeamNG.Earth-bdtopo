"""shops.py: the cell's own places layer put on the buildings that hold them.

Over a Bailleul cell the buildings' Overture class made 42 shopfronts of 1903 facades, while the places
layer of the same cell names 180 businesses; the join finds a building for 74 of them (31 places sit more
than NEAR_M from any footprint, in car parks and on roads, and 75 are trades that never face a street).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import buildings as bld                                             # noqa: E402
import mapgeom                                                      # noqa: E402
import shops                                                        # noqa: E402


def features(layer, geoms, attrs=None):
    coords, coord_off = [], [0]
    for g in geoms:
        coords.extend(map(tuple, np.asarray(g, np.float64).reshape(-1, 2)))
        coord_off.append(len(coords))
    k = len(geoms)
    return mapgeom.Features(layer=layer, ids=np.array([f"{layer[0]}{i:03d}" for i in range(k)]),
                            kind=np.full(k, mapgeom.POLYGON, np.uint8),
                            part_off=np.arange(k + 1, dtype=np.int64), ring_off=np.arange(k + 1, dtype=np.int64),
                            coord_off=np.array(coord_off, np.int64), lonlat=np.array(coords, np.float64),
                            attrs={k2: np.array(v) for k2, v in (attrs or {}).items()})


def square(lon, lat, m=20.0):
    """A footprint of about m metres a side, at (lon, lat)."""
    dx = m / (111320.0 * np.cos(np.radians(lat)) * 2)
    dy = m / 110540.0 / 2
    return np.array([(lon - dx, lat - dy), (lon + dx, lat - dy), (lon + dx, lat + dy), (lon - dx, lat + dy)])


def places(points, cats, names=None, status=None):
    n = len(points)
    a = {"basic_category": cats, "name": names or [f"Chez {i}" for i in range(n)],
         "taxonomy": [""] * n, "operating_status": status or ["open"] * n}
    return features("places", [np.array([p]) for p in points], a)


class CategoryTest(unittest.TestCase):
    """kind_of: 71 categories in one town, a dozen looks. The table is the judgement; this pins its edges."""

    def test_the_trades_a_street_shows(self):
        for cat, want in (("bakery", "bakery"), ("restaurant", "restaurant"), ("bar", "bar"),
                          ("pharmacy_and_drug_store", "pharmacy"), ("personal_or_beauty_service", "beauty"),
                          ("fashion_and_apparel_store", "fashion"), ("financial_service", "bank"),
                          ("hotel", "hotel"), ("automotive_service", "automotive"),
                          ("hardware_home_and_garden_store", "hardware"), ("food_and_beverage_store", "food")):
            self.assertEqual(shops.kind_of(cat), want, cat)

    def test_what_never_faces_a_street_takes_no_front(self):
        for cat in ("religious_organization", "school", "parking", "cemetery", "industrial",
                    "social_or_community_service", ""):
            self.assertIsNone(shops.kind_of(cat), cat)

    def test_an_unknown_retail_category_still_gets_a_plain_front(self):
        self.assertEqual(shops.kind_of("vinyl_record_store"), "shop")
        self.assertEqual(shops.kind_of("whatever_service"), "shop")
        self.assertEqual(shops.kind_of("sometrade", "eat_and_drink > x"), "restaurant")

    def test_a_category_nobody_can_place_is_left_alone(self):
        self.assertIsNone(shops.kind_of("quantum_flux"))

    def test_every_kind_has_a_look_in_the_mesh_builder(self):
        """A kind with no colour would draw an awning the look table cannot resolve."""
        for k in shops.KINDS:
            self.assertIn(k, bld.SHOP_COLOURS, k)
            self.assertIn(f"awning|{k}", bld.LOOKS, k)
            self.assertIn(f"shopboard|{k}", bld.LOOKS, k)

    def test_every_mapped_category_lands_on_a_known_kind(self):
        for cat, k in shops.CATEGORY_KIND.items():
            self.assertIn(k, shops.KINDS, cat)


class JoinTest(unittest.TestCase):
    LON, LAT = 2.735, 50.73

    def feats(self, n=3, step=40.0):
        dx = step / (111320.0 * np.cos(np.radians(self.LAT)))
        return features("buildings", [square(self.LON + i * dx, self.LAT) for i in range(n)]), dx

    def test_a_shop_inside_a_footprint_names_its_building(self):
        f, _dx = self.feats(1)
        named, lost = shops.join(f, [places([(self.LON, self.LAT)], ["bakery"], ["Boulangerie Louise"])])
        self.assertEqual((named, lost), (1, 0))
        self.assertEqual(str(f.attrs["shop_kind"][0]), "bakery")
        self.assertEqual(str(f.attrs["shop_name"][0]), "Boulangerie Louise")

    def test_a_point_on_the_pavement_still_finds_its_building(self):
        """Overture often puts a shop's point in front of its door, outside the walls."""
        f, _dx = self.feats(1)
        off = 14.0 / 110540.0                       # ~14 m north of the centre of a 20 m square: just outside
        named, lost = shops.join(f, [places([(self.LON, self.LAT + off)], ["restaurant"])])
        self.assertEqual((named, lost), (1, 0))
        self.assertEqual(str(f.attrs["shop_kind"][0]), "restaurant")

    def test_a_point_far_from_everything_is_counted_not_forced(self):
        f, _dx = self.feats(1)
        far = (shops.NEAR_M * 6) / 110540.0
        named, lost = shops.join(f, [places([(self.LON, self.LAT + far)], ["restaurant"])])
        self.assertEqual((named, lost), (0, 1), "better no shopfront than one on the wrong building")
        self.assertNotIn("shop_kind", f.attrs)

    def test_a_closed_business_is_not_a_shopfront(self):
        f, _dx = self.feats(1)
        named, _lost = shops.join(f, [places([(self.LON, self.LAT)], ["bakery"], status=["closed"])])
        self.assertEqual(named, 0)

    def test_a_building_with_several_shops_keeps_the_first(self):
        f, _dx = self.feats(1)
        named, _lost = shops.join(f, [places([(self.LON, self.LAT), (self.LON, self.LAT)],
                                             ["bakery", "bar"])])
        self.assertEqual(named, 1, "one front per building: the mesh draws one band")
        self.assertEqual(str(f.attrs["shop_kind"][0]), "bakery")

    def test_each_shop_goes_to_its_own_building(self):
        f, dx = self.feats(3)
        pts = [(self.LON + i * dx, self.LAT) for i in range(3)]
        named, lost = shops.join(f, [places(pts, ["bakery", "pharmacy_and_drug_store", "bar"])])
        self.assertEqual((named, lost), (3, 0))
        self.assertEqual([str(v) for v in f.attrs["shop_kind"]], ["bakery", "pharmacy", "bar"])

    def test_nothing_to_join(self):
        f, _dx = self.feats(1)
        self.assertEqual(shops.join(f, []), (0, 0))
        self.assertEqual(shops.join(f, [None]), (0, 0))
        self.assertEqual(shops.join(features("buildings", []), [places([(0, 0)], ["bakery"])]), (0, 0))


class MeshTest(unittest.TestCase):
    """What the named trade changes in the building that gets built."""

    RING = np.array([(0.0, 0.0), (8.0, 0.0), (8.0, 12.0), (0.0, 12.0)])
    ROAD = [(np.array([(-30.0, -4.0), (40.0, -4.0)]), 3.5, "residential", True)]

    def built(self, attrs):
        spec = bld.plan(self.RING, attrs, lat=50.73, lon=2.735)
        bld.doors([(spec, 0.0, 0.0)], self.ROAD, 0.0, 0.0)
        m = bld.Mesh()
        idx = bld._Looks(spec.seed)
        if spec.sign is not None:
            bld._signs(m, spec, self.RING, 6.0, idx)
            bld._awning(m, spec, self.RING, 6.0, idx)
        return spec, m

    def test_a_named_shop_makes_the_building_a_shop(self):
        spec, _m = self.built({"class": "", "height": 9.0, "shop_kind": "bakery"})
        self.assertTrue(spec.shop)
        self.assertEqual(spec.trade, "bakery")

    def test_a_pitched_roof_shop_gets_its_front(self):
        """A high street's shops sit in pitched-roof terraces; the flat-roof test alone gave them none."""
        spec, m = self.built({"class": "", "height": 9.0, "roof_shape": "gabled", "shop_kind": "bakery"})
        self.assertEqual(spec.roof, "gabled")
        self.assertIsNotNone(spec.sign, "a shop the data names faces its street whatever its roof")
        self.assertGreater(len(m.T), 0, "and something is drawn on it")

    def test_a_house_gets_no_front_at_all(self):
        spec, m = self.built({"class": "", "height": 9.0, "roof_shape": "gabled"})
        self.assertIsNone(spec.trade)
        self.assertEqual(len(m.T), 0)

    def test_the_awning_hangs_off_the_street_wall_and_slopes_down(self):
        spec, _m = self.built({"class": "", "height": 9.0, "shop_kind": "bakery"})
        m = bld.Mesh()
        bld._awning(m, spec, self.RING, 6.0, bld._Looks(spec.seed))
        P = np.array(m.P, np.float64).reshape(-1, 3)
        self.assertGreater(len(P), 0)
        self.assertAlmostEqual(P[:, 2].max() - P[:, 2].min(), bld.AWNING_DROP_M + 0.02, places=6)
        a, b = self.RING[spec.sign], self.RING[(spec.sign + 1) % 4]
        d = (b - a) / np.hypot(*(b - a))
        out = np.array([d[1], -d[0]])
        reach = ((P[:, :2] - a) * out).sum(1)
        self.assertAlmostEqual(reach.max(), bld.AWNING_OUT_M, places=6, msg="it reaches over the pavement")
        self.assertGreaterEqual(reach.min(), -1e-9, "and never back into the building")

    def test_each_trade_has_its_own_colours(self):
        seen = {}
        for k in shops.KINDS:
            board, awning = bld.SHOP_COLOURS[k]
            self.assertNotEqual(board, awning, k)
            self.assertNotIn(board, seen, "%s shares a board colour with %s" % (k, seen.get(board)))
            seen[board] = k

    def test_a_short_wall_takes_no_awning(self):
        tiny = np.array([(0.0, 0.0), (1.5, 0.0), (1.5, 2.0), (0.0, 2.0)])
        spec = bld.plan(tiny, {"class": "", "height": 6.0, "shop_kind": "bakery"}, lat=50.73, lon=2.735)
        spec.sign = 0
        m = bld.Mesh()
        bld._awning(m, spec, tiny, 5.0, bld._Looks(spec.seed))
        self.assertEqual(len(m.T), 0, "an awning as wide as a door is clutter")


if __name__ == "__main__":
    unittest.main()
