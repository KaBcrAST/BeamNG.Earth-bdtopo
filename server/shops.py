"""Shops on the buildings that hold them, from the Overture places already downloaded (the user, 2026-11-02,
beside a photograph of a hand-built Yokohama street: "comment peut-on nous rendre autant un rendu realiste").

A commercial street reads as real through its shopfronts: the awning, the board over it, the glazed ground
floor. buildings.py was inventing all three. Whether a building is a shop came from its own Overture class,
which almost nothing carries: over a Bailleul cell that made 42 shopfronts of 1903 facades, while the places
layer of the SAME cell holds 180 named businesses in 71 categories, 43 of them with a recognised brand
(ALDI, Atol, Boulangerie Louise). That layer is downloaded for every cell and was read by gasstations.py
alone, for fuel; the other 177 changed nothing in the world.

    named, lost = shops.join(buildings, [places])   # writes shop_kind / shop_name on the buildings

A place is a point. It belongs to the building whose footprint contains it, and failing that to the nearest
footprint within NEAR_M: a shop's point is often set on the pavement in front of its door, not inside the
walls. The category is reduced to one of KINDS, which is all the mesh builder reads -- it decides the
awning's colour and the board's, and nothing finer, because a hundred categories would be a hundred
inventions. The name is carried along for whoever can draw text one day; nothing does yet.
"""
import logging

import numpy as np

import bdtopo
import polygon

log = logging.getLogger("earthd.shops")

NEAR_M = 12.0               # m: a place just off its building (its point is often on the pavement)
STEP = bdtopo._Grid.STEP    # the same lon/lat bucketing as the BD TOPO join

# Overture's basic_category -> the look a shopfront takes. Anything else street-facing falls back to "shop".
KINDS = ("bakery", "food", "restaurant", "bar", "pharmacy", "beauty", "fashion", "bank", "hotel",
         "automotive", "hardware", "shop")
CATEGORY_KIND = {
    "bakery": "bakery", "patisserie": "bakery", "dessert_shop": "bakery",
    "cafe": "bar", "coffee_shop": "bar", "bar": "bar", "pub": "bar", "brewery": "bar", "nightlife": "bar",
    "restaurant": "restaurant", "fast_food": "restaurant", "food_and_beverage_service": "restaurant",
    "food_and_beverage_store": "food", "grocery_store": "food", "supermarket": "food", "butcher": "food",
    "farmers_market": "food", "wine_and_spirits_store": "food",
    "pharmacy_and_drug_store": "pharmacy", "pharmacy": "pharmacy", "health_and_medical": "pharmacy",
    "complementary_and_alternative_medicine": "pharmacy", "optometrist": "pharmacy",
    "personal_or_beauty_service": "beauty", "hair_salon": "beauty", "beauty_salon": "beauty",
    "barber": "beauty", "spa": "beauty",
    "fashion_and_apparel_store": "fashion", "shoe_store": "fashion", "jewelry_store": "fashion",
    "financial_service": "bank", "bank_credit_union": "bank", "insurance_agency": "bank",
    "professional_service": "bank", "real_estate": "bank",
    "hotel": "hotel", "accommodation": "hotel", "bed_and_breakfast": "hotel",
    "automotive_service": "automotive", "automotive_repair": "automotive", "car_dealer": "automotive",
    "gas_station": "automotive",
    "hardware_home_and_garden_store": "hardware", "furniture_store": "hardware", "home_service": "hardware",
    "home_improvement_store": "hardware",
}
# a business, but never a shopfront on a street: no awning, no board
NOT_A_SHOPFRONT = {"religious_organization", "school", "education", "public_service_government",
                   "social_or_community_service", "family_service", "sports_and_recreation",
                   "landmark_and_historical_building", "park", "parking", "transportation",
                   "industrial", "agriculture", "utilities", "cemetery"}


def kind_of(category, taxonomy=""):
    """The shopfront look for an Overture category, or None when it is not a shopfront at all."""
    c = (category or "").strip().lower()
    if not c or c in NOT_A_SHOPFRONT:
        return None
    k = CATEGORY_KIND.get(c)
    if k:
        return k
    # the taxonomy is a path, "eat_and_drink > restaurant > ...": its head tells the street-facing trades
    head = (taxonomy or "").strip().lower().split(">")[0].strip()
    if head == "eat_and_drink":
        return "restaurant"
    if head in ("retail", "shopping"):
        return "shop"
    if any(w in c for w in ("_store", "shop", "retail", "market")):
        return "shop"
    if any(w in c for w in ("service", "repair", "agency", "studio", "office", "salon")):
        return "shop"
    return None


def _points(places):
    """[(lon, lat, kind, name)] of the shopfront places of one layer."""
    out = []
    if places is None or len(places) == 0:
        return out
    a = places.attrs

    def col(k, i):
        v = a[k][i] if k in a else None
        return "" if v is None or str(v) == "None" else str(v)

    for i in range(len(places)):
        if col("operating_status", i).lower() in ("closed", "closed_permanently"):
            continue
        k = kind_of(col("basic_category", i), col("taxonomy", i))
        if k is None:
            continue
        j = places.coord_off[places.ring_off[places.part_off[i]]]
        out.append((float(places.lonlat[j, 0]), float(places.lonlat[j, 1]), k, col("name", i)))
    return out


def join(feats, place_layers):
    """Give every building that holds a shop its kind and the shop's name.

    Returns (buildings named, places that found no building); feats is changed in place."""
    n = len(feats)
    if not n:
        return 0, 0
    pts = []
    for p in place_layers:
        pts += _points(p)
    if not pts:
        return 0, 0
    rings, boxes = [], []
    for i in range(n):
        r = bdtopo._outer(feats, i)
        rings.append(r)
        if r is None:
            boxes.append((0.0, 0.0, -1.0, -1.0))        # an empty box: never a candidate
        else:
            boxes.append((r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()))
    buckets = {}
    for i, (w, s, e, nn) in enumerate(boxes):
        if e < w:
            continue
        for gx in range(int(np.floor(w / STEP)), int(np.floor(e / STEP)) + 1):
            for gy in range(int(np.floor(s / STEP)), int(np.floor(nn / STEP)) + 1):
                buckets.setdefault((gx, gy), []).append(i)
    kinds = bdtopo._text(feats, "shop_kind", n)
    names = bdtopo._text(feats, "shop_name", n)
    named = lost = 0
    for lon, lat, kind, name in pts:
        cand = []
        gx0, gy0 = int(np.floor(lon / STEP)), int(np.floor(lat / STEP))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += buckets.get((gx0 + dx, gy0 + dy), ())
        pt = np.array([[lon, lat]], np.float64)
        hit = None
        for i in cand:
            w, s, e, nn = boxes[i]
            if w <= lon <= e and s <= lat <= nn and rings[i] is not None \
                    and bool(polygon.contains_points(rings[i], pt)[0]):
                hit = i
                break
        if hit is None:                 # a point set on the pavement: the nearest footprint, if it is close
            # to the footprint's BOX, not its centre: a shop's point a few metres off the wall of a big
            # store is tens of metres from its centre, and measuring from there loses it
            kx = 111320.0 * float(np.cos(np.radians(lat)))
            best = None
            for i in set(cand):
                if rings[i] is None:
                    continue
                w, s_, e, nn = boxes[i]
                ddx = max(w - lon, 0.0, lon - e) * kx
                ddy = max(s_ - lat, 0.0, lat - nn) * 110540.0
                d = float(np.hypot(ddx, ddy))
                if d <= NEAR_M and (best is None or d < best[0]):
                    best = (d, i)
            hit = best[1] if best else None
        if hit is None:
            lost += 1
            continue
        if kinds[hit]:                  # a building holding several shops keeps the first
            continue
        kinds[hit] = kind
        names[hit] = name
        named += 1
    if named:
        feats.attrs["shop_kind"] = np.array([v or "" for v in kinds])
        feats.attrs["shop_name"] = np.array([v or "" for v in names])
    return named, lost
