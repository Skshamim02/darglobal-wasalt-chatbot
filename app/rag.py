"""Retrieval layer: BM25 over scraped DarGlobal + Wasalt text, plus a structured
filter over Wasalt listings for questions like "cheapest 3-bed apartment for
rent in Jeddah" (which keyword search alone answers badly)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

STOP = set("""a an the and or of to in on for with at by from is are was were be been it this that these those
what which who whom how when where why do does did can could would should will i you me my your we our us
about tell show give find list any some all there their they them as into than then so if not no yes please
get have has had more most much many""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Doc:
    text: str
    url: str
    title: str
    source: str  # "DarGlobal" | "Wasalt"
    meta: dict = field(default_factory=dict)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def chunk(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    parts, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):  # cut at a sentence / word boundary
            k = max(text.rfind(". ", i, j), text.rfind(" ", i + size // 2, j))
            j = k + 1 if k > i else j
        parts.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return parts


def fmt_price(v) -> str:
    if v is None:
        return "price on request"
    return f"{v:,.0f}"


def listing_text(l: dict) -> str:
    bits = [
        l.get("title") or "",
        f"for {l.get('purpose')}",
        f"{l.get('type')} in {l.get('district') or ''}, {l.get('city') or ''}",
        f"price {fmt_price(l.get('price'))} {l.get('currency') or 'SAR'}"
        + ({"monthly": " per month", "yearly": " per year"}.get(l.get("rent_frequency"), "") if l.get("purpose") == "rent" else ""),
    ]
    if l.get("bedrooms"):
        bits.append(f"{l['bedrooms']} bedrooms")
    if l.get("bathrooms"):
        bits.append(f"{l['bathrooms']} bathrooms")
    if l.get("built_up_area_sqm"):
        bits.append(f"{l['built_up_area_sqm']} sqm")
    if l.get("age") not in (None, ""):
        bits.append(f"age {l['age']}")
    if l.get("furnishing"):
        bits.append(str(l["furnishing"]))
    if l.get("amenities"):
        bits.append("amenities: " + ", ".join(map(str, l["amenities"][:8])))
    if l.get("project"):
        bits.append(f"project {l['project']}")
    bits.append(f"area {l.get('area_name') or ''}")
    return "; ".join(b for b in bits if b)


CITY_ALIASES = {
    "riyadh": "Riyadh", "jeddah": "Jeddah", "jedda": "Jeddah", "dammam": "Dammam",
    "makkah": "Makkah", "mecca": "Makkah", "madinah": "Madinah", "medina": "Madinah",
    "khobar": "Khobar",
}
TYPE_ALIASES = {
    "apartment": "Apartment", "apartments": "Apartment", "flat": "Apartment", "flats": "Apartment",
    "villa": "Villa", "villas": "Villa", "land": "Land", "lands": "Land", "plot": "Land", "plots": "Land",
    "building": "Building", "buildings": "Building", "floor": "Floor", "floors": "Floor",
    "rest house": "Rest House", "warehouse": "Warehouse",
}


def parse_money(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("k", "thousand"):
        v *= 1_000
    elif unit in ("m", "mn", "million"):
        v *= 1_000_000
    return v


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA):
        self.docs: list[Doc] = []
        self.listings: list[dict] = []
        self.darglobal_pages = 0
        self._load(data_dir)
        self.bm25 = BM25([tokenize(d.title + " " + d.text) for d in self.docs])

    def _load(self, data_dir: Path) -> None:
        dg = data_dir / "darglobal.json"
        if dg.exists():
            pages = json.loads(dg.read_text(encoding="utf-8"))
            self.darglobal_pages = len(pages)
            for p in pages:
                for c in chunk(p.get("text") or ""):
                    self.docs.append(Doc(c, p.get("url", ""), p.get("title", ""), "DarGlobal"))
        wp = data_dir / "wasalt_pages.json"
        if wp.exists():
            for p in json.loads(wp.read_text(encoding="utf-8")):
                for c in chunk(p.get("text") or ""):
                    self.docs.append(Doc(c, p.get("url", ""), p.get("title", ""), "Wasalt"))
        wl = data_dir / "wasalt_listings.json"
        if wl.exists():
            self.listings = json.loads(wl.read_text(encoding="utf-8"))
            for l in self.listings:
                self.docs.append(Doc(listing_text(l), l.get("url") or "", l.get("title") or "", "Wasalt", {"listing": l}))

    # ---------- structured listing search ----------
    def parse_filters(self, q: str) -> dict:
        ql = q.lower()
        f: dict = {}
        for k, v in CITY_ALIASES.items():
            if re.search(rf"\b{k}\b", ql):
                f["city"] = v
        for k, v in TYPE_ALIASES.items():
            if re.search(rf"\b{k}\b", ql):
                f["type"] = v
        if re.search(r"\b(rent|rental|renting|lease|per year|per month|yearly|monthly)\b", ql):
            f["purpose"] = "rent"
        elif re.search(r"\b(buy|sale|purchase|for sale|buying)\b", ql):
            f["purpose"] = "sale"
        m = re.search(r"(\d+)\s*(?:-|\s)?\s*(?:bed|beds|bedroom|bedrooms|br|bhk)\b", ql)
        if m:
            f["bedrooms"] = int(m.group(1))
        money = r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|mn|million|thousand)?"
        m = re.search(r"(?:under|below|less than|max(?:imum)?|up to|within|budget(?: of)?)\s*(?:sar|sr)?\s*" + money, ql)
        if m:
            f["max_price"] = parse_money(m.group(1), m.group(2))
        m = re.search(r"(?:over|above|more than|min(?:imum)?|at least|starting)\s*(?:sar|sr)?\s*" + money, ql)
        if m:
            f["min_price"] = parse_money(m.group(1), m.group(2))
        if re.search(r"\b(cheap|cheapest|lowest|affordable|budget|least expensive)\b", ql):
            f["sort"] = "price_asc"
        elif re.search(r"\b(expensive|priciest|highest|luxury|most expensive)\b", ql):
            f["sort"] = "price_desc"
        elif re.search(r"\b(largest|biggest|spacious)\b", ql):
            f["sort"] = "area_desc"
        return f

    def search_listings(self, f: dict, limit: int = 8) -> list[dict]:
        def area(l):
            try:
                return float(l.get("built_up_area_sqm") or 0)
            except ValueError:
                return 0.0

        res = []
        for l in self.listings:
            if f.get("city") and l.get("city") != f["city"]:
                continue
            if f.get("type") and l.get("type") != f["type"]:
                continue
            if f.get("purpose") and l.get("purpose") != f["purpose"]:
                continue
            if f.get("bedrooms") and l.get("bedrooms") != f["bedrooms"]:
                continue
            p = l.get("price")
            if f.get("max_price") and (p is None or p > f["max_price"]):
                continue
            if f.get("min_price") and (p is None or p < f["min_price"]):
                continue
            res.append(l)
        key = f.get("sort")
        if key == "price_asc":
            res.sort(key=lambda l: (l.get("price") is None or l["price"] <= 0, l.get("price") or 0))
        elif key == "price_desc":
            res.sort(key=lambda l: -(l.get("price") or 0))
        elif key == "area_desc":
            res.sort(key=lambda l: -area(l))
        return res[:limit]

    def is_listing_query(self, q: str, f: dict) -> bool:
        if "darglobal" in q.lower() or "dar global" in q.lower():
            return False
        signal = {"city", "type", "bedrooms", "max_price", "min_price"}
        return bool(signal & f.keys()) or ("purpose" in f and "wasalt" in q.lower())

    # ---------- main entry ----------
    def retrieve(self, q: str, k: int = 7) -> dict:
        qt = tokenize(q)
        f = self.parse_filters(q)
        listings = self.search_listings(f) if self.is_listing_query(q, f) else []
        scored = sorted(zip(self.bm25.scores(qt), range(len(self.docs))), reverse=True)
        want_dg = bool(re.search(r"dar\s?global|darglobal|trump|aida|missoni|lamborghini|pagani|elie saab|fendi|aston martin|w residences|mouawad|marbella|oman|qatar|london|spain|uk\b", q.lower()))
        chunks, seen_listing_ids = [], {l["id"] for l in listings}
        for s, i in scored:
            if s <= 0 or len(chunks) >= k:
                break
            d = self.docs[i]
            lid = d.meta.get("listing", {}).get("id")
            if lid and (lid in seen_listing_ids or listings):
                continue  # structured results already cover listings
            chunks.append(d)
        # make sure a DarGlobal-specific question gets DarGlobal context
        if want_dg and not any(d.source == "DarGlobal" for d in chunks):
            extra = [self.docs[i] for s, i in scored if s > 0 and self.docs[i].source == "DarGlobal"][:3]
            chunks = extra + chunks[: max(0, k - len(extra))]
        # pin overview pages that BM25 under-ranks (short pages / list-style text)
        ql = q.lower()
        pins = []
        if want_dg or "project" in ql or "portfolio" in ql:
            if re.search(r"project|portfolio|where|which|countr|location|dubai|uae|london|uk\b|spain|oman|qatar|saudi|jeddah|riyadh|muscat|doha", ql):
                pins.append("/projects")
        if re.search(r"contact|phone|call|email|office|address|reach|whatsapp|sales team", ql):
            pins.append("/get-in-touch")
        if re.search(r"investor|shares?|stock|revenue|profit|financial|listed|lse", ql):
            pins.append("/investor")
        for suffix in pins:
            pd = [d for d in self.docs if d.source == "DarGlobal" and d.url.rstrip("/").endswith(suffix)]
            chunks = pd[:3] + [d for d in chunks if d not in pd]
        chunks = chunks[: k + 3]
        return {"filters": f, "listings": listings, "chunks": chunks}

    def stats(self) -> dict:
        by_city = Counter(l.get("city") for l in self.listings)
        by_purpose = Counter(l.get("purpose") for l in self.listings)
        by_type = Counter(l.get("type") for l in self.listings)
        return {
            "darglobal_pages": self.darglobal_pages,
            "wasalt_listings": len(self.listings),
            "chunks_indexed": len(self.docs),
            "listings_by_city": dict(by_city.most_common()),
            "listings_by_purpose": dict(by_purpose),
            "listings_by_type": dict(by_type.most_common()),
        }
