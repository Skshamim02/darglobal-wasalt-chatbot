"""Scrape publicly available Wasalt (wasalt.sa) data.

Wasalt is a Next.js site: every listing page ships its data as JSON in
<script id="__NEXT_DATA__">. We read that instead of parsing HTML, which gives
clean structured fields (price, city, district, bedrooms, area, ...).

robots.txt allows "/" for all agents (only /search and /cdn-cgi/rum are
disallowed), so we only use the SEO landing pages such as
/en/properties-for-sale-in-riyadh and their paginated /en/{sale|rent}/... views,
with a polite delay between requests.

Output: data/wasalt_listings.json, data/wasalt_pages.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://wasalt.sa"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
OUT = Path(__file__).resolve().parent.parent / "data"
DELAY = 1.0

CITIES = ["riyadh", "jeddah", "dammam", "khobar", "makkah", "madinah"]
SLUGS = (
    [f"properties-for-sale-in-{c}" for c in CITIES]
    + [f"properties-for-rent-in-{c}" for c in CITIES]
    + [
        "villas-for-sale-in-riyadh", "apartments-for-sale-in-riyadh",
        "apartments-for-rent-in-riyadh", "villas-for-rent-in-riyadh",
        "villas-for-sale-in-jeddah", "apartments-for-sale-in-jeddah",
        "apartments-for-rent-in-jeddah", "lands-for-sale-in-riyadh",
        "apartments-for-rent-in-dammam", "villas-for-sale-in-dammam",
    ]
)

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language": "en"})


def next_data(url: str) -> dict | None:
    r = session.get(url, timeout=30)
    time.sleep(DELAY)
    if r.status_code != 200:
        print("  !", r.status_code, url)
        return None
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    return json.loads(m.group(1))["props"]["pageProps"] if m else None


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ")).strip()


CITY_NAMES = {"Aldammam": "Dammam", "Makkah Al Mukarramah": "Makkah", "Al Khobar": "Khobar", "Alkhobar": "Khobar"}


def rent_freq(rf) -> str | None:
    """rentFreq is {"yearly": {"amount": .., "default_freq": true}, ...}; keep the default period."""
    if isinstance(rf, dict) and rf:
        for k, v in rf.items():
            if isinstance(v, dict) and v.get("default_freq"):
                return k
        return next(iter(rf))
    return rf or None


def iter_props(items):
    """searchResult.properties mixes dicts with nested lists (grouped units)."""
    for it in items or []:
        if isinstance(it, list):
            yield from iter_props(it)
        elif isinstance(it, dict) and it.get("id") and it.get("propertyInfo"):
            yield it


def clean_listing(p: dict) -> dict:
    info = p.get("propertyInfo") or {}
    attrs = {a.get("key"): a.get("value") for a in (p.get("attributes") or [])}
    for_ = info.get("propertyFor") or ""
    price = info.get("salePrice") if for_ == "sale" else info.get("expectedRent")
    try:
        price = float(price) if price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    slug = info.get("slug") or ""
    return {
        "id": p.get("id"),
        "title": info.get("title"),
        "url": f"{BASE}/en/property/{for_}/{slug}" if slug else None,
        "purpose": for_,                      # sale | rent
        "type": info.get("propertySubType"),  # Villa, Apartment, Floor, Land...
        "category": info.get("propertyMainType"),
        "city": CITY_NAMES.get(info.get("city"), info.get("city")),
        "district": info.get("district") or info.get("zone"),
        "area_name": info.get("territory"),
        "address": info.get("address"),
        "project": info.get("projectName"),
        "price": price,
        "currency": info.get("currencyType") or "SAR",
        "rent_frequency": rent_freq(info.get("rentFreq")),
        "bedrooms": attrs.get("noOfBedrooms"),
        "bathrooms": attrs.get("noOfBathrooms"),
        "built_up_area_sqm": attrs.get("builtUpArea") or p.get("floorSize"),
        "age": attrs.get("completionYear"),
        "furnishing": info.get("furnishingType"),
        "amenities": [a.get("name") for a in (p.get("featureAmenities") or []) if isinstance(a, dict)],
        "verified": p.get("isVerified"),
        "rega_licensed": p.get("isRegaProp"),
        "updated": p.get("ageOfListing"),
        "lat": (p.get("location") or {}).get("lat"),
        "lon": (p.get("location") or {}).get("lon"),
    }


def main(pages_per_slug: int = 2) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    listings: dict[int, dict] = {}
    pages: list[dict] = []

    # Home page: about text, FAQ, services
    pp = next_data(f"{BASE}/en")
    if pp:
        cms = pp.get("cmsData") or {}
        faq = (cms.get("Faq") or {}).get("Questions") or []
        faq_text = " ".join(
            f"Q: {q.get('question') or q.get('title')} A: {html_to_text(q.get('answer') or q.get('description') or '')}"
            for q in faq if isinstance(q, dict)
        )
        about = cms.get("section_10_aboutWasalt") or {}
        body = (pp.get("bannerData") or {}).get("body_content") or {}
        sections = []
        for k in ("section_2", "section_5"):
            for s in cms.get(k) or []:
                sections.append(f"{s.get('heading')}: {s.get('content')} ({s.get('link')})")
        auction = cms.get("section_6_auction") or {}
        pages.append({
            "url": f"{BASE}/en",
            "title": "About Wasalt",
            "text": " ".join(filter(None, [
                (about.get("header") or {}).get("title"),
                (about.get("header") or {}).get("subTitle"),
                html_to_text(about.get("description")),
                html_to_text(body.get("para")),
                " ".join(sections),
                f"Auctions: {(auction.get('header') or {}).get('subTitle')} ({auction.get('redirectUrl')})",
            ])),
        })
        if faq_text:
            pages.append({"url": f"{BASE}/en", "title": "Wasalt FAQ", "text": faq_text})

    for slug in SLUGS:
        print(slug)
        pp = next_data(f"{BASE}/en/{slug}")
        if not pp:
            continue
        guide = html_to_text(pp.get("para"))
        if guide:
            pages.append({"url": f"{BASE}/en/{slug}", "title": slug.replace("-", " ").title() + " - guide", "text": guide})
        sr = pp.get("searchResult") or {}
        for p in iter_props(sr.get("properties")):
            listings[p["id"]] = clean_listing(p)
        # further pages through the paginated search view of the same filters
        fq = pp.get("filterQueries") or {}
        purpose = fq.get("propertyFor")
        if purpose and pages_per_slug > 1:
            q = {k: v for k, v in fq.items() if k not in ("page", "pageSize", "includeProperties")}
            for page in range(2, pages_per_slug + 1):
                qs = "&".join(f"{k}={v}" for k, v in q.items() if v not in (None, ""))
                pp2 = next_data(f"{BASE}/en/{purpose}/search?{qs}&page={page}")
                for p in iter_props(((pp2 or {}).get("searchResult") or {}).get("properties")):
                    listings[p["id"]] = clean_listing(p)
        print("  total listings:", len(listings))

    (OUT / "wasalt_listings.json").write_text(json.dumps(list(listings.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "wasalt_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(listings)} listings, {len(pages)} pages")


if __name__ == "__main__":
    main()
