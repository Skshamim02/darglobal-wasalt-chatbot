"""FastAPI app: RAG chatbot over scraped DarGlobal + Wasalt data, answered by a
free OpenRouter model."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import KnowledgeBase, listing_text

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Free models; OpenRouter tries them in order if one is rate-limited or down.
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "inclusionai/ling-3.2-flash:free,baidu/ernie-5.5-flash-preview:free,google/gemma-4-31b-it:free,openrouter/free",
).split(",") if m.strip()]

kb = KnowledgeBase()
app = FastAPI(title="DarGlobal & Wasalt Property Assistant")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

SYSTEM = """You are "Dar Assist", a property assistant answering ONLY from the scraped public data of two websites:
- DarGlobal (darglobal.co.uk): an international luxury real estate developer (projects, branded partners, company info).
- Wasalt (wasalt.sa): a Saudi property portal (listings for sale/rent in Saudi cities, platform info).

Rules:
- Use only the CONTEXT below. If the answer isn't there, say you don't have that in the scraped data and suggest the official site. Never invent prices, dates, units or contact details.
- Say which site a fact comes from (DarGlobal or Wasalt). Give prices with currency.
- The interface automatically shows photo cards (with price, size and links) for the matching Wasalt listings and DarGlobal projects under your answer. So do NOT output markdown tables or raw listing URLs; give a short summary in bullet points (name, location, price, beds/size) and highlight what stands out. You may link to DarGlobal pages with markdown links.
- Keep answers under about 220 words; no emojis or flag symbols. Be warm and professional, like a luxury real-estate concierge. Use markdown with bold for key facts. Answer in the user's language (English or Arabic).
- Wasalt listings are a snapshot taken on {date}; mention that availability may have changed when relevant."""

SCRAPED_ON = os.getenv("SCRAPED_ON", "20 Nov 2026")


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[Turn] = Field(default_factory=list, max_length=12)


# simple per-IP rate limit so the public demo can't burn the free quota
_hits: dict[str, deque] = defaultdict(deque)
RATE = int(os.getenv("RATE_PER_MIN", "12"))


def rate_ok(ip: str) -> bool:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        return False
    q.append(now)
    return True


def build_context(r: dict) -> tuple[str, list[dict]]:
    parts, sources = [], []
    if r["listings"]:
        parts.append(f"WASALT LISTINGS matching filters {json.dumps(r['filters'])}:")
        for l in r["listings"]:
            parts.append(f"- {listing_text(l)} | link: {l.get('url')}")
            sources.append({"title": l.get("title"), "url": l.get("url"), "source": "Wasalt"})
    elif r["filters"] and kb.is_listing_query("", r["filters"]):
        parts.append(f"WASALT LISTINGS: no scraped listing matches filters {json.dumps(r['filters'])}.")
    for i, d in enumerate(r["chunks"], 1):
        parts.append(f"[{d.source} | {d.title} | {d.url}]\n{d.text}")
        if d.url and not any(s["url"] == d.url for s in sources):
            sources.append({"title": d.title, "url": d.url, "source": d.source})
    s = kb.stats()
    parts.append(
        f"DATASET STATS: {s['darglobal_pages']} DarGlobal pages; {s['wasalt_listings']} Wasalt listings "
        f"by city {s['listings_by_city']}, by purpose {s['listings_by_purpose']}, by type {s['listings_by_type']}."
    )
    return "\n\n".join(parts), sources[:8]


PROJECTS = json.loads((ROOT / "data" / "darglobal_projects.json").read_text(encoding="utf-8")) \
    if (ROOT / "data" / "darglobal_projects.json").exists() else []
LOC_WORDS = ["dubai", "rak", "marjan", "uae", "emirates", "oman", "muscat", "aida", "qatar", "doha", "simaisma",
             "spain", "benahav", "marbella", "london", "uk", "jeddah", "riyadh", "saudi", "ksa", "rayana"]


def project_cards(q: str, r: dict, limit: int = 8) -> list[dict]:
    """DarGlobal project cards: by brand / location named in the question, else by retrieved pages."""
    ql = q.lower()
    if r["listings"] and not re.search(r"dar\s?global", ql):
        return []
    picked = []
    for p in PROJECTS:
        hay = f"{p['name']} {p['location']} {p.get('brand') or ''}".lower()
        brand = (p.get("brand") or "").lower().replace("the ", "").split(" ")[0]
        if brand and len(brand) > 2 and re.search(rf"\b{re.escape(brand)}", ql):
            picked.append(p)
        elif re.search(r"dar\s?global|project|develop|propert|villa|residen|tower", ql) and any(
            re.search(rf"\b{w}\b", ql) and w in hay for w in LOC_WORDS
        ):
            picked.append(p)
    if not picked:
        urls = [d.url.rstrip("/") for d in r["chunks"] if d.source == "DarGlobal"]
        for u in urls:
            picked += [p for p in PROJECTS if p["url"].rstrip("/") == u and p not in picked]
    seen, out = set(), []
    for p in picked:
        if p["url"] not in seen:
            seen.add(p["url"])
            out.append({"kind": "project", **p})
    return out[:limit]


def listing_cards(r: dict) -> list[dict]:
    keep = ("title", "url", "purpose", "type", "city", "district", "price", "currency", "rent_frequency",
            "bedrooms", "bathrooms", "built_up_area_sqm", "image", "updated")
    return [{"kind": "listing", **{k: l.get(k) for k in keep}} for l in r["listings"][:8]]


async def ask_llm(messages: list[dict]) -> tuple[str, str]:
    if not API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured on the server.")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": os.getenv("APP_URL", "https://github.com/Skshamim02/darglobal-wasalt-chatbot"),
        "X-Title": "DarGlobal Wasalt Assistant",
    }
    last_err = ""
    async with httpx.AsyncClient(timeout=45) as client:
        for attempt in range(4):
            # rotate the fallback list so each retry starts from a different free model
            order = MODELS[attempt % len(MODELS):] + MODELS[: attempt % len(MODELS)]
            payload = {"model": order[0], "models": order[:3], "messages": messages, "temperature": 0.2, "max_tokens": 1600}
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = ((choice.get("message") or {}).get("content") or "").strip()
                if text:
                    return text, data.get("model", MODELS[0])
                last_err = "empty response"
            else:
                last_err = f"{resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            await asyncio.sleep(1.0)
    raise HTTPException(502, f"The language model is unavailable right now ({last_err}). Please retry.")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


# Image proxy for Wasalt listing photos (served through our origin so they load even where the
# image CDN is blocked). Only Wasalt's image host is allowed.
IMG_HOSTS = ("https://imagedelivery.net/1DNKFJPRaeUdy_j8F7HT3w/",)
_img_cache: dict[str, tuple[bytes, str]] = {}


@app.get("/img")
async def img(u: str):
    if not u.startswith(IMG_HOSTS) or len(u) > 400:
        raise HTTPException(400, "host not allowed")
    hit = _img_cache.get(u)
    if not hit:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as c:
            r = await c.get(u, headers={"Accept": "image/webp,image/jpeg,image/*"})
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200 or not ctype.startswith("image/"):
            raise HTTPException(404, "image unavailable")
        hit = (r.content, ctype)
        if len(_img_cache) > 400:
            _img_cache.pop(next(iter(_img_cache)))
        _img_cache[u] = hit
    return Response(hit[0], media_type=hit[1], headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/health")
def health():
    return {"ok": True, "models": MODELS, "key_configured": bool(API_KEY), **kb.stats()}


@app.get("/api/search")
def search(q: str):
    """Debug view of what the retriever finds (no LLM call)."""
    r = kb.retrieve(q)
    return {
        "filters": r["filters"],
        "listings": r["listings"],
        "chunks": [{"source": d.source, "title": d.title, "url": d.url, "text": d.text[:300]} for d in r["chunks"]],
    }


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not rate_ok(ip):
        return JSONResponse({"error": "Too many requests, please wait a minute."}, status_code=429)
    # use the last user turn too, so follow-ups like "and in Jeddah?" keep their topic
    prev_user = next((t.content for t in reversed(body.history) if t.role == "user"), "")
    r = kb.retrieve(body.message)
    if not r["chunks"] and not r["listings"] and prev_user:
        r = kb.retrieve(prev_user + " " + body.message)
    context, sources = build_context(r)
    messages = [{"role": "system", "content": SYSTEM.format(date=SCRAPED_ON) + "\n\nCONTEXT:\n" + context}]
    for t in body.history[-8:]:
        if t.role in ("user", "assistant"):
            messages.append({"role": t.role, "content": t.content[:2000]})
    messages.append({"role": "user", "content": body.message})
    answer, model = await ask_llm(messages)
    cards = listing_cards(r) + project_cards(body.message, r)
    return {"answer": answer, "sources": sources, "model": model, "filters": r["filters"], "cards": cards}


@app.get("/api/stats")
def stats():
    s = kb.stats()
    countries = sorted({p["location"].split(",")[-1].strip() for p in PROJECTS})
    return {**s, "darglobal_projects": len(PROJECTS), "countries": countries,
            "cities": len(s["listings_by_city"]), "scraped_on": SCRAPED_ON}
