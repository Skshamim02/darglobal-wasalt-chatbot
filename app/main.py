"""FastAPI app: RAG chatbot over scraped DarGlobal + Wasalt data, answered by a
free OpenRouter model."""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import KnowledgeBase, listing_text

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Free models; OpenRouter tries them in order if one is rate-limited or down.
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "google/gemma-4-31b-it:free,nvidia/nemotron-3.5-lightning:free,openrouter/free",
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
- For listings, show a short markdown list: title, location, price, beds/size, and the link.
- Be concise and friendly. Use markdown. Answer in the user's language (English or Arabic).
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


async def ask_llm(messages: list[dict]) -> tuple[str, str]:
    if not API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured on the server.")
    payload = {"model": MODELS[0], "models": MODELS, "messages": messages, "temperature": 0.2, "max_tokens": 900}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": os.getenv("APP_URL", "https://huggingface.co/spaces"),
        "X-Title": "DarGlobal Wasalt Assistant",
    }
    last_err = ""
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(3):
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
            await asyncio.sleep(1.5 * (attempt + 1))
    raise HTTPException(502, f"The language model is unavailable right now ({last_err}). Please retry.")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


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
    return {"answer": answer, "sources": sources, "model": model, "filters": r["filters"]}
