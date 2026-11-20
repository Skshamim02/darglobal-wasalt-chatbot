# Dar Assist: DarGlobal & Wasalt property chatbot

A retrieval-augmented chatbot over publicly available data scraped from
**darglobal.co.uk** (luxury developer: projects, branded partners, company info)
and **wasalt.sa** (Saudi property portal: listings for sale and rent, platform
info). Answers come from a free OpenRouter model and cite the source pages.

## How it works

```
scraper/ ──► data/*.json ──► app/rag.py (BM25 + listing filters) ──► OpenRouter (free model) ──► FastAPI + web UI
```

1. **Scraping**
   - *Wasalt*: `scraper/scrape_wasalt.py` reads the Next.js `__NEXT_DATA__` JSON
     on the public SEO landing pages (e.g. `/en/properties-for-sale-in-riyadh`),
     so the fields come out structured (price, city, district, beds, area, rent
     period). That gives about 970 listings across Riyadh, Jeddah, Dammam, Makkah
     and Madinah, plus the city guides, the About text and the FAQ. Requests are
     sent one per second and stay within robots.txt.
   - *DarGlobal*: the site sits behind Incapsula bot protection, so plain HTTP
     clients get a challenge page. Rather than bypass it, I collected the public
     pages (home, about, every project and country listing, partner brands,
     hospitality, FAQ, contact, press) in a normal browser session and parsed the
     HTML text plus each page's `__NEXT_DATA__` into `data/darglobal.json`:
     77 pages and 38 projects across the UAE, Oman, Qatar, Saudi Arabia, Spain and the UK.
2. **Retrieval** (`app/rag.py`)
   - BM25 keyword index over chunked page text and one document per listing.
     It has no embedding model, so the container stays small and starts fast.
   - A small query parser pulls out structured filters (city, rent or sale,
     property type, bedrooms, price limits in "2 million" / "500k" form,
     cheapest/largest sorting) and queries the listings directly, which works far
     better than keyword search for questions like "cheapest 3-bed apartment for
     rent in Riyadh".
3. **Generation** (`app/main.py`)
   - The prompt is grounded in the retrieved context only. The model has to name
     the source site, give prices with currency, and link listings. Follow-up
     questions keep the chat history.
   - Models are free OpenRouter models with automatic fallback:
     `google/gemma-4-31b-it:free → nvidia/nemotron-3.5-lightning:free → openrouter/free`.
   - The API is rate-limited per IP so the free quota isn't burned.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| POST | `/api/chat` | `{"message": "...", "history": [...]}` → answer, sources, model |
| GET | `/api/search?q=` | Retrieval debug view (no LLM call) |
| GET | `/api/health` | Status and dataset stats |

## Deployment

It deploys as a Docker web service on Render (free tier) from this repo's
`Dockerfile`. `OPENROUTER_API_KEY` is set as a secret environment variable.
On the free tier, a sleeping instance takes about 50 s to wake on the first request.

## Run locally

```bash
docker build -t dar-assist .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... dar-assist
# open http://localhost:7860
```

Re-scrape Wasalt: `pip install -r requirements-scraper.txt && python scraper/scrape_wasalt.py`

## Configuration

| Env var | Default |
|---|---|
| `OPENROUTER_API_KEY` | required |
| `OPENROUTER_MODELS` | comma-separated fallback list above |
| `RATE_PER_MIN` | 12 |

## Limitations

- Listings are a snapshot from 20 Nov 2026, so prices and availability change.
- BM25 is lexical. With more time I'd add multilingual embeddings (Arabic and
  English) for hybrid search, and schedule re-scrapes.
