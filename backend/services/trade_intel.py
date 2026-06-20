"""
Trade Intelligence Service.

Given a ticker + trade date, this module:
  1. Fetches news articles from Polygon.io (±3 days around the gap)
     Falls back to yfinance if Polygon has no coverage (Indian tickers).
  2. Geocodes each publisher to (lat, lng) using a static lookup.
  3. Classifies articles into globe categories.
  4. Builds GeoEvent pins + GlobeArc connections to the exchange hub.
  5. Calls Claude (Haiku) to synthesize a 2-3 sentence reasoning for the gap.

Returns a dict ready for the frontend TradeIntel page.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

# ── Publisher → (lat, lng) static lookup ─────────────────────────────────────
PUBLISHER_GEO: dict[str, tuple[float, float]] = {
    # India
    "Economic Times":           (19.0760,  72.8777),
    "Business Standard":        (28.6139,  77.2090),
    "Moneycontrol":             (19.0760,  72.8777),
    "The Hindu Business Line":  (13.0827,  80.2707),
    "Livemint":                 (28.6139,  77.2090),
    "NDTV Profit":              (28.6139,  77.2090),
    "Financial Express":        (28.6139,  77.2090),
    "NSE India":                (19.0760,  72.8777),
    "BSE India":                (19.0760,  72.8777),
    "Mint":                     (28.6139,  77.2090),
    "ET Markets":               (19.0760,  72.8777),
    # US
    "Bloomberg":                (40.7128,  -74.0060),
    "Reuters":                  (40.7128,  -74.0060),
    "CNBC":                     (40.7592,  -73.9851),
    "Wall Street Journal":      (40.7128,  -74.0060),
    "MarketWatch":              (40.7128,  -74.0060),
    "Seeking Alpha":            (37.7749, -122.4194),
    "Yahoo Finance":            (37.3382, -121.8863),
    "Motley Fool":              (38.9072,  -77.0369),
    "Benzinga":                 (42.3314,  -83.0458),
    "Zacks":                    (41.8781,  -87.6298),
    "Investopedia":             (40.7128,  -74.0060),
    "Barron's":                 (40.7128,  -74.0060),
    "TheStreet":                (40.7128,  -74.0060),
    "AP":                       (40.7128,  -74.0060),
    # UK
    "Financial Times":          (51.5074,   -0.1278),
    "Guardian":                 (51.5074,   -0.1278),
    "BBC":                      (51.5074,   -0.1278),
    "The Telegraph":            (51.5074,   -0.1278),
    # Asia
    "South China Morning Post": (22.3193,  114.1694),
    "Straits Times":            ( 1.3521,  103.8198),
    "Nikkei":                   (35.6762,  139.6503),
}

# Fallback: India center
_DEFAULT_GEO = (20.5937, 78.9629)

# ── Category keyword matching ─────────────────────────────────────────────────
_CAT_KEYWORDS: dict[str, list[str]] = {
    "monetary_policy": ["rate", "rbi", "federal reserve", "fed ", "central bank",
                        "basis points", "bps", "repo", "interest rate", "hawkish",
                        "dovish", "monetary policy", "rate cut", "rate hike"],
    "energy":          ["oil", "crude", "natural gas", "energy", "petroleum",
                        "brent", "wti", "opec", "refinery", "coal"],
    "geopolitical":    ["war", "sanctions", "geopolit", "tension", "conflict",
                        "military", "election", "government", "border",
                        "india-pak", "china-india", "russia"],
    "trade":           ["trade", "export", "import", "tariff", "duty", "wto",
                        "supply chain", "logistics", "customs"],
    "inflation":       ["inflation", "cpi", "wpi", "consumer price",
                        "wholesale price", "iip", "gdp", "recession"],
    "markets":         ["earnings", "revenue", "profit", "results", "q1", "q2",
                        "q3", "q4", "quarterly", "guidance", "upgrade",
                        "downgrade", "target", "buy rating", "sell rating",
                        "ipo", "merger", "acquisition"],
}

EXCHANGE_COORDS: dict[str, tuple[float, float]] = {
    "NSE":    (19.0760,  72.8777),   # Mumbai
    "NYSE":   (40.7128,  -74.0060),  # New York
    "NASDAQ": (40.7128,  -74.0060),
}

CATEGORY_COLORS: dict[str, str] = {
    "monetary_policy": "#0a84ff",
    "energy":          "#ff8c00",
    "geopolitical":    "#ff2d55",
    "trade":           "#ffd60a",
    "inflation":       "#bf5af2",
    "markets":         "#00ff88",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_category(text: str) -> str:
    t = text.lower()
    for cat, keywords in _CAT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return cat
    return "markets"


def _publisher_coords(publisher_name: str) -> tuple[float, float]:
    name = publisher_name.lower()
    for key, coords in PUBLISHER_GEO.items():
        if key.lower() in name or name in key.lower():
            return coords
    return _DEFAULT_GEO


def _poly_sentiment_score(sentiment: str) -> float:
    return {"positive": 0.6, "negative": -0.6, "neutral": 0.0}.get(sentiment, 0.0)


# ── News fetching ─────────────────────────────────────────────────────────────

async def _fetch_polygon(ticker: str, date_str: str, window: int = 3) -> list:
    """Polygon.io news — best for US tickers."""
    if not POLYGON_API_KEY:
        return []
    clean = ticker.split(".")[0]           # strip .NS / .BO
    try:
        date  = datetime.strptime(date_str, "%Y-%m-%d")
        start = (date - timedelta(days=window)).strftime("%Y-%m-%dT00:00:00Z")
        end   = (date + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")
        url   = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker":               clean,
            "published_utc.gte":    start,
            "published_utc.lte":    end,
            "order":                "desc",
            "limit":                15,
            "apiKey":               POLYGON_API_KEY,
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            logger.info(f"Polygon.io: {len(results)} articles for {clean}")
            return results
        logger.warning(f"Polygon.io {resp.status_code} for {clean}")
    except Exception as e:
        logger.warning(f"Polygon.io fetch error: {e}")
    return []


def _fetch_yfinance(ticker: str) -> list:
    """
    Fetch recent news for a specific ticker from yfinance.
    Primary source for NSE stocks — returns company-specific news.
    """
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        raw = ticker_obj.news or []

        # yfinance v0.2+ nests content differently — handle both formats
        normalized = []
        for n in raw[:15]:
            # New format: content is nested under 'content' key
            content = n.get("content", n)
            title   = (content.get("title", "") or n.get("title", "")).strip()
            if not title:
                continue

            # Timestamp: new format uses pubDate string, old uses providerPublishTime int
            pub_date = ""
            if "pubDate" in content:
                pub_date = content["pubDate"][:10]
            elif "providerPublishTime" in n:
                ts = n.get("providerPublishTime", 0)
                if ts:
                    pub_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

            url = (
                content.get("canonicalUrl", {}).get("url", "")
                or content.get("clickThroughUrl", {}).get("url", "")
                or n.get("link", "")
            ).strip()

            provider = (
                content.get("provider", {}).get("displayName", "")
                or n.get("publisher", "Yahoo Finance")
            )

            normalized.append({
                "title":         title,
                "published_utc": pub_date,
                "article_url":   url,
                "publisher":     {"name": provider},
                "description":   "",
                "insights":      [],
                "_ticker_specific": True,   # mark as directly about this ticker
            })

        logger.info(f"yfinance: {len(normalized)} articles for {ticker}")
        return normalized
    except Exception as e:
        logger.warning(f"yfinance news failed for {ticker}: {e}")
        return []


# ── Globe data builder ────────────────────────────────────────────────────────

def _build_globe_data(
    articles: list,
    ticker: str,
    date_str: str,
) -> tuple[list, list]:
    """Convert articles → GeoEvent pins + GlobeArcs to exchange hub."""
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    exch      = "NSE" if is_indian else "NYSE"
    hub_lat, hub_lng = EXCHANGE_COORDS[exch]

    events: list = []
    arcs:   list = []

    # Exchange "hub" pin
    hub_id = f"hub_{ticker}_{date_str}"
    events.append({
        "id":            hub_id,
        "title":         f"{ticker.replace('.NS','').replace('.BO','')} — Gap-Up",
        "summary":       f"Gapped up at open on {exch} ({date_str})",
        "published_at":  date_str,
        "category":      "markets",
        "lat":           hub_lat,
        "lng":           hub_lng,
        "location_name": "Mumbai" if is_indian else "New York",
        "severity":      1.0,
        "is_analyzed":   True,
        "source_name":   exch,
        "source_url":    "",
    })

    for i, article in enumerate(articles[:10]):
        pub  = article.get("publisher", {})
        name = pub.get("name", "Unknown") if isinstance(pub, dict) else str(pub)
        lat, lng = _publisher_coords(name)

        title = article.get("title", "")
        desc  = article.get("description", "") or ""
        cat   = _detect_category(f"{title} {desc}")

        # Use Polygon's per-ticker sentiment insight if available
        poly_score = 0.0
        for ins in article.get("insights", []):
            clean = ticker.split(".")[0].upper()
            if ins.get("ticker", "").upper() == clean:
                poly_score = _poly_sentiment_score(ins.get("sentiment", "neutral"))
                break

        severity = round(min(1.0, max(0.15, 0.45 + poly_score * 0.35 + max(0, 0.05 * (9 - i)))), 3)
        ev_id    = f"news_{i}_{date_str}"

        events.append({
            "id":            ev_id,
            "title":         title[:120],
            "summary":       desc[:280] if desc else None,
            "source_url":    article.get("article_url", ""),
            "source_name":   name,
            "published_at":  article.get("published_utc", date_str),
            "category":      cat,
            "lat":           lat,
            "lng":           lng,
            "location_name": name,
            "severity":      severity,
            "is_analyzed":   False,
        })

        # Arc: news source city → exchange hub
        arcs.append({
            "id":               f"arc_{i}_{date_str}",
            "source_event_id":  ev_id,
            "target_event_id":  hub_id,
            "source_lat":       lat,
            "source_lng":       lng,
            "target_lat":       hub_lat,
            "target_lng":       hub_lng,
            "relationship_type":"causal",
            "strength":         severity,
            "category":         cat,
        })

    return events, arcs


# ── Groq reasoning ────────────────────────────────────────────────────────────

def _fmt_headline(a: dict) -> str:
    pub = (a.get("publisher") or {})
    pub_name = pub.get("name", "") if isinstance(pub, dict) else str(pub)
    return f"• {a.get('title','')[:115]}  [{pub_name}]"


async def _generate_reasoning(
    ticker: str,
    date_str: str,
    gap_pct: float,
    articles: list,
    ctx: dict,
) -> str:
    """
    Generate Groq reasoning grounded in article context.

    Articles are split into two tiers:
      - Ticker-specific: directly about this company (from yfinance or RAG high-score)
      - Market context:  broader India/sector news (lower relevance)

    The prompt explicitly instructs the LLM NOT to fabricate connections to
    unrelated companies (Nvidia, Tesla etc.) unless causal links exist.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    if not GROQ_API_KEY:
        return "Add GROQ_API_KEY to your .env file to enable AI reasoning."

    clean = ticker.replace(".NS", "").replace(".BO", "").upper()

    # ── Split articles by relevance tier ──────────────────────────────────────
    # Tier 1: explicitly fetched for this ticker, OR mentions ticker name in title
    ticker_articles = [
        a for a in articles
        if a.get("_ticker_specific")
        or clean.lower() in (a.get("title", "") or "").lower()
    ]
    # Tier 2: broader market / macro context (not directly about this stock)
    macro_articles = [
        a for a in articles
        if a not in ticker_articles
    ]

    ticker_block = (
        "\n".join(_fmt_headline(a) for a in ticker_articles[:6])
        or "No company-specific news found in database."
    )
    macro_block = (
        "\n".join(_fmt_headline(a) for a in macro_articles[:4])
        or "None."
    )

    # ── Trade context ─────────────────────────────────────────────────────────
    ctx_lines = [f"Stock: {clean} (NSE India)", f"Gap at open: +{gap_pct:.1f}% on {date_str}"]
    if ctx.get("nifty_ret") is not None:
        ctx_lines.append(f"Nifty 50 that day: {ctx['nifty_ret']:+.2f}%")
    if ctx.get("sector"):
        ctx_lines.append(f"Sector: {ctx['sector']}")
    if ctx.get("volume_ratio") is not None:
        ctx_lines.append(f"Volume: {ctx['volume_ratio']:.1f}× the 20-day average ({"high" if ctx['volume_ratio'] > 2 else "moderate"} conviction)")
    if ctx.get("sector_rank") is not None:
        ctx_lines.append(f"Sector rank: top {100 - int(ctx['sector_rank'] * 100)}% of peers")

    prompt = f"""You are a senior equity analyst explaining a gap-up trade to a retail investor.

TRADE DETAILS:
{chr(10).join(f"  {l}" for l in ctx_lines)}

COMPANY-SPECIFIC NEWS (primary source — these are about {clean} directly):
{ticker_block}

BROADER MARKET CONTEXT (secondary — only reference if it directly explains {clean}'s move):
{macro_block}

TASK: In 2–3 sentences, explain WHY {clean} gapped up +{gap_pct:.1f}% at open on {date_str}.

RULES (follow strictly):
1. If company-specific news exists → lead with that catalyst (earnings beat, analyst upgrade, deal, order win, etc.)
2. If no company news found → explain using sector momentum, Nifty direction, or technical breakout above key levels
3. NEVER invent connections to unrelated foreign companies (Nvidia, Tesla, Apple etc.) unless there is a clear, direct supply-chain or macro link to {clean}
4. Be specific and concise — no filler phrases like "amid positive market sentiment" or "on the back of strong buying"
5. If you genuinely cannot identify a catalyst, say so honestly and give the most likely technical reason"""

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e)
        logger.warning(f"Groq reasoning error for {clean}: {e}")
        # Parse common API errors into human-readable messages
        if "401" in err or "invalid_api_key" in err.lower():
            return "Invalid Groq API key — check GROQ_API_KEY in your .env file."
        if "429" in err or "rate_limit" in err.lower():
            return "Groq rate limit reached. Wait a moment and try again."
        if "503" in err or "service_unavailable" in err.lower():
            return "Groq service temporarily unavailable. Try again shortly."
        return f"AI reasoning unavailable ({type(e).__name__}). Check backend logs for details."


# ── Public entry point ────────────────────────────────────────────────────────

async def get_trade_intel(
    ticker: str,
    date_str: str,
    gap_pct: float         = 0.0,
    nifty_ret: Optional[float]  = None,
    sector: Optional[str]       = None,
    volume_ratio: Optional[float] = None,
    sector_rank: Optional[float]  = None,
) -> dict:
    """
    Full trade intelligence for one trade.
    RAG-first: retrieves articles from ChromaDB vector store.
    Falls back to Polygon.io / yfinance if RAG store is empty or sparse.
    Fresh articles are ingested back into RAG for future queries.
    """
    from backend.services.rag_engine import RAGEngine
    rag = RAGEngine.get()

    # ── 1. Always fetch yfinance for the specific ticker ──────────────────────
    # yfinance returns company-specific news (earnings, upgrades, results etc.)
    # This is the PRIMARY source — especially critical for Indian stocks where
    # the RAG store holds mostly global macro news (SPY/QQQ/AAPL tickers).
    articles = _fetch_yfinance(ticker)
    logger.info(f"yfinance primary: {len(articles)} articles for {ticker}")

    # Ingest ticker-specific articles into RAG for future queries
    if articles:
        rag.ingest(articles)

    # ── 2. Try Polygon for dated window (good for US tickers / recent events) ─
    if not articles:
        poly = await _fetch_polygon(ticker, date_str)
        if poly:
            rag.ingest(poly)
            articles = poly

    # ── 3. RAG supplement — only high-relevance articles (min_score=0.18) ─────
    # min_score filters out generic "Nvidia earnings" / "Tesla stock" articles
    # that have superficial semantic similarity but zero relevance to the trade.
    seen_urls = {a.get("article_url", "") for a in articles}
    rag_articles = rag.retrieve_for_trade(ticker, date_str, n=8, min_score=0.18)
    added_from_rag = 0
    for a in rag_articles:
        url = a.get("article_url", "")
        if url and url not in seen_urls:
            articles.append(a)
            seen_urls.add(url)
            added_from_rag += 1
    logger.info(f"RAG added {added_from_rag} additional relevant articles (filtered by score >= 0.18)")

    # ── 3. Globe data ─────────────────────────────────────────────────────────
    events, arcs = _build_globe_data(articles, ticker, date_str)

    # ── 4. Technical context ──────────────────────────────────────────────────
    ctx = {
        "nifty_ret":    nifty_ret,
        "sector":       sector or "Unknown",
        "volume_ratio": volume_ratio,
        "sector_rank":  sector_rank,
    }

    # ── 5. Groq reasoning (uses RAG-sourced articles as context) ─────────────
    reasoning = await _generate_reasoning(ticker, date_str, gap_pct, articles, ctx)

    # ── 6. Globe focus point (exchange hub) ───────────────────────────────────
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    focus_lat = 19.0760 if is_indian else 40.7128
    focus_lng = 72.8777 if is_indian else -74.0060

    return {
        "ticker":       ticker,
        "date":         date_str,
        "gap_pct":      gap_pct,
        "reasoning":    reasoning,
        "events":       events,
        "arcs":         arcs,
        "focus_lat":    focus_lat,
        "focus_lng":    focus_lng,
        "n_articles":   len(articles),
        "rag_count":    rag.count,
    }
