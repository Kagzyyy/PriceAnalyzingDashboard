"""
God's Eye View — Macro Prediction Engine.

Fetches the past 30 days of major financial headlines, then uses Groq LLaMA
to generate forward-looking macro predictions with:
  • lat/lng origin of each catalyst
  • Confidence score (0-1)
  • Direction (bullish / bearish / neutral / risk)
  • Affected markets
  • Sector outlook
  • Key risks
  • India-specific impact

Returns GeoEvent pins + GlobeArcs for react-globe.gl.
"""
import os
import json
import logging
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

# Broad tickers that pull diverse global macro news
PULSE_TICKERS = [
    "SPY", "QQQ", "GLD", "USO", "TLT",
    "AAPL", "NVDA", "MSFT", "AMZN", "JPM",
]

# Direction → hex color (for globe pin coloring)
DIRECTION_COLORS: dict[str, str] = {
    "bullish":  "#00e676",
    "bearish":  "#ff1744",
    "neutral":  "#00e5ff",
    "risk":     "#ffd740",
}

PUBLISHER_GEO: dict[str, tuple[float, float]] = {
    "Bloomberg":                (40.7128,  -74.0060),
    "Reuters":                  (40.7128,  -74.0060),
    "CNBC":                     (40.7592,  -73.9851),
    "Wall Street Journal":      (40.7128,  -74.0060),
    "Financial Times":          (51.5074,   -0.1278),
    "MarketWatch":              (40.7128,  -74.0060),
    "Seeking Alpha":            (37.7749, -122.4194),
    "Yahoo Finance":            (37.3382, -121.8863),
    "Benzinga":                 (42.3314,  -83.0458),
    "Motley Fool":              (38.9072,  -77.0369),
    "Zacks":                    (41.8781,  -87.6298),
    "Barron's":                 (40.7128,  -74.0060),
    "Economic Times":           (19.0760,   72.8777),
    "Business Standard":        (28.6139,   77.2090),
    "Moneycontrol":             (19.0760,   72.8777),
    "Livemint":                 (28.6139,   77.2090),
    "South China Morning Post": (22.3193,  114.1694),
    "Nikkei":                   (35.6762,  139.6503),
    "Guardian":                 (51.5074,   -0.1278),
    "AP":                       (40.7128,  -74.0060),
    "BBC":                      (51.5074,   -0.1278),
}
_DEFAULT_GEO = (40.7128, -74.0060)


def _publisher_coords(name: str) -> tuple[float, float]:
    nl = name.lower()
    for key, coords in PUBLISHER_GEO.items():
        if key.lower() in nl or nl in key.lower():
            return coords
    return _DEFAULT_GEO


def _empty_predictions() -> dict:
    return {
        "macro_regime":      "neutral",
        "regime_confidence": 0.5,
        "regime_reasoning":  "Insufficient data for analysis.",
        "predictions":       [],
        "sector_outlook":    {},
        "key_risks":         [],
        "india_impact":      "Add GROQ_API_KEY for AI-powered predictions.",
    }


# ── News fetching ──────────────────────────────────────────────────────────────

async def _fetch_polygon_news(days_back: int = 30) -> list:
    """Fetch ~30 days of global market news from Polygon.io."""
    if not POLYGON_API_KEY:
        return []
    seen, articles = set(), []
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")

    async with httpx.AsyncClient(timeout=12.0) as client:
        for ticker in PULSE_TICKERS:
            try:
                resp = await client.get(
                    "https://api.polygon.io/v2/reference/news",
                    params={
                        "ticker":              ticker,
                        "published_utc.gte":   since,
                        "order":               "desc",
                        "limit":               10,
                        "apiKey":              POLYGON_API_KEY,
                    },
                )
                if resp.status_code == 200:
                    for item in resp.json().get("results", []):
                        url = item.get("article_url", item.get("id", ""))
                        if url not in seen:
                            seen.add(url)
                            articles.append(item)
            except Exception as e:
                logger.warning(f"Polygon news error for {ticker}: {e}")

    logger.info(f"Market pulse: {len(articles)} articles fetched ({days_back}d)")
    return articles[:50]


def _fetch_yfinance_news() -> list:
    """Fallback: yfinance news for major indices."""
    try:
        import yfinance as yf
        tickers = ["^GSPC", "^IXIC", "GC=F", "CL=F", "^TNX", "^NSEI"]
        seen, out = set(), []
        for t in tickers:
            try:
                for n in (yf.Ticker(t).news or [])[:6]:
                    url = n.get("link", "")
                    ts  = n.get("providerPublishTime", 0)
                    if url not in seen and ts > 0:
                        seen.add(url)
                        out.append({
                            "title":         n.get("title", ""),
                            "published_utc": datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "article_url":   url,
                            "publisher":     {"name": n.get("publisher", "Yahoo Finance")},
                            "description":   "",
                            "insights":      [],
                        })
            except Exception:
                pass
        return out[:30]
    except Exception as e:
        logger.warning(f"yfinance pulse fallback error: {e}")
        return []


# ── Groq prediction engine ─────────────────────────────────────────────────────

async def _groq_predictions(articles: list) -> dict:
    """
    Single Groq call: analyze 30-day headlines → forward predictions with
    lat/lng, confidence, direction, affected markets, sector outlook, key risks.
    """
    if not GROQ_API_KEY or not articles:
        return _empty_predictions()

    try:
        from groq import Groq

        # Build headline block (up to 40 headlines)
        headlines_block = "\n".join(
            f"{i + 1}. {a.get('title', '')[:130]}"
            f"  [{(a.get('publisher') or {}).get('name', '') if isinstance(a.get('publisher'), dict) else ''}]"
            for i, a in enumerate(articles[:40])
        )

        current_month = datetime.utcnow().strftime("%B %Y")
        prompt = f"""You are a senior macro analyst at a top hedge fund. Below are the major global financial headlines from the past 30 days ({current_month}).

{headlines_block}

Analyze these headlines and generate forward-looking predictions for the next 2–6 weeks.

Return ONLY valid JSON with this exact structure:
{{
  "macro_regime": "risk_on|risk_off|neutral|transitional",
  "regime_confidence": 0.75,
  "regime_reasoning": "One sentence explanation of current regime",
  "predictions": [
    {{
      "title": "Short specific prediction — max 12 words",
      "detail": "1-2 sentences: specific catalyst and expected outcome",
      "confidence": 0.72,
      "timeframe": "2-4 weeks",
      "direction": "bullish|bearish|neutral|risk",
      "category": "monetary_policy|energy|geopolitical|trade|inflation|markets",
      "affected_markets": ["US equities", "Crude oil", "INR"],
      "lat": 38.9,
      "lng": -77.0,
      "evidence": "Most relevant supporting headline (max 15 words)"
    }}
  ],
  "sector_outlook": {{
    "Technology": {{"direction": "bullish", "reasoning": "brief one-liner"}},
    "Energy":     {{"direction": "neutral", "reasoning": "brief one-liner"}},
    "Banking":    {{"direction": "bearish", "reasoning": "brief one-liner"}},
    "Metals":     {{"direction": "bullish", "reasoning": "brief one-liner"}},
    "Pharma":     {{"direction": "neutral", "reasoning": "brief one-liner"}},
    "IT":         {{"direction": "bullish", "reasoning": "brief one-liner"}}
  }},
  "key_risks": [
    "Specific risk event 1 with timing",
    "Specific risk event 2 with timing",
    "Specific risk event 3 with timing",
    "Specific risk event 4 with timing"
  ],
  "india_impact": "2-3 sentences on what these global trends specifically mean for Nifty 50, INR, and Indian sectors."
}}

Rules:
- Generate 5–8 predictions. Be specific — not generic.
- lat/lng: use the city/country where the primary catalyst originates.
  Examples: Fed decision → Washington DC (38.9, -77.0), OPEC → Riyadh (24.7, 46.7),
  China data → Beijing (39.9, 116.4), ECB → Frankfurt (50.1, 8.7), India → Mumbai (19.1, 72.9).
- confidence: 0.45–0.90 range only. Don't use 1.0 or round numbers like 0.70.
- direction: 'risk' means elevated uncertainty / tail risk, not directional."""

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2200,
            temperature=0.25,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        result = json.loads(raw)
        logger.info(f"Groq predictions: {len(result.get('predictions', []))} generated")
        return result
    except Exception as e:
        logger.warning(f"Groq prediction error: {e}")
        return _empty_predictions()


# ── Globe data builder ─────────────────────────────────────────────────────────

def _predictions_to_events(predictions: list) -> tuple[list, list]:
    """Convert prediction objects → GeoEvent pins + GlobeArcs for react-globe.gl."""
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Central intelligence hub pin (New York)
    hub_id = "macro_hub"
    events: list = [{
        "id":            hub_id,
        "title":         "Global Macro Intelligence Hub",
        "summary":       "Forward predictions based on 30-day global headline analysis",
        "published_at":  now_str,
        "category":      "markets",
        "lat":           40.7128,
        "lng":           -74.0060,
        "location_name": "New York",
        "severity":      1.0,
        "is_analyzed":   True,
        "source_name":   "Alpha Edge",
        "direction":     "neutral",
        "direction_color": "#00e5ff",
    }]
    arcs: list = []

    for i, pred in enumerate(predictions):
        ev_id    = f"pred_{i}"
        direction = pred.get("direction", "neutral")
        confidence = float(pred.get("confidence", 0.5))
        lat = float(pred.get("lat", 40.7128))
        lng = float(pred.get("lng", -74.0060))

        events.append({
            "id":             ev_id,
            "title":          pred.get("title", ""),
            "summary":        pred.get("detail", ""),
            "detail":         pred.get("detail", ""),
            "published_at":   now_str,
            "category":       pred.get("category", "markets"),
            "lat":            lat,
            "lng":            lng,
            "location_name":  pred.get("category", ""),
            "severity":       confidence,
            "is_analyzed":    True,
            "source_name":    "Groq / LLaMA 3.3",
            "direction":      direction,
            "direction_color": DIRECTION_COLORS.get(direction, "#00e5ff"),
            "timeframe":      pred.get("timeframe", ""),
            "affected_markets": pred.get("affected_markets", []),
            "evidence":       pred.get("evidence", ""),
        })

        arcs.append({
            "id":                f"arc_{i}",
            "source_event_id":   ev_id,
            "target_event_id":   hub_id,
            "source_lat":        lat,
            "source_lng":        lng,
            "target_lat":        40.7128,
            "target_lng":        -74.0060,
            "relationship_type": "causal",
            "strength":          confidence,
            "category":          pred.get("category", "markets"),
            "direction":         direction,
            "direction_color":   DIRECTION_COLORS.get(direction, "#00e5ff"),
        })

    return events, arcs


# ── Public entry point ─────────────────────────────────────────────────────────

async def get_market_pulse() -> dict:
    """Full macro prediction pipeline."""
    # Fetch last 30 days of global headlines
    articles = await _fetch_polygon_news(days_back=30)
    if not articles:
        articles = _fetch_yfinance_news()

    # ── Ingest into RAG store so trade intel can query it later ───────────────
    if articles:
        try:
            from backend.services.rag_engine import RAGEngine
            n_rag = RAGEngine.get().ingest(articles)
            logger.info(f"Market pulse: ingested {n_rag} articles into RAG store")
        except Exception as _rag_err:
            logger.warning(f"RAG ingest skipped: {_rag_err}")

    # Generate forward predictions via Groq
    analysis  = await _groq_predictions(articles)
    predictions = analysis.get("predictions", [])

    # Build globe data from predictions
    events, arcs = _predictions_to_events(predictions)

    return {
        "macro_regime":      analysis.get("macro_regime",      "neutral"),
        "regime_confidence": analysis.get("regime_confidence", 0.5),
        "regime_reasoning":  analysis.get("regime_reasoning",  ""),
        "predictions":       predictions,
        "sector_outlook":    analysis.get("sector_outlook",    {}),
        "key_risks":         analysis.get("key_risks",         []),
        "india_impact":      analysis.get("india_impact",      ""),
        "events":            events,
        "arcs":              arcs,
        "generated_at":      datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "n_articles":        len(articles),
    }
