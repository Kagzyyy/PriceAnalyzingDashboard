"""
Google News RSS scraper for NSE stocks.
Fetches 14 days of headlines per stock — no API key required.
Uses primary + fallback query for better coverage.
"""
import time
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_GNEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
_DELAY = 0.2   # seconds between requests — polite crawling


def _parse_date(entry) -> datetime:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(entry.published)
        return dt.replace(tzinfo=None)
    except Exception:
        return datetime.now()


def _fetch_feed(query: str, cutoff: datetime) -> list[dict]:
    try:
        import feedparser
    except ImportError:
        return []

    url = _GNEWS_URL.format(query=query)
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:30]:
            pub = _parse_date(entry)
            if pub < cutoff:
                continue
            title = getattr(entry, "title", "") or ""
            if title:
                results.append({"headline": title[:256], "pub_date": pub.date()})
        return results
    except Exception:
        return []


def fetch_stock_headlines(ticker: str, company_name: str, symbol: str, days: int = 14) -> list[dict]:
    """
    Fetch headlines for a stock using primary + fallback queries.
    Extends to 14-day window per research recommendation.
    """
    try:
        import feedparser  # noqa — just checking install
    except ImportError:
        logger.warning("feedparser not installed — run: pip install feedparser")
        return []

    cutoff = datetime.now() - timedelta(days=days)
    headlines = []

    # Primary: company name
    primary_q = f'"{company_name}" NSE stock'.replace(" ", "+")
    rows = _fetch_feed(primary_q, cutoff)
    for r in rows:
        r["ticker"] = ticker
        headlines.append(r)

    # Fallback: symbol only (catches more results for less-covered stocks)
    if len(headlines) < 3:
        fallback_q = f'"{symbol}" NSE India stock'.replace(" ", "+")
        rows2 = _fetch_feed(fallback_q, cutoff)
        seen = {h["headline"] for h in headlines}
        for r in rows2:
            if r["headline"] not in seen:
                r["ticker"] = ticker
                headlines.append(r)

    return headlines


def scrape_all(symbol_map: dict, days: int = 14) -> pd.DataFrame:
    """
    Scrape Google News for every stock in symbol_map.
    symbol_map: { "HDFCBANK.NS": {"symbol": "HDFCBANK", "name": "HDFC Bank", ...} }
    Returns DataFrame: ticker, headline, pub_date
    """
    try:
        import feedparser  # noqa
    except ImportError:
        logger.warning("feedparser not installed — run: pip install feedparser")
        return pd.DataFrame(columns=["ticker", "headline", "pub_date"])

    all_rows = []
    items = list(symbol_map.items())
    logger.info(f"Scraping news for {len(items)} stocks ({days}-day window)...")

    for i, (yf_ticker, info) in enumerate(items):
        name   = info.get("name",   info.get("symbol", yf_ticker))
        symbol = info.get("symbol", yf_ticker.replace(".NS", ""))
        rows = fetch_stock_headlines(yf_ticker, name, symbol, days=days)
        all_rows.extend(rows)
        time.sleep(_DELAY)

        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(items)} scraped — {len(all_rows)} headlines so far")

    if not all_rows:
        logger.warning("No headlines scraped.")
        return pd.DataFrame(columns=["ticker", "headline", "pub_date"])

    df = pd.DataFrame(all_rows)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    logger.info(f"Scraped {len(df)} headlines for {df['ticker'].nunique()} stocks")
    return df
