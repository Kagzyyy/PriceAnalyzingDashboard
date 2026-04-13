"""
Builds ChromaDB documents from daily prediction data.
Runs after each data refresh.
"""
import logging
import pandas as pd
from datetime import date
from backend.data.storage import get_latest_predictions, get_sector_summary
from backend.data.symbols import get_symbol_map
from backend.rag.embeddings import upsert_documents

logger = logging.getLogger(__name__)


def _clean(v):
    """Convert numpy/pandas types to plain Python types for ChromaDB."""
    import math
    if hasattr(v, 'item'):
        v = v.item()
    # Force to native Python float/int to avoid float32/int64 rejection
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return float(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        return str(v)
    # Fallback: try converting to float, then str
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return str(v)


def _clean_meta(d: dict) -> dict:
    """Apply _clean() to every value in a metadata dict."""
    return {k: _clean(v) for k, v in d.items()}


def build_stock_documents(predictions: pd.DataFrame, symbol_map: dict) -> list[dict]:
    """Create one text document per stock for RAG indexing."""
    docs = []
    today = str(date.today())

    for _, row in predictions.iterrows():
        ticker = row.get("ticker", "")
        info = symbol_map.get(ticker, {})
        symbol = ticker.replace(".NS", "")
        name = info.get("name", symbol)
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        prob_up = _clean(float(row.get("prob_up", 0.5)))
        direction = str(row.get("direction", "NEUTRAL"))
        price = _clean(float(row.get("price", 0) or 0))
        change_pct = _clean(float(row.get("change_pct", 0) or 0))
        _vol = row.get("volume", 0)
        volume = int(_clean(float(_vol if _vol == _vol else 0)))  # NaN-safe int
        rsi = _clean(float(row.get("rsi", 0) or 0))
        macd = _clean(float(row.get("macd", 0) or 0))
        bb_pct = _clean(float(row.get("bb_pct", 0.5) or 0.5))
        volume_ratio = _clean(float(row.get("volume_ratio", 1) or 1))

        confidence = round(prob_up * 100 if direction == "UP" else (1 - prob_up) * 100, 1)

        # Textual signal summaries
        rsi_desc = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        vol_desc = f"{volume_ratio:.1f}x average" if volume_ratio > 0 else "normal"
        macd_desc = "bullish crossover" if macd > 0 else "bearish crossover"
        change_desc = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"

        text = (
            f"{name} ({symbol}) in the {sector} sector ({industry}): "
            f"Model predicts {direction} direction with {confidence}% confidence. "
            f"Current price ₹{price:.2f}, today's change {change_desc}. "
            f"RSI is {rsi:.1f} ({rsi_desc}), MACD is {macd_desc}, "
            f"Bollinger Band %B is {bb_pct:.2f}, volume is {vol_desc}. "
            f"{'Strong buy candidate.' if prob_up > 0.75 else 'Strong sell candidate.' if prob_up < 0.25 else ''}"
        )

        docs.append({
            "id": f"stock_{symbol}_{today}",
            "text": text,
            "metadata": _clean_meta({
                "type": "stock",
                "symbol": symbol,
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "direction": direction,
                "prob_up": prob_up,
                "confidence": confidence,
                "price": price,
                "change_pct": change_pct,
                "date": today,
            }),
        })

    return docs


def build_sector_documents(sector_summary: pd.DataFrame) -> list[dict]:
    """Create one text document per sector."""
    docs = []
    today = str(date.today())

    for _, row in sector_summary.iterrows():
        sector = row.get("sector", "Unknown")
        bullish_pct = float(row.get("bullish_pct", 50))
        longs = int(row.get("longs", 0))
        shorts = int(row.get("shorts", 0))
        count = int(row.get("count", 0))
        avg_prob = float(row.get("avg_prob_up", 0.5))

        sentiment = "strongly bullish" if avg_prob > 0.65 else \
                    "bullish" if avg_prob > 0.55 else \
                    "bearish" if avg_prob < 0.45 else \
                    "strongly bearish" if avg_prob < 0.35 else "neutral"

        text = (
            f"{sector} sector analysis for {today}: "
            f"The sector is {sentiment} with an average bullish confidence of {bullish_pct}%. "
            f"Out of {count} stocks analyzed, {longs} are predicted to go UP and {shorts} predicted to go DOWN. "
            f"{'This sector shows the most upward momentum today.' if avg_prob > 0.65 else ''}"
            f"{'This sector shows significant downward pressure today.' if avg_prob < 0.35 else ''}"
        )

        docs.append({
            "id": f"sector_{sector.lower().replace(' ', '_')}_{today}",
            "text": text,
            "metadata": _clean_meta({
                "type": "sector",
                "sector": sector,
                "bullish_pct": bullish_pct,
                "longs": longs,
                "shorts": shorts,
                "count": count,
                "avg_prob_up": avg_prob,
                "date": today,
            }),
        })

    return docs


def build_market_summary_document(predictions: pd.DataFrame, sector_summary: pd.DataFrame) -> list[dict]:
    """Create a single market-wide summary document."""
    today = str(date.today())
    total = len(predictions)
    if total == 0:
        return []

    longs = len(predictions[predictions["direction"] == "UP"])
    shorts = len(predictions[predictions["direction"] == "DOWN"])
    avg_prob = predictions["prob_up"].mean()

    top_sectors = sector_summary.head(3)["sector"].tolist() if not sector_summary.empty else []
    worst_sectors = sector_summary.tail(3)["sector"].tolist() if not sector_summary.empty else []

    sentiment = "BULLISH" if avg_prob > 0.53 else "BEARISH" if avg_prob < 0.47 else "NEUTRAL"

    text = (
        f"Market overview for {today}: "
        f"Overall market sentiment is {sentiment}. "
        f"Of {total} NSE stocks analyzed, {longs} are predicted UP and {shorts} predicted DOWN. "
        f"Market breadth (advance/decline): {longs}/{shorts}. "
        f"Top performing sectors today: {', '.join(top_sectors)}. "
        f"Underperforming sectors today: {', '.join(worst_sectors)}. "
        f"Average bullish probability across all stocks: {avg_prob*100:.1f}%."
    )

    return [{
        "id": f"market_overview_{today}",
        "text": text,
        "metadata": _clean_meta({
            "type": "market_overview",
            "date": today,
            "sentiment": sentiment,
            "total_stocks": total,
            "longs": longs,
            "shorts": shorts,
            "avg_prob_up": avg_prob,
        }),
    }]


def run_indexing() -> int:
    """Full indexing pipeline. Returns number of documents indexed."""
    logger.info("Starting RAG indexing...")
    symbol_map = get_symbol_map()
    predictions = get_latest_predictions()

    if predictions.empty:
        logger.warning("No predictions to index")
        return 0

    sector_summary = get_sector_summary(symbol_map)

    stock_docs = build_stock_documents(predictions, symbol_map)
    sector_docs = build_sector_documents(sector_summary)
    market_docs = build_market_summary_document(predictions, sector_summary)

    all_docs = stock_docs + sector_docs + market_docs
    upsert_documents(all_docs)

    logger.info(f"Indexed {len(all_docs)} documents ({len(stock_docs)} stocks, {len(sector_docs)} sectors, 1 market)")
    return len(all_docs)
