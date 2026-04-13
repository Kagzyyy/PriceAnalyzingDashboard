"""
Long/short ranking engine.
Filters and ranks predictions by confidence and liquidity.
"""
import pandas as pd
from backend.config import MIN_VOLUME, MIN_PRICE, TOP_N
from backend.data.symbols import get_symbol_map


def rank_predictions(predictions: pd.DataFrame) -> dict:
    """
    Takes the latest predictions DataFrame.
    Returns dict with 'longs' and 'shorts' ranked lists.
    """
    if predictions.empty:
        return {"longs": [], "shorts": []}

    symbol_map = get_symbol_map()
    df = predictions.copy()

    # Enrich with name/sector
    df["symbol"] = df["ticker"].str.replace(".NS", "", regex=False)
    df["name"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("name", t))
    df["sector"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("sector", "Unknown"))
    df["industry"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("industry", "Unknown"))

    # Liquidity filters
    df = df[df["price"].fillna(0) >= MIN_PRICE]
    df = df[df["volume"].fillna(0) >= MIN_VOLUME]

    # --- LONGS: top N by prob_up (percentile ranking — no fixed threshold) ---
    # Cross-sectional model outputs probabilities near 0.50; fixed thresholds
    # like >0.60 would return zero stocks. Rank and take top N instead.
    longs_df = df.nlargest(TOP_N, "prob_up").copy()
    longs_df["rank"] = range(1, len(longs_df) + 1)
    longs_df["confidence_pct"] = (longs_df["prob_up"] * 100).round(1)

    # --- SHORTS: bottom N by prob_up ---
    shorts_df = df.nsmallest(TOP_N, "prob_up").copy()
    shorts_df["rank"] = range(1, len(shorts_df) + 1)
    shorts_df["confidence_pct"] = (shorts_df["prob_down"] * 100).round(1)

    return {
        "longs": _to_records(longs_df),
        "shorts": _to_records(shorts_df),
    }


def _to_records(df: pd.DataFrame) -> list[dict]:
    display_cols = [
        "rank", "ticker", "symbol", "name", "sector", "industry",
        "prob_up", "prob_down", "confidence_pct", "direction",
        "price", "change_pct", "volume", "rsi", "macd", "bb_pct", "volume_ratio",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    return df[existing_cols].fillna(0).to_dict(orient="records")


def get_market_breadth(predictions: pd.DataFrame) -> dict:
    """Overall market statistics from predictions."""
    if predictions.empty:
        return {}

    total = len(predictions)
    avg_prob_up = predictions["prob_up"].mean()
    # Use percentile-based breadth: top/bottom 20% as longs/shorts
    pct_rank = predictions["prob_up"].rank(pct=True)
    longs = int((pct_rank >= 0.80).sum())
    shorts = int((pct_rank <= 0.20).sum())
    neutral = total - longs - shorts

    return {
        "total_stocks": total,
        "longs": longs,
        "shorts": shorts,
        "neutral": neutral,
        "avg_prob_up": round(avg_prob_up, 3),
        "market_sentiment": "BULLISH" if avg_prob_up > 0.53 else "BEARISH" if avg_prob_up < 0.47 else "NEUTRAL",
    }
