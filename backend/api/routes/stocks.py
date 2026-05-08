from fastapi import APIRouter, HTTPException, Query
from backend.data.storage import get_latest_predictions, get_ohlcv
from backend.data.symbols import get_symbol_map

router = APIRouter()


@router.get("/stocks/{symbol}")
def get_stock_detail(symbol: str, days: int = Query(180, ge=30, le=730)):
    ticker = f"{symbol.upper()}.NS"
    # Use full symbol map so any stock in the DB can be looked up by the user.
    symbol_map = get_symbol_map(universe="all")
    info = symbol_map.get(ticker, {})

    # Prediction
    preds = get_latest_predictions()
    pred_row = preds[preds["ticker"] == ticker]

    if pred_row.empty:
        raise HTTPException(status_code=404, detail=f"No data for symbol {symbol}")

    row = pred_row.iloc[0]
    prob_up = float(row.get("prob_up", 0.5))
    direction = str(row.get("direction", "NEUTRAL"))
    confidence_pct = round(prob_up * 100 if direction == "UP" else (1 - prob_up) * 100, 1)

    # OHLCV for chart
    ohlcv_df = get_ohlcv(ticker=ticker, days=days)
    ohlcv = []
    if not ohlcv_df.empty:
        ohlcv = [
            {
                "date": str(r["date"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["volume"]),
            }
            for _, r in ohlcv_df.iterrows()
        ]

    return {
        "ticker": ticker,
        "symbol": symbol.upper(),
        "name": info.get("name", symbol.upper()),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "prob_up": prob_up,
        "prob_down": float(row.get("prob_down", 1 - prob_up)),
        "direction": direction,
        "confidence_pct": confidence_pct,
        "price": float(row.get("price", 0)),
        "change_pct": float(row.get("change_pct", 0)),
        "rsi": float(row.get("rsi", 0)) if row.get("rsi") else None,
        "macd": float(row.get("macd", 0)) if row.get("macd") else None,
        "bb_pct": float(row.get("bb_pct", 0)) if row.get("bb_pct") else None,
        "volume_ratio": float(row.get("volume_ratio", 1)) if row.get("volume_ratio") else None,
        "ohlcv": ohlcv,
    }


@router.get("/stocks")
def search_stocks(q: str = Query("", min_length=0)):
    # Search over full list so users can find any stock, not just nifty50.
    symbol_map = get_symbol_map(universe="all")
    q = q.lower()
    results = []
    for ticker, info in symbol_map.items():
        sym = ticker.replace(".NS", "")
        if q in sym.lower() or q in info.get("name", "").lower() or q in info.get("sector", "").lower():
            results.append({
                "ticker": ticker,
                "symbol": sym,
                "name": info.get("name", sym),
                "sector": info.get("sector", "Unknown"),
            })
    return {"data": results[:50]}
