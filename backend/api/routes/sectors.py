from fastapi import APIRouter
from backend.data.storage import get_sector_summary, get_latest_predictions
from backend.data.symbols import get_symbol_map

router = APIRouter()


@router.get("/sectors")
def get_sectors():
    symbol_map = get_symbol_map()
    summary = get_sector_summary(symbol_map)
    if summary.empty:
        return {"data": []}
    return {"data": summary.fillna(0).to_dict(orient="records")}


@router.get("/sectors/{sector}/stocks")
def get_sector_stocks(sector: str):
    """Get all predictions for stocks in a given sector."""
    symbol_map = get_symbol_map()
    preds = get_latest_predictions()

    if preds.empty:
        return {"data": []}

    preds["sector"] = preds["ticker"].map(
        lambda t: symbol_map.get(t, {}).get("sector", "Unknown")
    )
    preds["symbol"] = preds["ticker"].str.replace(".NS", "", regex=False)
    preds["name"] = preds["ticker"].map(lambda t: symbol_map.get(t, {}).get("name", t))
    preds["confidence_pct"] = (preds["prob_up"] * 100).round(1)

    filtered = preds[preds["sector"].str.lower() == sector.lower()]
    filtered = filtered.sort_values("prob_up", ascending=False)

    return {
        "sector": sector,
        "count": len(filtered),
        "data": filtered.fillna(0).to_dict(orient="records"),
    }
