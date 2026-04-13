from fastapi import APIRouter, Query
from backend.data.storage import get_latest_predictions
from backend.data.symbols import get_symbol_map
from backend.ml.ranking import rank_predictions, get_market_breadth

router = APIRouter()


def _enrich(df, symbol_map):
    df = df.copy()
    df["symbol"] = df["ticker"].str.replace(".NS", "", regex=False)
    df["name"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("name", t))
    df["sector"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("sector", "Unknown"))
    df["industry"] = df["ticker"].map(lambda t: symbol_map.get(t, {}).get("industry", "Unknown"))
    df["confidence_pct"] = (df["prob_up"] * 100).round(1)
    df["volume"] = df["volume"].fillna(0).astype(int)
    return df


@router.get("/predictions")
def get_predictions(
    sector: str = Query(None),
    direction: str = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
):
    preds = get_latest_predictions()
    symbol_map = get_symbol_map()
    preds = _enrich(preds, symbol_map)

    if sector:
        preds = preds[preds["sector"].str.lower() == sector.lower()]
    if direction:
        preds = preds[preds["direction"].str.upper() == direction.upper()]

    total = len(preds)
    start = (page - 1) * page_size
    preds = preds.iloc[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": preds.fillna(0).to_dict(orient="records"),
    }


@router.get("/rankings/longs")
def get_longs():
    preds = get_latest_predictions()
    rankings = rank_predictions(preds)
    return {"data": rankings["longs"], "count": len(rankings["longs"])}


@router.get("/rankings/shorts")
def get_shorts():
    preds = get_latest_predictions()
    rankings = rank_predictions(preds)
    return {"data": rankings["shorts"], "count": len(rankings["shorts"])}


@router.get("/breadth")
def get_breadth():
    preds = get_latest_predictions()
    return get_market_breadth(preds)
