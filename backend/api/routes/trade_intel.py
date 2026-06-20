from fastapi import APIRouter, Query
from typing import Optional
from backend.services.trade_intel import get_trade_intel

router = APIRouter()


@router.get("/trade-intel/{ticker}/{date_str}")
async def trade_intel(
    ticker:       str,
    date_str:     str,
    gap_pct:      float          = Query(0.0,  description="Gap % at open"),
    nifty_ret:    Optional[float]= Query(None, description="Nifty 50 return that day (%)"),
    sector:       Optional[str]  = Query(None, description="Stock sector"),
    volume_ratio: Optional[float]= Query(None, description="Volume vs 20-day avg"),
    sector_rank:  Optional[float]= Query(None, description="Sector momentum rank 0-1"),
):
    """
    Trade intelligence for a single gap-up trade.
    Returns AI reasoning + globe GeoEvents + GlobeArcs explaining
    why the stock gapped on the given date.
    """
    return await get_trade_intel(
        ticker=ticker,
        date_str=date_str,
        gap_pct=gap_pct,
        nifty_ret=nifty_ret,
        sector=sector,
        volume_ratio=volume_ratio,
        sector_rank=sector_rank,
    )
