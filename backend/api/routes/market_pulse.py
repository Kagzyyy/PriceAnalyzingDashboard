from fastapi import APIRouter
from backend.services.market_pulse import get_market_pulse

router = APIRouter()

@router.get("/market-pulse")
async def market_pulse():
    """
    Today's global macro headlines with Groq sentiment analysis,
    next-week outlook, watchlist, and globe data.
    """
    return await get_market_pulse()
