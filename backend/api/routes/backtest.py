from fastapi import APIRouter, Query, HTTPException
from backend.ml.backtester import run_backtest
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/backtest")
def get_backtest(
    threshold:   float = Query(0.10, ge=0.01, le=0.50, description="Top/bottom percentile of stocks to trade (0.10 = top & bottom 10%)"),
    stop_loss:   float = Query(1.0,  ge=0.1,  le=5.0,  description="Stop loss % per trade"),
    take_profit: float = Query(1.5,  ge=0.1,  le=10.0, description="Take profit % per trade"),
    max_pos:     int   = Query(20,   ge=1,    le=50,   description="Max simultaneous positions"),
    capital:     float = Query(100000.0, ge=1000,      description="Starting capital (₹)"),
    direction:   str   = Query("both", description="Trade direction: long | short | both"),
):
    try:
        result = run_backtest(
            confidence_threshold=threshold,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_positions=max_pos,
            initial_capital=capital,
            trade_direction=direction,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")
