from fastapi import APIRouter, Query
from backend.ml.pairs_backtester import run_pairs_backtest

router = APIRouter()


@router.get("/pairs-backtest")
def pairs_backtest(
    z_entry:           float = Query(2.0,       ge=1.0,   le=4.0,   description="Z-score entry threshold"),
    z_exit:            float = Query(0.5,       ge=0.0,   le=1.5,   description="Z-score exit threshold (mean reversion)"),
    z_stop:            float = Query(3.0,       ge=2.0,   le=6.0,   description="Z-score stop loss threshold"),
    max_pairs:         int   = Query(5,         ge=1,     le=20,    description="Max simultaneous pairs"),
    max_holding_days:  int   = Query(20,        ge=5,     le=60,    description="Max holding period (trading days)"),
    sector_only:       bool  = Query(True,                          description="Only trade same-sector pairs"),
    use_ml_confirm:    bool  = Query(True,                          description="Use XGBoost to confirm pair direction"),
    use_sentiment:     bool  = Query(True,                          description="Use FinBERT to filter fundamental divergence"),
    use_regime_filter: bool  = Query(True,                          description="Suppress trades in crisis regimes"),
    capital:           float = Query(100_000.0, ge=1000.0,          description="Initial capital in INR"),
):
    return run_pairs_backtest(
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
        max_pairs=max_pairs,
        max_holding_days=max_holding_days,
        initial_capital=capital,
        sector_only=sector_only,
        use_ml_confirm=use_ml_confirm,
        use_sentiment=use_sentiment,
        use_regime_filter=use_regime_filter,
    )
