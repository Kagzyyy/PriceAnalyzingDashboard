from fastapi import APIRouter, Query, BackgroundTasks
from backend.ml.gap_and_go import run_gap_and_go_backtest
from backend.ml.gap_trainer import train_gap_model, GAP_MODEL_PATH

router = APIRouter()


@router.post("/train-gap-model")
def train_gap_model_endpoint(background_tasks: BackgroundTasks):
    """Train the gap-continuation XGBoost model (Tier 3). Runs in background ~60s."""
    if GAP_MODEL_PATH.exists():
        background_tasks.add_task(train_gap_model)
        return {"status": "retraining", "message": "Gap model retraining in background."}
    background_tasks.add_task(train_gap_model)
    return {"status": "training", "message": "Gap model training started in background (~60s)."}


@router.get("/gap-backtest")
def gap_backtest(
    gap_threshold:     float = Query(1.5,       ge=0.5,  le=5.0,   description="Minimum gap-up % to consider a trade"),
    gap_max_pct:       float = Query(4.0,       ge=2.0,  le=20.0,  description="Cap — skip gaps above this % (large gaps tend to fade)"),
    target_pct:        float = Query(2.0,       ge=0.5,  le=10.0,  description="Profit target %"),
    stop_pct:          float = Query(0.5,       ge=0.1,  le=5.0,   description="Stop loss %"),
    ml_threshold:      float = Query(0.50,      ge=0.50, le=0.90,  description="XGBoost prob_up threshold (0.50=binary UP filter, optimal)"),
    sent_threshold:    float = Query(-0.10,     ge=-1.0, le=0.5,   description="FinBERT sentiment threshold"),
    max_positions:     int   = Query(5,         ge=1,    le=20,    description="Max simultaneous trades per day"),
    volume_min_ratio:  float = Query(1.5,       ge=0.5,  le=5.0,   description="Min volume ratio vs 20-day avg (institutional confirmation)"),
    rsi_low:           float = Query(35.0,      ge=10.0, le=50.0,  description="RSI floor — below = falling knife"),
    rsi_high:          float = Query(68.0,      ge=55.0, le=85.0,  description="RSI ceiling — above = overbought, gap fades"),
    nifty_min_ret:     float = Query(-0.3,      ge=-3.0, le=0.0,   description="Skip trade days when Nifty is down more than this %"),
    use_ml_confirm:    bool  = Query(True,                          description="XGBoost filter (prob_up > threshold)"),
    use_sentiment:     bool  = Query(True,                          description="FinBERT sentiment filter"),
    use_sector_filter: bool  = Query(True,                          description="Sector momentum rank filter"),
    use_volume_filter: bool  = Query(True,                          description="Volume confirmation filter (T1)"),
    use_rsi_filter:    bool  = Query(False,                         description="RSI range filter — disabled by default (over-filters momentum gaps)"),
    use_nifty_filter:  bool  = Query(True,                          description="Skip down-market days (T1)"),
    use_trend_filter:   bool  = Query(True,   description="Stock must be above EMA20 (T2)"),
    use_gap_ml_confirm: bool  = Query(False,  description="Gap-continuation XGBoost (T3) — train model first via POST /train-gap-model"),
    gap_ml_threshold:   float = Query(0.50,   ge=0.40, le=0.90, description="Gap ML probability threshold"),
    use_walkforward:    bool  = Query(False,  description="Walk-forward validation — retrain per year, OOS predictions only (2023-2025). Honest Sharpe/WR."),
    slippage_pct:       float = Query(0.0,    ge=0.0, le=1.0, description="One-way execution slippage % at open (0.10 = 0.10%). Models fill friction on gap stocks."),
    capital:            float = Query(100_000.0, ge=1000.0, description="Initial capital in INR"),
):
    return run_gap_and_go_backtest(
        gap_threshold=gap_threshold,
        gap_max_pct=gap_max_pct,
        target_pct=target_pct,
        stop_pct=stop_pct,
        ml_threshold=ml_threshold,
        sent_threshold=sent_threshold,
        max_positions=max_positions,
        volume_min_ratio=volume_min_ratio,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        nifty_min_ret=nifty_min_ret,
        initial_capital=capital,
        use_ml_confirm=use_ml_confirm,
        use_sentiment=use_sentiment,
        use_sector_filter=use_sector_filter,
        use_volume_filter=use_volume_filter,
        use_rsi_filter=use_rsi_filter,
        use_nifty_filter=use_nifty_filter,
        use_trend_filter=use_trend_filter,
        use_gap_ml_confirm=use_gap_ml_confirm,
        gap_ml_threshold=gap_ml_threshold,
        use_walkforward=use_walkforward,
        slippage_pct=slippage_pct,
    )
