"""
Daily pipeline: fetch → features → predict → rank → index RAG.
Runs automatically at 8:45 AM IST via APScheduler.

Smart caching: each step checks whether its output is already fresh for today
and skips itself if so. Opening the app multiple times in one day costs nothing.
"""
import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# NOTE: heavy imports (xgboost, sentence_transformers, etc.) are intentionally
# kept INSIDE run_full_pipeline() so they don't slow down uvicorn startup.
from backend.data.storage import (get_last_prediction_date, OHLCV_FILE, SENTIMENT_FILE)
from backend.config import MODEL_PATH

logger = logging.getLogger(__name__)

_MIN_ROWS_PER_TICKER = 600   # ~3 years of trading days; triggers re-fetch if data is too short


# ── Freshness helpers ──────────────────────────────────────────────────────

def _is_today(date_str: str | None) -> bool:
    """Return True if date_str is today's date (YYYY-MM-DD or datetime)."""
    if not date_str:
        return False
    try:
        return str(date_str)[:10] == str(date.today())
    except Exception:
        return False


def _predictions_are_fresh() -> bool:
    last = get_last_prediction_date()
    fresh = _is_today(last)
    if fresh:
        logger.info(f"Predictions already fresh for today ({last}). Skipping pipeline.")
    return fresh


def _sentiment_is_fresh() -> bool:
    """Returns True if we already scraped sentiment today for a reasonable number of stocks."""
    if not SENTIMENT_FILE.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_parquet(SENTIMENT_FILE)
        if df.empty:
            return False
        df["date"] = pd.to_datetime(df["date"])
        today_rows = df[df["date"].dt.date == date.today()]
        # Consider fresh if we have sentiment for ≥50 stocks today
        fresh = today_rows["ticker"].nunique() >= 50
        if fresh:
            logger.info(f"Sentiment already scraped today ({today_rows['ticker'].nunique()} stocks). Skipping.")
        return fresh
    except Exception:
        return False


def _needs_full_history() -> bool:
    if not OHLCV_FILE.exists():
        return True
    try:
        import pandas as pd
        df = pd.read_parquet(OHLCV_FILE)
        if df.empty:
            return True
        median_rows = df.groupby("ticker").size().median()
        logger.info(f"Median rows per ticker in DB: {median_rows:.0f}")
        return median_rows < _MIN_ROWS_PER_TICKER
    except Exception as e:
        logger.warning(f"Could not read OHLCV file: {e}")
        return True


# ── Pipeline ───────────────────────────────────────────────────────────────

async def run_full_pipeline(force: bool = False):
    """
    Complete daily pipeline with smart caching.
    Set force=True to bypass all freshness checks (e.g., manual refresh button).
    Heavy imports happen here, not at module level, so startup stays fast.
    """
    logger.info(f"=== Starting daily pipeline (force={force}) ===")

    # ── Fast-exit: everything already done today ───────────────────────────
    if not force and _predictions_are_fresh():
        logger.info("All data is fresh for today. Nothing to do.")
        return

    # Lazy imports — only loaded when the pipeline actually runs
    from backend.data.fetcher import fetch_refresh_data, fetch_training_data
    from backend.data.storage import (upsert_ohlcv, upsert_predictions,
                                      get_all_ohlcv_for_features, upsert_sentiment,
                                      get_sentiment)
    from backend.ml.model import train_model, predict
    from backend.rag.indexer import run_indexing

    needs_history = _needs_full_history()
    model_exists  = Path(MODEL_PATH).exists()

    # ── Step 1: Fetch data ─────────────────────────────────────────────────
    if needs_history or not model_exists:
        logger.info("Step 1/5: First run — fetching 2-year training data...")
        df = fetch_training_data()
        if df.empty:
            logger.error("No training data fetched. Pipeline aborted.")
            return
        upsert_ohlcv(df)
        logger.info(f"Stored {len(df)} rows for {df['ticker'].nunique()} stocks")
    else:
        logger.info("Step 1/5: Fetching 60-day refresh data...")
        df = fetch_refresh_data()
        if df.empty:
            logger.error("No data fetched. Pipeline aborted.")
            return
        upsert_ohlcv(df)
        logger.info(f"Fetched {len(df)} rows for {df['ticker'].nunique()} stocks")

    # ── Step 2: Train or load model ────────────────────────────────────────
    logger.info("Step 2/5: Loading/training model...")
    if not model_exists or needs_history:
        logger.info("Training model on full history...")
        all_df  = get_all_ohlcv_for_features()
        metrics = train_model(all_df)
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.2%}")
    else:
        logger.info("Model loaded from disk — no retraining needed.")

    # ── Step 3: Sentiment (skip if already done today) ─────────────────────
    if not force and _sentiment_is_fresh():
        logger.info("Step 3/5: Sentiment already fresh. Skipping scraping.")
    else:
        logger.info("Step 3/5: Running FinBERT sentiment analysis...")
        try:
            from backend.data.symbols import get_symbol_map
            from backend.sentiment.scraper import scrape_all
            from backend.sentiment.analyzer import analyze
            symbol_map   = get_symbol_map()
            headlines_df = scrape_all(symbol_map, days=3)
            if not headlines_df.empty:
                sent_df = analyze(headlines_df)
                upsert_sentiment(sent_df)
                logger.info(f"Sentiment saved for {sent_df['ticker'].nunique()} stocks")
            else:
                logger.warning("No headlines scraped — using cached sentiment.")
        except Exception as e:
            logger.warning(f"Sentiment step failed (non-fatal): {e}")

    # ── Step 4: Generate predictions ───────────────────────────────────────
    logger.info("Step 4/5: Generating predictions...")
    all_df       = get_all_ohlcv_for_features()
    sentiment_df = get_sentiment(days=30)
    pred_df      = predict(all_df, sentiment_df=sentiment_df)
    if not pred_df.empty:
        upsert_predictions(pred_df)
        logger.info(f"Predictions saved for {len(pred_df)} stocks")

    # ── Step 5: Index RAG ──────────────────────────────────────────────────
    logger.info("Step 5/5: Indexing RAG documents...")
    n_docs = run_indexing()
    logger.info(f"Indexed {n_docs} documents")

    logger.info("=== Pipeline complete ===")


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        run_full_pipeline,
        trigger=CronTrigger(hour=8, minute=45, timezone="Asia/Kolkata"),
        id="daily_refresh",
        name="Daily NSE data refresh",
        replace_existing=True,
        misfire_grace_time=300,
    )
    return scheduler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from backend.data.storage import init_db
    init_db()
    asyncio.run(run_full_pipeline())
