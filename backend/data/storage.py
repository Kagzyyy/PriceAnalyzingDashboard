"""
File-based storage using parquet + JSON. No database required.
"""
import json
import logging
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Use absolute path anchored to project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
OHLCV_FILE       = DATA_DIR / "ohlcv.parquet"
PREDICTIONS_FILE = DATA_DIR / "predictions.parquet"
SENTIMENT_FILE   = DATA_DIR / "sentiment.parquet"
META_FILE        = DATA_DIR / "model_meta.json"


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage initialized at {DATA_DIR}")


# ── OHLCV ─────────────────────────────────────────────────────────────────────

def upsert_ohlcv(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if OHLCV_FILE.exists():
        existing = pd.read_parquet(OHLCV_FILE)
        df = pd.concat([existing, df]).drop_duplicates(subset=["ticker", "date"], keep="last")
    df.to_parquet(OHLCV_FILE, index=False)
    logger.info(f"Saved {len(df)} OHLCV rows")


def get_ohlcv(ticker: str = None, days: int = 365) -> pd.DataFrame:
    if not OHLCV_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(OHLCV_FILE)
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    df = df[pd.to_datetime(df["date"]) >= cutoff]
    if ticker:
        df = df[df["ticker"] == ticker]
    return df.sort_values(["ticker", "date"])


def get_all_ohlcv_for_features() -> pd.DataFrame:
    from backend.config import LOOKBACK_DAYS
    return get_ohlcv(days=LOOKBACK_DAYS)


# ── Predictions ───────────────────────────────────────────────────────────────

def upsert_predictions(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if PREDICTIONS_FILE.exists():
        existing = pd.read_parquet(PREDICTIONS_FILE)
        df = pd.concat([existing, df]).drop_duplicates(subset=["ticker", "prediction_date"], keep="last")
    df.to_parquet(PREDICTIONS_FILE, index=False)
    logger.info(f"Saved {len(df)} predictions")


def get_latest_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PREDICTIONS_FILE)
    idx = df.groupby("ticker")["prediction_date"].idxmax()
    return df.loc[idx].sort_values("prob_up", ascending=False).reset_index(drop=True)


def get_sector_summary(symbol_map: dict) -> pd.DataFrame:
    preds = get_latest_predictions()
    if preds.empty:
        return pd.DataFrame()
    preds["sector"] = preds["ticker"].map(
        lambda t: symbol_map.get(t, {}).get("sector", "Unknown")
    )
    summary = preds.groupby("sector").agg(
        avg_prob_up=("prob_up", "mean"),
        count=("ticker", "count"),
        longs=("direction", lambda x: (x == "UP").sum()),
        shorts=("direction", lambda x: (x == "DOWN").sum()),
    ).reset_index()
    summary["bullish_pct"] = (summary["avg_prob_up"] * 100).round(1)
    return summary.sort_values("avg_prob_up", ascending=False)


# ── Model meta ────────────────────────────────────────────────────────────────

def save_model_meta(accuracy: float, n_samples: int, n_stocks: int,
                    train_accuracy: float = 0.0, precision: float = 0.0, recall: float = 0.0) -> None:
    META_FILE.parent.mkdir(exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "accuracy": accuracy,
            "train_accuracy": train_accuracy,
            "precision": precision,
            "recall": recall,
            "overfit_gap": round(train_accuracy - accuracy, 4),
            "n_samples": n_samples,
            "n_stocks": n_stocks,
        }, f)


def get_model_meta() -> dict:
    if not META_FILE.exists():
        return {}
    with open(META_FILE) as f:
        return json.load(f)


# ── Sentiment ─────────────────────────────────────────────────────────────────

def upsert_sentiment(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if SENTIMENT_FILE.exists():
        existing = pd.read_parquet(SENTIMENT_FILE)
        existing["date"] = pd.to_datetime(existing["date"])
        df = pd.concat([existing, df]).drop_duplicates(subset=["ticker", "date"], keep="last")
    df.to_parquet(SENTIMENT_FILE, index=False)
    logger.info(f"Saved {len(df)} sentiment rows for {df['ticker'].nunique()} stocks")


def get_sentiment(days: int = 60) -> pd.DataFrame:
    """Return sentiment data. Default 60 days to give z-score rolling window enough history."""
    if not SENTIMENT_FILE.exists():
        return pd.DataFrame(columns=[
            "ticker", "date", "sentiment_score",
            "sent_pos", "sent_neg", "sent_neu", "sent_zscore", "news_count"
        ])
    df = pd.read_parquet(SENTIMENT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    # Ensure new columns exist even for old data
    for col in ["sent_pos", "sent_neg", "sent_neu", "sent_zscore"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[df["date"] >= cutoff].sort_values(["ticker", "date"])


def get_last_prediction_date() -> str | None:
    if not PREDICTIONS_FILE.exists():
        return None
    df = pd.read_parquet(PREDICTIONS_FILE)
    if df.empty:
        return None
    return str(df["prediction_date"].max())
