"""
Nifty 50 data helper — fetches and caches index OHLCV.

Used for:
  1. Relative return label: stock_5d_ret > nifty_5d_ret (outperformance)
  2. Market context features for the model (regime, momentum, volatility)
"""
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CACHE_FILE   = _PROJECT_ROOT / "data" / "nifty.parquet"


def get_nifty_data(days: int = 1825) -> pd.DataFrame:
    """
    Returns Nifty 50 daily OHLCV as a DataFrame.
    Caches to parquet; refreshes automatically if cache is > 20 hours old.
    """
    if _CACHE_FILE.exists():
        age_h = (datetime.now() - datetime.fromtimestamp(
            _CACHE_FILE.stat().st_mtime)).total_seconds() / 3600
        if age_h < 20:
            df = pd.read_parquet(_CACHE_FILE)
            df["date"] = pd.to_datetime(df["date"])
            return df

    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days)
        raw   = yf.download(
            "^NSEI",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logger.warning("Nifty yfinance returned empty — falling back to cache")
            raise ValueError("empty")

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.reset_index().rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"])

        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE_FILE, index=False)
        logger.info(f"Nifty data cached: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Nifty fetch failed: {e}")
        if _CACHE_FILE.exists():
            df = pd.read_parquet(_CACHE_FILE)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()


def get_nifty_features(days: int = 1825) -> pd.DataFrame:
    """
    Returns a date-indexed DataFrame of Nifty-derived features.
    Includes nifty_fwd_5d (forward 5-day return %) for label computation.
    """
    df = get_nifty_data(days=days)
    if df.empty:
        return pd.DataFrame()

    df   = df.sort_values("date").reset_index(drop=True)
    c    = df["close"]

    feat = pd.DataFrame()
    feat["date"] = df["date"]

    # Context features (used as model inputs)
    feat["nifty_ret_1d"]    = c.pct_change(1)  * 100
    feat["nifty_ret_5d"]    = c.pct_change(5)  * 100
    feat["nifty_ret_20d"]   = c.pct_change(20) * 100
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    feat["nifty_vs_ema20"]  = (c / ema20 - 1) * 100
    feat["nifty_vs_ema50"]  = (c / ema50 - 1) * 100
    feat["nifty_regime"]    = (c > ema50).astype(float)
    feat["nifty_volatility"] = c.pct_change().rolling(20).std() * 100

    # Forward 5-day return — used ONLY for label construction (not a model feature)
    feat["nifty_fwd_5d"] = (c.shift(-5) / c - 1) * 100

    return feat
