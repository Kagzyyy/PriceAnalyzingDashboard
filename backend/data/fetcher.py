"""
yfinance batch data fetcher for all NSE stocks.
"""
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from backend.config import LOOKBACK_DAYS, REFRESH_DAYS, BATCH_SIZE
from backend.data.symbols import get_tickers, get_symbol_map

logger = logging.getLogger(__name__)


def fetch_history(tickers: list[str], days: int) -> pd.DataFrame:
    """
    Fetch OHLCV history for a list of tickers in batches.
    Returns a long-format DataFrame with columns:
    [ticker, date, open, high, low, close, volume]
    """
    end = datetime.today()
    start = end - timedelta(days=days)
    all_frames = []

    failed_tickers = []   # tickers that failed in batch — retried individually below

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        logger.info(f"Fetching batch {i // BATCH_SIZE + 1}/{-(-len(tickers) // BATCH_SIZE)}: {len(batch)} tickers")
        try:
            raw = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                failed_tickers.extend(batch)
                continue

            # Handle single vs multi ticker response
            if isinstance(raw.columns, pd.MultiIndex):
                # Multi-ticker: columns are (price_type, ticker)
                fetched = set(raw.columns.get_level_values(1))
                for ticker in batch:
                    if ticker not in fetched:
                        failed_tickers.append(ticker)
                        continue
                    df = raw.xs(ticker, level=1, axis=1).copy()
                    df = _clean_ohlcv(df, ticker)
                    if df is not None:
                        all_frames.append(df)
                    else:
                        failed_tickers.append(ticker)
            else:
                # Single ticker
                df = raw.copy()
                df = _clean_ohlcv(df, batch[0])
                if df is not None:
                    all_frames.append(df)
                else:
                    failed_tickers.append(batch[0])
        except Exception as e:
            logger.warning(f"Batch fetch error: {e}")
            failed_tickers.extend(batch)
            continue

    # ── Retry failed tickers one-by-one with repair=True ─────────────────────
    # yfinance sometimes drops tickers with timezone issues in batch mode.
    # Individual download with repair=True usually recovers them (ZOMATO, TATAMOTORS, etc.)
    if failed_tickers:
        logger.info(f"Retrying {len(failed_tickers)} failed tickers individually: {failed_tickers}")
        for ticker in failed_tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    repair=True,
                )
                if raw.empty:
                    logger.warning(f"No data for {ticker} even after retry — skipping")
                    continue
                df = _clean_ohlcv(raw, ticker)
                if df is not None:
                    all_frames.append(df)
                    logger.info(f"Recovered {ticker}: {len(df)} rows")
                else:
                    logger.warning(f"Could not clean data for {ticker} — skipping")
            except Exception as e:
                logger.warning(f"Retry failed for {ticker}: {e}")

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    return result


def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalize and validate a single ticker's OHLCV dataframe."""
    try:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(df) < 20:
            return None
        df = df.reset_index()
        df.rename(columns={"Date": "date", "Datetime": "date"}, inplace=True)
        if "date" not in df.columns:
            df["date"] = df.index
        df["ticker"] = ticker
        df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        logger.debug(f"Clean error for {ticker}: {e}")
        return None


def fetch_training_data() -> pd.DataFrame:
    """
    Fetch 3 years of OHLCV for the Nifty 100 universe.

    Why nifty100 (not nifty50)?
      - The ML model trains on the nifty50 subset (filtered later in the
        training pipeline), but the pairs backtester needs nifty100 data to
        find enough cointegrated pairs: C(100,2)=4950 vs C(50,2)=1225.
      - nifty50 ⊂ nifty100, so nifty50 model training data is fully covered.
    """
    logger.info(f"Fetching training data ({LOOKBACK_DAYS} days, Nifty 100 universe)...")
    tickers = get_tickers(universe="nifty100")
    logger.info(f"Universe: {len(tickers)} tickers")
    return fetch_history(tickers, LOOKBACK_DAYS)


def fetch_refresh_data() -> pd.DataFrame:
    """Fetch recent 60 days for Nifty 100 (covers both model + pairs)."""
    logger.info("Fetching refresh data (60 days, Nifty 100 universe)...")
    tickers = get_tickers(universe="nifty100")
    return fetch_history(tickers, REFRESH_DAYS)


def fetch_single_stock(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch data for a single NSE stock for detail view."""
    ticker = f"{symbol}.NS"
    return fetch_history([ticker], days)
