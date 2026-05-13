"""
Earnings surprise + analyst data fetcher for Nifty 100 stocks.

Two signals:
  1. Earnings surprise (PEAD source):
       surprisePercent = (epsActual - epsEstimate) / |epsEstimate|
       Stocks that beat estimates tend to drift upward for 20-60 days (PEAD).

  2. Analyst price target upside:
       analyst_upside = (mean_target - current_price) / current_price
       Stocks with high analyst upside tend to outperform.

Limitation: yfinance only provides 4 quarters of earnings history for NSE.
This covers roughly Sep 2024 → Mar 2026, so it's most useful for:
  - Last few months of the training period
  - All of the test period
  - All production inference (always have recent quarter)

Data is cached to data/earnings.parquet and data/analyst_targets.parquet.
Cache is refreshed if > 6 hours old.
"""
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent.parent
_EARN_CACHE  = _ROOT / "data" / "earnings.parquet"
_ANALYST_CACHE = _ROOT / "data" / "analyst_targets.parquet"
_CACHE_TTL_H = 6   # hours


def _cache_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_h = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
    return age_h > _CACHE_TTL_H


# ── Earnings Surprise ─────────────────────────────────────────────────────────

def fetch_earnings(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch last 4 quarters of earnings surprise for each ticker.
    Returns long-format: [ticker, earnings_date, surprise_pct, eps_actual, eps_estimate]
    """
    records = []
    for i, ticker in enumerate(tickers):
        try:
            tk = yf.Ticker(ticker)
            eh = tk.earnings_history
            if eh is None or (hasattr(eh, "empty") and eh.empty):
                continue
            for quarter_date, row in eh.iterrows():
                surprise = row.get("surprisePercent", None)
                if surprise is None or (isinstance(surprise, float) and np.isnan(surprise)):
                    surprise = 0.0
                records.append({
                    "ticker":        ticker,
                    "earnings_date": pd.Timestamp(quarter_date),
                    "surprise_pct":  float(surprise),
                    "eps_actual":    float(row.get("epsActual",   0) or 0),
                    "eps_estimate":  float(row.get("epsEstimate", 0) or 0),
                })
        except Exception as e:
            logger.debug(f"Earnings fetch failed for {ticker}: {e}")
        # Polite rate limit — yfinance has per-IP throttling
        if i % 20 == 19:
            time.sleep(2)

    if not records:
        return pd.DataFrame(columns=["ticker", "earnings_date", "surprise_pct", "eps_actual", "eps_estimate"])
    df = pd.DataFrame(records)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)


def get_earnings(tickers: list[str]) -> pd.DataFrame:
    """Load from cache or fetch fresh."""
    if not _cache_stale(_EARN_CACHE):
        df = pd.read_parquet(_EARN_CACHE)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df
    logger.info(f"Fetching earnings history for {len(tickers)} tickers…")
    df = fetch_earnings(tickers)
    if not df.empty:
        _EARN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_EARN_CACHE, index=False)
        logger.info(f"Earnings cached: {len(df)} rows for {df['ticker'].nunique()} stocks")
    return df


def build_earnings_features(ohlcv_df: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, date) row in ohlcv_df, attach the most recent
    earnings surprise that occurred BEFORE that date.

    Features:
      earnings_surprise_pct  — most recent quarter surprise %
      days_since_earnings    — trading days since last earnings (PEAD decay proxy)
      earnings_beat_streak   — consecutive quarters of positive surprises (1-4)
    """
    if earnings_df.empty:
        ohlcv_df["earnings_surprise_pct"] = 0.0
        ohlcv_df["days_since_earnings"]   = 999.0
        ohlcv_df["earnings_beat_streak"]  = 0.0
        return ohlcv_df

    ohlcv_df = ohlcv_df.copy()
    # Normalise both to microsecond precision to avoid merge dtype mismatch
    # (pandas uses ns internally; yfinance timestamps can be seconds or microseconds)
    ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"]).astype("datetime64[us]")
    earnings_df = earnings_df.copy()
    earnings_df["earnings_date"] = pd.to_datetime(earnings_df["earnings_date"]).astype("datetime64[us]")

    result_frames = []
    for ticker, grp in ohlcv_df.groupby("ticker"):
        grp = grp.copy().sort_values("date")
        earn = earnings_df[earnings_df["ticker"] == ticker].sort_values("earnings_date")

        if earn.empty:
            grp["earnings_surprise_pct"] = 0.0
            grp["days_since_earnings"]   = 999.0
            grp["earnings_beat_streak"]  = 0.0
            result_frames.append(grp)
            continue

        # For each row, find the most recent earnings date BEFORE that date
        # Use pd.merge_asof (sorted join on nearest past key)
        grp_sorted  = grp.sort_values("date").reset_index(drop=True)
        # Rename earnings_date → date so merge_asof can join on a single key
        earn_sorted = (
            earn[["earnings_date", "surprise_pct"]]
            .sort_values("earnings_date")
            .rename(columns={"earnings_date": "date", "surprise_pct": "earnings_surprise_pct"})
            .reset_index(drop=True)
        )

        merged = pd.merge_asof(
            grp_sorted[["date"]],
            earn_sorted,            # already has "date" column
            on="date",
            direction="backward",   # use the most recent past earnings
        )
        grp_sorted["earnings_surprise_pct"] = merged["earnings_surprise_pct"].fillna(0.0).values

        # Days since earnings (calendar days, capped at 999)
        last_earn_dates = pd.merge_asof(
            grp_sorted[["date"]],
            earn_sorted[["date"]].rename(columns={"date": "earnings_date"}),
            left_on="date", right_on="earnings_date",
            direction="backward",
        )["earnings_date"]
        grp_sorted["days_since_earnings"] = (
            (grp_sorted["date"].values - last_earn_dates.values) / np.timedelta64(1, "D")
        ).clip(0, 999)
        grp_sorted["days_since_earnings"] = grp_sorted["days_since_earnings"].fillna(999.0)

        # Consecutive beat streak (count of consecutive positive surprises up to today)
        # E.g. if last 3 quarters all beat → streak = 3 → high-quality momentum
        def beat_streak(row_date):
            past = earn[earn["earnings_date"] <= row_date]["surprise_pct"].values
            if len(past) == 0:
                return 0
            streak = 0
            for s in reversed(past):
                if s > 0:
                    streak += 1
                else:
                    break
            return streak

        grp_sorted["earnings_beat_streak"] = grp_sorted["date"].apply(beat_streak)

        result_frames.append(grp_sorted)

    if not result_frames:
        return ohlcv_df

    out = pd.concat(result_frames, ignore_index=True)
    return out


# ── Analyst Price Targets ─────────────────────────────────────────────────────

def fetch_analyst_targets(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch current analyst price targets and recommendation distribution.
    Returns one row per ticker: [ticker, analyst_upside_pct, analyst_buy_pct, analyst_count]
    """
    records = []
    for i, ticker in enumerate(tickers):
        try:
            tk      = yf.Ticker(ticker)
            targets = tk.analyst_price_targets
            rec     = tk.recommendations_summary

            if not targets or "mean" not in targets or "current" not in targets:
                continue

            current = float(targets.get("current", 0) or 0)
            mean_t  = float(targets.get("mean",    0) or 0)

            if current <= 0 or mean_t <= 0:
                continue

            upside = (mean_t - current) / current

            # Recommendation distribution
            buy_pct    = 0.5  # neutral default
            analyst_n  = 0
            if rec is not None and not rec.empty:
                curr_row = rec[rec["period"] == "0m"]
                if not curr_row.empty:
                    r = curr_row.iloc[0]
                    total = r[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum()
                    if total > 0:
                        buy_pct   = (r["strongBuy"] + r["buy"]) / total
                        analyst_n = int(total)

            records.append({
                "ticker":          ticker,
                "analyst_upside_pct":  float(upside),
                "analyst_buy_pct":     float(buy_pct),
                "analyst_count":       analyst_n,
            })
        except Exception as e:
            logger.debug(f"Analyst target fetch failed for {ticker}: {e}")
        if i % 20 == 19:
            time.sleep(2)

    if not records:
        return pd.DataFrame(columns=["ticker", "analyst_upside_pct", "analyst_buy_pct", "analyst_count"])
    return pd.DataFrame(records)


def get_analyst_targets(tickers: list[str]) -> pd.DataFrame:
    """Load from cache or fetch fresh."""
    if not _cache_stale(_ANALYST_CACHE):
        return pd.read_parquet(_ANALYST_CACHE)
    logger.info(f"Fetching analyst targets for {len(tickers)} tickers…")
    df = fetch_analyst_targets(tickers)
    if not df.empty:
        _ANALYST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_ANALYST_CACHE, index=False)
        logger.info(f"Analyst targets cached: {len(df)} stocks")
    return df
