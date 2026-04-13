"""
Feature engineering using the `ta` library.
Computes technical indicators + FinBERT sentiment + Nifty market context features.

Label (cross-sectional relative return):
  label = 1  if stock_return_5d > nifty_return_5d  (outperforms market)
  label = 0  otherwise

Why relative return instead of absolute direction:
  - Absolute next-day direction is ~50/50 regardless of features (noise dominates)
  - Relative 5-day outperformance is ~50/50 in ALL market regimes (bull or bear)
  - FinBERT sentiment has documented alpha at 2-5 day horizon for relative returns
  - Weekly rebalancing cuts transaction costs by 5x vs daily trading
"""
import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

_SENT_COLS  = ["sentiment_score", "sent_pos", "sent_neg", "sent_neu", "sent_zscore", "news_count"]
_NIFTY_COLS = ["nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d",
               "nifty_vs_ema20", "nifty_vs_ema50", "nifty_regime", "nifty_volatility",
               "nifty_fwd_5d"]


def engineer_features(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    nifty_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Takes long-format OHLCV dataframe (ticker, date, open, high, low, close, volume).
    Optionally accepts sentiment_df and nifty_df.
    Returns dataframe with all features + label column.
    Label = 1 if stock's 5-day return beats Nifty's 5-day return, else 0.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Merge sentiment ────────────────────────────────────────────────────
    if sentiment_df is not None and not sentiment_df.empty:
        sent = sentiment_df.copy()
        sent["date"] = pd.to_datetime(sent["date"])
        available_cols = ["ticker", "date"] + [c for c in _SENT_COLS if c in sent.columns]
        df = df.merge(sent[available_cols], on=["ticker", "date"], how="left")

    for col in _SENT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # 1-day sentiment lag — news published on day D is usable on day D+1
    if sentiment_df is not None and not sentiment_df.empty:
        for col in _SENT_COLS:
            if col in df.columns:
                df[col] = df.groupby("ticker")[col].shift(1).fillna(0.0)

    # ── Merge Nifty features (market context + forward return for label) ───
    if nifty_df is not None and not nifty_df.empty:
        nifty = nifty_df.copy()
        nifty["date"] = pd.to_datetime(nifty["date"])
        merge_cols = ["date"] + [c for c in _NIFTY_COLS if c in nifty.columns]
        df = df.merge(nifty[merge_cols], on="date", how="left")

    for col in _NIFTY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # ── Per-ticker feature engineering ─────────────────────────────────────
    result_frames = []
    errors = []
    for ticker in df["ticker"].unique():
        try:
            t_df = df[df["ticker"] == ticker].copy().reset_index(drop=True)
            if len(t_df) < 100:
                continue
            t_df = _add_features(t_df)
            result_frames.append(t_df)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            continue

    if not result_frames:
        if errors:
            logger.error(f"Feature engineering failed for all tickers. First 3: {errors[:3]}")
        return pd.DataFrame()

    if errors:
        logger.warning(f"Skipped {len(errors)} tickers. Example: {errors[0]}")

    out = pd.concat(result_frames, ignore_index=True)
    out = out.dropna(subset=["label"])
    return out


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    # --- Momentum ---
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(c, window=7).rsi()
    df["roc_5"]  = ta.momentum.ROCIndicator(c, window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(c, window=20).roc()

    macd_ind = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["willr"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # --- Trend ---
    df["ema_9"]   = ta.trend.EMAIndicator(c, window=9).ema_indicator()
    df["ema_20"]  = ta.trend.EMAIndicator(c, window=20).ema_indicator()
    df["ema_50"]  = ta.trend.EMAIndicator(c, window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(c, window=200).ema_indicator()

    df["price_vs_ema20"]  = (c / df["ema_20"]  - 1) * 100
    df["price_vs_ema50"]  = (c / df["ema_50"]  - 1) * 100
    df["price_vs_ema200"] = (c / df["ema_200"] - 1) * 100
    df["ema9_vs_ema20"]   = (df["ema_9"]  / df["ema_20"] - 1) * 100
    df["ema20_vs_ema50"]  = (df["ema_20"] / df["ema_50"] - 1) * 100

    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"] = adx_ind.adx()

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"]          = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["daily_range_pct"] = (h - l) / c.replace(0, np.nan) * 100

    # --- Volume ---
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    vol_ma20 = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma20.replace(0, np.nan)
    df["volume_ma20"]  = vol_ma20

    df["vwap"]     = (h + l + c) / 3
    df["vwap_dev"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan) * 100

    # --- Pattern ---
    df["gap_pct"]    = (o / c.shift(1) - 1) * 100
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    hl_range = (h - l).replace(0, np.nan)
    df["upper_wick"] = (h - df[["open", "close"]].max(axis=1)) / hl_range * 100
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - l) / hl_range * 100
    df["body_pct"]   = abs(c - o) / hl_range * 100

    # --- Market position ---
    high_52w = c.rolling(252, min_periods=50).max()
    low_52w  = c.rolling(252, min_periods=50).min()
    df["pct_from_52w_high"] = (c / high_52w.replace(0, np.nan) - 1) * 100
    df["pct_from_52w_low"]  = (c / low_52w.replace(0, np.nan)  - 1) * 100

    # Returns
    df["ret_1d"]  = c.pct_change(1)  * 100
    df["ret_5d"]  = c.pct_change(5)  * 100
    df["ret_10d"] = c.pct_change(10) * 100
    df["ret_20d"] = c.pct_change(20) * 100

    # --- FinBERT Sentiment ---
    s = df["sentiment_score"].astype(float)
    df["sent_today"]    = s
    df["sent_7d"]       = s.rolling(7,  min_periods=1).mean()
    df["sent_14d"]      = s.rolling(14, min_periods=1).mean()
    df["sent_momentum"] = s - df["sent_14d"]

    df["sent_pos_today"] = df["sent_pos"].astype(float)
    df["sent_neg_today"] = df["sent_neg"].astype(float)
    df["sent_neu_today"] = df["sent_neu"].astype(float)
    df["sent_zscore"]    = df["sent_zscore"].astype(float)
    df["news_count"]     = df["news_count"].astype(float)

    # --- Label: 5-day relative outperformance vs Nifty 50 ─────────────────
    # label = 1 if this stock beats the Nifty 50 over the next 5 trading days
    # This is always ~50/50 balanced (regime-independent) and has more signal
    # than next-day absolute direction.
    stock_5d_ret  = (c.shift(-5) / c - 1) * 100    # in %
    nifty_5d_ret  = df["nifty_fwd_5d"].astype(float)  # in %, pre-merged
    df["label"]   = (stock_5d_ret > nifty_5d_ret).astype(float)

    return df


FEATURE_COLS = [
    # Momentum
    "rsi_14", "rsi_7", "roc_5", "roc_10", "roc_20",
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "willr",
    # Trend
    "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
    "ema9_vs_ema20", "ema20_vs_ema50", "adx",
    # Volatility
    "bb_pct", "bb_width", "atr_14", "daily_range_pct",
    # Volume
    "volume_ratio", "vwap_dev", "obv",
    # Pattern
    "gap_pct", "inside_bar", "upper_wick", "lower_wick", "body_pct",
    # Market position
    "pct_from_52w_high", "pct_from_52w_low",
    # Returns
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # FinBERT sentiment
    "sent_today", "sent_7d", "sent_14d", "sent_momentum",
    "sent_pos_today", "sent_neg_today", "sent_neu_today",
    "sent_zscore", "news_count",
    # Nifty market context (regime, momentum, volatility of the broad market)
    "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d",
    "nifty_vs_ema20", "nifty_vs_ema50", "nifty_regime", "nifty_volatility",
]
