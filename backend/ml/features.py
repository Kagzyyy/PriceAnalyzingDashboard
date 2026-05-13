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
    use_fundamental: bool = True,
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

    # ── Fundamental features (earnings surprise + analyst targets) ─────────
    if use_fundamental:
        out = _add_fundamental_features(out)

    # ── Cross-sectional sector-relative features ───────────────────────────
    # These MUST be computed after the full concat — they compare each stock
    # against its sector peers on the same date, which requires all tickers
    # to be present simultaneously. Computing them per-ticker would just
    # give zeros everywhere.
    out = _add_sector_relative_features(out)

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

    # --- PEAD proxy (Post-Earnings Announcement Drift via price gaps) ------
    # Large price gaps (>3%) often signal earnings surprises. After a positive
    # earnings gap, stocks tend to drift upward for 20-60 days. This encodes
    # that drift signal from OHLCV alone — no EPS data needed.
    large_gap_up   = (df["gap_pct"] >  3.0).astype(float)
    large_gap_down = (df["gap_pct"] < -3.0).astype(float)
    df["pead_gap_up_5d"]    = large_gap_up.rolling(5,  min_periods=1).sum()
    df["pead_gap_up_20d"]   = large_gap_up.rolling(20, min_periods=1).sum()
    df["pead_gap_up_60d"]   = large_gap_up.rolling(60, min_periods=1).sum()
    df["pead_gap_down_5d"]  = large_gap_down.rolling(5,  min_periods=1).sum()
    df["pead_gap_down_20d"] = large_gap_down.rolling(20, min_periods=1).sum()
    # Net gap momentum: recent positive gaps minus negative gaps (signed signal)
    df["pead_net_20d"] = df["pead_gap_up_20d"] - df["pead_gap_down_20d"]

    # --- Label: 5-day relative outperformance vs Nifty 50 ─────────────────
    # label = 1 if this stock beats the Nifty 50 over the next 5 trading days
    # This is always ~50/50 balanced (regime-independent) and has more signal
    # than next-day absolute direction.
    stock_5d_ret  = (c.shift(-5) / c - 1) * 100    # in %
    nifty_5d_ret  = df["nifty_fwd_5d"].astype(float)  # in %, pre-merged
    df["label"]   = (stock_5d_ret > nifty_5d_ret).astype(float)

    return df


def _add_fundamental_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Attach earnings surprise and analyst target features.

    Earnings surprise: PEAD (Post-Earnings Announcement Drift)
      Stocks that beat EPS estimates tend to drift upward for 20-60 days.
      Feature is the most recent quarter's surprise % for each stock on each date.

    Analyst price targets: stocks far below mean analyst target tend to outperform.
      Feature is (mean_target - price) / price = upside %.

    NOTE: yfinance NSE earnings history only covers ~4 quarters (last ~1 year).
    Features are 0.0 for dates before available earnings data.
    """
    try:
        from backend.data.symbols import get_tickers
        from backend.data.earnings import get_earnings, build_earnings_features, get_analyst_targets

        tickers = list(out["ticker"].unique())

        # ── Earnings surprise ──────────────────────────────────────────────
        earnings_df = get_earnings(tickers)
        if not earnings_df.empty:
            out = build_earnings_features(out, earnings_df)
            logger.info(f"Earnings features merged: {(out['earnings_surprise_pct'] != 0).sum()} non-zero rows")
        else:
            out["earnings_surprise_pct"] = 0.0
            out["days_since_earnings"]   = 999.0
            out["earnings_beat_streak"]  = 0.0

        # Derived earnings features
        # PEAD decay: surprise fades after ~60 days → weight by recency
        out["pead_weighted_surprise"] = (
            out["earnings_surprise_pct"]
            * np.exp(-out["days_since_earnings"].clip(0, 999) / 60.0)
        )
        # Categorical: strong beat / beat / miss / strong miss
        s = out["earnings_surprise_pct"]
        out["earnings_strong_beat"] = (s >  0.10).astype(float)  # beat by >10%
        out["earnings_beat"]        = ((s > 0) & (s <= 0.10)).astype(float)
        out["earnings_miss"]        = (s <  0.0).astype(float)

        # ── Analyst targets ────────────────────────────────────────────────
        analyst_df = get_analyst_targets(tickers)
        if not analyst_df.empty:
            # Static feature: merge on ticker only (same value for all dates)
            out = out.merge(
                analyst_df[["ticker", "analyst_upside_pct", "analyst_buy_pct", "analyst_count"]],
                on="ticker", how="left",
            )
            out["analyst_upside_pct"] = out["analyst_upside_pct"].fillna(0.0)
            out["analyst_buy_pct"]    = out["analyst_buy_pct"].fillna(0.5)
            out["analyst_count"]      = out["analyst_count"].fillna(0.0)
            logger.info(f"Analyst features merged for {analyst_df['ticker'].nunique()} stocks")
        else:
            out["analyst_upside_pct"] = 0.0
            out["analyst_buy_pct"]    = 0.5
            out["analyst_count"]      = 0.0

    except Exception as e:
        logger.warning(f"Fundamental features failed (non-fatal): {e}")
        for col in [
            "earnings_surprise_pct", "days_since_earnings", "earnings_beat_streak",
            "pead_weighted_surprise", "earnings_strong_beat", "earnings_beat", "earnings_miss",
            "analyst_upside_pct", "analyst_buy_pct", "analyst_count",
        ]:
            if col not in out.columns:
                out[col] = 0.0

    return out


def _add_sector_relative_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional features: how is each stock performing vs its sector peers?

    Why this matters:
      - RSI=65 on its own is ambiguous. RSI=65 when every peer is at 45 is bullish.
      - XGBoost cannot derive cross-sectional context from per-stock features alone.
      - Sector-relative signals are harder to arbitrage because they require
        knowing the entire sector simultaneously — a genuine information edge.

    Features added (all computed within [date × sector] groups):
      sector_ret_1d_vs_peers  — stock 1d return minus sector median 1d return
      sector_ret_5d_vs_peers  — stock 5d return minus sector median 5d return
      sector_ret_20d_vs_peers — stock 20d return minus sector median 20d return
      sector_rsi_rank         — RSI percentile rank within sector  (0=worst, 1=best)
      sector_momentum_rank    — 20d return percentile rank within sector
      sector_volume_rank      — volume_ratio percentile rank within sector
      sector_bb_rank          — Bollinger %B percentile rank within sector
    """
    try:
        from backend.data.symbols import get_symbol_map
        sym_map = get_symbol_map(universe="nifty100")
        # Map ticker → sector
        out["_sector"] = out["ticker"].map(
            lambda t: sym_map.get(t, {}).get("sector", "Unknown")
        )

        grp = out.groupby(["date", "_sector"])

        # 1. Sector-demeaned returns (stock value − sector median)
        for ret_col in ["ret_1d", "ret_5d", "ret_20d"]:
            sector_med = grp[ret_col].transform("median")
            out[f"sector_{ret_col}_vs_peers"] = (out[ret_col] - sector_med).fillna(0.0)

        # 2. Percentile ranks within sector on each date
        for raw_col, feat_name in [
            ("rsi_14",       "sector_rsi_rank"),
            ("ret_20d",      "sector_momentum_rank"),
            ("volume_ratio", "sector_volume_rank"),
            ("bb_pct",       "sector_bb_rank"),
        ]:
            out[feat_name] = grp[raw_col].transform(
                lambda x: x.rank(pct=True, method="average")
            ).fillna(0.5)   # 0.5 = neutral for single-stock sectors

        out.drop(columns=["_sector"], inplace=True)
        logger.info("Sector-relative features added successfully")

    except Exception as e:
        logger.warning(f"Sector-relative features failed (non-fatal): {e}")
        # Fill with neutral values so training still proceeds
        for col in [
            "sector_ret_1d_vs_peers", "sector_ret_5d_vs_peers", "sector_ret_20d_vs_peers",
            "sector_rsi_rank", "sector_momentum_rank", "sector_volume_rank", "sector_bb_rank",
        ]:
            if col not in out.columns:
                out[col] = 0.0

    return out


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
    # Sector-relative strength (cross-sectional — stock vs sector peers)
    # RSI=65 when peers are at 45 is much more bullish than RSI=65 in isolation.
    "sector_ret_1d_vs_peers", "sector_ret_5d_vs_peers", "sector_ret_20d_vs_peers",
    "sector_rsi_rank", "sector_momentum_rank", "sector_volume_rank", "sector_bb_rank",
    # PEAD proxy (Post-Earnings Announcement Drift via price gaps)
    # Large gaps (>3%) are proxies for earnings surprises. After an earnings beat
    # gap, stocks drift up for 20-60 days. Full 5-year history from OHLCV.
    "pead_gap_up_5d", "pead_gap_up_20d", "pead_gap_up_60d",
    "pead_gap_down_5d", "pead_gap_down_20d", "pead_net_20d",
    # Earnings surprise (yfinance — last 4 quarters, covers ~Sep 2024 onward)
    "earnings_surprise_pct", "days_since_earnings", "earnings_beat_streak",
    "pead_weighted_surprise", "earnings_strong_beat", "earnings_beat", "earnings_miss",
    # Analyst price targets (static per stock — current snapshot)
    "analyst_upside_pct", "analyst_buy_pct", "analyst_count",
]
