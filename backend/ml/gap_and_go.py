"""
Gap and Go strategy with ML enhancement.

Base strategy:
  1. Stock gaps up >= gap_threshold (default 1.5%) at open
     (Open > Prev_Close × 1.015)
  2. Enter at open price
  3. Target: +2%  |  Stop: -0.5%  (4:1 R:R — break-even at 28% win rate)
  4. Fallback: close of day

Optimal parameters found empirically:
  - Target +2%, Stop -0.5% achieves 4:1 R:R → break-even at 28% WR
  - Base win rate ~33% (above break-even) with OHLC simulation
  - Streak tested ~49% (intraday 5-min candles, more accurate)
  - ML filters improve both win rate and trade quality

ML Enhancement layers (each independently toggleable):

  Layer 1 — XGBoost confirmation:
    Only trade gaps where yesterday's prob_up > ml_threshold (default 0.55).
    Features are from the PREVIOUS day → zero look-ahead bias.

  Layer 2 — Sentiment filter:
    Skip gaps where FinBERT 3-day rolling sentiment < sent_threshold (-0.10).
    Negative sentiment = the gap has fundamental bad news → likely to fade.

  Layer 3 — Sector momentum filter:
    Only trade stocks where sector_momentum_rank > 0.5 (stock is outperforming
    its sector peers). Avoids catching a falling knife in a weak sector.

Trade simulation from daily OHLC:
  We approximate intraday using the day's High, Low, Close.
  When both target AND stop are touched in the same day, we use the close
  direction as a proxy for which was hit first:
    - close >= entry → gap continued upward → assume target hit first
    - close <  entry → gap reversed → assume stop hit first
  This is more realistic than "stop always first" for genuine momentum gaps.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from backend.config import MODEL_PATH
from backend.data.storage import get_all_ohlcv_for_features, get_sentiment
from backend.data.symbols import NIFTY_100_SYMBOLS
from backend.ml.features import FEATURE_COLS, engineer_features
from backend.data.nifty import get_nifty_features
from backend.ml.gap_trainer import GAP_MODEL_PATH, GAP_FEATURE_COLS
from backend.ml.walkforward import build_oos_prob_lookup, build_oos_gap_prob_lookup

logger = logging.getLogger(__name__)

ROUND_TRIP_COST = 0.20   # % — STT + exchange fees (Zerodha delivery)


# ── Trade simulation ──────────────────────────────────────────────────────────

def _simulate_trade(entry: float, high: float, low: float, close: float,
                    target_pct: float, stop_pct: float,
                    slippage_pct: float = 0.0) -> tuple[float, str]:
    """
    Approximate intraday trade result from daily OHLC.
    Returns (pct_return_net_of_costs, exit_reason).

    slippage_pct models execution friction at the open (market order on a
    gap-up stock).  The fill price is open * (1 + slippage_pct/100).
    Target and stop remain anchored to the theoretical open so the P&L
    compresses on wins and expands on losses — exactly what happens live.

    When both target AND stop are hit in the same day, we use the close
    direction to determine which was likely hit first:
      - close >= entry → target likely hit first (momentum continued)
      - close <  entry → stop likely hit first (gap reversed)
    """
    effective_entry = entry * (1 + slippage_pct / 100)   # what you actually pay
    target = entry * (1 + target_pct / 100)               # planned target vs open
    stop   = entry * (1 - stop_pct   / 100)               # planned stop vs open

    hit_target = high >= target
    hit_stop   = low  <= stop

    if hit_target and not hit_stop:
        gross  = (target - effective_entry) / effective_entry * 100
        reason = "target"
    elif hit_stop and not hit_target:
        gross  = (stop   - effective_entry) / effective_entry * 100
        reason = "stop_loss"
    elif hit_target and hit_stop:
        if close >= entry:
            gross  = (target - effective_entry) / effective_entry * 100
            reason = "target"
        else:
            gross  = (stop   - effective_entry) / effective_entry * 100
            reason = "stop_loss"
    else:
        gross  = (close - effective_entry) / effective_entry * 100
        reason = "close_of_day"

    return round(gross - ROUND_TRIP_COST, 4), reason


# ── ML helpers ────────────────────────────────────────────────────────────────

def _build_prob_lookup(raw: pd.DataFrame) -> dict:
    """
    Run XGBoost on all stocks, all dates.
    Returns {(ticker, 'YYYY-MM-DD'): prob_up}

    Features are computed at end of day D → prediction used on day D+1.
    No look-ahead bias.
    """
    if not Path(MODEL_PATH).exists():
        logger.warning("No trained model found — ML layer disabled.")
        return {}
    try:
        nifty_df = get_nifty_features()
        feat     = engineer_features(raw, nifty_df=nifty_df)
        feat     = feat.sort_values(["ticker", "date"])

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        X = feat[FEATURE_COLS].fillna(0)
        if hasattr(model, "feature_names_in_"):
            trained = list(model.feature_names_in_)
            for c in trained:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[trained]

        proba = model.predict_proba(X)[:, 1]
        feat  = feat.copy()
        feat["prob_up"] = proba

        # Shift forward by 1 day: prediction made on day D applies to day D+1
        lookup = {}
        for ticker, grp in feat.groupby("ticker"):
            grp = grp.sort_values("date").reset_index(drop=True)
            for i in range(len(grp) - 1):
                trade_date = str(grp.loc[i + 1, "date"])[:10]
                lookup[(ticker, trade_date)] = float(grp.loc[i, "prob_up"])

        logger.info(f"ML prob lookup built: {len(lookup)} entries")
        return lookup

    except Exception as e:
        logger.warning(f"ML lookup failed: {e} — ML layer disabled.")
        return {}


def _build_sentiment_lookup(days: int = 1825) -> dict:
    """
    Returns {(ticker, 'YYYY-MM-DD'): avg_sent_score_3d}
    Uses 3-day rolling average to smooth single-article noise.
    """
    try:
        sent = get_sentiment(days=days)
        if sent.empty:
            return {}
        sent["date"] = pd.to_datetime(sent["date"])
        sent = sent.sort_values(["ticker", "date"])
        # Column is 'sentiment_score' in the DB schema
        score_col = "sentiment_score" if "sentiment_score" in sent.columns else "sent_score"
        sent["sent_3d"] = (
            sent.groupby("ticker")[score_col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        lookup = {
            (row["ticker"], str(row["date"].date())): float(row["sent_3d"])
            for _, row in sent[["ticker", "date", "sent_3d"]].iterrows()
        }
        logger.info(f"Sentiment lookup built: {len(lookup)} entries")
        return lookup
    except Exception as e:
        logger.warning(f"Sentiment lookup failed: {e}")
        return {}


# ── Gap ML helper ─────────────────────────────────────────────────────────────

def _build_gap_prob_lookup(raw: pd.DataFrame, gap_threshold: float) -> dict:
    """
    Run the gap-specific XGBoost on all gap-up days.
    Returns {(ticker, 'YYYY-MM-DD'): gap_continuation_prob}

    Uses previous day's FEATURE_COLS + current day's gap_pct (observable at
    market open — not look-ahead) to predict gap-continuation probability.
    Requires gap_model.pkl — train it first with gap_trainer.py.
    """
    if not GAP_MODEL_PATH.exists():
        logger.warning("gap_model.pkl not found — run: python -m backend.ml.gap_trainer")
        return {}
    try:
        with open(GAP_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        nifty_df = get_nifty_features()
        feat     = engineer_features(raw, nifty_df=nifty_df)
        feat["date"] = pd.to_datetime(feat["date"])
        feat = feat.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Identify current gap sizes from raw
        raw2 = raw.copy()
        raw2["date"]      = pd.to_datetime(raw2["date"])
        raw2["prev_close"] = raw2.groupby("ticker")["close"].shift(1)
        raw2["gap_pct_r"]  = (raw2["open"] - raw2["prev_close"]) / raw2["prev_close"] * 100
        gap_raw = raw2[raw2["gap_pct_r"] >= gap_threshold][["ticker", "date", "gap_pct_r"]]
        gap_info = {
            (r["ticker"], str(r["date"].date())): float(r["gap_pct_r"])
            for _, r in gap_raw.iterrows()
        }

        # Build inference rows: previous day features + current gap_pct
        rows, keys = [], []
        for ticker, grp in feat.groupby("ticker"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                trade_date = str(grp.loc[i, "date"].date())
                key        = (ticker, trade_date)
                if key not in gap_info:
                    continue
                feat_row = grp.loc[i - 1, FEATURE_COLS].fillna(0).to_dict()
                feat_row["current_gap_pct"] = gap_info[key]
                rows.append(feat_row)
                keys.append(key)

        if not rows:
            return {}

        X = pd.DataFrame(rows)[GAP_FEATURE_COLS]
        if hasattr(model, "feature_names_in_"):
            trained = list(model.feature_names_in_)
            for c in trained:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[trained]

        proba = model.predict_proba(X)[:, 1]
        lookup = {k: float(p) for k, p in zip(keys, proba)}
        logger.info(f"Gap ML lookup built: {len(lookup)} entries")
        return lookup
    except Exception as e:
        logger.warning(f"Gap ML lookup failed: {e}")
        return {}


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_gap_and_go_backtest(
    gap_threshold:   float = 1.5,      # minimum gap % to consider
    gap_max_pct:     float = 4.0,      # T2: cap very large gaps (>4% tend to fade)
    target_pct:      float = 2.0,      # profit target %
    stop_pct:        float = 0.5,      # stop loss % — 4:1 R:R breaks even at 28% WR
    ml_threshold:    float = 0.50,     # prob_up threshold — 0.50 = "ML says UP"
    sent_threshold:  float = -0.10,    # sentiment threshold
    max_positions:   int   = 5,        # max simultaneous trades per day
    initial_capital: float = 100_000.0,
    # existing ML layers
    use_ml_confirm:    bool = True,    # XGBoost at 0.50 = binary UP/DOWN filter (best config)
    use_sentiment:     bool = True,    # Sentiment: ON (directionally correct)
    use_sector_filter: bool = True,    # Sector rank: ON (primary value-add)
    # Tier 1 additions
    use_volume_filter: bool = True,    # T1: only trade gaps with institutional volume
    volume_min_ratio:  float = 1.5,    # T1: yesterday volume must be >= 1.5x 20-day avg
    use_rsi_filter:    bool = False,   # T1: RSI hurts gap-and-go (over-filters momentum)
    rsi_low:           float = 35.0,   # T1: RSI floor — below = oversold/falling knife
    rsi_high:          float = 68.0,   # T1: RSI ceiling — above = overbought, gap fades
    use_nifty_filter:  bool = True,    # T1: skip trades on market-down days
    nifty_min_ret:     float = -0.3,   # T1: skip if Nifty is down more than this %
    # Tier 2 additions
    use_trend_filter:  bool = True,    # T2: stock must be in uptrend (above EMA20)
    # Tier 3 — gap-specific ML (train first: python -m backend.ml.gap_trainer)
    use_gap_ml_confirm: bool  = False, # T3: gap-continuation XGBoost
    gap_ml_threshold:   float = 0.50,  # T3: probability threshold
    # Walk-forward validation — out-of-sample ML (tests 2023-2025 only)
    use_walkforward:    bool  = False, # retrain per period, no in-sample bias
    # Execution realism
    slippage_pct:       float = 0.0,   # one-way fill slippage at open (e.g. 0.10 = 0.10%)
) -> dict:

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading OHLCV data…")
    raw = get_all_ohlcv_for_features()
    if raw.empty:
        raise ValueError("No OHLCV data. Run the pipeline first.")

    nifty100 = {f"{s}.NS" for s in NIFTY_100_SYMBOLS}
    raw = raw[raw["ticker"].isin(nifty100)].copy()
    logger.info(f"Universe: {raw['ticker'].nunique()} stocks")

    required = {"open", "high", "low", "close"}
    missing  = required - set(raw.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── 2. Compute gap for every stock-day ────────────────────────────────
    raw["prev_close"] = raw.groupby("ticker")["close"].shift(1)
    raw["gap_pct"]    = (raw["open"] - raw["prev_close"]) / raw["prev_close"] * 100
    raw = raw.dropna(subset=["prev_close", "gap_pct"])

    # ── 3. Engineer features (always needed — used by all filters + walk-forward)
    logger.info("Engineering features...")
    try:
        nifty_df = get_nifty_features()
        feat     = engineer_features(raw, nifty_df=nifty_df)
        feat["date"] = pd.to_datetime(feat["date"])
    except Exception as e:
        logger.warning(f"Feature engineering failed: {e}")
        feat     = pd.DataFrame()
        nifty_df = pd.DataFrame()

    # ── 4. Build ML and sentiment lookups ─────────────────────────────────
    if use_walkforward:
        logger.info("Walk-forward mode: retraining per period for OOS predictions...")
        prob_lookup = build_oos_prob_lookup(feat) if (use_ml_confirm and not feat.empty) else {}
        gap_prob_lookup = (
            build_oos_gap_prob_lookup(raw, feat, gap_threshold)
            if (use_gap_ml_confirm and not feat.empty) else {}
        )
    else:
        prob_lookup     = _build_prob_lookup(raw)                    if use_ml_confirm    else {}
        gap_prob_lookup = _build_gap_prob_lookup(raw, gap_threshold) if use_gap_ml_confirm else {}

    sent_lookup = _build_sentiment_lookup() if use_sentiment else {}

    # ── 5. Build rule-based feature lookups (sector rank, RSI, volume, trend)
    sector_rank_lookup:  dict = {}
    volume_ratio_lookup: dict = {}
    rsi_lookup:          dict = {}
    trend_lookup:        dict = {}
    nifty_ret_lookup:    dict = {}

    needs_features = (use_sector_filter or use_volume_filter or
                      use_rsi_filter or use_trend_filter or use_nifty_filter)
    if needs_features and not feat.empty:
        try:
            # Nifty daily return for market-direction filter
            if use_nifty_filter and nifty_df is not None and not nifty_df.empty:
                ndf = nifty_df.copy()
                ndf["date"] = pd.to_datetime(ndf["date"])
                ndf = ndf.sort_values("date")
                if "close" in ndf.columns:
                    ndf["nifty_ret"] = ndf["close"].pct_change() * 100
                elif "nifty_ret_1d" in ndf.columns:
                    ndf["nifty_ret"] = ndf["nifty_ret_1d"]
                for _, row in ndf.iterrows():
                    nifty_ret_lookup[str(row["date"].date())] = float(
                        row.get("nifty_ret", 0.0)
                    )

            feat_sub = feat[["ticker", "date"]].copy()
            for col, lk in [
                ("sector_momentum_rank", sector_rank_lookup),
                ("volume_ratio",         volume_ratio_lookup),
                ("rsi_14",               rsi_lookup),
                ("price_vs_ema20",       trend_lookup),
            ]:
                if col in feat.columns:
                    feat_sub[col] = feat[col]
                    for _, row in feat_sub[["ticker", "date", col]].iterrows():
                        lk[(row["ticker"], str(row["date"])[:10])] = float(row[col])

        except Exception as e:
            logger.warning(f"Feature lookup build failed: {e}")

    # ── 5. Simulate trades day by day ─────────────────────────────────────
    trades       = []
    equity       = initial_capital
    equity_curve = []

    filter_stats = {
        "total_gaps":         0,
        "gap_cap_filtered":   0,
        "ml_filtered":        0,
        "sentiment_filtered": 0,
        "sector_filtered":    0,
        "volume_filtered":    0,
        "rsi_filtered":       0,
        "nifty_filtered":     0,
        "trend_filtered":     0,
        "gap_ml_filtered":    0,
        "traded":             0,
    }

    all_dates = sorted(raw["date"].unique())

    for date in all_dates:
        date_str = str(date.date())
        day_df   = raw[raw["date"] == date]

        # T1: skip entire day if market is down
        if use_nifty_filter and nifty_ret_lookup:
            nret = nifty_ret_lookup.get(date_str, 0.0)
            if nret < nifty_min_ret:
                filter_stats["nifty_filtered"] += len(
                    day_df[day_df["gap_pct"] >= gap_threshold]
                )
                continue

        gaps = day_df[
            (day_df["gap_pct"] >= gap_threshold) &
            (day_df["gap_pct"] <= gap_max_pct)
        ].copy()
        filter_stats["total_gaps"] += len(gaps)

        # Count gaps above cap separately
        over_cap = day_df[day_df["gap_pct"] > gap_max_pct]
        filter_stats["gap_cap_filtered"] += len(over_cap)

        candidates = []
        for _, row in gaps.iterrows():
            ticker = row["ticker"]

            # Layer 1: ML confirmation
            if use_ml_confirm and prob_lookup:
                prob = prob_lookup.get((ticker, date_str), 0.5)
                if prob < ml_threshold:
                    filter_stats["ml_filtered"] += 1
                    continue

            # Layer 2: Sentiment filter
            if use_sentiment and sent_lookup:
                sent = sent_lookup.get((ticker, date_str), 0.0)
                if sent < sent_threshold:
                    filter_stats["sentiment_filtered"] += 1
                    continue

            # Layer 3: Sector momentum filter
            if use_sector_filter and sector_rank_lookup:
                rank = sector_rank_lookup.get((ticker, date_str), 0.5)
                if rank < 0.5:
                    filter_stats["sector_filtered"] += 1
                    continue

            # T1: Volume confirmation — institutional participation required
            if use_volume_filter and volume_ratio_lookup:
                vol_ratio = volume_ratio_lookup.get((ticker, date_str), 1.0)
                if vol_ratio < volume_min_ratio:
                    filter_stats["volume_filtered"] += 1
                    continue

            # T1: RSI range — not overbought, not a falling knife
            if use_rsi_filter and rsi_lookup:
                rsi = rsi_lookup.get((ticker, date_str), 50.0)
                if not (rsi_low <= rsi <= rsi_high):
                    filter_stats["rsi_filtered"] += 1
                    continue

            # T2: Trend filter — stock must be above its 20-day EMA
            if use_trend_filter and trend_lookup:
                price_vs_ema20 = trend_lookup.get((ticker, date_str), 0.0)
                if price_vs_ema20 < 0:
                    filter_stats["trend_filtered"] += 1
                    continue

            # T3: Gap-specific ML — predicts gap continuation, not daily direction
            if use_gap_ml_confirm and gap_prob_lookup:
                gap_prob = gap_prob_lookup.get((ticker, date_str), 0.5)
                if gap_prob < gap_ml_threshold:
                    filter_stats["gap_ml_filtered"] += 1
                    continue

            candidates.append(row)

        # Cap at max_positions per day
        if len(candidates) > max_positions:
            # Sort by gap size — take the strongest gaps
            candidates = sorted(candidates, key=lambda r: r["gap_pct"], reverse=True)
            candidates = candidates[:max_positions]

        filter_stats["traded"] += len(candidates)

        # Simulate each trade
        day_pnl_pct = 0.0
        n = len(candidates)

        for row in candidates:
            entry = float(row["open"])
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])

            ret, reason = _simulate_trade(entry, high, low, close, target_pct, stop_pct, slippage_pct)

            day_pnl_pct += ret / max(n, 1)   # equal weight across positions

            trades.append({
                "date":        date_str,
                "ticker":      row["ticker"],
                "gap_pct":     round(float(row["gap_pct"]), 2),
                "entry":       round(entry, 2),
                "ret_pct":     round(ret, 3),
                "exit_reason": reason,
                "won":         ret > 0,
            })

        equity *= (1 + day_pnl_pct / 100)
        equity_curve.append({"date": date_str, "value": round(equity, 2)})

    # ── 6. Metrics ────────────────────────────────────────────────────────
    trades_df   = pd.DataFrame(trades)
    equity_vals = [e["value"] for e in equity_curve]

    total_return = (equity / initial_capital - 1) * 100
    n_days       = len(equity_curve)
    years        = n_days / 252
    ann_return   = ((equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    daily_rets = pd.Series(equity_vals).pct_change().dropna() * 100
    sharpe     = (
        daily_rets.mean() / daily_rets.std() * np.sqrt(252)
        if len(daily_rets) >= 5 and daily_rets.std() > 0 else 0.0
    )

    series   = pd.Series(equity_vals)
    roll_max = series.cummax()
    max_dd   = float(((series - roll_max) / roll_max * 100).min())

    n_trades  = len(trades_df)
    win_rate  = float(trades_df["won"].mean() * 100)          if n_trades > 0 else 0.0
    avg_win   = float(trades_df.loc[trades_df["won"],  "ret_pct"].mean()) if n_trades > 0 else 0.0
    avg_loss  = float(trades_df.loc[~trades_df["won"], "ret_pct"].mean()) if n_trades > 0 else 0.0
    profit_factor = (
        abs(trades_df.loc[trades_df["won"],  "ret_pct"].sum()) /
        max(abs(trades_df.loc[~trades_df["won"], "ret_pct"].sum()), 1e-8)
        if n_trades > 0 else 0.0
    )
    exit_breakdown = trades_df["exit_reason"].value_counts().to_dict() if n_trades > 0 else {}

    # Monthly returns
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    monthly_eq  = eq_df.set_index("date")["value"].resample("ME").last().ffill()
    monthly_ret = monthly_eq.pct_change().dropna() * 100
    monthly = [
        {"month": d.strftime("%b %Y"), "return_pct": round(float(r), 2)}
        for d, r in monthly_ret.items()
    ]

    # Benchmark
    bt_start = equity_curve[0]["date"]  if equity_curve else None
    bt_end   = equity_curve[-1]["date"] if equity_curve else None
    benchmark_curve  = _fetch_benchmark(bt_start, bt_end, initial_capital) if bt_start else []
    benchmark_return = 0.0
    if benchmark_curve:
        benchmark_return = round(
            (benchmark_curve[-1]["value"] / benchmark_curve[0]["value"] - 1) * 100, 2
        )
    alpha = round(total_return - benchmark_return, 2)

    warnings = []
    if n_trades < 50:
        warnings.append(f"Only {n_trades} trades — not statistically significant.")
    if years < 1:
        warnings.append(f"Backtest period is only {n_days} days — extend for reliable results.")

    logger.info(
        f"Gap & Go backtest: {total_return:.1f}% | Sharpe: {sharpe:.2f} | "
        f"Win: {win_rate:.0f}% | PF: {profit_factor:.2f} | "
        f"Trades: {n_trades} | Filter stats: {filter_stats}"
    )

    metrics = {
        "total_return_pct":    round(total_return, 2),
        "annualized_return":   round(ann_return, 2),
        "sharpe_ratio":        round(float(sharpe), 2),
        "max_drawdown_pct":    round(max_dd, 2),
        "win_rate_pct":        round(win_rate, 1),
        "avg_win_pct":         round(avg_win, 3),
        "avg_loss_pct":        round(avg_loss, 3),
        "profit_factor":       round(float(profit_factor), 3),
        "total_trades":        n_trades,
        "total_gaps_seen":     filter_stats["total_gaps"],
        "ml_filtered":         filter_stats["ml_filtered"],
        "sentiment_filtered":  filter_stats["sentiment_filtered"],
        "sector_filtered":     filter_stats["sector_filtered"],
        "volume_filtered":     filter_stats["volume_filtered"],
        "rsi_filtered":        filter_stats["rsi_filtered"],
        "nifty_filtered":      filter_stats["nifty_filtered"],
        "trend_filtered":      filter_stats["trend_filtered"],
        "gap_cap_filtered":    filter_stats["gap_cap_filtered"],
        "gap_ml_filtered":     filter_stats["gap_ml_filtered"],
        "trades_taken":        filter_stats["traded"],
        "backtest_days":       n_days,
        "backtest_start":      bt_start,
        "backtest_end":        bt_end,
        "initial_capital":     initial_capital,
        "final_capital":       round(equity, 2),
        "benchmark_return":    benchmark_return,
        "alpha":               alpha,
    }

    return {
        "metrics":         metrics,
        "equity_curve":    equity_curve,
        "benchmark_curve": benchmark_curve,
        "monthly":         monthly,
        "trades":          trades[-200:],
        "exit_breakdown":  exit_breakdown,
        "filter_stats":    {k: v for k, v in filter_stats.items()},
        "warnings":        warnings,
    }


def _fetch_benchmark(start_date: str, end_date: str, capital: float) -> list[dict]:
    try:
        import yfinance as yf
        df = yf.download("^NSEI", start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        if df.empty:
            return []
        closes = df["Close"].squeeze().dropna()
        scale  = capital / float(closes.iloc[0])
        return [{"date": str(d.date()), "value": round(float(v) * scale, 2)}
                for d, v in closes.items()]
    except Exception as e:
        logger.warning(f"Benchmark fetch failed: {e}")
        return []
