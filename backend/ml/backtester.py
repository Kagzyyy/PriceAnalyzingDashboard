"""
Walk-forward backtester — Cross-sectional Relative Return Strategy.

Strategy:
  - Label: stock_5d_return > nifty_5d_return (outperforms market in 5 days)
  - Signal: every 5 trading days (weekly rebalancing)
  - Entry:  close price on signal day
  - Exit:   close price 5 trading days later
  - Stop:   triggered if MIN low over next 5 days breaches stop price (longs)
            triggered if MAX high over next 5 days breaches stop price (shorts)
  - Sizing: equal-weight, max `max_positions` per signal day
  - Costs:  0.20% round-trip (brokerage + STT + exchange + SEBI + slippage)

Why weekly rebalancing:
  - 5x fewer transactions → 5x less drag from costs
  - 5-day return has meaningfully more signal than next-day (momentum + sentiment effects)
  - Market-neutral relative return label is stable in all regimes (always ~50/50)

NSE round-trip cost: ~0.20% (brokerage 0.03% + STT 0.025% + exchange 0.005% +
                               SEBI 0.0002% + GST 0.006% + slippage 0.05% = ~0.12-0.20%)
"""
import logging
import pickle
import numpy as np
import pandas as pd

from backend.config import MODEL_PATH, UNIVERSE
from backend.data.storage import get_all_ohlcv_for_features
from backend.data.nifty import get_nifty_features
from backend.ml.features import FEATURE_COLS, engineer_features

logger = logging.getLogger(__name__)

TRANSACTION_COST_PCT = 0.20


def _fetch_benchmark(start_date: str, end_date: str, initial_capital: float) -> list[dict]:
    try:
        import yfinance as yf
        for ticker in ["^NSEI", "NIFTYBEES.NS"]:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
            if not df.empty:
                break
        if df.empty:
            return []
        closes = df["Close"].squeeze().dropna()
        if len(closes) == 0:
            return []
        scale = initial_capital / float(closes.iloc[0])
        return [{"date": str(d.date()), "value": round(float(v) * scale, 2)}
                for d, v in closes.items()]
    except Exception as e:
        logger.warning(f"Benchmark fetch failed: {e}")
        return []


def run_backtest(
    confidence_threshold: float = 0.55,
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 5.0,
    max_positions: int = 10,
    initial_capital: float = 100_000.0,
    trade_direction: str = "both",
) -> dict:
    """
    Weekly-rebalancing cross-sectional relative return backtest.
    Signals are taken every 5 trading days. Positions held for 5 days.
    Stop/target checked against 5-day min-low / max-high.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run the pipeline first.")

    # ── 1. Load data & features ────────────────────────────────────────────
    logger.info("Loading OHLCV data…")
    raw = get_all_ohlcv_for_features()
    if raw.empty:
        raise ValueError("No OHLCV data. Run the pipeline first.")

    # Filter to same universe the model was trained on (default: nifty50).
    # Parquet may contain nifty100 data (for pairs), but the backtest must be
    # consistent with the production model universe.
    if UNIVERSE != "all":
        from backend.data.symbols import get_tickers
        allowed = set(get_tickers(universe=UNIVERSE))
        raw = raw[raw["ticker"].isin(allowed)]
        logger.info(f"Universe filter ({UNIVERSE}): {raw['ticker'].nunique()} stocks in backtest")

    logger.info("Loading Nifty features…")
    nifty_df = get_nifty_features()

    logger.info("Engineering features…")
    feat = engineer_features(raw, nifty_df=nifty_df)
    if feat.empty:
        raise ValueError("Feature engineering produced no rows.")

    feat = feat.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── 2. 5-day forward returns & stop simulation columns ─────────────────
    feat["next_5d_ret_pct"] = (
        feat.groupby("ticker")["close"].shift(-5) / feat["close"] - 1
    ) * 100

    # Min LOW and Max HIGH over the next 5 trading days (for intraday stop/target)
    low_fwd  = pd.concat([feat.groupby("ticker")["low"].shift(-i)  for i in range(1, 6)], axis=1)
    high_fwd = pd.concat([feat.groupby("ticker")["high"].shift(-i) for i in range(1, 6)], axis=1)
    feat["next_5d_min_low"]  = low_fwd.min(axis=1)
    feat["next_5d_max_high"] = high_fwd.max(axis=1)

    feat = feat.dropna(subset=["next_5d_ret_pct"])

    # ── 3. 80/20 date split (identical to training split) ─────────────────
    all_dates  = sorted(feat["date"].unique())
    split_idx  = int(len(all_dates) * 0.80)
    test_start = all_dates[split_idx]
    test_df    = feat[feat["date"] >= test_start].copy()
    logger.info(
        f"Backtest: {str(test_start)[:10]} → {str(all_dates[-1])[:10]} "
        f"({len(test_df)} rows, {test_df['ticker'].nunique()} tickers)"
    )

    # ── 4. Model predictions ───────────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = test_df[FEATURE_COLS].fillna(0)
    if hasattr(model, "feature_names_in_"):
        trained_cols = list(model.feature_names_in_)
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[trained_cols]

    proba = model.predict_proba(X)
    test_df = test_df.copy()
    test_df["prob_up"]   = proba[:, 1]   # prob stock beats Nifty in 5 days
    test_df["prob_down"] = proba[:, 0]   # prob stock underperforms Nifty

    # ── 5. Weekly signal dates (every 5th trading day) ─────────────────────
    test_dates   = sorted(test_df["date"].unique())
    signal_dates = set(test_dates[i] for i in range(0, len(test_dates), 5))
    logger.info(f"Signal days: {len(signal_dates)} (rebalancing every 5 trading days)")

    # ── 6. Simulate trades ────────────────────────────────────────────────
    # confidence_threshold is reinterpreted as top_pct:
    #   0.05 = trade top & bottom 5% of ranked stocks
    #   0.10 = trade top & bottom 10%  (default)
    #   0.15 = trade top & bottom 15%
    #   0.20 = trade top & bottom 20%
    top_pct = confidence_threshold

    equity       = initial_capital
    equity_curve = []
    trades       = []

    for trade_date in test_dates:
        day_str = str(trade_date)[:10]

        if trade_date not in signal_dates:
            equity_curve.append({"date": day_str, "value": round(equity, 2)})
            continue

        day = test_df[test_df["date"] == trade_date].copy()

        # ── Z-score percentile ranking ─────────────────────────────────────
        # Rank prob_up across all stocks on this day (0=worst, 1=best).
        # This always generates signals regardless of absolute probability
        # values — we're asking "which stocks are best/worst TODAY", not
        # "which stocks exceed an arbitrary threshold".
        day["pct_rank"] = day["prob_up"].rank(pct=True)

        # Top X% → long (highest relative probability of beating Nifty)
        long_cands  = day[day["pct_rank"] >= (1 - top_pct)].copy()
        long_cands["_dir"]     = "LONG"
        long_cands["strength"] = (long_cands["prob_up"] - 0.5).clip(lower=1e-6)

        # Bottom X% → short (lowest relative probability = predicted underperformers)
        short_cands = day[day["pct_rank"] <= top_pct].copy()
        short_cands["_dir"]     = "SHORT"
        short_cands["strength"] = (0.5 - short_cands["prob_up"]).clip(lower=1e-6)

        if trade_direction == "long":
            signals = long_cands
        elif trade_direction == "short":
            signals = short_cands
        else:
            signals = pd.concat([long_cands, short_cands])

        if signals.empty:
            equity_curve.append({"date": day_str, "value": round(equity, 2)})
            continue

        # Take the strongest signals up to max_positions
        signals = signals.nlargest(max_positions, "strength").copy()

        # ── Signal-proportional sizing ─────────────────────────────────────
        # Weight each position proportional to its signal strength
        # (distance of prob_up from 0.5). Stronger signals get more capital.
        # Cap any single position at 20% to avoid concentration.
        total_strength  = signals["strength"].sum()
        signals["weight"] = (signals["strength"] / total_strength).clip(upper=0.20)
        signals["weight"] = signals["weight"] / signals["weight"].sum()  # re-normalise

        day_ret = 0.0

        for _, row in signals.iterrows():
            next_5d_ret   = row["next_5d_ret_pct"]
            direction     = row["_dir"]
            entry         = row["close"]
            next_min_low  = row["next_5d_min_low"]
            next_max_high = row["next_5d_max_high"]
            weight        = row["weight"]

            # ── Stop/target simulation over 5-day window ──────────────────
            # Checks if the worst/best price over 5 days triggers stop/target.
            # Conservative: if both stop and target are hit, assume stop first.
            if direction == "LONG":
                stop_px   = entry * (1 - stop_loss_pct   / 100)
                target_px = entry * (1 + take_profit_pct / 100)

                if pd.notna(next_min_low) and next_min_low <= stop_px:
                    gross_ret = -stop_loss_pct
                elif pd.notna(next_max_high) and next_max_high >= target_px:
                    gross_ret = take_profit_pct
                else:
                    gross_ret = next_5d_ret

            else:  # SHORT
                stop_px   = entry * (1 + stop_loss_pct   / 100)
                target_px = entry * (1 - take_profit_pct / 100)

                if pd.notna(next_max_high) and next_max_high >= stop_px:
                    gross_ret = -stop_loss_pct
                elif pd.notna(next_min_low) and next_min_low <= target_px:
                    gross_ret = take_profit_pct
                else:
                    gross_ret = -next_5d_ret

            net_ret  = gross_ret - TRANSACTION_COST_PCT
            day_ret += weight * net_ret

            trades.append({
                "ticker":     row["ticker"].replace(".NS", ""),
                "date":       day_str,
                "direction":  direction,
                "confidence": round(float(row["pct_rank"]) * 100, 1),
                "next_ret":   round(float(next_5d_ret), 3),
                "gross_ret":  round(float(gross_ret), 3),
                "trade_ret":  round(float(net_ret), 3),
                "won":        net_ret > 0,
            })

        equity *= (1 + day_ret / 100)
        equity_curve.append({"date": day_str, "value": round(equity, 2)})

    # ── 7. Metrics ─────────────────────────────────────────────────────────
    trades_df   = pd.DataFrame(trades)
    equity_vals = [e["value"] for e in equity_curve]

    total_return_pct = (equity / initial_capital - 1) * 100
    n_days  = len(equity_curve)
    years   = n_days / 252
    ann_return = ((equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    daily_rets = pd.Series(equity_vals).pct_change().dropna() * 100
    if len(daily_rets) >= 5 and daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    signal_dates_str = {str(d)[:10] for d in signal_dates}
    active_days = len(signal_dates_str & set(t["date"] for t in trades)) if trades else 0

    series   = pd.Series(equity_vals)
    roll_max = series.cummax()
    max_dd   = ((series - roll_max) / roll_max * 100).min()

    win_rate  = trades_df["won"].mean() * 100 if not trades_df.empty else 0
    avg_win   = trades_df.loc[trades_df["won"],  "trade_ret"].mean() if not trades_df.empty else 0
    avg_loss  = trades_df.loc[~trades_df["won"], "trade_ret"].mean() if not trades_df.empty else 0
    n_trades  = len(trades_df)
    n_longs   = int((trades_df["direction"] == "LONG").sum())  if not trades_df.empty else 0
    n_shorts  = int((trades_df["direction"] == "SHORT").sum()) if not trades_df.empty else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    max_loss  = trades_df["trade_ret"].min() if not trades_df.empty else 0.0

    # ── Monthly returns ────────────────────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")
    monthly_eq  = eq_df["value"].resample("ME").last().ffill()
    monthly_ret = monthly_eq.pct_change().dropna() * 100
    monthly = [
        {"month": d.strftime("%b %Y"), "return_pct": round(float(r), 2)}
        for d, r in monthly_ret.items()
    ]

    # ── Quarterly returns ──────────────────────────────────────────────────
    quarterly_eq  = eq_df["value"].resample("QE").last().ffill()
    quarterly_ret = quarterly_eq.pct_change().dropna() * 100
    quarterly = [
        {"quarter": d.to_period("Q").strftime("Q%q %Y"), "return_pct": round(float(r), 2)}
        for d, r in quarterly_ret.items()
    ]

    # ── Feature importance ─────────────────────────────────────────────────
    fi = sorted(zip(FEATURE_COLS, model.feature_importances_),
                key=lambda x: x[1], reverse=True)[:20]
    feature_importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in fi]

    # ── Benchmark ─────────────────────────────────────────────────────────
    bt_start = equity_curve[0]["date"]  if equity_curve else None
    bt_end   = equity_curve[-1]["date"] if equity_curve else None
    benchmark_curve  = _fetch_benchmark(bt_start, bt_end, initial_capital) if bt_start else []
    benchmark_return = 0.0
    if benchmark_curve:
        benchmark_return = round(
            (benchmark_curve[-1]["value"] / benchmark_curve[0]["value"] - 1) * 100, 2
        )
    alpha = round(total_return_pct - benchmark_return, 2)

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings = []
    if n_days < 180:
        warnings.append(
            f"Short test window ({n_days} days). Annualized return extrapolates "
            "this period to a full year — treat it as indicative only."
        )
    if active_days < 10:
        warnings.append(
            f"Only {active_days} active rebalancing days. Lower the confidence "
            "threshold to generate more signals."
        )
    if sharpe > 2.5:
        warnings.append(
            f"Portfolio Sharpe of {sharpe:.2f} is above 2.5. Live execution with "
            "real slippage will reduce this — treat as upper bound."
        )
    if benchmark_return > 0 and alpha < 2:
        warnings.append(
            f"Nifty 50 returned {benchmark_return:.1f}% in the same period. "
            f"Alpha over buy-and-hold is {alpha:.1f}%."
        )

    metrics = {
        "total_return_pct":     round(total_return_pct, 2),
        "annualized_return":    round(ann_return, 2),
        "sharpe_ratio":         round(float(sharpe), 2),
        "max_drawdown_pct":     round(float(max_dd), 2),
        "win_rate_pct":         round(float(win_rate), 1),
        "avg_win_pct":          round(float(avg_win), 3),
        "avg_loss_pct":         round(float(avg_loss), 3),
        "expectancy_pct":       round(float(expectancy), 3),
        "total_trades":         n_trades,
        "long_trades":          n_longs,
        "short_trades":         n_shorts,
        "backtest_days":        n_days,
        "active_days":          active_days,
        "max_single_loss_pct":  round(float(max_loss), 3),
        "backtest_start":       bt_start,
        "backtest_end":         bt_end,
        "initial_capital":      initial_capital,
        "final_capital":        round(equity, 2),
        "benchmark_return":     benchmark_return,
        "alpha":                alpha,
        "transaction_cost_pct": TRANSACTION_COST_PCT,
    }

    logger.info(
        f"Backtest: {total_return_pct:.1f}% | Nifty: {benchmark_return:.1f}% | "
        f"Alpha: {alpha:+.1f}% | Sharpe: {sharpe:.2f} | DD: {max_dd:.1f}% | "
        f"Win: {win_rate:.0f}% ({n_trades} trades, {len(signal_dates)} rebalance days)"
    )

    return {
        "metrics":            metrics,
        "equity_curve":       equity_curve,
        "benchmark_curve":    benchmark_curve,
        "monthly":            monthly,
        "quarterly":          quarterly,
        "trades":             trades[-100:],
        "feature_importance": feature_importance,
        "warnings":           warnings,
    }
