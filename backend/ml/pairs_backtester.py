"""
ML-enhanced walk-forward pairs trading backtester.

Pure stat-arb baseline:
  Enter when spread z-score crosses ±z_entry.
  Exit when |z-score| < z_exit or |z-score| > z_stop or max hold exceeded.

Three ML enhancement layers (each independently toggleable):

Layer 1 — Sentiment filter (FinBERT):
  Before entering, check if sentiment EXPLAINS the spread divergence.
  If sent_score_A and sent_score_B diverge strongly, the spread has a
  fundamental reason and may NOT mean-revert. Skip the trade.

  Example: HDFC spread diverges from ICICI.
    sent_HDFC = -0.82 (CEO exit news)
    sent_ICICI = +0.10 (neutral)
    |diff| = 0.72 > threshold → fundamental reason → SKIP

Layer 2 — XGBoost confirmation:
  Check if the model's individual stock predictions agree with the pairs signal.
  LONG spread (buy A, sell B) is confirmed if XGBoost also predicts A will
  outperform and B will underperform. Disagreement shrinks position size.

  Conviction levels:
    Both legs confirmed   → 100% position size
    One leg confirmed     → 50% position size
    Neither confirmed     → SKIP (signals conflict)

Layer 3 — Regime filter (Nifty volatility):
  During crisis regimes (Nifty 20-day vol > 2.5× long-run average),
  cointegration relationships break down. All pairs signals suppressed.

  Example: COVID March 2020 — all sector correlations collapsed temporarily.
  Without this filter, the model would keep entering pairs as spreads widen,
  only to see them widen further (divergence, not reversion).
"""
import logging
import pickle
import numpy as np
import pandas as pd

from backend.config import MODEL_PATH
from backend.data.storage import get_all_ohlcv_for_features, get_sentiment
from backend.data.symbols import get_symbol_map, NIFTY_100_SYMBOLS
from backend.data.nifty import get_nifty_features
from backend.ml.pairs import find_cointegrated_pairs
from backend.ml.features import FEATURE_COLS, engineer_features

logger = logging.getLogger(__name__)

TRANSACTION_COST_PER_LEG    = 0.20
ROUND_TRIP_COST             = TRANSACTION_COST_PER_LEG * 2 * 2   # both legs, entry + exit

SENTIMENT_DIVERGENCE_THRESH = 0.35   # |sent_A - sent_B| above this → skip
CRISIS_VOL_MULTIPLIER       = 2.5    # Nifty vol > 2.5× long-run → crisis regime


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _build_ml_lookup(raw: pd.DataFrame, split_date) -> dict:
    """
    Run XGBoost on the test period and return a lookup:
      {(ticker, 'YYYY-MM-DD'): prob_up}
    Used for ML confirmation in the trade loop.
    """
    if not MODEL_PATH.exists():
        logger.warning("No trained model found — ML confirmation disabled.")
        return {}

    try:
        logger.info("Generating XGBoost predictions for ML confirmation layer…")
        nifty_df = get_nifty_features()
        feat     = engineer_features(raw, nifty_df=nifty_df)
        feat     = feat.sort_values(["ticker", "date"])
        test_df  = feat[feat["date"] >= split_date].copy()

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        X = test_df[FEATURE_COLS].fillna(0)
        if hasattr(model, "feature_names_in_"):
            trained = list(model.feature_names_in_)
            for c in trained:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[trained]

        proba = model.predict_proba(X)
        test_df = test_df.copy()
        test_df["prob_up"] = proba[:, 1]

        lookup = {
            (row["ticker"], str(row["date"])[:10]): float(row["prob_up"])
            for _, row in test_df[["ticker", "date", "prob_up"]].iterrows()
        }
        logger.info(f"ML lookup built: {len(lookup)} (ticker, date) entries")
        return lookup

    except Exception as e:
        logger.warning(f"ML lookup failed ({e}) — ML confirmation disabled.")
        return {}


def _build_sentiment_lookup(split_date) -> dict:
    """
    Build 3-day rolling average sentiment per (ticker, date).
    Returns {(ticker, 'YYYY-MM-DD'): avg_sent_score_3d}
    """
    try:
        sent = get_sentiment(days=1825)
        if sent.empty:
            return {}

        sent["date"] = pd.to_datetime(sent["date"])
        sent = sent.sort_values(["ticker", "date"])

        # 3-day rolling average per ticker
        sent["sent_3d"] = (
            sent.groupby("ticker")["sent_score"]
                .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

        lookup = {
            (row["ticker"], str(row["date"].date())): float(row["sent_3d"])
            for _, row in sent[sent["date"] >= pd.Timestamp(split_date)][
                ["ticker", "date", "sent_3d"]
            ].iterrows()
        }
        logger.info(f"Sentiment lookup built: {len(lookup)} entries")
        return lookup

    except Exception as e:
        logger.warning(f"Sentiment lookup failed ({e}) — sentiment filter disabled.")
        return {}


def _build_crisis_days(nifty_df: pd.DataFrame) -> set:
    """
    Flag dates where Nifty 20-day volatility exceeds 2.5× its long-run average.
    These are crisis/OOD regime days where cointegration breaks down.
    """
    try:
        nifty = nifty_df.copy()
        nifty["date"] = pd.to_datetime(nifty["date"])
        nifty = nifty.sort_values("date").set_index("date")

        if "close" not in nifty.columns:
            return set()

        ret         = nifty["close"].pct_change().dropna()
        vol_20      = ret.rolling(20).std()
        vol_longrun = ret.rolling(252).std()

        crisis = vol_20[vol_20 > CRISIS_VOL_MULTIPLIER * vol_longrun].index
        crisis_days = {str(d.date()) for d in crisis}
        logger.info(f"Crisis regime days identified: {len(crisis_days)}")
        return crisis_days

    except Exception as e:
        logger.warning(f"Crisis day detection failed ({e}) — regime filter disabled.")
        return set()


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_pairs_backtest(
    z_entry:            float = 2.0,
    z_exit:             float = 0.5,
    z_stop:             float = 3.0,
    max_pairs:          int   = 5,
    max_holding_days:   int   = 20,
    initial_capital:    float = 100_000.0,
    sector_only:        bool  = True,
    use_ml_confirm:     bool  = True,
    use_sentiment:      bool  = True,
    use_regime_filter:  bool  = True,
) -> dict:

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading OHLCV data…")
    raw = get_all_ohlcv_for_features()
    if raw.empty:
        raise ValueError("No OHLCV data. Run the pipeline first.")

    prices = (
        raw.pivot_table(index="date", columns="ticker", values="close")
           .sort_index()
    )
    prices.index = pd.to_datetime(prices.index)

    # Filter to Nifty 100 universe — drops thin mid/small-caps that produce
    # spurious cointegration signals and have poor data quality.
    nifty100_tickers = {f"{s}.NS" for s in NIFTY_100_SYMBOLS}
    available_n100   = [c for c in prices.columns if c in nifty100_tickers]
    if len(available_n100) >= 30:   # need enough for pairs
        prices = prices[available_n100]
        logger.info(f"Filtered to Nifty 100: {len(available_n100)} tickers available")
    else:
        logger.warning(f"Only {len(available_n100)} Nifty 100 tickers found — using all {len(prices.columns)}")

    # ── 2. 80/20 split ────────────────────────────────────────────────────
    all_dates  = sorted(prices.index)
    split_idx  = int(len(all_dates) * 0.80)
    split_date = all_dates[split_idx]

    formation_px = prices.loc[prices.index <= split_date]
    trading_px   = prices.loc[prices.index >  split_date]

    logger.info(
        f"Formation: {str(all_dates[0])[:10]} → {str(split_date)[:10]} | "
        f"Trading: {str(split_date)[:10]} → {str(all_dates[-1])[:10]}"
    )

    # ── 3. Find cointegrated pairs on formation period ────────────────────
    # Pairs needs wider universe (Nifty 100) to find enough cointegrated pairs.
    # The directional model uses nifty50 (set in config.UNIVERSE), but we
    # override to nifty100 here so we have C(100,2)=4950 candidate pairs.
    symbol_map = get_symbol_map(universe="nifty100")
    sector_map = {t: symbol_map.get(t, {}).get("sector", "Unknown") for t in prices.columns}

    logger.info("Running cointegration tests…")
    pairs_df = find_cointegrated_pairs(
        formation_px, sector_map,
        sector_only=sector_only,
        max_pairs=max_pairs * 10,
    )

    if pairs_df.empty:
        # If same-sector yielded nothing, automatically retry cross-sector
        if sector_only:
            logger.warning("No cointegrated pairs found with sector_only=True — retrying cross-sector…")
            pairs_df = find_cointegrated_pairs(
                formation_px, sector_map,
                sector_only=False,
                max_pairs=max_pairs * 10,
            )
        if pairs_df.empty:
            raise ValueError(
                "No cointegrated pairs found even cross-sector. "
                "The formation period may contain structural breaks (mergers, crises). "
                "Try a shorter lookback or different universe."
            )

    # Half-life filter — pairs that revert too fast (noise) or too slow (no signal)
    hl_filtered = pairs_df[
        pairs_df["half_life"].notna() &
        (pairs_df["half_life"] >= 2) &
        (pairs_df["half_life"] <= 60)
    ].head(max_pairs).reset_index(drop=True)

    # If half-life filter removes everything, relax it rather than crashing
    if hl_filtered.empty:
        logger.warning("Half-life filter (2–60 days) removed all pairs — using top pairs without HL filter")
        pairs_df = pairs_df.head(max_pairs).reset_index(drop=True)
    else:
        pairs_df = hl_filtered

    logger.info(f"Selected {len(pairs_df)} pairs")

    # ── 4. Formation period spread statistics ─────────────────────────────
    pair_stats = []
    for _, row in pairs_df.iterrows():
        t1, t2 = row["ticker_a"], row["ticker_b"]
        beta   = row["hedge_ratio"]
        f_data = formation_px[[t1, t2]].dropna()
        if f_data.empty:
            continue
        spread = np.log(f_data[t1].astype(float)) - beta * np.log(f_data[t2].astype(float))
        pair_stats.append({
            **row.to_dict(),
            "formation_mean": float(spread.mean()),
            "formation_std":  float(spread.std()),
        })

    if not pair_stats:
        raise ValueError("Could not compute spread stats for any pair.")

    # ── 5. Build enhancement layer lookups ────────────────────────────────
    ml_lookup   = _build_ml_lookup(raw, split_date)    if use_ml_confirm    else {}
    sent_lookup = _build_sentiment_lookup(split_date)  if use_sentiment     else {}

    crisis_days = set()
    if use_regime_filter:
        try:
            nifty_df    = get_nifty_features()
            crisis_days = _build_crisis_days(nifty_df)
        except Exception as e:
            logger.warning(f"Regime filter setup failed: {e}")

    # Track filter statistics
    filter_stats = {
        "total_signals":    0,
        "regime_blocked":   0,
        "sentiment_blocked": 0,
        "ml_skipped":       0,
        "ml_half_size":     0,
        "traded":           0,
    }

    # ── 6. Walk-forward simulation ────────────────────────────────────────
    equity         = initial_capital
    equity_curve   = []
    trades         = []
    open_positions = {}

    trading_dates = sorted(trading_px.index)

    for date in trading_dates:
        date_str    = str(date.date())
        day_pnl_pct = 0.0

        for ps in pair_stats:
            pair_id = ps["pair_id"]
            t1, t2  = ps["ticker_a"], ps["ticker_b"]
            beta    = ps["hedge_ratio"]
            f_mean  = ps["formation_mean"]
            f_std   = ps["formation_std"]

            try:
                p1 = float(trading_px.loc[date, t1])
                p2 = float(trading_px.loc[date, t2])
            except KeyError:
                continue
            if np.isnan(p1) or np.isnan(p2) or p1 <= 0 or p2 <= 0:
                continue

            z = (np.log(p1) - beta * np.log(p2) - f_mean) / max(f_std, 1e-8)
            pair_weight = 1.0 / len(pair_stats)

            # ── Manage open position ──────────────────────────────────────
            if pair_id in open_positions:
                pos = open_positions[pair_id]
                pos["days_held"] += 1

                exit_reason = None
                if abs(z) > z_stop:
                    exit_reason = "stop_loss"
                elif abs(z) < z_exit:
                    exit_reason = "mean_reversion"
                elif pos["days_held"] >= max_holding_days:
                    exit_reason = "max_holding"

                if exit_reason:
                    direction  = pos["direction"]
                    entry_p1   = pos["entry_p1"]
                    entry_p2   = pos["entry_p2"]
                    conviction = pos["conviction"]

                    ret_p1 = (p1 / entry_p1 - 1) * 100
                    ret_p2 = (p2 / entry_p2 - 1) * 100

                    gross_ret = (ret_p1 - ret_p2) / 2 if direction == "LONG" else (ret_p2 - ret_p1) / 2
                    net_ret   = gross_ret - ROUND_TRIP_COST

                    day_pnl_pct += pair_weight * conviction * net_ret

                    trades.append({
                        "pair":        pair_id,
                        "sector":      ps["sector"],
                        "date_entry":  pos["entry_date"],
                        "date_exit":   date_str,
                        "direction":   direction,
                        "days_held":   pos["days_held"],
                        "entry_z":     round(float(pos["entry_z"]), 2),
                        "exit_z":      round(float(z), 2),
                        "gross_ret":   round(float(gross_ret), 3),
                        "trade_ret":   round(float(net_ret), 3),
                        "conviction":  round(float(conviction), 2),
                        "exit_reason": exit_reason,
                        "ml_confirmed":  pos.get("ml_confirmed", "N/A"),
                        "sent_filtered": pos.get("sent_filtered", False),
                        "won":         net_ret > 0,
                    })
                    del open_positions[pair_id]

            # ── Check for new entry ───────────────────────────────────────
            else:
                if abs(z) <= z_entry:
                    continue

                direction = "SHORT" if z > z_entry else "LONG"
                filter_stats["total_signals"] += 1

                # ── Layer 3: Regime filter ────────────────────────────────
                if use_regime_filter and date_str in crisis_days:
                    filter_stats["regime_blocked"] += 1
                    continue

                # ── Layer 1: Sentiment filter ─────────────────────────────
                sent_filtered = False
                if use_sentiment and sent_lookup:
                    s1 = sent_lookup.get((t1, date_str), None)
                    s2 = sent_lookup.get((t2, date_str), None)
                    if s1 is not None and s2 is not None:
                        if abs(s1 - s2) > SENTIMENT_DIVERGENCE_THRESH:
                            filter_stats["sentiment_blocked"] += 1
                            sent_filtered = True
                            continue

                # ── Layer 2: ML confirmation ──────────────────────────────
                conviction   = 1.0
                ml_confirmed = "no_model"

                if use_ml_confirm and ml_lookup:
                    prob_a = ml_lookup.get((t1, date_str), 0.5)
                    prob_b = ml_lookup.get((t2, date_str), 0.5)

                    if direction == "LONG":
                        # Long A, Short B → XGBoost confirms if A predicted to outperform, B underperform
                        agrees_a = prob_a > 0.5
                        agrees_b = prob_b < 0.5
                    else:
                        # Short A, Long B → XGBoost confirms if A predicted to underperform, B outperform
                        agrees_a = prob_a < 0.5
                        agrees_b = prob_b > 0.5

                    if agrees_a and agrees_b:
                        conviction   = 1.0
                        ml_confirmed = "full"
                    elif agrees_a or agrees_b:
                        conviction   = 0.5
                        ml_confirmed = "partial"
                        filter_stats["ml_half_size"] += 1
                    else:
                        filter_stats["ml_skipped"] += 1
                        continue   # both legs conflict → skip

                filter_stats["traded"] += 1
                open_positions[pair_id] = {
                    "direction":     direction,
                    "entry_z":       float(z),
                    "entry_date":    date_str,
                    "entry_p1":      p1,
                    "entry_p2":      p2,
                    "days_held":     0,
                    "conviction":    conviction,
                    "ml_confirmed":  ml_confirmed,
                    "sent_filtered": sent_filtered,
                }

        equity *= (1 + day_pnl_pct / 100)
        equity_curve.append({"date": date_str, "value": round(equity, 2)})

    # ── 7. Force-close open positions ─────────────────────────────────────
    for pair_id, pos in list(open_positions.items()):
        ps = next((p for p in pair_stats if p["pair_id"] == pair_id), None)
        if not ps:
            continue
        trades.append({
            "pair":         pair_id,
            "sector":       ps["sector"],
            "date_entry":   pos["entry_date"],
            "date_exit":    str(trading_dates[-1].date()),
            "direction":    pos["direction"],
            "days_held":    pos["days_held"],
            "entry_z":      round(float(pos["entry_z"]), 2),
            "exit_z":       0.0,
            "gross_ret":    0.0,
            "trade_ret":    round(-ROUND_TRIP_COST, 3),
            "conviction":   pos.get("conviction", 1.0),
            "exit_reason":  "end_of_period",
            "ml_confirmed": pos.get("ml_confirmed", "N/A"),
            "sent_filtered": False,
            "won":          False,
        })

    # ── 8. Metrics ────────────────────────────────────────────────────────
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
    max_dd   = ((series - roll_max) / roll_max * 100).min()

    n_trades   = len(trades_df)
    win_rate   = trades_df["won"].mean() * 100          if not trades_df.empty else 0.0
    avg_win    = trades_df.loc[trades_df["won"],  "trade_ret"].mean() if not trades_df.empty else 0.0
    avg_loss   = trades_df.loc[~trades_df["won"], "trade_ret"].mean() if not trades_df.empty else 0.0
    avg_hold   = trades_df["days_held"].mean()          if not trades_df.empty else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    exit_breakdown = (
        trades_df["exit_reason"].value_counts().to_dict()
        if not trades_df.empty else {}
    )

    # Monthly returns
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    monthly_eq  = eq_df.set_index("date")["value"].resample("ME").last().ffill()
    monthly_ret = monthly_eq.pct_change().dropna() * 100
    monthly = [
        {"month": d.strftime("%b %Y"), "return_pct": round(float(r), 2)}
        for d, r in monthly_ret.items()
    ]

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
    if n_trades < 10:
        warnings.append(f"Only {n_trades} trades. Lower z_entry or use sector_only=False.")
    if filter_stats["traded"] == 0:
        warnings.append("All signals were filtered out. Try disabling some filters.")
    if sharpe > 2.5:
        warnings.append(f"Sharpe {sharpe:.2f} > 2.5 — treat as upper bound.")
    if n_days < 180:
        warnings.append(f"Test window is only {n_days} days.")

    metrics = {
        "total_return_pct":     round(total_return, 2),
        "annualized_return":    round(ann_return, 2),
        "sharpe_ratio":         round(float(sharpe), 2),
        "max_drawdown_pct":     round(float(max_dd), 2),
        "win_rate_pct":         round(float(win_rate), 1),
        "avg_win_pct":          round(float(avg_win), 3),
        "avg_loss_pct":         round(float(avg_loss), 3),
        "expectancy_pct":       round(float(expectancy), 3),
        "total_trades":         n_trades,
        "avg_holding_days":     round(float(avg_hold), 1),
        "n_pairs":              len(pair_stats),
        "backtest_days":        n_days,
        "backtest_start":       bt_start,
        "backtest_end":         bt_end,
        "initial_capital":      initial_capital,
        "final_capital":        round(equity, 2),
        "benchmark_return":     benchmark_return,
        "alpha":                alpha,
        "transaction_cost_pct": ROUND_TRIP_COST,
    }

    logger.info(
        f"Pairs backtest: {total_return:.1f}% | Nifty: {benchmark_return:.1f}% | "
        f"Alpha: {alpha:+.1f}% | Sharpe: {sharpe:.2f} | Win: {win_rate:.0f}% | "
        f"Trades: {n_trades} | Filter stats: {filter_stats}"
    )

    pairs_summary = [
        {
            "pair_id":     ps["pair_id"],
            "symbol_a":    ps["symbol_a"],
            "symbol_b":    ps["symbol_b"],
            "sector":      ps["sector"],
            "pvalue":      ps["pvalue"],
            "hedge_ratio": ps["hedge_ratio"],
            "half_life":   ps["half_life"],
            "correlation": ps["correlation"],
            "n_obs":       ps["n_obs"],
            "trades":      len([t for t in trades if t["pair"] == ps["pair_id"]]),
        }
        for ps in pair_stats
    ]

    return {
        "metrics":         metrics,
        "equity_curve":    equity_curve,
        "benchmark_curve": benchmark_curve,
        "monthly":         monthly,
        "trades":          trades[-200:],
        "pairs":           pairs_summary,
        "exit_breakdown":  exit_breakdown,
        "filter_stats":    filter_stats,
        "warnings":        warnings,
    }
