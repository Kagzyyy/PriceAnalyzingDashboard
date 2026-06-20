# -*- coding: utf-8 -*-
"""
Side-by-side comparison of filter tiers for Gap & Go, including
walk-forward (OOS) validation to show honest out-of-sample numbers.

Configs compared:
  0. Raw gaps only (no ML, no filters)
  1. + Daily XGBoost (in-sample)
  2. Baseline: + Sector momentum filter (in-sample)
  3. Tier 1:   Baseline + Volume + Nifty (in-sample)
  4. Tier 2:   T1 + Trend + Gap cap 4% (in-sample)
  5. Tier 3:   T1 + Gap ML only (in-sample)
  6. All:      T1 + T2 + T3 (in-sample)
  ---
  7. WF T1:    Tier 1 with walk-forward XGBoost (OOS)
  8. WF T3:    T1 + Gap ML walk-forward (OOS)
  9. WF All:   T1 + T2 + T3 walk-forward (OOS)

Usage:
    python compare_tiers.py
"""
import sys
import os
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from backend.ml.gap_trainer import GAP_MODEL_PATH, train_gap_model
from backend.ml.gap_and_go import run_gap_and_go_backtest

W = 92  # table width


def run(label, **kwargs):
    print("  Running: {} ...".format(label), flush=True)
    result = run_gap_and_go_backtest(**kwargs)
    m = result["metrics"]
    return {
        "label":  label,
        "trades": m["total_trades"],
        "wr":     m["win_rate_pct"],
        "sharpe": m["sharpe_ratio"],
        "ret":    m["total_return_pct"],
        "maxdd":  m["max_drawdown_pct"],
        "alpha":  m["alpha"],
    }


def row(r):
    ret_s   = ("+{:.1f}%".format(r["ret"])   if r["ret"]   >= 0 else "{:.1f}%".format(r["ret"]))
    alpha_s = ("+{:.1f}%".format(r["alpha"]) if r["alpha"] >= 0 else "{:.1f}%".format(r["alpha"]))
    return (
        "{:<40}{:>7}{:>8.1f}%{:>9.2f}{:>9}{:>9.1f}%{:>9}".format(
            r["label"], r["trades"], r["wr"], r["sharpe"],
            ret_s, r["maxdd"], alpha_s
        )
    )


def main():
    if not GAP_MODEL_PATH.exists():
        print("Gap model not found - training now (~60s)...")
        _, stats = train_gap_model()
        print("  Done. Samples: {} | positive rate: {:.1%} | train AUC: {:.3f}\n".format(
            stats["n_samples"], stats["positive_rate"], stats["train_auc"]
        ))
    else:
        print("Gap model found at {}\n".format(GAP_MODEL_PATH))

    import argparse
    parser = argparse.ArgumentParser(description="Gap & Go tier comparison")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="One-way slippage %% per trade (e.g. 0.10 = 0.10%%)")
    args = parser.parse_args()
    slip = args.slippage
    if slip > 0:
        print("  Slippage mode: {:.2f}%% one-way applied to all configs\n".format(slip))

    base = dict(
        gap_threshold    = 1.5,
        gap_max_pct      = 10.0,
        target_pct       = 2.0,
        stop_pct         = 0.5,
        ml_threshold     = 0.50,
        sent_threshold   = -0.10,
        max_positions    = 5,
        initial_capital  = 100_000.0,
        use_ml_confirm     = False,
        use_sentiment      = False,
        use_sector_filter  = False,
        use_volume_filter  = False,
        use_rsi_filter     = False,
        use_nifty_filter   = False,
        use_trend_filter   = False,
        use_gap_ml_confirm = False,
        use_walkforward    = False,
        slippage_pct       = slip,
    )

    print("Running 10 configs - please wait (walk-forward configs take ~3 min each)...\n")

    results = []

    # ── In-sample configs ──────────────────────────────────────────────────
    results.append(run("0. Raw gaps (no filters)", **base))

    results.append(run("1. + Daily XGBoost",
        **{**base, "use_ml_confirm": True}))

    results.append(run("2. Baseline (XGB + Sector)",
        **{**base, "use_ml_confirm": True, "use_sector_filter": True}))

    results.append(run("3. Tier 1 (+ Volume + Nifty)",
        **{**base,
           "use_ml_confirm": True,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True}))

    results.append(run("4. Tier 2 (T1 + Trend + Gap cap 4%)",
        **{**base,
           "use_ml_confirm": True,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_trend_filter": True,
           "gap_max_pct": 4.0}))

    results.append(run("5. Tier 3 (T1 + Gap ML only)",
        **{**base,
           "use_ml_confirm": False,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_gap_ml_confirm": True}))

    results.append(run("6. All Tiers (T1 + T2 + T3)",
        **{**base,
           "use_ml_confirm": True,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_trend_filter": True,
           "use_gap_ml_confirm": True,
           "gap_max_pct": 4.0}))

    # ── Walk-forward (OOS) configs — honest numbers ────────────────────────
    print("\n  [Walk-forward configs — each retrains XGBoost 3x on expanding windows]")

    results.append(run("7. WF: T1 XGBoost (OOS 2023-25)",
        **{**base,
           "use_ml_confirm": True,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_walkforward": True}))

    results.append(run("8. WF: T1 + Gap ML (OOS 2023-25)",
        **{**base,
           "use_ml_confirm": False,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_gap_ml_confirm": True,
           "use_walkforward": True}))

    results.append(run("9. WF: All Tiers (OOS 2023-25)",
        **{**base,
           "use_ml_confirm": True,
           "use_sector_filter": True,
           "use_volume_filter": True,
           "use_nifty_filter": True,
           "use_trend_filter": True,
           "use_gap_ml_confirm": True,
           "gap_max_pct": 4.0,
           "use_walkforward": True}))

    header = "{:<40}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "Config", "Trades", "WR", "Sharpe", "Return", "MaxDD", "Alpha")

    sep_in   = "-" * W
    sep_wf   = "~" * W

    print()
    print("=" * W)
    print("  Gap & Go - Tier Comparison (In-sample vs Walk-Forward OOS)")
    print("=" * W)
    print(header)
    print(sep_in)
    for r in results[:7]:
        print(row(r))
    print(sep_wf)
    print("  ^ In-sample (train + test on same data)       v Walk-forward OOS (honest)")
    print(sep_wf)
    for r in results[7:]:
        print(row(r))
    print("=" * W)
    print()
    print("  Walk-forward covers 2023-2025 only (3 years OOS).")
    print("  In-sample covers 2020-2025 (5 years, inflated).")
    print("  A Sharpe > 1.0 OOS with reasonable trade count = deployable edge.")
    print()


if __name__ == "__main__":
    main()
