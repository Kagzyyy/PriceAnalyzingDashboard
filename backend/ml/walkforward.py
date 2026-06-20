# -*- coding: utf-8 -*-
"""
Walk-forward validation for Gap and Go ML models.

Fixes the in-sample bias where the model was trained AND tested on the
same 5-year dataset (producing artificially high WR / Sharpe).

How it works:
  Round 1: Train on 2020-01-01 to 2022-12-31  ->  predict 2023 only
  Round 2: Train on 2020-01-01 to 2023-12-31  ->  predict 2024 only
  Round 3: Train on 2020-01-01 to 2024-12-31  ->  predict 2025 only

The backtest then uses ONLY the out-of-sample predictions (2023-2025).
Results cover 3 years instead of 5, but every trade was predicted by a
model that had never seen that day during training.
"""
import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backend.ml.features import FEATURE_COLS
from backend.ml.gap_trainer import GAP_FEATURE_COLS, gap_continuation_label

logger = logging.getLogger(__name__)

# Expanding window periods: (train_end, test_start, test_end)
WALK_FORWARD_PERIODS = [
    ("2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-12-31"),
]


def _make_xgb(scale_pos_weight=1.0) -> XGBClassifier:
    """Same architecture as the production model — shallow, regularised."""
    return XGBClassifier(
        n_estimators       = 500,
        max_depth          = 4,
        learning_rate      = 0.02,
        subsample          = 0.7,
        colsample_bytree   = 0.6,
        colsample_bylevel  = 0.7,
        min_child_weight   = 15,
        gamma              = 2,
        reg_alpha          = 2.0,
        reg_lambda         = 5.0,
        scale_pos_weight   = scale_pos_weight,
        eval_metric        = "logloss",
        tree_method        = "hist",
        n_jobs             = -1,
        random_state       = 42,
        early_stopping_rounds = 30,
        verbosity          = 0,
    )


# ---------------------------------------------------------------------------
# Walk-forward for daily direction XGBoost
# ---------------------------------------------------------------------------

def build_oos_prob_lookup(feat: pd.DataFrame) -> dict:
    """
    Train daily-direction XGBoost on expanding windows, predict out-of-sample.

    Parameters
    ----------
    feat : pd.DataFrame
        Full engineered feature DataFrame (all tickers, all dates).
        Must contain FEATURE_COLS + 'label' + 'ticker' + 'date'.

    Returns
    -------
    dict  {(ticker, 'YYYY-MM-DD'): oos_prob_up}
        Only contains entries for 2023-2025 (the OOS years).
        Feature day D -> prediction applied to trade day D+1 (no look-ahead).
    """
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values(["ticker", "date"]).reset_index(drop=True)

    oos_lookup: dict = {}
    total_oos = 0

    for train_end, test_start, test_end in WALK_FORWARD_PERIODS:
        logger.info(f"WF daily XGB: train <= {train_end}  |  test {test_start} -> {test_end}")

        train_mask = feat["date"] <= train_end
        test_mask  = (feat["date"] >= test_start) & (feat["date"] <= test_end)

        train_df = feat[train_mask].dropna(subset=["label"])
        test_df  = feat[test_mask]

        if len(train_df) < 500 or len(test_df) < 10:
            logger.warning(f"  Skipping period: insufficient data")
            continue

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["label"].astype(int)

        # Validation set: last 15% of training dates (chronological)
        train_dates = sorted(train_df["date"].unique())
        val_split   = train_dates[int(len(train_dates) * 0.85)]
        val_mask    = train_df["date"] >= val_split
        tr_mask     = train_df["date"] <  val_split

        X_tr  = X_train[tr_mask.values]
        X_val = X_train[val_mask.values]
        y_tr  = y_train[tr_mask.values]
        y_val = y_train[val_mask.values]

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        spw = neg / max(pos, 1)

        model = _make_xgb(scale_pos_weight=spw)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Predict on test period
        X_test = test_df[FEATURE_COLS].fillna(0)
        if hasattr(model, "feature_names_in_"):
            for c in model.feature_names_in_:
                if c not in X_test.columns:
                    X_test[c] = 0.0
            X_test = X_test[model.feature_names_in_]

        proba = model.predict_proba(X_test)[:, 1]
        test_df = test_df.copy()
        test_df["prob_up"] = proba

        # Shift: features on day D -> probability applied to trade on day D+1
        n_added = 0
        for ticker, grp in test_df.groupby("ticker"):
            grp = grp.sort_values("date").reset_index(drop=True)
            for i in range(len(grp) - 1):
                trade_date = str(grp.loc[i + 1, "date"].date())
                oos_lookup[(ticker, trade_date)] = float(grp.loc[i, "prob_up"])
                n_added += 1

        logger.info(f"  -> {n_added} OOS prob entries added")
        total_oos += n_added

    logger.info(f"WF daily XGB complete: {total_oos} total OOS entries across 3 years")
    return oos_lookup


# ---------------------------------------------------------------------------
# Walk-forward for gap-continuation XGBoost
# ---------------------------------------------------------------------------

def build_oos_gap_prob_lookup(raw: pd.DataFrame,
                               feat: pd.DataFrame,
                               gap_threshold: float = 1.5) -> dict:
    """
    Train gap-continuation XGBoost on expanding windows, predict out-of-sample.

    Parameters
    ----------
    raw  : pd.DataFrame  Full OHLCV (ticker, date, open, high, low, close)
    feat : pd.DataFrame  Full engineered features (same index as raw after merge)
    gap_threshold : float  Minimum gap % to label as a gap-up day

    Returns
    -------
    dict  {(ticker, 'YYYY-MM-DD'): oos_gap_continuation_prob}
        Only contains entries for gap-up days in 2023-2025.
    """
    # Build gap labels for all gap-up days
    raw2 = raw.copy()
    raw2["date"] = pd.to_datetime(raw2["date"])
    raw2 = raw2.sort_values(["ticker", "date"]).reset_index(drop=True)
    raw2["prev_close"] = raw2.groupby("ticker")["close"].shift(1)
    raw2["gap_pct"]    = (raw2["open"] - raw2["prev_close"]) / raw2["prev_close"] * 100
    raw2 = raw2.dropna(subset=["prev_close", "gap_pct"])

    gap_days = raw2[raw2["gap_pct"] >= gap_threshold].copy()
    gap_days["label"] = gap_days.apply(
        lambda r: gap_continuation_label(r["open"], r["high"], r["low"], r["close"]),
        axis=1
    )

    # Build lookup: (ticker, date_str) -> (label, gap_pct)
    gap_info = {
        (row["ticker"], str(row["date"].date())): (int(row["label"]), float(row["gap_pct"]))
        for _, row in gap_days.iterrows()
    }

    # Align features
    feat2 = feat.copy()
    feat2["date"] = pd.to_datetime(feat2["date"])
    feat2 = feat2.sort_values(["ticker", "date"]).reset_index(drop=True)

    oos_lookup: dict = {}
    total_oos = 0

    for train_end, test_start, test_end in WALK_FORWARD_PERIODS:
        logger.info(f"WF gap ML: train <= {train_end}  |  test {test_start} -> {test_end}")

        # Build training rows: gap days in train period, features from prev day
        train_rows = []
        test_rows  = []
        test_keys  = []

        for ticker, grp in feat2.groupby("ticker"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                trade_date = str(grp.loc[i, "date"].date())
                key = (ticker, trade_date)
                if key not in gap_info:
                    continue
                label, cur_gap = gap_info[key]
                feat_row = grp.loc[i - 1, FEATURE_COLS].fillna(0).to_dict()
                feat_row["current_gap_pct"] = cur_gap

                row_date = grp.loc[i, "date"]
                if row_date <= pd.Timestamp(train_end):
                    feat_row["label"] = label
                    train_rows.append(feat_row)
                elif pd.Timestamp(test_start) <= row_date <= pd.Timestamp(test_end):
                    test_rows.append(feat_row)
                    test_keys.append(key)

        if len(train_rows) < 50 or len(test_rows) < 5:
            logger.warning(f"  Skipping: train={len(train_rows)} test={len(test_rows)} gap samples")
            continue

        train_df = pd.DataFrame(train_rows)
        test_df  = pd.DataFrame(test_rows)

        X_train = train_df[GAP_FEATURE_COLS].fillna(0)
        y_train = train_df["label"].astype(int)

        # Chronological val split (last 15% of training gap days)
        n_val   = max(1, int(len(X_train) * 0.15))
        X_tr    = X_train.iloc[:-n_val]
        X_val   = X_train.iloc[-n_val:]
        y_tr    = y_train.iloc[:-n_val]
        y_val   = y_train.iloc[-n_val:]

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        spw = neg / max(pos, 1)

        model = _make_xgb(scale_pos_weight=spw)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        X_test = test_df[GAP_FEATURE_COLS].fillna(0)
        if hasattr(model, "feature_names_in_"):
            for c in model.feature_names_in_:
                if c not in X_test.columns:
                    X_test[c] = 0.0
            X_test = X_test[model.feature_names_in_]

        proba = model.predict_proba(X_test)[:, 1]
        for key, p in zip(test_keys, proba):
            oos_lookup[key] = float(p)

        logger.info(f"  -> {len(test_keys)} OOS gap prob entries  "
                    f"(train WR={y_train.mean():.1%})")
        total_oos += len(test_keys)

    logger.info(f"WF gap ML complete: {total_oos} total OOS entries")
    return oos_lookup
