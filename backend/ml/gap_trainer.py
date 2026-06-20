"""
Train a gap-continuation XGBoost classifier (Tier 3).

The standard daily-direction model answers: "will this stock close higher today?"
This model answers: "given the stock gapped up, will it hit +2% target
before reversing to -0.5% stop?" — the actual question gap-and-go needs answered.

A stock can gap up, hit +2% by 11am, then reverse and close flat.
The daily model marks that as a loss and avoids it. This model correctly
calls it a win because the gap continued.

Usage:
    python -m backend.ml.gap_trainer          # train and save
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from backend.config import MODEL_PATH
from backend.data.storage import get_all_ohlcv_for_features
from backend.data.symbols import NIFTY_100_SYMBOLS
from backend.ml.features import FEATURE_COLS, engineer_features
from backend.data.nifty import get_nifty_features

logger = logging.getLogger(__name__)

GAP_MODEL_PATH = Path(MODEL_PATH).parent / "gap_model.pkl"
ROUND_TRIP_COST = 0.20

# Gap model uses all standard features + current day's gap size.
# current_gap_pct is observable at market open (not look-ahead).
GAP_FEATURE_COLS = list(FEATURE_COLS) + ["current_gap_pct"]


def gap_continuation_label(open_: float, high: float, low: float, close: float,
                            target_pct: float = 2.0, stop_pct: float = 0.5) -> int:
    """
    Binary label: 1 = gap trade won, 0 = gap trade lost.
    Uses identical OHLC simulation logic as _simulate_trade in gap_and_go.py.
    """
    target    = open_ * (1 + target_pct / 100)
    stop      = open_ * (1 - stop_pct  / 100)
    hit_target = high >= target
    hit_stop   = low  <= stop

    if hit_target and not hit_stop:
        return 1
    elif hit_stop and not hit_target:
        return 0
    elif hit_target and hit_stop:
        return 1 if close >= open_ else 0
    else:
        net = (close - open_) / open_ * 100 - ROUND_TRIP_COST
        return 1 if net > 0 else 0


def train_gap_model(gap_threshold: float = 1.5,
                    target_pct:    float = 2.0,
                    stop_pct:      float = 0.5) -> tuple:
    """
    Train and save a gap-continuation XGBoost classifier.
    Returns (model, stats_dict).
    """
    logger.info("Loading OHLCV for gap model training…")
    raw      = get_all_ohlcv_for_features()
    nifty100 = {f"{s}.NS" for s in NIFTY_100_SYMBOLS}
    raw      = raw[raw["ticker"].isin(nifty100)].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    raw["prev_close"] = raw.groupby("ticker")["close"].shift(1)
    raw["gap_pct"]    = (raw["open"] - raw["prev_close"]) / raw["prev_close"] * 100
    raw = raw.dropna(subset=["prev_close", "gap_pct"])

    gap_days = raw[raw["gap_pct"] >= gap_threshold].copy()
    gap_days["label"] = gap_days.apply(
        lambda r: gap_continuation_label(r["open"], r["high"], r["low"], r["close"],
                                         target_pct, stop_pct), axis=1
    )
    logger.info(f"Gap-up days: {len(gap_days)} | "
                f"win rate (label=1): {gap_days['label'].mean():.1%}")

    gap_info: dict = {
        (row["ticker"], str(row["date"].date())): (int(row["label"]), float(row["gap_pct"]))
        for _, row in gap_days.iterrows()
    }

    logger.info("Engineering features…")
    nifty_df = get_nifty_features()
    feat     = engineer_features(raw, nifty_df=nifty_df)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values(["ticker", "date"]).reset_index(drop=True)

    # For each gap day use PREVIOUS day's features (no look-ahead bias).
    # current_gap_pct is added separately — it IS observable at market open.
    rows = []
    for ticker, grp in feat.groupby("ticker"):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            trade_date = str(grp.loc[i, "date"].date())
            key        = (ticker, trade_date)
            if key not in gap_info:
                continue
            label, cur_gap = gap_info[key]
            feat_row = grp.loc[i - 1, FEATURE_COLS].fillna(0).to_dict()
            feat_row["current_gap_pct"] = cur_gap
            feat_row["label"]           = label
            rows.append(feat_row)

    if not rows:
        raise ValueError("No gap training rows — check data availability.")

    df = pd.DataFrame(rows)
    X  = df[GAP_FEATURE_COLS]
    y  = df["label"]

    pos = int(y.sum())
    neg = len(y) - pos
    logger.info(f"Training samples: {len(df)} | positive={pos} ({y.mean():.1%}) | negative={neg}")

    model = XGBClassifier(
        n_estimators       = 300,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = neg / max(pos, 1),
        random_state       = 42,
        eval_metric        = "logloss",
        verbosity          = 0,
    )
    model.fit(X, y)

    preds      = model.predict(X)
    probs      = model.predict_proba(X)[:, 1]
    train_acc  = float((preds == y).mean())
    try:
        train_auc = float(roc_auc_score(y, probs))
    except Exception:
        train_auc = 0.5

    logger.info(f"Gap model ready — acc {train_acc:.1%}  AUC {train_auc:.3f}")

    GAP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GAP_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved → {GAP_MODEL_PATH}")

    return model, {
        "n_samples":      len(df),
        "positive_rate":  float(y.mean()),
        "train_accuracy": train_acc,
        "train_auc":      train_auc,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _, stats = train_gap_model()
    print("\n=== Gap Model Training Complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
