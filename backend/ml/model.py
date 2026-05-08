"""
XGBoost universal model — trained once on all NSE stocks.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from backend.config import MODEL_PATH, UNIVERSE
from backend.ml.features import FEATURE_COLS, engineer_features
from backend.data.storage import save_model_meta

logger = logging.getLogger(__name__)


def _filter_to_universe(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter OHLCV data to the configured universe (default: nifty50).
    The parquet may contain nifty100 data (needed for pairs), but the
    directional model only trains/predicts on the configured universe.
    """
    if UNIVERSE == "all":
        return ohlcv_df
    from backend.data.symbols import get_tickers
    allowed = set(get_tickers(universe=UNIVERSE))
    filtered = ohlcv_df[ohlcv_df["ticker"].isin(allowed)]
    logger.info(
        f"Universe filter ({UNIVERSE}): {ohlcv_df['ticker'].nunique()} → "
        f"{filtered['ticker'].nunique()} stocks"
    )
    return filtered


def train_model(ohlcv_df: pd.DataFrame) -> dict:
    """
    Train XGBoost on the configured universe (default: nifty50).
    Returns accuracy metrics dict.
    """
    ohlcv_df = _filter_to_universe(ohlcv_df)
    logger.info("Engineering features for training...")
    from backend.data.nifty import get_nifty_features
    nifty_df = get_nifty_features()
    feat_df = engineer_features(ohlcv_df, nifty_df=nifty_df)

    if feat_df.empty:
        raise ValueError("No feature data generated")

    # Drop rows with too many NaNs
    feat_df = feat_df.dropna(subset=FEATURE_COLS + ["label"], thresh=len(FEATURE_COLS) // 2 + 1)

    X = feat_df[FEATURE_COLS].fillna(0)
    y = feat_df["label"].astype(int)

    logger.info(f"Training on {len(X)} samples, {X.shape[1]} features, {feat_df['ticker'].nunique()} stocks")

    # ── Date-based split (critical: prevents data leakage) ────────────────
    # Data is sorted by [ticker, date]. A row-index split (shuffle=False) would
    # put alphabetically-early tickers' recent dates into the training set while
    # the backtester tests on those same dates → inflated metrics.
    # Splitting by unique date ensures every ticker's recent data is held out.
    all_dates  = sorted(feat_df["date"].unique())
    split_date = all_dates[int(len(all_dates) * 0.80)]
    train_mask = feat_df["date"] < split_date
    test_mask  = feat_df["date"] >= split_date

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    logger.info(
        f"Date split at {str(split_date)[:10]} — "
        f"train: {len(X_train)} rows, test: {len(X_test)} rows"
    )

    # Validation set carved from the END of training data (chronological)
    val_dates  = sorted(feat_df[train_mask]["date"].unique())
    val_split  = val_dates[int(len(val_dates) * 0.85)]
    val_mask   = feat_df[train_mask]["date"] >= val_split
    tr_mask    = feat_df[train_mask]["date"] <  val_split

    X_tr  = X_train[tr_mask.values]
    X_val = X_train[val_mask.values]
    y_tr  = y_train[tr_mask.values]
    y_val = y_train[val_mask.values]

    model = XGBClassifier(
        # Capacity — kept shallow to avoid memorising noise
        n_estimators=1000,        # high ceiling; early stopping will cut it
        max_depth=4,              # was 6 — shallower trees generalise better
        learning_rate=0.02,       # was 0.05 — slower learning, less overfitting

        # Sampling regularisation
        subsample=0.7,            # was 0.8 — use 70% of rows per tree
        colsample_bytree=0.6,     # was 0.8 — use 60% of features per tree
        colsample_bylevel=0.7,    # additionally subsample per depth level

        # Splitting regularisation
        min_child_weight=20,      # was 5 — require more samples per leaf
        gamma=2,                  # was 1 — higher min-split gain needed

        # Weight regularisation (L1 + L2)
        reg_alpha=2.0,            # was 0.1 — stronger L1 (sparsity)
        reg_lambda=5.0,           # was 1.0 — stronger L2 (weight shrinkage)

        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50, # stop if val logloss doesn't improve for 50 rounds
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    train_pred = model.predict(X_train)
    y_pred     = model.predict(X_test)

    train_acc  = accuracy_score(y_train, train_pred)
    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    gap        = train_acc - accuracy

    logger.info(f"Train accuracy : {train_acc:.4f}")
    logger.info(f"Test  accuracy : {accuracy:.4f}  (gap={gap:+.4f})")
    logger.info(f"Precision      : {precision:.4f}")
    logger.info(f"Recall         : {recall:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save metadata
    save_model_meta(
        accuracy=round(accuracy, 4),
        train_accuracy=round(train_acc, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        n_samples=len(X),
        n_stocks=feat_df["ticker"].nunique(),
    )

    return {
        "accuracy": accuracy,
        "train_accuracy": train_acc,
        "precision": precision,
        "recall": recall,
        "overfit_gap": gap,
        "n_samples": len(X),
        "n_stocks": feat_df["ticker"].nunique(),
    }


def load_model() -> XGBClassifier:
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train_model() first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(ohlcv_df: pd.DataFrame, sentiment_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Given recent OHLCV data, generate predictions for universe stocks only.
    Returns DataFrame: [ticker, date, prob_up, prob_down, direction, price, change_pct, volume, ...]
    """
    ohlcv_df = _filter_to_universe(ohlcv_df)
    model = load_model()

    logger.info("Engineering features for prediction...")
    from backend.data.nifty import get_nifty_features
    nifty_df = get_nifty_features()
    feat_df = engineer_features(ohlcv_df, sentiment_df=sentiment_df, nifty_df=nifty_df)

    if feat_df.empty:
        logger.warning("No feature data for prediction")
        return pd.DataFrame()

    # Use only the LATEST row per ticker (predict tomorrow from today's data)
    latest = feat_df.sort_values("date").groupby("ticker").last().reset_index()
    latest = latest.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)

    # Ensure feature columns exactly match training order (guards against any
    # column ordering differences between engineer_features calls)
    X = latest[FEATURE_COLS].fillna(0)

    # Align column names to what the model was trained on (handles XGBoost
    # feature_names_in_ validation introduced in XGBoost >= 1.6)
    if hasattr(model, "feature_names_in_"):
        trained_cols = list(model.feature_names_in_)
        missing = [c for c in trained_cols if c not in X.columns]
        if missing:
            logger.warning(f"Adding {len(missing)} missing feature columns as zeros: {missing}")
            for c in missing:
                X[c] = 0.0
        extra = [c for c in X.columns if c not in trained_cols]
        if extra:
            logger.warning(f"Dropping {len(extra)} extra feature columns not in training: {extra}")
        X = X[trained_cols]

    proba = model.predict_proba(X)
    latest["prob_up"] = proba[:, 1]
    latest["prob_down"] = proba[:, 0]
    latest["direction"] = latest["prob_up"].apply(lambda p: "UP" if p >= 0.5 else "DOWN")
    latest["price"] = latest["close"]
    latest["change_pct"] = latest["ret_1d"].fillna(0)
    latest["prediction_date"] = date.today()

    # Pick columns for storage
    cols = [
        "ticker", "prediction_date", "prob_up", "prob_down", "direction",
        "price", "change_pct", "volume",
        "rsi_14", "macd", "bb_pct", "volume_ratio",
    ]
    # Rename rsi_14 → rsi for storage
    out = latest[cols].copy()
    out = out.rename(columns={"rsi_14": "rsi"})

    logger.info(f"Predictions generated for {len(out)} stocks")
    return out


def get_feature_importance() -> pd.DataFrame:
    """Return feature importances from the trained model."""
    model = load_model()
    importance = model.feature_importances_
    return pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False)


if __name__ == "__main__":
    import sys
    from backend.data.fetcher import fetch_training_data
    from backend.data.storage import init_db, upsert_ohlcv

    logging.basicConfig(level=logging.INFO)
    init_db()

    print("Fetching training data...")
    df = fetch_training_data()
    print(f"Fetched {len(df)} rows")

    upsert_ohlcv(df)

    print("Training model...")
    metrics = train_model(df)
    print(f"Done! Accuracy: {metrics['accuracy']:.2%}")
