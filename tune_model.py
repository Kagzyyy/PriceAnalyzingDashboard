"""
Hyperparameter tuning for XGBoost — finds the best combination that
minimises overfitting and maximises test accuracy.

Uses DATE-BASED cross-validation throughout to prevent data leakage.
Data is sorted [ticker, date], so a row-index split leaks recent data
from alphabetically-early tickers into the training set. A date split
gives every ticker the same cutoff date at every fold boundary.

Label: next-day direction (close[t+1] > close[t]).
  Stays ~50/50 balanced regardless of market regime.

Minimum bars: 200 per ticker (required for reliable EMA200, 52w high/low).

Run:  C:\Python314\python.exe tune_model.py
"""
import logging
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from backend.data.storage import init_db, get_all_ohlcv_for_features, save_model_meta
from backend.ml.features import FEATURE_COLS, engineer_features
from backend.config import MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_DIST = {
    "max_depth"        : [3, 4, 5, 6],
    "learning_rate"    : [0.01, 0.02, 0.05, 0.1],
    "n_estimators"     : [300, 500, 700, 1000],
    "reg_alpha"        : [0.1, 0.5, 1.0],
    "reg_lambda"       : [1.0, 3.0, 5.0, 10.0],
    "min_child_weight" : [5, 10, 15, 20],
    "gamma"            : [0.1, 0.5, 1.0],
    "subsample"        : [0.6, 0.7, 0.8],
    "colsample_bytree" : [0.6, 0.7, 0.8],
}

N_ITER   = 40
CV_FOLDS = 5


def load_data():
    logger.info("Loading OHLCV data...")
    raw = get_all_ohlcv_for_features()
    if raw.empty:
        logger.error("No data found. Run a Refresh first.")
        sys.exit(1)

    logger.info(f"Engineering features on {len(raw)} rows...")
    from backend.data.nifty import get_nifty_features
    nifty_df = get_nifty_features()
    feat = engineer_features(raw, nifty_df=nifty_df)
    feat = feat.dropna(subset=FEATURE_COLS + ["label"], thresh=len(FEATURE_COLS) // 2 + 1)
    feat = feat.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = feat[FEATURE_COLS].fillna(0)
    y = feat["label"].astype(int)
    dates = pd.to_datetime(feat["date"])

    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Date range: {dates.min().date()} → {dates.max().date()}")
    return X, y, dates


def make_date_splits(dates: pd.Series, n_splits: int):
    """
    Build walk-forward CV splits based on unique dates, not row indices.
    Each fold trains on all dates before the fold boundary and tests on
    the next block of dates. This prevents any ticker's recent data from
    appearing in the training set of a fold whose test set contains it.
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    # Reserve last 20% for final hold-out — only split the training 80%
    train_dates = unique_dates[:int(n_dates * 0.80)]
    fold_size   = len(train_dates) // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = train_dates[fold_size * (i + 1)]
        test_end  = train_dates[min(fold_size * (i + 2), len(train_dates) - 1)]

        train_idx = np.where(dates < train_end)[0]
        test_idx  = np.where((dates >= train_end) & (dates < test_end))[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
        logger.info(
            f"  Fold {i+1}: train up to {str(train_end)[:10]}, "
            f"test {str(train_end)[:10]} → {str(test_end)[:10]} "
            f"({len(train_idx)} / {len(test_idx)} rows)"
        )

    return splits


def run_search(X: pd.DataFrame, y: pd.Series, dates: pd.Series):
    import random
    random.seed(42)
    np.random.seed(42)

    splits = make_date_splits(dates, CV_FOLDS)
    logger.info(f"\nRunning {N_ITER} random parameter combinations × {len(splits)} date-based folds...")
    logger.info("This will take ~20–40 minutes.\n")

    # Sample random combinations
    keys   = list(PARAM_DIST.keys())
    combos = []
    for _ in range(N_ITER):
        combos.append({k: random.choice(PARAM_DIST[k]) for k in keys})

    results = []
    for i, params in enumerate(combos):
        fold_train_accs = []
        fold_test_accs  = []

        for train_idx, test_idx in splits:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]

            # Carve a small validation set from end of training fold for early stopping
            val_split = int(len(X_tr) * 0.85)
            X_tr2, X_val = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
            y_tr2, y_val = y_tr.iloc[:val_split], y_tr.iloc[val_split:]

            try:
                model = XGBClassifier(
                    **params,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42,
                    early_stopping_rounds=30,
                )
                model.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
                fold_train_accs.append(accuracy_score(y_tr, model.predict(X_tr)))
                fold_test_accs.append(accuracy_score(y_te, model.predict(X_te)))
            except Exception as e:
                logger.warning(f"Combo {i+1} fold failed: {e}")
                continue

        if not fold_test_accs:
            continue

        mean_train = np.mean(fold_train_accs)
        mean_test  = np.mean(fold_test_accs)
        gap        = mean_train - mean_test

        results.append({
            "params":     params,
            "train_acc":  mean_train,
            "test_acc":   mean_test,
            "overfit_gap": gap,
        })

        logger.info(
            f"  [{i+1:2d}/{N_ITER}] test={mean_test:.4f} train={mean_train:.4f} "
            f"gap={gap:+.4f}  depth={params['max_depth']} "
            f"lr={params['learning_rate']} alpha={params['reg_alpha']}"
        )

    if not results:
        logger.error("All combinations failed.")
        sys.exit(1)

    # Sort by test accuracy, prefer lower overfit gap when tied
    results.sort(key=lambda r: (-r["test_acc"], r["overfit_gap"]))
    return results


def report_and_save(results: list, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
    print("\n" + "="*70)
    print("TOP 10 COMBINATIONS (date-based CV, no leakage)")
    print("="*70)
    print(f"{'Test Acc':>9} {'Train Acc':>9} {'Gap':>7}  Parameters")
    print("-"*70)
    for r in results[:10]:
        p = r["params"]
        print(
            f"  {r['test_acc']:.4f}    {r['train_acc']:.4f}  {r['overfit_gap']:+.4f}  "
            f"depth={p['max_depth']} lr={p['learning_rate']} "
            f"n_est={p['n_estimators']} alpha={p['reg_alpha']} "
            f"lambda={p['reg_lambda']} mcw={p['min_child_weight']}"
        )

    best_params = results[0]["params"]

    # ── Final hold-out: strict date-based 80/20 split ─────────────────────
    unique_dates = sorted(dates.unique())
    split_date   = unique_dates[int(len(unique_dates) * 0.80)]
    train_mask   = dates < split_date
    test_mask    = dates >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Validation set from end of training data
    train_dates_only = sorted(dates[train_mask].unique())
    val_date = train_dates_only[int(len(train_dates_only) * 0.85)]
    tr_mask  = dates[train_mask] < val_date
    val_mask = dates[train_mask] >= val_date

    X_tr,  X_val  = X_train[tr_mask.values],  X_train[val_mask.values]
    y_tr,  y_val  = y_train[tr_mask.values],  y_train[val_mask.values]

    logger.info(
        f"\nFinal holdout: train up to {str(split_date)[:10]}, "
        f"test from {str(split_date)[:10]} ({len(X_test)} rows)"
    )

    best_model = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
    )
    best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc  = accuracy_score(y_test,  best_model.predict(X_test))
    precision = precision_score(y_test, best_model.predict(X_test), zero_division=0)
    recall    = recall_score(y_test,    best_model.predict(X_test), zero_division=0)
    gap       = train_acc - test_acc

    print("\n" + "="*70)
    print("BEST MODEL — FINAL HOLD-OUT (date-based, zero leakage)")
    print("="*70)
    print(f"  Split date     : {str(split_date)[:10]}")
    print(f"  Train samples  : {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  Overfit gap    : {gap:+.4f}  ({'OK' if gap < 0.05 else 'OVERFITTING'})")
    print(f"  Precision      : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall         : {recall:.4f}  ({recall*100:.1f}%)")
    print()
    print("BEST PARAMETERS:")
    for k, v in sorted(best_params.items()):
        print(f"  {k:25s} = {v}")

    answer = input("\nSave this as the production model? (y/n): ").strip().lower()
    if answer == "y":
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)
        save_model_meta(
            accuracy=round(test_acc, 4),
            train_accuracy=round(train_acc, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            n_samples=len(X),
            n_stocks=0,
        )
        print(f"\nModel saved to {MODEL_PATH}")
        print("Run the Backtest page to see honest results.")
    else:
        print("Model not saved.")


if __name__ == "__main__":
    init_db()
    X, y, dates = load_data()
    results = run_search(X, y, dates)
    report_and_save(results, X, y, dates)
