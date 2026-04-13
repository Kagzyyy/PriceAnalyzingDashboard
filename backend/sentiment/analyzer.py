"""
FinBERT sentiment analyzer.
Model: ProsusAI/finbert — runs fully locally, no API key needed.
Downloads ~500 MB on first use, then cached permanently.

Based on research findings:
- Extract all 3 scores (positive, negative, neutral) separately — better than single compound
- Z-score normalization using 30-day rolling window
- 14-day lookback window for rolling averages
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_pipeline = None
_BATCH_SIZE = 16   # smaller batches = fewer parsing edge cases
_MAX_LEN = 512


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT (first time downloads ~500 MB, cached after)...")
            _pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
                device=-1,         # CPU
                truncation=True,
                max_length=_MAX_LEN,
            )
            logger.info("FinBERT ready.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            _pipeline = None
    return _pipeline


def _parse_result(result) -> tuple[float, float, float, float]:
    """
    Parse FinBERT output into (compound, positive, negative, neutral).
    Handles all output formats across transformers versions.
    Returns (compound, pos, neg, neu) where compound = pos - neg in [-1, 1].
    """
    try:
        # Unwrap double nesting: [[{...}]] → [{...}]
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]

        # Normalise: single dict → list of dicts
        if isinstance(result, dict):
            result = [result]

        if isinstance(result, list):
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            for item in result:
                # Guard against string iteration edge-case
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "neutral")).lower().strip()
                score = float(item.get("score", 0.0))
                if label in scores:
                    scores[label] = score

            pos = scores["positive"]
            neg = scores["negative"]
            neu = scores["neutral"]
            compound = pos - neg   # [-1, +1]
            return compound, pos, neg, neu

    except Exception as e:
        logger.debug(f"FinBERT parse error: {e} | type={type(result)}")

    return 0.0, 0.0, 0.0, 1.0


def score_headlines(texts: list[str]) -> pd.DataFrame:
    """
    Run FinBERT on a list of headlines.
    Returns DataFrame with columns: compound, sent_pos, sent_neg, sent_neu
    """
    pipe = _load_pipeline()
    empty = pd.DataFrame({
        "compound": [0.0] * len(texts),
        "sent_pos":  [0.0] * len(texts),
        "sent_neg":  [0.0] * len(texts),
        "sent_neu":  [1.0] * len(texts),
    })

    if pipe is None:
        return empty

    rows = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = [str(t)[:_MAX_LEN] for t in texts[i: i + _BATCH_SIZE]]
        try:
            results = pipe(batch)
            for r in results:
                compound, pos, neg, neu = _parse_result(r)
                rows.append({"compound": compound, "sent_pos": pos, "sent_neg": neg, "sent_neu": neu})
        except Exception as e:
            logger.warning(f"FinBERT batch error: {e}")
            for _ in batch:
                rows.append({"compound": 0.0, "sent_pos": 0.0, "sent_neg": 0.0, "sent_neu": 1.0})

    return pd.DataFrame(rows)


def analyze(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FinBERT on scraped headlines and aggregate to daily scores per ticker.

    Input:  DataFrame with columns [ticker, headline, pub_date]
    Output: DataFrame with columns:
            ticker, date, sentiment_score (compound), sent_pos, sent_neg, sent_neu,
            news_count, sent_zscore
    """
    if headlines_df.empty:
        logger.warning("No headlines to analyze.")
        return pd.DataFrame(columns=[
            "ticker", "date", "sentiment_score",
            "sent_pos", "sent_neg", "sent_neu", "news_count", "sent_zscore"
        ])

    logger.info(f"Running FinBERT on {len(headlines_df)} headlines...")
    texts = headlines_df["headline"].fillna("").tolist()
    scores_df = score_headlines(texts)

    df = headlines_df.copy().reset_index(drop=True)
    df = pd.concat([df, scores_df], axis=1)

    # Aggregate to ticker + date level (equal-weight average per paper recommendation)
    agg = (
        df.groupby(["ticker", "pub_date"])
        .agg(
            sentiment_score=("compound",  "mean"),
            sent_pos       =("sent_pos",  "mean"),
            sent_neg       =("sent_neg",  "mean"),
            sent_neu       =("sent_neu",  "mean"),
            news_count     =("compound",  "count"),
        )
        .reset_index()
        .rename(columns={"pub_date": "date"})
    )

    for col in ["sentiment_score", "sent_pos", "sent_neg", "sent_neu"]:
        agg[col] = agg[col].astype(float).round(4)

    agg["date"] = pd.to_datetime(agg["date"])

    # Z-score normalization per ticker (30-day rolling window)
    agg = agg.sort_values(["ticker", "date"])
    roll_mean = agg.groupby("ticker")["sentiment_score"].transform(lambda x: x.rolling(30, min_periods=3).mean())
    roll_std  = agg.groupby("ticker")["sentiment_score"].transform(lambda x: x.rolling(30, min_periods=3).std())
    agg["sent_zscore"] = ((agg["sentiment_score"] - roll_mean) / roll_std.replace(0, 1)).fillna(0.0).round(4)

    logger.info(
        f"Sentiment computed for {agg['ticker'].nunique()} stocks — "
        f"avg compound: {agg['sentiment_score'].mean():.3f} | "
        f"avg pos: {agg['sent_pos'].mean():.3f} | "
        f"avg neg: {agg['sent_neg'].mean():.3f}"
    )
    return agg
