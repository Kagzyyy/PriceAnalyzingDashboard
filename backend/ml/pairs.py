"""
Pairs identification using Engle-Granger cointegration testing.

What is cointegration?
  Two stocks are cointegrated if their prices share a long-run equilibrium.
  They can diverge short-term but the spread (log(p1) - β*log(p2)) always
  mean-reverts. This is stronger than correlation — two stocks can be highly
  correlated yet not cointegrated (both trend up but at different rates).

Why log prices?
  log(p1) - β*log(p2) creates a stationary spread even when both prices are
  non-stationary (random walks). Stationarity is required for mean reversion.

Hedge ratio (β):
  Found via OLS: log(p1) = β*log(p2) + α + ε
  β tells us: for every 1% move in stock B, stock A moves β%.
  We hold β units of B for every 1 unit of A — this makes the spread
  dollar-neutral (market-neutral).

Half-life of mean reversion:
  Estimated from the Ornstein-Uhlenbeck process:
    Δspread(t) = λ * spread(t-1) + ε
    half_life  = -ln(2) / λ
  A half-life of 5 days means the spread decays 50% toward its mean every 5 days.
  Too short (<2 days) = too much noise. Too long (>60 days) = too slow to trade.
"""
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS = 200   # at least ~10 months of overlapping data
COINT_PVALUE     = 0.10  # Engle-Granger threshold — 0.10 is standard in pairs literature
                          # Engle-Granger has low power on ~800 obs; 0.05 rejects too many real pairs
MIN_CORRELATION  = 0.60  # reject pairs with weak linear relationship (incl. negative β pairs)
                          # Negative β means both stocks move in OPPOSITE directions — not a real pair


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    sector_map: dict,
    sector_only: bool = True,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """
    Test all candidate pairs for cointegration on the formation period prices.

    Args:
        prices:      Wide DataFrame — index=date, columns=tickers, values=close price
        sector_map:  ticker → sector string
        sector_only: If True, only test pairs within the same sector
                     (economically justified, fewer spurious pairs)
        max_pairs:   Maximum pairs to return (ranked by p-value)

    Returns:
        DataFrame with columns: ticker_a, ticker_b, symbol_a, symbol_b, sector,
        pvalue, hedge_ratio, half_life, correlation, n_obs, pair_id
    """
    tickers = [
        c for c in prices.columns
        if prices[c].dropna().shape[0] >= MIN_OBSERVATIONS
    ]

    results = []
    tested  = 0

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]

            # Sector filter — only pairs with an economic relationship
            if sector_only:
                s1 = sector_map.get(t1, "Unknown")
                s2 = sector_map.get(t2, "Unknown")
                if s1 != s2 or s1 == "Unknown":
                    continue

            # Align both series
            common = prices[[t1, t2]].dropna()
            if len(common) < MIN_OBSERVATIONS:
                continue

            log_p1 = np.log(common[t1].values.astype(float))
            log_p2 = np.log(common[t2].values.astype(float))
            tested += 1

            try:
                # Engle-Granger cointegration test
                _, pvalue, _ = coint(log_p1, log_p2)
                if pvalue >= COINT_PVALUE:
                    continue

                # Economic sanity: reject pairs with weak or negative correlation.
                # Negative correlation → β will be negative → not a market-neutral pair.
                # Low correlation → cointegration is likely spurious.
                corr = float(np.corrcoef(log_p1, log_p2)[0, 1])
                if corr < MIN_CORRELATION:
                    continue

                # Hedge ratio via OLS: log(p1) = β*log(p2) + α
                X    = add_constant(log_p2)
                res  = OLS(log_p1, X).fit()
                beta = float(res.params[1])

                # Reject negative hedge ratio: means the two stocks move in OPPOSITE
                # directions — not a real economic relationship for pairs trading.
                if beta <= 0:
                    continue

                # Spread = log(p1) - β*log(p2)
                spread = pd.Series(log_p1 - beta * log_p2)

                # Half-life via Ornstein-Uhlenbeck:
                # Δspread = λ*spread(t-1) + ε  →  half_life = -ln(2)/λ
                # Use .values to get numpy arrays so add_constant produces
                # integer-indexed params (params[0]=const, params[1]=lambda)
                spread_lag  = spread.shift(1).dropna().values
                spread_diff = spread.diff().dropna().values
                hl_res      = OLS(spread_diff, add_constant(spread_lag)).fit()
                lam         = float(hl_res.params[1])
                half_life   = round(-np.log(2) / lam, 1) if lam < 0 else None

                results.append({
                    "ticker_a":    t1,
                    "ticker_b":    t2,
                    "symbol_a":    t1.replace(".NS", ""),
                    "symbol_b":    t2.replace(".NS", ""),
                    "sector":      sector_map.get(t1, "Unknown"),
                    "pvalue":      round(float(pvalue), 4),
                    "hedge_ratio": round(float(beta), 4),
                    "half_life":   half_life,
                    "correlation": round(corr, 4),
                    "n_obs":       len(common),
                })

            except Exception:
                continue

    logger.info(
        f"Cointegration: tested {tested} pairs, "
        f"found {len(results)} significant (p < {COINT_PVALUE})"
    )

    if not results:
        return pd.DataFrame()

    df = (
        pd.DataFrame(results)
        .sort_values("pvalue")
        .head(max_pairs)
        .reset_index(drop=True)
    )
    df["pair_id"] = df["symbol_a"] + "/" + df["symbol_b"]
    return df


def compute_zscore(
    log_p1: pd.Series,
    log_p2: pd.Series,
    hedge_ratio: float,
    formation_mean: float,
    formation_std:  float,
) -> pd.Series:
    """
    Compute z-score of the spread using formation-period statistics.

    z-score > +2: spread is above its historical mean → expect reversion DOWN
    z-score < -2: spread is below its historical mean → expect reversion UP
    """
    spread = log_p1 - hedge_ratio * log_p2
    return (spread - formation_mean) / max(formation_std, 1e-8)
