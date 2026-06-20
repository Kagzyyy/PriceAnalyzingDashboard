"""Quick diagnostic — run this to find exactly why cointegration finds 0 pairs."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from backend.data.storage import get_all_ohlcv_for_features

print("=== 1. Loading OHLCV data ===")
raw = get_all_ohlcv_for_features()
print(f"Rows: {len(raw)}, Tickers: {raw['ticker'].nunique()}")
print(f"Date range: {raw['date'].min()} to {raw['date'].max()}")
print(f"Duplicate (ticker,date) pairs: {raw.duplicated(['ticker','date']).sum()}")
print()

print("=== 2. Building price matrix ===")
prices = raw.pivot_table(index="date", columns="ticker", values="close").sort_index()
prices.index = pd.to_datetime(prices.index)
print(f"Price matrix shape: {prices.shape}")
print(f"NaN count per ticker (sample): {prices.isnull().sum().sort_values(ascending=False).head(10).to_dict()}")
print()

print("=== 3. Formation period prices ===")
all_dates = sorted(prices.index)
split_idx  = int(len(all_dates) * 0.80)
split_date = all_dates[split_idx]
formation_px = prices.loc[prices.index <= split_date]
print(f"Formation rows: {len(formation_px)}, split at {str(split_date)[:10]}")
print()

print("=== 4. Testing cointegration on a known pair (HDFCBANK vs ICICIBANK) ===")
from statsmodels.tsa.stattools import coint

for t1, t2 in [("HDFCBANK.NS", "ICICIBANK.NS"), ("TCS.NS", "INFY.NS"), ("HINDUNILVR.NS", "BRITANNIA.NS")]:
    if t1 not in formation_px.columns or t2 not in formation_px.columns:
        print(f"{t1}/{t2}: not in price matrix")
        continue
    common = formation_px[[t1, t2]].dropna()
    print(f"{t1}/{t2}: {len(common)} common observations")
    if len(common) < 50:
        print("  → too few observations")
        continue
    log_p1 = np.log(common[t1].values.astype(float))
    log_p2 = np.log(common[t2].values.astype(float))
    try:
        score, pvalue, crits = coint(log_p1, log_p2)
        print(f"  score={score:.4f}, p-value={pvalue:.4f}, crits={crits}")
    except Exception as e:
        print(f"  ERROR: {e}")
print()

print("=== 5. P-value distribution across 20 random pairs ===")
tickers = [c for c in formation_px.columns if formation_px[c].dropna().shape[0] >= 200]
print(f"Eligible tickers: {len(tickers)}")
pvalues = []
import random; random.seed(42)
pairs_sample = [(tickers[i], tickers[j]) for i in range(len(tickers)) for j in range(i+1, len(tickers))]
random.shuffle(pairs_sample)
for t1, t2 in pairs_sample[:20]:
    common = formation_px[[t1, t2]].dropna()
    if len(common) < 200: continue
    log_p1 = np.log(common[t1].values.astype(float))
    log_p2 = np.log(common[t2].values.astype(float))
    try:
        _, pvalue, _ = coint(log_p1, log_p2)
        pvalues.append(pvalue)
        print(f"  {t1.replace('.NS','')} / {t2.replace('.NS','')}: p={pvalue:.4f}")
    except Exception as e:
        print(f"  {t1}/{t2}: ERROR {e}")

if pvalues:
    print(f"\nMin p-value: {min(pvalues):.4f}")
    print(f"Max p-value: {max(pvalues):.4f}")
    print(f"Mean p-value: {np.mean(pvalues):.4f}")
    print(f"Pairs with p < 0.10: {sum(p < 0.10 for p in pvalues)}/{len(pvalues)}")
