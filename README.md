# NSE Stock Prediction Dashboard

XGBoost-powered stock direction predictor for all NSE stocks with a React dashboard and RAG research assistant.

## Features
- **XGBoost ML model** trained on 34 technical indicators across all NSE stocks
- **Long/Short rankings** with confidence scores (updated daily at 8:45 AM IST)
- **Sector heatmap** showing bullish/bearish pressure per sector
- **Candlestick charts** with EMA and volume overlays per stock
- **RAG Research Assistant** — ask natural language questions about today's market
- **Auto-scheduler** — runs pipeline every morning before market open

## Quick Start

### 1. Add your Anthropic API key
Edit `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

### 2. First-time model training
Double-click `train_model.bat`
- Downloads 2 years of NSE data (~10-30 min)
- Trains universal XGBoost on ~1M data points
- Saves model to `models/xgb_universal.pkl`

### 3. Launch dashboard
Double-click `start.bat`
- Opens http://localhost:5173 in your browser
- API docs at http://localhost:8000/docs

### 4. Daily use
- Dashboard auto-refreshes at **8:45 AM IST** before NSE opens at 9:15 AM
- Click **Refresh** button in the navbar for manual refresh

## Architecture

```
Backend (FastAPI)        Frontend (React + Vite)
├── Data Pipeline        ├── Dashboard (heatmap + top picks)
│   ├── yfinance         ├── Rankings (long/short table)
│   └── DuckDB           ├── Stock Detail (chart + indicators)
├── ML Pipeline          └── Chat (RAG assistant)
│   ├── 34 features
│   └── XGBoost
└── RAG
    ├── ChromaDB
    └── Claude API
```

## Model Details

**Algorithm:** XGBoost (universal model trained on all stocks)

**Features (34 total):**
- Momentum: RSI(7,14), MACD, ROC(5,10,20), Stochastic, Williams %R
- Trend: EMA(9,20,50,200) ratios, ADX
- Volatility: Bollinger %B, ATR, daily range
- Volume: OBV, volume ratio vs 20-day avg, VWAP deviation
- Pattern: Gap %, inside bar, candle body/wick ratios
- Position: % from 52-week high/low

**Expected accuracy:**
- All predictions: ~54-58%
- >60% confidence: ~58-64%
- >70% confidence: ~62-68%

## Requirements
- Python 3.11+
- Node.js 18+
- 4GB RAM (for model training)
- Internet connection (for yfinance data)
