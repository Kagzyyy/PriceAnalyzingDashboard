# Alpha Edge — NSE Intelligence Platform
### Full Project Documentation (v2.0)
> Last updated: 2026-05-19

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Design System](#3-design-system)
4. [Backend — Python / FastAPI](#4-backend)
   - 4.1 [Entry Point](#41-entry-point--mainpy)
   - 4.2 [Config](#42-config--configpy)
   - 4.3 [Data Layer](#43-data-layer)
   - 4.4 [ML Layer](#44-ml-layer)
   - 4.5 [Services Layer](#45-services-layer)
   - 4.6 [API Routes](#46-api-routes)
5. [Frontend — React / TypeScript](#5-frontend)
   - 5.1 [App Shell & Routing](#51-app-shell--routing)
   - 5.2 [Pages](#52-pages)
   - 5.3 [Components](#53-components)
   - 5.4 [Types](#54-types)
   - 5.5 [API Client](#55-api-client)
6. [Gap & Go Strategy — Full Details](#6-gap--go-strategy)
7. [God's Eye View — Macro Prediction Engine](#7-gods-eye-view)
8. [Trade Intelligence System](#8-trade-intelligence-system)
9. [ML Models](#9-ml-models)
10. [Planned: Local RAG + Ollama Engine](#10-planned-local-rag--ollama-engine)
11. [Environment Variables](#11-environment-variables)
12. [Running the Project](#12-running-the-project)
13. [Pending Tasks](#13-pending-tasks)

---

## 1. Project Overview

**Alpha Edge** is a Palantir/Bloomberg-quality NSE (National Stock Exchange of India) intelligence platform. It is a two-tab web app with:

| Tab | Purpose |
|-----|---------|
| **God's Eye View** | Full-viewport 3D globe with AI-generated forward macro predictions |
| **Gap & Go** | Backtester for the gap-continuation intraday strategy with AI trade reasoning |

**Core idea:** Fetch real market data → run ML models → surface actionable insights on a premium dark-terminal UI.

**Tech stack:**
- **Backend:** Python 3.11, FastAPI, XGBoost, DuckDB, yfinance, APScheduler
- **Frontend:** React 18, TypeScript, Vite, react-globe.gl, Recharts, Lucide Icons
- **AI:** Groq (LLaMA 3.3 70B) for predictions + Anthropic Claude Haiku for trade reasoning
- **Styling:** Inter + JetBrains Mono, Palantir dark theme

---

## 2. Architecture

```
PriceAnalyzingDashboard/
├── backend/
│   ├── main.py                    # FastAPI app entry point + lifespan
│   ├── config.py                  # All env vars, paths, constants
│   ├── scheduler.py               # APScheduler background jobs
│   ├── api/
│   │   ├── routes/
│   │   │   ├── predictions.py     # GET /predictions, /rankings/longs, /rankings/shorts
│   │   │   ├── stocks.py          # GET /stocks/:symbol, /stocks?q=
│   │   │   ├── sectors.py         # GET /sectors
│   │   │   ├── assistant.py       # POST /assistant/query, GET /assistant/suggestions
│   │   │   ├── system.py          # GET /status, POST /refresh, GET /breadth
│   │   │   ├── backtest.py        # GET /backtest
│   │   │   ├── pairs_backtest.py  # GET /pairs-backtest
│   │   │   ├── gap_backtest.py    # GET /gap-backtest, POST /train-gap-model
│   │   │   ├── trade_intel.py     # GET /trade-intel/:ticker/:date
│   │   │   └── market_pulse.py    # GET /market-pulse
│   │   └── schemas.py
│   ├── data/
│   │   ├── fetcher.py             # yfinance OHLCV download
│   │   ├── storage.py             # DuckDB read/write
│   │   ├── nifty.py               # Nifty 50 index features
│   │   ├── symbols.py             # NSE ticker lists (Nifty50, 100, all)
│   │   └── earnings.py            # Earnings calendar
│   ├── ml/
│   │   ├── model.py               # XGBoost direction model (train + predict)
│   │   ├── features.py            # Feature engineering (ta library)
│   │   ├── ranking.py             # Stock ranking by prob_up
│   │   ├── backtester.py          # Directional model backtester
│   │   ├── pairs.py               # Statistical pairs detection (ADF/cointegration)
│   │   ├── pairs_backtester.py    # Pairs trading backtester
│   │   ├── gap_and_go.py          # Gap & Go strategy + ML filters
│   │   ├── gap_trainer.py         # Tier-3 gap-continuation XGBoost model
│   │   └── walkforward.py         # Walk-forward OOS validation
│   ├── services/
│   │   ├── market_pulse.py        # God's Eye View — 30-day macro predictions
│   │   └── trade_intel.py         # Per-trade AI reasoning + globe events
│   ├── sentiment/
│   │   ├── analyzer.py            # FinBERT sentiment scoring
│   │   └── scraper.py             # News scraping for sentiment
│   └── rag/
│       ├── assistant.py           # RAG-based Q&A assistant
│       ├── indexer.py             # ChromaDB document indexer
│       └── embeddings.py          # Embedding utilities
│
├── frontend/
│   └── src/
│       ├── App.tsx                # Sidebar layout + routing
│       ├── index.css              # Palantir design system (CSS custom props)
│       ├── main.tsx               # React root
│       ├── api/
│       │   └── client.ts          # All API calls + TypeScript interfaces
│       ├── types/
│       │   └── globe.ts           # GeoEvent, GlobeArc, CATEGORY_COLORS types
│       ├── components/
│       │   └── TradeGlobe.tsx     # Reusable react-globe.gl component
│       └── pages/
│           ├── GodEyeView.tsx     # God's Eye View page (globe + predictions)
│           └── GapAndGo.tsx       # Gap & Go backtester page
│
├── data/                          # DuckDB + parquet + ChromaDB (auto-created)
├── models/                        # Trained model .pkl files (auto-created)
├── .env                           # API keys (not committed)
└── PROJECT_DOCS.md                # This file
```

---

## 3. Design System

File: `frontend/src/index.css`

### Design Language
Inspired by Palantir Foundry and Bloomberg Terminal:
- Near-black backgrounds, no gradients
- Cyan accent (`#00e5ff`) — the only "bright" color
- 2px border-radius everywhere (sharp, terminal feel)
- Uppercase micro-labels at 10px, 0.08em letter-spacing
- JetBrains Mono for all numbers
- Left-border accent on metric cards instead of shadows
- No hover shadows — use border color changes only

### CSS Custom Properties (Design Tokens)

```css
--bg:           #000508;   /* near-black page background */
--surface:      #03080e;   /* cards, sidebar */
--surface-2:    #060e18;   /* nested panels */
--surface-3:    #091522;   /* deepest nesting */
--border:       rgba(255,255,255,0.055);
--border-2:     rgba(255,255,255,0.10);
--border-3:     rgba(255,255,255,0.18);

--text-1:       #ccd8e8;   /* primary text */
--text-2:       #3d5268;   /* secondary text */
--text-3:       #1c2e3e;   /* muted text */

--accent:       #00e5ff;   /* cyan — primary interactive */
--accent-dim:   rgba(0,229,255,0.07);
--accent-glow:  rgba(0,229,255,0.18);

--green:        #00e676;   /* bullish / positive */
--red:          #ff1744;   /* bearish / negative */
--amber:        #ffd740;   /* risk / warning */
--violet:       #7c4dff;   /* neutral / structural */
```

### Component Classes

| Class | Description |
|-------|-------------|
| `.surface` | Primary card (bg + border + 2px radius) |
| `.surface-2` | Nested card |
| `.label` | 10px uppercase 0.08em tracking secondary text |
| `.heading` | 12px semibold primary text |
| `.num` | JetBrains Mono number formatting |
| `.metric` | Metric card with left-border accent |
| `.nav-item` | Sidebar navigation link |
| `.filter-pill` | Compact toggle pill for strategy filters |
| `.pred-card` | Prediction list item with active state |
| `.overlay-panel` | Semi-transparent glassmorphism panel over globe |
| `.conf-bar` | Confidence bar container |
| `.conf-bar-fill` | Confidence bar fill (colored by direction) |
| `.status-dot` | Colored dot prefix (pulsing when `.live`) |
| `.tab-btn` | Tab switcher button |

### Direction Colors (Globe + UI)
```
bullish  →  #00e676  (green)
bearish  →  #ff1744  (red)
neutral  →  #00e5ff  (cyan)
risk     →  #ffd740  (amber)
```

---

## 4. Backend

### 4.1 Entry Point — `main.py`

```
FastAPI app with lifespan context manager:
  startup:  init_db() → scheduler.start()
  shutdown: scheduler.shutdown()

CORS: localhost:5173, localhost:3000, 127.0.0.1:5173
Registered routers (all prefixed /api):
  predictions, stocks, sectors, assistant, system,
  backtest, pairs_backtest, gap_backtest, trade_intel, market_pulse
```

### 4.2 Config — `config.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `data/market.duckdb` | DuckDB database path |
| `MODEL_PATH` | `models/xgb_universal.pkl` | Directional XGBoost model |
| `CHROMA_PATH` | `data/chroma` | ChromaDB vector store |
| `UNIVERSE` | `nifty100` | Stock universe (`nifty50` / `nifty100` / `all`) |
| `LOOKBACK_DAYS` | `1825` | Training data lookback (5 years) |
| `REFRESH_DAYS` | `60` | Days fetched on daily refresh |
| `LONG_THRESHOLD` | `0.60` | prob_up > 60% → LONG signal |
| `SHORT_THRESHOLD` | `0.40` | prob_up < 40% → SHORT signal |
| `TOP_N` | `50` | Top N longs/shorts returned |

### 4.3 Data Layer

#### `data/fetcher.py`
- Downloads OHLCV data from yfinance in batches of 50 tickers
- Appends `.NS` suffix for NSE stocks
- Saves to DuckDB + Parquet

#### `data/storage.py`
- DuckDB-based storage for OHLCV, predictions, sentiment, model metadata
- Key functions: `init_db()`, `get_all_ohlcv_for_features()`, `get_sentiment()`, `save_model_meta()`

#### `data/nifty.py`
- Fetches `^NSEI` (Nifty 50 index) data
- Computes: `nifty_ret_1d`, `nifty_ret_5d`, `nifty_ret_20d`, `nifty_vs_ema20`, `nifty_vs_ema50`, `nifty_regime`, `nifty_volatility`
- These are merged into every stock's feature row as market context

#### `data/symbols.py`
- `NIFTY_50_SYMBOLS` — 50 large caps
- `NIFTY_100_SYMBOLS` — 100 stocks (used for pairs, more pair combinations)
- `get_tickers(universe)` — returns symbol list for configured universe

### 4.4 ML Layer

#### `ml/features.py` — Feature Engineering

**Label definition:**
```
label = 1  if stock_return_5d > nifty_return_5d
label = 0  otherwise
```
Why relative return? Absolute direction is ~50/50 noise. Relative outperformance has documented alpha from FinBERT sentiment at 2–5 day horizon.

**Feature columns (FEATURE_COLS):**

| Category | Features |
|----------|---------|
| Price returns | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d` |
| Momentum | `rsi_14`, `macd`, `macd_signal`, `macd_hist` |
| Volatility | `atr_14`, `bb_pct`, `bb_width` |
| Volume | `volume_ratio` (vs 20d avg), `obv`, `vwap_ret` |
| Trend | `ema_9_20_cross`, `price_vs_ema20`, `price_vs_ema50` |
| Sentiment | `sentiment_score`, `sent_pos`, `sent_neg`, `sent_neu`, `sent_zscore`, `news_count` |
| Market context | `nifty_ret_1d`, `nifty_ret_5d`, `nifty_vs_ema20`, `nifty_regime`, `nifty_volatility` |
| Fundamental | `pe_ratio`, `pb_ratio`, `analyst_score`, `earnings_proximity` |

Sentiment is lagged by 1 day (`shift(1)`) to avoid look-ahead bias.

#### `ml/model.py` — XGBoost Direction Model

```python
XGBClassifier(
    n_estimators=500, max_depth=4,
    learning_rate=0.02, subsample=0.7,
    colsample_bytree=0.6, min_child_weight=15,
    gamma=2, reg_alpha=2.0, reg_lambda=5.0,
    tree_method='hist'
)
```

- **Date-based train/test split** — no data leakage (critical)
- Trained on `nifty100` universe by default
- Serialized to `models/xgb_universal.pkl`

#### `ml/gap_trainer.py` — Tier-3 Gap Model

Separate XGBoost model answering: *"Given this stock gapped up, will it hit +2% target before reversing to -0.5% stop?"*

Why separate? The daily direction model marks a win as "closed higher than open" — but a gap trade that hit +2% by 11am and then closed flat is a **win** for gap-and-go. The gap model uses gap-specific labels.

```python
GAP_FEATURE_COLS = FEATURE_COLS + ["current_gap_pct"]
# current_gap_pct is observable at open — zero look-ahead
```

Label generation:
```python
def gap_continuation_label(open_, high, low, close, target_pct=2.0, stop_pct=0.5):
    hit_target = high >= open_ * 1.02
    hit_stop   = low  <= open_ * 0.995
    if hit_target and not hit_stop: return 1
    if hit_stop   and not hit_target: return 0
    if hit_target and hit_stop:
        return 1 if close >= open_ else 0  # close direction as tiebreaker
    net = (close - open_) / open_ * 100 - 0.20  # net of costs
    return 1 if net > 0 else 0
```

#### `ml/walkforward.py` — Walk-Forward Validation

Fixes in-sample bias where model is trained and tested on same data.

```
Round 1: Train 2020–2022  →  predict 2023 OOS
Round 2: Train 2020–2023  →  predict 2024 OOS
Round 3: Train 2020–2024  →  predict 2025 OOS
Backtest uses ONLY 2023–2025 OOS predictions.
```

Results cover 3 years, but every trade was predicted by a model that never saw that year during training. Sharpe/WR from walk-forward is the honest number.

#### `ml/gap_and_go.py` — Strategy Execution

See [Section 6](#6-gap--go-strategy) for full details.

#### `ml/pairs.py` + `ml/pairs_backtester.py` — Statistical Pairs

- ADF test + Engle-Granger cointegration to find cointegrated pairs
- Z-score mean-reversion trading
- Sector-constrained pair selection
- Entry: z > 2.0, Exit: z < 0, Stop: z > 3.0

### 4.5 Services Layer

#### `services/market_pulse.py` — God's Eye View Engine

See [Section 7](#7-gods-eye-view) for full details.

#### `services/trade_intel.py` — Trade Intelligence

See [Section 8](#8-trade-intelligence-system) for full details.

### 4.6 API Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Model metadata, last refresh, sentiment |
| `/api/refresh` | POST | Trigger background data + model refresh |
| `/api/predictions` | GET | All predictions (paginated, filterable) |
| `/api/rankings/longs` | GET | Top 50 LONG signals |
| `/api/rankings/shorts` | GET | Top 50 SHORT signals |
| `/api/sectors` | GET | Sector-level bullish/bearish summary |
| `/api/stocks/:symbol` | GET | OHLCV + prediction for one stock |
| `/api/stocks?q=` | GET | Search stocks |
| `/api/breadth` | GET | Market breadth (advance/decline) |
| `/api/assistant/query` | POST | RAG-based Q&A about the portfolio |
| `/api/assistant/suggestions` | GET | Suggested questions |
| `/api/backtest` | GET | Directional model backtest |
| `/api/pairs-backtest` | GET | Pairs trading backtest |
| `/api/gap-backtest` | GET | Gap & Go backtest |
| `/api/train-gap-model` | POST | Trigger gap model training (background) |
| `/api/trade-intel/:ticker/:date` | GET | AI reasoning for one gap trade |
| `/api/market-pulse` | GET | 30-day macro prediction engine |

---

## 5. Frontend

### 5.1 App Shell & Routing

File: `frontend/src/App.tsx`

**Layout:** `display: flex, height: 100vh, overflow: hidden`

```
┌──────────────────────────────────────────────────────┐
│  Sidebar (210px fixed)   │  Main content (flex-1)    │
│  • ALPHA EDGE logo       │  Route: /  → GodEyeView   │
│  • Nav: God's Eye View   │  Route: /gap → GapAndGo   │
│  • Nav: Gap & Go         │                           │
│  • Contextual info panel │  overflow: hidden (globe) │
│  • Live status footer    │  overflow: auto (gap)     │
└──────────────────────────────────────────────────────┘
```

**Sidebar contextual info:**
- On `/` (Globe): Shows direction color legend (bullish/bearish/neutral/risk)
- On `/gap` (Gap & Go): Shows strategy parameters (gap min, target, stop, R:R, cost) + Globe icon explanation

**Key detail:** `overflowY` on main is `hidden` for the globe page (fills full height), `auto` for Gap & Go (scrollable content).

### 5.2 Pages

#### `pages/GodEyeView.tsx` — God's Eye View

**Layout:** `display: flex, height: 100%, overflow: hidden`

```
┌────────────────────────────────────────────────────────────────────┐
│  Left Panel (310px)      │         Globe Area (flex-1)             │
│  ┌──────────────────┐    │  ┌──────────────────────────────────┐   │
│  │ Regime header    │    │  │  Top fade overlay                │   │
│  │ (macro_regime,   │    │  │  TradeGlobe (position: absolute) │   │
│  │  confidence bar, │    │  │  Bottom sector overlay           │   │
│  │  reasoning)      │    │  │    ├── 6 sector blocks           │   │
│  ├──────────────────┤    │  │    └── India outlook             │   │
│  │ Prediction list  │    │  │  Right detail overlay            │   │
│  │ (PredCard ×N,    │    │  │    (shown on click/select)       │   │
│  │  scrollable)     │    │  └──────────────────────────────────┘   │
│  ├──────────────────┤    │                                         │
│  │ Key Risks        │    │                                         │
│  ├──────────────────┤    │                                         │
│  │ India Outlook    │    │                                         │
│  └──────────────────┘    │                                         │
└────────────────────────────────────────────────────────────────────┘
```

**PredCard component:**
- Compact prediction list item
- Left-border colored by direction
- Confidence bar (animated fill)
- Click → flies globe to that prediction's lat/lng via `selectedEventId` prop
- Active state: brighter border + background tint

**State flow:**
```typescript
const [data, setData]           = useState<MarketPulseResult | null>(null)
const [selectedPred, setSelected] = useState<string | null>(null)
const [activeEvent,  setEvent]    = useState<GeoEvent | null>(null)

// Clicking PredCard → sets selectedPred → TradeGlobe flies to lat/lng
// Clicking globe pin → sets activeEvent → shows right detail overlay
```

#### `pages/GapAndGo.tsx` — Gap & Go Backtester

**Structure (top to bottom):**

```
1. Header bar (title + Run Backtest button)
2. Filter pills row — compact toggles for all 9 filters:
   [ML Confirm] [Sentiment] [Sector] [Volume] [Nifty] [Trend] [Gap ML] [RSI] [Walk-Fwd]
3. Parameters grid (5 per row):
   gap_threshold, gap_max_pct, target_pct, stop_pct, ml_threshold,
   sent_threshold, max_positions, volume_min_ratio, slippage_pct, capital
4. Metric cards row (9 cards):
   Total Return, Annualized, Sharpe, Max DD, Win Rate, Profit Factor,
   Trades, Gap Filter Rate, Alpha
5. Tab nav: [Equity Curve] [Monthly] [Trade Log] [Exit Breakdown] [Filters]
6. Tab content (charts / tables)
```

**FPill component:**
```typescript
const FPill = ({ label, value, onChange }) => (
  <button
    className="filter-pill"
    style={{ borderColor: value ? 'var(--accent)' : 'var(--border-2)', ... }}
    onClick={() => onChange(!value)}
  >
    {value ? '●' : '○'} {label}
  </button>
)
```

**Trade Log row:**
- Each row has a `<Globe size={13} />` icon
- Clicking it calls `openIntel(trade)` → fetches `/api/trade-intel/:ticker/:date`
- Opens a fixed overlay modal with left panel (reasoning + articles list) + right panel (3D globe)
- Modal globe flies to the ticker's home exchange location

**Charts:** Recharts `ComposedChart` with Palantir color scheme
- Equity curve: Area chart with cyan fill + gradient
- Monthly returns: Bar chart, green/red by sign
- Exit breakdown: PieChart with direction colors

### 5.3 Components

#### `components/TradeGlobe.tsx`

Self-contained `react-globe.gl` wrapper.

**Props:**
```typescript
interface TradeGlobeProps {
  events:           GeoEvent[]       // pins on the globe
  arcs:             GlobeArc[]       // connection lines between pins
  focusLat:         number           // initial camera lat
  focusLng:         number           // initial camera lng
  focusAltitude?:   number           // initial zoom (default 1.8)
  onEventClick?:    (event) => void  // called when pin is clicked
  selectedEventId?: string | null    // external control: fly to this event
}
```

**Globe configuration:**
```
autoRotate: true, speed: 0.25
zoom: minDistance 140, maxDistance 700
Globe texture: NASA blue marble (dark)
Point size: scaled by severity (0.3–1.5)
Point color: direction_color > direction > category color
Arc color: direction_color if present, else category color
```

**External fly-to (key feature):**
```typescript
useEffect(() => {
  if (selectedEventId === undefined) return
  setSelected(selectedEventId)
  if (!selectedEventId || !globeRef.current) return
  const ev = events.find(e => e.id === selectedEventId)
  if (!ev) return
  globeRef.current.pointOfView({ lat: ev.lat, lng: ev.lng, altitude: 1.5 }, 1000)
}, [selectedEventId, events])
```

**ResizeObserver:** Watches container div for size changes → updates `<Globe width/height>` → prevents globe from overflowing on layout changes.

### 5.4 Types

File: `frontend/src/types/globe.ts`

```typescript
interface GeoEvent {
  id, title, summary?, source_url?, source_name?,
  published_at, category: GlobeCategory,
  lat, lng, location_name?,
  severity,           // 0–1, used for pin size
  is_analyzed,
  // Direction (God's Eye View)
  direction?:         'bullish' | 'bearish' | 'neutral' | 'risk'
  direction_color?:   string
  timeframe?:         string
  affected_markets?:  string[]
  evidence?:          string
  detail?:            string
  // Legacy (Trade Intel)
  sentiment?:         'bullish' | 'bearish' | 'neutral'
  affected?:          string
}

interface GlobeArc {
  id, source_event_id, target_event_id,
  source_lat, source_lng, target_lat, target_lng,
  relationship_type: 'causal' | 'correlated',
  strength,          // arc opacity/color intensity
  category, direction?, direction_color?
}

interface TradeIntelResult {
  ticker, date, gap_pct, reasoning,
  events: GeoEvent[], arcs: GlobeArc[],
  focus_lat, focus_lng, n_articles
}
```

### 5.5 API Client

File: `frontend/src/api/client.ts`

All API calls go through an Axios instance with `baseURL: '/api'` (Vite proxy to FastAPI on `:8000`).

**Key interfaces:**

| Interface | Used by |
|-----------|---------|
| `Prediction` | Rankings, predictions table |
| `MarketStatus` | Status bar |
| `StockDetail` | Stock detail page |
| `BacktestResult` | Backtest tab |
| `GapResult` | Gap & Go tab |
| `GapMetrics` | Gap & Go metric cards |
| `GapTradeRecord` | Trade log table |
| `GapFilterStats` | Filter breakdown tab |
| `GapParams` | Backtest parameter form |
| `MarketPulseResult` | God's Eye View |
| `MacroPrediction` | Prediction cards |
| `TradeIntelParams` | Trade intel modal |

---

## 6. Gap & Go Strategy

File: `backend/ml/gap_and_go.py`

### Base Strategy

```
Entry:   Stock gaps up ≥ gap_threshold% at open
         (Open > Prev_Close × (1 + gap_threshold/100))
Target:  +2.0% above open price  →  profit
Stop:    −0.5% below open price  →  loss
R:R:     4 : 1  →  break-even win rate = 1/(1+4) = 20%
Cost:    0.20% round-trip (STT + exchange fees)
```

### Default Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gap_threshold` | 1.5% | 0.5–5.0 | Minimum gap-up to trade |
| `gap_max_pct` | 4.0% | 2.0–20.0 | Skip massive gaps (tend to fade) |
| `target_pct` | 2.0% | 0.5–10.0 | Profit target |
| `stop_pct` | 0.5% | 0.1–5.0 | Stop loss |
| `max_positions` | 5 | 1–20 | Max trades per day |
| `capital` | 100,000 INR | — | Initial capital |

### ML Filter Layers

All filters are independently toggleable:

| Layer | Filter | Default | Logic |
|-------|--------|---------|-------|
| T1 | **XGBoost confirm** | ON | `prob_up > ml_threshold (0.50)` from previous day's features |
| T1 | **Sentiment** | ON | Skip if FinBERT 3d rolling sentiment < −0.10 |
| T1 | **Sector momentum** | ON | Only trade if sector_momentum_rank > 0.5 |
| T1 | **Volume confirm** | ON | Min volume ratio vs 20d avg ≥ 1.5× |
| T1 | **Nifty filter** | ON | Skip day if Nifty down > −0.3% (bad market day) |
| T2 | **Trend filter** | ON | Stock must be above EMA-20 |
| T3 | **Gap ML confirm** | OFF | Tier-3 gap model prob > 0.50 (train first) |
| — | **RSI filter** | OFF | RSI must be 35–68 (disabled: over-filters momentum gaps) |
| — | **Walk-forward** | OFF | OOS-only predictions (honest backtest) |

### Trade Simulation (OHLC Approximation)

We approximate intraday from daily OHLC:

```python
effective_entry = open × (1 + slippage/100)   # fill price
target          = open × (1 + target_pct/100)  # planned exit
stop            = open × (1 - stop_pct/100)    # planned stop

# Both target AND stop hit same day → use close as tiebreaker
if close >= open:  target hit first  → WIN
if close < open:   stop hit first    → LOSS

# Net return
net_pct = (exit_price - effective_entry) / effective_entry × 100 - 0.20
```

### Filter Statistics Returned

```typescript
interface GapFilterStats {
  total_gaps,        // all detected gap-ups
  gap_cap_filtered,  // removed because gap > gap_max_pct
  ml_filtered,       // XGBoost said no
  sentiment_filtered,
  sector_filtered,
  volume_filtered,
  rsi_filtered,
  nifty_filtered,    // bad market day
  trend_filtered,    // below EMA20
  gap_ml_filtered,   // Tier-3 gap model said no
  traded             // actually executed
}
```

---

## 7. God's Eye View

File: `backend/services/market_pulse.py`

### Pipeline

```
1. _fetch_polygon_news(days_back=30)   → up to 50 articles from 10 global tickers
   Fallback: _fetch_yfinance_news()    → up to 30 articles from 6 index tickers

2. _groq_predictions(articles)
   → Formats top 40 headlines
   → Sends to Groq LLaMA-3.3-70B with structured JSON prompt
   → Returns macro analysis

3. _predictions_to_events(predictions)
   → Converts predictions to GeoEvent pins
   → Creates GlobeArc from each prediction to the NYC "macro hub" pin

4. Returns full MarketPulseResult dict
```

### Groq Prompt (LLaMA 3.3 70B)

```
Role: Senior macro analyst at a top hedge fund
Input: 40 headlines from past 30 days
Output: Forward-looking predictions for next 2-6 weeks

JSON schema:
{
  "macro_regime": "risk_on|risk_off|neutral|transitional",
  "regime_confidence": 0.75,
  "regime_reasoning": "One sentence",
  "predictions": [{
    "title": "Max 12 words",
    "detail": "1-2 sentences: catalyst + expected outcome",
    "confidence": 0.72,      // range: 0.45–0.90
    "timeframe": "2-4 weeks",
    "direction": "bullish|bearish|neutral|risk",
    "category": "monetary_policy|energy|geopolitical|trade|inflation|markets",
    "affected_markets": ["US equities", "Crude oil", "INR"],
    "lat": 38.9, "lng": -77.0,  // where catalyst originates
    "evidence": "Max 15 words supporting headline"
  }],
  "sector_outlook": {
    "Technology": { "direction": "bullish", "reasoning": "one-liner" },
    "Energy": ..., "Banking": ..., "Metals": ..., "Pharma": ..., "IT": ...
  },
  "key_risks": ["Risk 1 with timing", "Risk 2", "Risk 3", "Risk 4"],
  "india_impact": "2-3 sentences on Nifty50, INR, Indian sectors"
}
```

### Globe Data Structure

- **Hub pin:** New York (40.71, −74.00) = "Global Macro Intelligence Hub"
- **Prediction pins:** One per prediction at the catalyst's lat/lng
- **Arcs:** Each prediction → NYC hub (causal relationship)
- **Pin colors:** bullish=#00e676, bearish=#ff1744, neutral=#00e5ff, risk=#ffd740

### News Sources

**Polygon.io tickers used:**
`SPY, QQQ, GLD, USO, TLT, AAPL, NVDA, MSFT, AMZN, JPM`

**yfinance fallback tickers:**
`^GSPC, ^IXIC, GC=F, CL=F, ^TNX, ^NSEI`

**Publisher → lat/lng geocoding (static lookup):**
Bloomberg/Reuters/WSJ → New York, FT/BBC/Guardian → London,
ET/Moneycontrol/Livemint → Mumbai, Nikkei → Tokyo, SCMP → Hong Kong

---

## 8. Trade Intelligence System

File: `backend/services/trade_intel.py`

### What it does

Given a `ticker` and `date` from the Gap & Go trade log:
1. Fetches news ±3 days around the gap (Polygon → yfinance fallback)
2. Geocodes each article's publisher to lat/lng
3. Classifies articles into globe categories
4. Builds GeoEvent pins + GlobeArc connections to the exchange hub
5. Calls **Groq LLaMA-3.3-70B** (or Claude Haiku) to synthesize a 2–3 sentence reasoning for why the stock gapped

### Frontend Integration

- Globe icon on each trade row in GapAndGo
- Click → `openIntel(trade)` → `GET /api/trade-intel/:ticker/:date`
- Opens fixed overlay modal:
  - **Left panel:** AI reasoning text + article list with dates
  - **Right panel:** TradeGlobe focused on the stock's exchange location
- Globe pins show which global catalysts were relevant to this specific gap

---

## 9. ML Models

### Model 1: Universal Direction Model (`models/xgb_universal.pkl`)

- **Purpose:** Predict if a stock will outperform Nifty over 5 days
- **Universe:** Nifty 100 (default)
- **Features:** 30+ technical + sentiment + market context features
- **Label:** Cross-sectional relative return vs Nifty 5d
- **Output:** `prob_up` (0–1), `direction` (UP/DOWN/NEUTRAL)
- **Threshold:** LONG if `prob_up > 0.60`, SHORT if `prob_up < 0.40`

### Model 2: Gap Continuation Model (`models/gap_model.pkl`)

- **Purpose:** Predict if a gap-up trade will hit +2% target before −0.5% stop
- **Features:** All direction model features + `current_gap_pct`
- **Label:** Binary win/loss based on OHLC simulation (target vs stop hit first)
- **Trained via:** `POST /api/train-gap-model` (background job, ~60s)
- **Used as:** Optional Tier-3 filter in Gap & Go (disabled by default)

### Sentiment Model: FinBERT

- **Source:** `ProsusAI/finbert` via HuggingFace
- **Input:** News article titles/summaries
- **Output:** `positive`, `negative`, `neutral` probabilities
- **Usage:** 3-day rolling average → `sentiment_score`, `sent_zscore`
- **In features:** Lagged by 1 day (news on day D used on day D+1)

---

## 10. Planned: Local RAG + Ollama Engine

**Status:** Designed, not yet implemented. This will replace the Groq API for God's Eye View predictions.

### Why replace Groq?

- Polygon free tier: 5 requests/min limit → 10 tickers = rate limit hit → 0 articles → no Groq call → 0 predictions shown
- Groq API requires internet + API key
- Local RAG is free, private, unlimited

### Planned Architecture

```
News articles
    ↓
ChromaDB (PersistentClient)
  collection: "financial_news"
  embedder: all-MiniLM-L6-v2 (ONNX, built into chromadb)
    ↓
Multi-query retrieval (7 semantic queries)
  → up to 25 relevant chunks
    ↓
Ollama (localhost:11434)
  model: llama3.2:3b (default) or mistral:7b
    ↓
Same JSON output format as current Groq path
→ Zero frontend changes needed
```

### Planned Files

| File | Status | Description |
|------|--------|-------------|
| `backend/services/rag_engine.py` | TO CREATE | RAGEngine class: ingest, retrieve, predict |
| `backend/api/routes/rag.py` | TO CREATE | `/api/rag/status` endpoint |
| `backend/services/market_pulse.py` | TO MODIFY | Replace `_groq_predictions()` with RAGEngine |
| `backend/main.py` | TO MODIFY | Register rag router |
| `frontend/src/api/client.ts` | TO ADD | `fetchRagStatus()` + `RagStatus` interface |
| `frontend/src/App.tsx` | TO MODIFY | Ollama status dot in sidebar footer |

### Setup (when implemented)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model:
ollama pull llama3.2:3b    # ~2GB, runs on 4GB RAM

# 3. Install Python dependency:
pip install chromadb       # includes ONNX embeddings — no GPU needed
```

### Model Selection Guide

| Model | RAM | Speed | Quality | Recommended for |
|-------|-----|-------|---------|-----------------|
| `llama3.2:1b` | 2GB | Very fast | Basic | Low-RAM systems |
| `llama3.2:3b` | 4GB | Fast | Good | **Default recommendation** |
| `phi4-mini:3.8b` | 4GB | Fast | Great | Alternative to 3b |
| `mistral:7b` | 8GB | Medium | Great | High-quality analysis |

---

## 11. Environment Variables

File: `.env` (project root — not committed to git)

```env
# Required for core features
POLYGON_API_KEY=your_key      # news for God's Eye View + Trade Intel
GROQ_API_KEY=your_key         # LLM predictions (current, being replaced)

# Optional
ANTHROPIC_API_KEY=your_key    # Claude Haiku for trade reasoning
GEMINI_API_KEY=your_key       # alternative LLM

# Paths (defaults work for most setups)
DB_PATH=data/market.duckdb
MODEL_PATH=models/xgb_universal.pkl
CHROMA_PATH=data/chroma

# Stock universe
UNIVERSE=nifty100             # nifty50 | nifty100 | all

# When local RAG is implemented:
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 12. Running the Project

### Backend

```bash
cd PriceAnalyzingDashboard

# Activate venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# or
source venv/bin/activate           # Linux/Mac

# Install dependencies (first time)
pip install -r requirements.txt

# Start FastAPI
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install    # first time
npm run dev    # Vite dev server on :5173
```

### Access

| URL | Description |
|-----|-------------|
| `http://localhost:5173` | Frontend (Vite proxy → FastAPI) |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:8000/health` | Health check |

### Training the Models

```bash
# Train the universal direction model (one-time, ~5-10 min)
python -m backend.ml.model

# Train the gap continuation model (optional, ~60s)
# Or via the UI: Gap & Go → "Train Gap Model" button
python -m backend.ml.gap_trainer
```

---

## 13. Pending Tasks

### High Priority

- [ ] **Implement Local RAG + Ollama** (replaces Groq for God's Eye View)
  - `backend/services/rag_engine.py` — RAGEngine class
  - `backend/api/routes/rag.py` — `/api/rag/status`
  - Modify `market_pulse.py` to use RAGEngine
  - Update `main.py` + `client.ts` + `App.tsx`

- [ ] **Fix news source reliability** (root cause of 0 predictions)
  - Reduce Polygon calls from 10 → 4 tickers (avoid rate limit)
  - Add RSS feed fallback (Yahoo Finance, Reuters RSS — no API key)
  - Ensure yfinance fallback works with latest API structure

### Medium Priority

- [ ] Add Ollama status indicator to sidebar footer
- [ ] Add `/api/rag/ingest` endpoint to manually trigger re-ingestion
- [ ] Add "no articles" graceful degradation (LLM generates from training knowledge)
- [ ] Auto-refresh God's Eye View every 4 hours (background scheduler job)

### Nice to Have

- [ ] Add more news sources (NewsAPI, Alpha Vantage News)
- [ ] Add sector-level sentiment overlay on globe
- [ ] Add portfolio-level P&L tracking tab
- [ ] Dark/light theme toggle (extend CSS custom props)

---

*Documentation generated from codebase inspection + session history.*
*Copy this file to your Second Brain vault: `C:\Users\ASUS\Desktop\Second Brain\Projects\`*
