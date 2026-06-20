import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export interface Prediction {
  ticker: string
  symbol: string
  name: string
  sector: string
  industry: string
  prob_up: number
  prob_down: number
  confidence_pct: number
  direction: 'UP' | 'DOWN' | 'NEUTRAL'
  price: number
  change_pct: number
  volume: number
  rsi?: number
  macd?: number
  bb_pct?: number
  volume_ratio?: number
  rank?: number
}

export interface SectorSummary {
  sector: string
  avg_prob_up: number
  bullish_pct: number
  count: number
  longs: number
  shorts: number
}

export interface MarketStatus {
  last_refresh: string | null
  model_accuracy: number | null
  n_stocks: number | null
  n_samples: number | null
  trained_at: string | null
  doc_count: number | null
  market_sentiment: string | null
  total_predictions: number | null
  longs: number | null
  shorts: number | null
  pipeline_running: boolean
}

export interface OHLCVPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface StockDetail extends Prediction {
  ohlcv: OHLCVPoint[]
}

export interface AssistantResponse {
  answer: string
  sources: Record<string, unknown>[]
  query: string
}

export const fetchStatus = () => api.get<MarketStatus>('/status').then(r => r.data)

export const fetchLongs = () => api.get<{ data: Prediction[]; count: number }>('/rankings/longs').then(r => r.data)
export const fetchShorts = () => api.get<{ data: Prediction[]; count: number }>('/rankings/shorts').then(r => r.data)

export const fetchSectors = () => api.get<{ data: SectorSummary[] }>('/sectors').then(r => r.data)

export const fetchPredictions = (params?: { sector?: string; direction?: string; page?: number }) =>
  api.get<{ total: number; data: Prediction[] }>('/predictions', { params }).then(r => r.data)

export const fetchStockDetail = (symbol: string, days = 180) =>
  api.get<StockDetail>(`/stocks/${symbol}`, { params: { days } }).then(r => r.data)

export const searchStocks = (q: string) =>
  api.get<{ data: Prediction[] }>('/stocks', { params: { q } }).then(r => r.data)

export const askAssistant = (query: string) =>
  api.post<AssistantResponse>('/assistant/query', { query }).then(r => r.data)

export const fetchSuggestions = () =>
  api.get<{ questions: string[] }>('/assistant/suggestions').then(r => r.data)

export const triggerRefresh = () => api.post('/refresh').then(r => r.data)

export const fetchMarketBreadth = () => api.get('/breadth').then(r => r.data)

// ── Backtest ───────────────────────────────────────────────────────────────
export interface BacktestMetrics {
  total_return_pct: number
  annualized_return: number
  sharpe_ratio: number
  max_drawdown_pct: number
  win_rate_pct: number
  avg_win_pct: number
  avg_loss_pct: number
  total_trades: number
  long_trades: number
  short_trades: number
  backtest_days: number
  backtest_start: string
  backtest_end: string
  initial_capital: number
  final_capital: number
  benchmark_return: number
  alpha: number
  expectancy_pct: number
  transaction_cost_pct: number
  active_days: number
  max_single_loss_pct: number
}

export interface EquityPoint    { date: string; value: number }
export interface MonthlyReturn  { month: string; return_pct: number }
export interface QuarterlyReturn { quarter: string; return_pct: number }
export interface TradeRecord {
  ticker: string; date: string; direction: 'LONG' | 'SHORT'
  confidence: number; next_ret: number; trade_ret: number; won: boolean
}
export interface FeatureImportance { feature: string; importance: number }

export interface BacktestResult {
  metrics: BacktestMetrics
  equity_curve: EquityPoint[]
  benchmark_curve: EquityPoint[]
  monthly: MonthlyReturn[]
  quarterly: QuarterlyReturn[]
  trades: TradeRecord[]
  feature_importance: FeatureImportance[]
  warnings: string[]
}

export interface BacktestParams {
  threshold?: number
  stop_loss?: number
  take_profit?: number
  max_pos?: number
  capital?: number
  direction?: string
}

export const fetchBacktest = (params?: BacktestParams) =>
  api.get<BacktestResult>('/backtest', { params }).then(r => r.data)

// ── Pairs Backtest ─────────────────────────────────────────────────────────
export interface PairsMetrics {
  total_return_pct:     number
  annualized_return:    number
  sharpe_ratio:         number
  max_drawdown_pct:     number
  win_rate_pct:         number
  avg_win_pct:          number
  avg_loss_pct:         number
  expectancy_pct:       number
  total_trades:         number
  avg_holding_days:     number
  n_pairs:              number
  backtest_days:        number
  backtest_start:       string
  backtest_end:         string
  initial_capital:      number
  final_capital:        number
  benchmark_return:     number
  alpha:                number
  transaction_cost_pct: number
}

export interface PairSummary {
  pair_id:     string
  symbol_a:    string
  symbol_b:    string
  sector:      string
  pvalue:      number
  hedge_ratio: number
  half_life:   number | null
  correlation: number
  n_obs:       number
  trades:      number
}

export interface PairsTradeRecord {
  pair:         string
  sector:       string
  date_entry:   string
  date_exit:    string
  direction:    'LONG' | 'SHORT'
  days_held:    number
  entry_z:      number
  exit_z:       number
  gross_ret:    number
  trade_ret:    number
  conviction:   number
  exit_reason:  string
  ml_confirmed: string
  won:          boolean
}

export interface FilterStats {
  total_signals:     number
  regime_blocked:    number
  sentiment_blocked: number
  ml_skipped:        number
  ml_half_size:      number
  traded:            number
}

export interface PairsResult {
  metrics:         PairsMetrics
  equity_curve:    EquityPoint[]
  benchmark_curve: EquityPoint[]
  monthly:         MonthlyReturn[]
  trades:          PairsTradeRecord[]
  pairs:           PairSummary[]
  exit_breakdown:  Record<string, number>
  filter_stats:    FilterStats
  warnings:        string[]
}

export interface PairsParams {
  z_entry?:           number
  z_exit?:            number
  z_stop?:            number
  max_pairs?:         number
  max_holding_days?:  number
  sector_only?:       boolean
  use_ml_confirm?:    boolean
  use_sentiment?:     boolean
  use_regime_filter?: boolean
  capital?:           number
}

export const fetchPairsBacktest = (params?: PairsParams) =>
  api.get<PairsResult>('/pairs-backtest', { params }).then(r => r.data)

// ── Gap and Go Backtest ────────────────────────────────────────────────────
export interface GapMetrics {
  total_return_pct:   number
  annualized_return:  number
  sharpe_ratio:       number
  max_drawdown_pct:   number
  win_rate_pct:       number
  avg_win_pct:        number
  avg_loss_pct:       number
  profit_factor:      number
  total_trades:       number
  total_gaps_seen:    number
  ml_filtered:        number
  sentiment_filtered: number
  sector_filtered:    number
  trades_taken:       number
  backtest_days:      number
  backtest_start:     string
  backtest_end:       string
  initial_capital:    number
  final_capital:      number
  benchmark_return:   number
  alpha:              number
}

export interface GapTradeRecord {
  date:        string
  ticker:      string
  gap_pct:     number
  entry:       number
  ret_pct:     number
  exit_reason: string
  won:         boolean
}

export interface GapFilterStats {
  total_gaps:         number
  gap_cap_filtered:   number
  ml_filtered:        number
  sentiment_filtered: number
  sector_filtered:    number
  volume_filtered:    number
  rsi_filtered:       number
  nifty_filtered:     number
  trend_filtered:     number
  gap_ml_filtered:    number
  traded:             number
}

export interface GapResult {
  metrics:         GapMetrics
  equity_curve:    EquityPoint[]
  benchmark_curve: EquityPoint[]
  monthly:         MonthlyReturn[]
  trades:          GapTradeRecord[]
  exit_breakdown:  Record<string, number>
  filter_stats:    GapFilterStats
  warnings:        string[]
}

export interface GapParams {
  gap_threshold?:      number
  gap_max_pct?:        number
  target_pct?:         number
  stop_pct?:           number
  ml_threshold?:       number
  sent_threshold?:     number
  max_positions?:      number
  volume_min_ratio?:   number
  rsi_low?:            number
  rsi_high?:           number
  nifty_min_ret?:      number
  use_ml_confirm?:     boolean
  use_sentiment?:      boolean
  use_sector_filter?:  boolean
  use_volume_filter?:  boolean
  use_rsi_filter?:     boolean
  use_nifty_filter?:   boolean
  use_trend_filter?:   boolean
  use_gap_ml_confirm?: boolean
  gap_ml_threshold?:   number
  use_walkforward?:    boolean
  slippage_pct?:       number
  capital?:            number
}

export const trainGapModel = () =>
  api.post('/train-gap-model').then(r => r.data)

export const fetchGapBacktest = (params?: GapParams) =>
  api.get<GapResult>('/gap-backtest', { params }).then(r => r.data)

// ── Trade Intelligence ─────────────────────────────────────────────────────
export interface TradeIntelParams {
  gap_pct?:      number
  nifty_ret?:    number
  sector?:       string
  volume_ratio?: number
  sector_rank?:  number
}

export const fetchTradeIntel = (ticker: string, date: string, params?: TradeIntelParams) =>
  api.get(`/trade-intel/${encodeURIComponent(ticker)}/${date}`, { params }).then(r => r.data)

// ── God's Eye View / Market Pulse ─────────────────────────────────────────
import type { GeoEvent, GlobeArc } from '../types/globe'

export interface MacroPrediction {
  title:            string
  detail:           string
  confidence:       number
  timeframe:        string
  direction:        'bullish' | 'bearish' | 'neutral' | 'risk'
  category:         string
  affected_markets: string[]
  lat:              number
  lng:              number
  evidence:         string
}

export interface SectorOutlookItem {
  direction: 'bullish' | 'bearish' | 'neutral'
  reasoning: string
}

export interface MarketPulseResult {
  macro_regime:      'risk_on' | 'risk_off' | 'neutral' | 'transitional'
  regime_confidence: number
  regime_reasoning:  string
  predictions:       MacroPrediction[]
  sector_outlook:    Record<string, SectorOutlookItem>
  key_risks:         string[]
  india_impact:      string
  events:            GeoEvent[]
  arcs:              GlobeArc[]
  generated_at:      string
  n_articles:        number
}

export const fetchMarketPulse = () =>
  api.get<MarketPulseResult>('/market-pulse').then(r => r.data)
