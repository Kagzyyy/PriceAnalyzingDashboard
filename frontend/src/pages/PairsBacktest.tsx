import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell, ReferenceLine,
} from 'recharts'
import {
  fetchPairsBacktest,
  type PairsResult, type PairsParams, type FilterStats,
} from '../api/client'
import { AlertTriangle, TrendingUp, TrendingDown, Brain, Shield, MessageSquare } from 'lucide-react'

// ── Metric card ─────────────────────────────────────────────────────────────
function MetricCard({ label, value, sub, accent = 'text-white' }: {
  label: string; value: string | number; sub?: string; accent?: string
}) {
  return (
    <div className="card-hover">
      <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-3">{label}</p>
      <p className={`text-3xl font-bold mono ${accent}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-2">{sub}</p>}
    </div>
  )
}

// ── Equity tooltip ───────────────────────────────────────────────────────────
function EquityTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-black border border-white/10 rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-gray-500 mb-1">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.dataKey} style={{ color: p.stroke }} className="mono font-semibold">
          {p.dataKey === 'value' ? 'Strategy' : 'Nifty 50'}:{' '}
          ₹{Number(p.value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </p>
      ))}
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function PairsBacktest() {
  const [params, setParams] = useState<PairsParams>({
    z_entry:           2.0,
    z_exit:            0.5,
    z_stop:            3.0,
    max_pairs:         5,
    max_holding_days:  20,
    sector_only:       true,
    use_ml_confirm:    true,
    use_sentiment:     true,
    use_regime_filter: true,
    capital:           100000,
  })

  const [result,  setResult]  = useState<PairsResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [tab,     setTab]     = useState<'equity' | 'monthly' | 'pairs' | 'trades'>('equity')

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchPairsBacktest(params)
      setResult(data)
      setTab('equity')
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const m = result?.metrics
  const accent = (v: number) => v >= 0 ? 'text-emerald-400' : 'text-red-400'
  const fmt    = (v: number, d = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`

  // Merge equity + benchmark for chart
  const chartData = (() => {
    if (!result) return []
    const bMap = new Map(result.benchmark_curve.map(b => [b.date, b.value]))
    return result.equity_curve.map(e => ({
      date:      e.date,
      value:     e.value,
      benchmark: bMap.get(e.date) ?? null,
    }))
  })()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
        <h1 className="text-2xl font-bold text-white tracking-tight">Pairs Backtester</h1>
        <p className="text-sm text-gray-600 mt-1">
          Statistical arbitrage — finds cointegrated NSE stock pairs and trades mean reversion of the spread.
          Formation period: first 80% of data. Out-of-sample test: last 20%.
        </p>
      </div>

      {/* ML Enhancement Toggles */}
      <div className="card">
        <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">ML Enhancement Layers</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* ML Confirmation */}
          <button
            onClick={() => setParams(p => ({ ...p, use_ml_confirm: !p.use_ml_confirm }))}
            className={`flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
              params.use_ml_confirm
                ? 'border-blue-500/40 bg-blue-500/10'
                : 'border-white/[0.06] bg-white/[0.02] opacity-50'
            }`}
          >
            <Brain size={16} className={params.use_ml_confirm ? 'text-blue-400' : 'text-gray-600'} />
            <div>
              <p className={`text-xs font-semibold ${params.use_ml_confirm ? 'text-blue-400' : 'text-gray-500'}`}>
                XGBoost Confirmation {params.use_ml_confirm ? 'ON' : 'OFF'}
              </p>
              <p className="text-xs text-gray-600 mt-0.5">
                Uses ML predictions to confirm pair direction. Conflicts reduce position size or skip trade.
              </p>
            </div>
          </button>

          {/* Sentiment Filter */}
          <button
            onClick={() => setParams(p => ({ ...p, use_sentiment: !p.use_sentiment }))}
            className={`flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
              params.use_sentiment
                ? 'border-purple-500/40 bg-purple-500/10'
                : 'border-white/[0.06] bg-white/[0.02] opacity-50'
            }`}
          >
            <MessageSquare size={16} className={params.use_sentiment ? 'text-purple-400' : 'text-gray-600'} />
            <div>
              <p className={`text-xs font-semibold ${params.use_sentiment ? 'text-purple-400' : 'text-gray-500'}`}>
                Sentiment Filter {params.use_sentiment ? 'ON' : 'OFF'}
              </p>
              <p className="text-xs text-gray-600 mt-0.5">
                FinBERT detects if news explains the spread divergence. Fundamental moves are skipped.
              </p>
            </div>
          </button>

          {/* Regime Filter */}
          <button
            onClick={() => setParams(p => ({ ...p, use_regime_filter: !p.use_regime_filter }))}
            className={`flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
              params.use_regime_filter
                ? 'border-emerald-500/40 bg-emerald-500/10'
                : 'border-white/[0.06] bg-white/[0.02] opacity-50'
            }`}
          >
            <Shield size={16} className={params.use_regime_filter ? 'text-emerald-400' : 'text-gray-600'} />
            <div>
              <p className={`text-xs font-semibold ${params.use_regime_filter ? 'text-emerald-400' : 'text-gray-500'}`}>
                Regime Filter {params.use_regime_filter ? 'ON' : 'OFF'}
              </p>
              <p className="text-xs text-gray-600 mt-0.5">
                Suppresses all pairs trades during crisis regimes (Nifty vol spike). Prevents OOD trading.
              </p>
            </div>
          </button>
        </div>
      </div>

      {/* Parameters */}
      <div className="card">
        <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-5">Strategy Parameters</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-4">

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Entry Z-Score</span>
            <select className="input-base text-xs py-1.5" value={params.z_entry}
              onChange={e => setParams({ ...params, z_entry: Number(e.target.value) })}>
              <option value={1.5}>1.5σ</option>
              <option value={2.0}>2.0σ</option>
              <option value={2.5}>2.5σ</option>
              <option value={3.0}>3.0σ</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Exit Z-Score</span>
            <select className="input-base text-xs py-1.5" value={params.z_exit}
              onChange={e => setParams({ ...params, z_exit: Number(e.target.value) })}>
              <option value={0.0}>0.0σ</option>
              <option value={0.5}>0.5σ</option>
              <option value={1.0}>1.0σ</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Stop Z-Score</span>
            <select className="input-base text-xs py-1.5" value={params.z_stop}
              onChange={e => setParams({ ...params, z_stop: Number(e.target.value) })}>
              <option value={2.5}>2.5σ</option>
              <option value={3.0}>3.0σ</option>
              <option value={3.5}>3.5σ</option>
              <option value={4.0}>4.0σ</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Max Pairs</span>
            <select className="input-base text-xs py-1.5" value={params.max_pairs}
              onChange={e => setParams({ ...params, max_pairs: Number(e.target.value) })}>
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={15}>15</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Max Hold (days)</span>
            <select className="input-base text-xs py-1.5" value={params.max_holding_days}
              onChange={e => setParams({ ...params, max_holding_days: Number(e.target.value) })}>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={30}>30</option>
              <option value={45}>45</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs text-gray-600">Universe</span>
            <select className="input-base text-xs py-1.5"
              value={params.sector_only ? 'sector' : 'all'}
              onChange={e => setParams({ ...params, sector_only: e.target.value === 'sector' })}>
              <option value="sector">Same sector</option>
              <option value="all">All pairs</option>
            </select>
          </label>

          <div className="flex items-end">
            <button
              onClick={run}
              disabled={loading}
              className="w-full bg-white text-black text-sm font-semibold py-1.5 px-4 rounded-lg
                         hover:bg-gray-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? 'Running…' : 'Run Backtest'}
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="card flex items-center justify-center py-16 space-x-3">
          <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
          <p className="text-sm text-gray-500">
            Running cointegration tests and simulating trades… (~30-60 seconds)
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card border border-red-500/20 bg-red-500/5">
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle size={16} />
            <p className="text-sm font-medium">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="space-y-2">
              {result.warnings.map((w, i) => (
                <div key={i} className="card border border-yellow-500/20 bg-yellow-500/5 flex items-start gap-2 py-3">
                  <AlertTriangle size={14} className="text-yellow-400 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-yellow-300">{w}</p>
                </div>
              ))}
            </div>
          )}

          {/* Metric cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Total Return"
              value={`${m!.total_return_pct >= 0 ? '+' : ''}${m!.total_return_pct.toFixed(1)}%`}
              sub={`₹${m!.initial_capital.toLocaleString('en-IN')} → ₹${m!.final_capital.toLocaleString('en-IN')}`}
              accent={accent(m!.total_return_pct)}
            />
            <MetricCard
              label="Annualized Return"
              value={`${m!.annualized_return >= 0 ? '+' : ''}${m!.annualized_return.toFixed(1)}%`}
              sub="CAGR"
              accent={accent(m!.annualized_return)}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={m!.sharpe_ratio.toFixed(2)}
              sub={m!.sharpe_ratio >= 1 ? 'Good' : m!.sharpe_ratio >= 0 ? 'Marginal' : 'Negative'}
              accent={m!.sharpe_ratio >= 1 ? 'text-emerald-400' : m!.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400'}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${m!.max_drawdown_pct.toFixed(1)}%`}
              sub="Peak-to-trough"
              accent="text-red-400"
            />
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Win Rate"
              value={`${m!.win_rate_pct.toFixed(1)}%`}
              sub={`${m!.total_trades} trades`}
              accent={m!.win_rate_pct >= 50 ? 'text-emerald-400' : 'text-red-400'}
            />
            <MetricCard
              label="Nifty 50 (Same Period)"
              value={`${m!.benchmark_return >= 0 ? '+' : ''}${m!.benchmark_return.toFixed(1)}%`}
              sub="Buy & hold benchmark"
              accent={accent(m!.benchmark_return)}
            />
            <MetricCard
              label="Alpha vs Nifty"
              value={`${m!.alpha >= 0 ? '+' : ''}${m!.alpha.toFixed(1)}%`}
              sub="Strategy − benchmark"
              accent={accent(m!.alpha)}
            />
            <MetricCard
              label="Avg Holding"
              value={`${m!.avg_holding_days.toFixed(1)}d`}
              sub={`${m!.n_pairs} pairs · ${m!.total_trades} trades`}
            />
          </div>

          {/* Filter stats */}
          {result.filter_stats && result.filter_stats.total_signals > 0 && (
            <div className="card">
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">Signal Filter Breakdown</p>
              <div className="grid grid-cols-3 sm:grid-cols-6 gap-4">
                {[
                  { label: 'Total Signals',      value: result.filter_stats.total_signals,     color: 'text-white' },
                  { label: 'Regime Blocked',      value: result.filter_stats.regime_blocked,    color: 'text-emerald-400' },
                  { label: 'Sentiment Blocked',   value: result.filter_stats.sentiment_blocked, color: 'text-purple-400' },
                  { label: 'ML Skipped',          value: result.filter_stats.ml_skipped,        color: 'text-blue-400' },
                  { label: 'Half Size (ML)',       value: result.filter_stats.ml_half_size,      color: 'text-yellow-400' },
                  { label: 'Traded',              value: result.filter_stats.traded,            color: 'text-emerald-400' },
                ].map(s => (
                  <div key={s.label} className="text-center">
                    <p className={`text-2xl font-bold mono ${s.color}`}>{s.value}</p>
                    <p className="text-xs text-gray-600 mt-1">{s.label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Footer info */}
          <p className="text-xs text-gray-700">
            Backtest period: {m!.backtest_start} → {m!.backtest_end} · {m!.backtest_days} calendar days ·
            Round-trip cost: {m!.transaction_cost_pct.toFixed(2)}% per trade (both legs)
          </p>

          {/* Tabs */}
          <div className="border-b border-white/[0.06]">
            <div className="flex gap-6">
              {(['equity', 'monthly', 'pairs', 'trades'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`pb-3 text-sm font-medium capitalize transition-colors ${
                    tab === t
                      ? 'text-white border-b-2 border-white'
                      : 'text-gray-600 hover:text-gray-400'
                  }`}
                >
                  {t === 'pairs' ? `Pairs (${result.pairs.length})` : t === 'trades' ? `Trade History` : t === 'equity' ? 'Equity Curve' : 'Monthly'}
                </button>
              ))}
            </div>
          </div>

          {/* Equity curve */}
          {tab === 'equity' && (
            <div className="card">
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">Portfolio Value</p>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickFormatter={d => d.slice(2, 7)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickFormatter={v => `₹${(v / 1000).toFixed(0)}k`} width={55} />
                  <Tooltip content={<EquityTooltip />} />
                  <Line type="monotone" dataKey="value" stroke="#ffffff" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="benchmark" stroke="#6b7280" strokeWidth={1}
                    dot={false} strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex gap-6 mt-3 justify-end">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <div className="w-6 h-px bg-white" />Strategy
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <div className="w-6 h-px bg-gray-500 border-dashed border-t border-gray-500" />Nifty 50
                </div>
              </div>
            </div>
          )}

          {/* Monthly returns */}
          {tab === 'monthly' && (
            <div className="card">
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">Monthly Returns</p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={result.monthly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="month" tick={{ fill: '#4b5563', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickFormatter={v => `${v.toFixed(1)}%`} width={45} />
                  <Tooltip
                    formatter={(v: number) => [`${v >= 0 ? '+' : ''}${v.toFixed(2)}%`, 'Return']}
                    contentStyle={{ background: '#000', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11 }}
                    labelStyle={{ color: '#6b7280' }}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
                  <Bar dataKey="return_pct" radius={[2, 2, 0, 0]}>
                    {result.monthly.map((m, i) => (
                      <Cell key={i} fill={m.return_pct >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Pairs table */}
          {tab === 'pairs' && (
            <div className="card">
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">
                Cointegrated Pairs — Formation Period Analysis
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.06]">
                      {['Pair', 'Sector', 'P-Value', 'Hedge β', 'Half-Life', 'Correlation', 'Trades'].map(h => (
                        <th key={h} className="text-left text-gray-600 uppercase tracking-wider pb-2 pr-4 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.pairs.map((p, i) => (
                      <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                        <td className="py-2.5 pr-4 font-semibold text-white mono">{p.pair_id}</td>
                        <td className="py-2.5 pr-4 text-gray-400">{p.sector}</td>
                        <td className={`py-2.5 pr-4 mono ${p.pvalue < 0.01 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                          {p.pvalue.toFixed(4)}
                        </td>
                        <td className="py-2.5 pr-4 mono text-gray-300">{p.hedge_ratio.toFixed(3)}</td>
                        <td className="py-2.5 pr-4 mono text-gray-300">
                          {p.half_life != null ? `${p.half_life}d` : '—'}
                        </td>
                        <td className={`py-2.5 pr-4 mono ${p.correlation > 0.8 ? 'text-emerald-400' : 'text-gray-300'}`}>
                          {p.correlation.toFixed(3)}
                        </td>
                        <td className="py-2.5 pr-4 mono text-gray-300">{p.trades}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Exit breakdown */}
              {Object.keys(result.exit_breakdown).length > 0 && (
                <div className="mt-6 pt-4 border-t border-white/[0.06]">
                  <p className="text-xs text-gray-600 uppercase tracking-wider mb-3">Exit Reasons</p>
                  <div className="flex gap-6">
                    {Object.entries(result.exit_breakdown).map(([reason, count]) => (
                      <div key={reason} className="text-center">
                        <p className="text-lg font-bold mono text-white">{count}</p>
                        <p className="text-xs text-gray-600 capitalize">{reason.replace('_', ' ')}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Trade history */}
          {tab === 'trades' && (
            <div className="card">
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">
                Trade History (last {result.trades.length})
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.06]">
                      {['Pair', 'Direction', 'Entry', 'Exit', 'Days', 'Entry Z', 'Exit Z', 'Conviction', 'ML', 'Return', 'Exit Reason'].map(h => (
                        <th key={h} className="text-left text-gray-600 uppercase tracking-wider pb-2 pr-3 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.slice().reverse().map((t, i) => (
                      <tr key={i} className={`border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors ${
                        t.won ? '' : 'opacity-70'
                      }`}>
                        <td className="py-2 pr-3 font-semibold mono text-white">{t.pair}</td>
                        <td className={`py-2 pr-3 font-semibold ${t.direction === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>
                          {t.direction === 'LONG'
                            ? <span className="flex items-center gap-1"><TrendingUp size={10} /> LONG</span>
                            : <span className="flex items-center gap-1"><TrendingDown size={10} /> SHORT</span>
                          }
                        </td>
                        <td className="py-2 pr-3 text-gray-500">{t.date_entry}</td>
                        <td className="py-2 pr-3 text-gray-500">{t.date_exit}</td>
                        <td className="py-2 pr-3 mono text-gray-400">{t.days_held}</td>
                        <td className="py-2 pr-3 mono text-gray-400">{t.entry_z.toFixed(2)}σ</td>
                        <td className="py-2 pr-3 mono text-gray-400">{t.exit_z.toFixed(2)}σ</td>
                        <td className={`py-2 pr-3 mono text-xs ${t.conviction >= 1 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                          {(t.conviction * 100).toFixed(0)}%
                        </td>
                        <td className={`py-2 pr-3 text-xs ${
                          t.ml_confirmed === 'full' ? 'text-emerald-400' :
                          t.ml_confirmed === 'partial' ? 'text-yellow-400' :
                          'text-gray-600'
                        }`}>
                          {t.ml_confirmed === 'full' ? '✓ full' :
                           t.ml_confirmed === 'partial' ? '~ partial' :
                           '—'}
                        </td>
                        <td className={`py-2 pr-3 mono font-semibold ${t.trade_ret >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {t.trade_ret >= 0 ? '+' : ''}{t.trade_ret.toFixed(2)}%
                        </td>
                        <td className={`py-2 pr-3 text-gray-500 capitalize ${
                          t.exit_reason === 'stop_loss' ? 'text-red-400' :
                          t.exit_reason === 'mean_reversion' ? 'text-emerald-400' : ''
                        }`}>
                          {t.exit_reason.replace('_', ' ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
