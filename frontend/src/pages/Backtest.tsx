import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, Cell, ReferenceLine, Legend,
} from 'recharts'
import {
  fetchBacktest,
  type BacktestResult, type BacktestParams, type QuarterlyReturn,
} from '../api/client'
import { TrendingUp, TrendingDown, Activity, AlertCircle, AlertTriangle } from 'lucide-react'

// ── Shared card ────────────────────────────────────────────────────────────
function MetricCard({
  label, value, sub, accent = 'text-white', delay = 0,
}: {
  label: string; value: string | number; sub?: string; accent?: string; delay?: number
}) {
  return (
    <div
      className="card-hover opacity-0 animate-fade-up"
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-3">{label}</p>
      <p className={`text-3xl font-bold mono ${accent}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-2">{sub}</p>}
    </div>
  )
}

// ── Custom tooltip for equity chart ────────────────────────────────────────
function EquityTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-black border border-white/10 rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-gray-500 mb-1">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.dataKey} style={{ color: p.stroke }} className="mono font-semibold">
          {p.dataKey === 'value' ? 'Strategy' : 'Nifty 50'}: ₹{Number(p.value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </p>
      ))}
    </div>
  )
}

// ── Custom tooltip for monthly bar ─────────────────────────────────────────
function MonthTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const val = payload[0].value as number
  return (
    <div className="bg-black border border-white/10 rounded-lg px-3 py-2 text-xs">
      <p className="text-gray-500 mb-1">{label}</p>
      <p className={`font-semibold mono ${val >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {val >= 0 ? '+' : ''}{val.toFixed(2)}%
      </p>
    </div>
  )
}

// ── Settings panel ─────────────────────────────────────────────────────────
function SettingsPanel({
  params, onChange, onRun, loading,
}: {
  params: BacktestParams
  onChange: (p: BacktestParams) => void
  onRun: () => void
  loading: boolean
}) {
  return (
    <div className="card opacity-0 animate-fade-up" style={{ animationDelay: '0ms', animationFillMode: 'forwards' }}>
      <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-5">Strategy Parameters</p>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-gray-600">Signal Width</span>
          <select
            className="input-base text-xs py-1.5"
            value={params.threshold}
            onChange={e => onChange({ ...params, threshold: Number(e.target.value) })}
          >
            <option value={0.05}>Top / Bottom 5%</option>
            <option value={0.10}>Top / Bottom 10%</option>
            <option value={0.15}>Top / Bottom 15%</option>
            <option value={0.20}>Top / Bottom 20%</option>
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-gray-600">Stop Loss</span>
          <select
            className="input-base text-xs py-1.5"
            value={params.stop_loss}
            onChange={e => onChange({ ...params, stop_loss: Number(e.target.value) })}
          >
            <option value={1.0}>1.0%</option>
            <option value={2.0}>2.0%</option>
            <option value={3.0}>3.0%</option>
            <option value={5.0}>5.0%</option>
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-gray-600">Take Profit</span>
          <select
            className="input-base text-xs py-1.5"
            value={params.take_profit}
            onChange={e => onChange({ ...params, take_profit: Number(e.target.value) })}
          >
            <option value={2.0}>2.0%</option>
            <option value={3.0}>3.0%</option>
            <option value={5.0}>5.0%</option>
            <option value={8.0}>8.0%</option>
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-gray-600">Max Positions</span>
          <select
            className="input-base text-xs py-1.5"
            value={params.max_pos}
            onChange={e => onChange({ ...params, max_pos: Number(e.target.value) })}
          >
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={15}>15</option>
            <option value={20}>20</option>
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-gray-600">Direction</span>
          <select
            className="input-base text-xs py-1.5"
            value={params.direction}
            onChange={e => onChange({ ...params, direction: e.target.value })}
          >
            <option value="both">Both</option>
            <option value="long">Long only</option>
            <option value="short">Short only</option>
          </select>
        </label>

        <div className="flex items-end">
          <button
            onClick={onRun}
            disabled={loading}
            className="w-full bg-white text-black text-xs font-semibold py-1.5 px-4 rounded-lg
                       hover:bg-gray-200 transition-colors duration-200 disabled:opacity-40"
          >
            {loading ? 'Running…' : 'Run Backtest'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function Backtest() {
  const [params, setParams] = useState<BacktestParams>({
    threshold: 0.10,
    stop_loss: 3.0,
    take_profit: 5.0,
    max_pos: 10,
    direction: 'both',
  })
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState<string | null>(null)
  const [tab, setTab]       = useState<'equity' | 'monthly' | 'quarterly' | 'features' | 'trades'>('equity')

  const runBacktest = () => {
    setLoading(true)
    setError(null)
    fetchBacktest(params)
      .then(setResult)
      .catch(e => setError(e?.response?.data?.detail ?? 'Backtest failed'))
      .finally(() => setLoading(false))
  }

  // Auto-run on first load
  useEffect(() => { runBacktest() }, [])

  const m = result?.metrics

  // Merge equity + benchmark curves by date, thin to 200 points
  const equityCurve = result?.equity_curve
    ? (() => {
        const raw = result.equity_curve
        const step = Math.max(1, Math.floor(raw.length / 200))
        const thinned = raw.filter((_, i) => i % step === 0 || i === raw.length - 1)

        // Build benchmark lookup
        const bmMap: Record<string, number> = {}
        ;(result.benchmark_curve ?? []).forEach(p => { bmMap[p.date] = p.value })

        return thinned.map(p => ({
          ...p,
          benchmark: bmMap[p.date] ?? null,
        }))
      })()
    : []

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
        <h1 className="text-2xl font-bold text-white tracking-tight">Backtester</h1>
        <p className="text-sm text-gray-600 mt-1">
          Cross-sectional relative return strategy — predicts which stocks beat Nifty 50 over 5 days. Weekly rebalancing. Out-of-sample on the last 20% of data.
        </p>
      </div>

      {/* Settings */}
      <SettingsPanel params={params} onChange={setParams} onRun={runBacktest} loading={loading} />

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 card border-red-500/20 bg-red-500/5 text-red-400 text-sm">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Warnings */}
      {result?.warnings && result.warnings.length > 0 && (
        <div className="space-y-2 opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
          {result.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-3 px-4 py-3 rounded-xl border border-yellow-500/20 bg-yellow-500/5 text-yellow-400 text-xs">
              <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
              {w}
            </div>
          ))}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-4 animate-fade-in">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="card">
                <div className="skeleton h-3 w-20 mb-4" />
                <div className="skeleton h-8 w-28" />
              </div>
            ))}
          </div>
          <div className="card h-72 flex items-center justify-center text-gray-700 text-sm">
            Computing strategy performance…
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && m && (
        <>
          {/* Metric cards — row 1 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Total Return"
              value={`${m.total_return_pct >= 0 ? '+' : ''}${m.total_return_pct.toFixed(1)}%`}
              sub={`₹${m.initial_capital.toLocaleString('en-IN')} → ₹${m.final_capital.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
              accent={m.total_return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}
              delay={0}
            />
            <MetricCard
              label="Annualized Return"
              value={`${m.annualized_return >= 0 ? '+' : ''}${m.annualized_return.toFixed(1)}%`}
              sub="CAGR"
              accent={m.annualized_return >= 0 ? 'text-emerald-400' : 'text-red-400'}
              delay={60}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={m.sharpe_ratio.toFixed(2)}
              sub={m.sharpe_ratio >= 1 ? 'Good (≥1.0)' : m.sharpe_ratio >= 0 ? 'Below avg' : 'Negative'}
              accent={m.sharpe_ratio >= 1 ? 'text-indigo-400' : 'text-gray-300'}
              delay={120}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${m.max_drawdown_pct.toFixed(1)}%`}
              sub="Peak-to-trough"
              accent="text-red-400"
              delay={180}
            />
          </div>

          {/* Metric cards — row 2 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Win Rate"
              value={`${m.win_rate_pct.toFixed(1)}%`}
              sub={`${m.total_trades.toLocaleString()} trades`}
              accent={m.win_rate_pct >= 50 ? 'text-emerald-400' : 'text-red-400'}
              delay={0}
            />
            <MetricCard
              label="Nifty 50 (same period)"
              value={`${m.benchmark_return >= 0 ? '+' : ''}${m.benchmark_return.toFixed(1)}%`}
              sub="Buy & hold benchmark"
              accent="text-gray-300"
              delay={60}
            />
            <MetricCard
              label="Alpha vs Nifty"
              value={`${m.alpha >= 0 ? '+' : ''}${m.alpha.toFixed(1)}%`}
              sub="Strategy − benchmark"
              accent={m.alpha >= 0 ? 'text-emerald-400' : 'text-red-400'}
              delay={120}
            />
            <MetricCard
              label="Trades"
              value={m.total_trades.toLocaleString()}
              sub={`${m.long_trades} long · ${m.short_trades} short`}
              delay={180}
            />
          </div>

          {/* Period info */}
          <p className="text-xs text-gray-700 opacity-0 animate-fade-up" style={{ animationDelay: '240ms', animationFillMode: 'forwards' }}>
            Backtest period: {m.backtest_start} → {m.backtest_end} &nbsp;·&nbsp; {m.backtest_days} calendar days &nbsp;·&nbsp; <span className={m.active_days < 30 ? 'text-yellow-600' : 'text-gray-500'}>{m.active_days} active trading days</span> &nbsp;·&nbsp; Sharpe computed across full portfolio period (incl. cash days)
          </p>

          {/* Tab switcher */}
          <div className="opacity-0 animate-fade-up flex gap-1 border-b border-white/[0.06]" style={{ animationDelay: '260ms', animationFillMode: 'forwards' }}>
            {(['equity', 'monthly', 'quarterly', 'features', 'trades'] as const).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2.5 text-sm font-medium transition-colors duration-200 border-b-2 -mb-px ${
                  tab === t
                    ? 'text-white border-white'
                    : 'text-gray-600 border-transparent hover:text-gray-400'
                }`}
              >
                {t === 'equity'    ? 'Equity Curve'
                : t === 'monthly'  ? 'Monthly'
                : t === 'quarterly'? 'Quarterly'
                : t === 'features' ? 'Feature Importance'
                : 'Trade History'}
              </button>
            ))}
          </div>

          {/* ── Equity Curve ─────────────────────────────────────────── */}
          {tab === 'equity' && (
            <div className="card opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-6">
                Portfolio Value over Time
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={equityCurve} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={d => d.slice(0, 7)}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={v => `₹${(v / 1000).toFixed(0)}k`}
                    width={52}
                  />
                  <Tooltip content={<EquityTooltip />} />
                  <ReferenceLine
                    y={m.initial_capital}
                    stroke="rgba(255,255,255,0.08)"
                    strokeDasharray="4 4"
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: '#6b7280', paddingTop: 8 }}
                    formatter={(v) => v === 'value' ? 'Strategy' : 'Nifty 50'}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    name="value"
                    stroke={m.total_return_pct >= 0 ? '#34d399' : '#f87171'}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: '#fff' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmark"
                    name="benchmark"
                    stroke="#6366f1"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    dot={false}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-700 mt-3 text-center">
                Dashed line = initial capital ₹{m.initial_capital.toLocaleString('en-IN')}
              </p>
            </div>
          )}

          {/* ── Monthly Returns ─────────────────────────────────────── */}
          {tab === 'monthly' && (
            <div className="card opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-6">
                Monthly Returns (%)
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={result.monthly} margin={{ top: 4, right: 8, left: 8, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                  <XAxis
                    dataKey="month"
                    tick={{ fill: '#4b5563', fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis
                    tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={v => `${v}%`}
                  />
                  <Tooltip content={<MonthTooltip />} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                  <Bar dataKey="return_pct" radius={[3, 3, 0, 0]}>
                    {result.monthly.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.return_pct >= 0 ? '#34d399' : '#f87171'}
                        fillOpacity={0.85}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Quarterly Returns ───────────────────────────────────── */}
          {tab === 'quarterly' && (
            <div className="card opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-6">
                Quarterly Returns (%)
              </p>
              {(!result.quarterly || result.quarterly.length === 0) ? (
                <p className="text-gray-600 text-sm text-center py-12">
                  Not enough data for quarterly breakdown (need ≥ 2 quarters)
                </p>
              ) : (
                <>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={result.quarterly} margin={{ top: 4, right: 8, left: 8, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                      <XAxis
                        dataKey="quarter"
                        tick={{ fill: '#4b5563', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        tick={{ fill: '#4b5563', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={v => `${v}%`}
                      />
                      <Tooltip
                        formatter={(v: number) => [`${v >= 0 ? '+' : ''}${v.toFixed(2)}%`, 'Return']}
                        contentStyle={{ background: '#000', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                        labelStyle={{ color: '#9ca3af', fontSize: 11 }}
                        itemStyle={{ color: '#fff', fontSize: 11 }}
                      />
                      <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                      <Bar dataKey="return_pct" radius={[3, 3, 0, 0]}>
                        {result.quarterly.map((entry: QuarterlyReturn, i: number) => (
                          <Cell
                            key={i}
                            fill={entry.return_pct >= 0 ? '#34d399' : '#f87171'}
                            fillOpacity={0.85}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  {/* Quarterly summary table */}
                  <div className="mt-6 overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-white/[0.06]">
                          <th className="py-2 px-3 text-left text-gray-700 font-medium">Quarter</th>
                          <th className="py-2 px-3 text-right text-gray-700 font-medium">Return</th>
                          <th className="py-2 px-3 text-right text-gray-700 font-medium">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.quarterly.map((q: QuarterlyReturn, i: number) => (
                          <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                            <td className="py-2 px-3 text-gray-400 mono">{q.quarter}</td>
                            <td className={`py-2 px-3 text-right mono font-semibold ${q.return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {q.return_pct >= 0 ? '+' : ''}{q.return_pct.toFixed(2)}%
                            </td>
                            <td className="py-2 px-3 text-right">
                              <span className={`px-2 py-0.5 rounded-full text-xs ${q.return_pct >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                {q.return_pct >= 0 ? 'Positive' : 'Negative'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-gray-700 mt-4">
                    Positive quarters: {result.quarterly.filter((q: QuarterlyReturn) => q.return_pct >= 0).length} / {result.quarterly.length}
                    &nbsp;·&nbsp;
                    Avg quarterly return: {(result.quarterly.reduce((s: number, q: QuarterlyReturn) => s + q.return_pct, 0) / result.quarterly.length).toFixed(2)}%
                  </p>
                </>
              )}
            </div>
          )}

          {/* ── Feature Importance ──────────────────────────────────── */}
          {tab === 'features' && (
            <div className="card opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-6">
                Top 20 Features by Importance
              </p>
              <ResponsiveContainer width="100%" height={480}>
                <BarChart
                  layout="vertical"
                  data={result.feature_importance}
                  margin={{ top: 4, right: 16, left: 100, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: '#4b5563', fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    width={96}
                  />
                  <Tooltip
                    formatter={(v: number) => [v.toFixed(4), 'Importance']}
                    contentStyle={{ background: '#000', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                    labelStyle={{ color: '#9ca3af', fontSize: 11 }}
                    itemStyle={{ color: '#fff', fontSize: 11 }}
                  />
                  <Bar dataKey="importance" fill="#6366f1" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-700 mt-3">
                Higher = model relies on this feature more for making predictions
              </p>
            </div>
          )}

          {/* ── Trade History ───────────────────────────────────────── */}
          {tab === 'trades' && (
            <div className="card p-0 overflow-hidden opacity-0 animate-fade-up" style={{ animationFillMode: 'forwards' }}>
              <div className="px-5 py-4 border-b border-white/[0.05] flex items-center justify-between">
                <p className="text-xs text-gray-600 uppercase tracking-widest font-medium">
                  Last {result.trades.length} Trades
                </p>
                <div className="flex items-center gap-3 text-xs text-gray-600">
                  <span className="flex items-center gap-1">
                    <TrendingUp size={11} className="text-emerald-400" />
                    {result.trades.filter(t => t.won).length} wins
                  </span>
                  <span className="flex items-center gap-1">
                    <TrendingDown size={11} className="text-red-400" />
                    {result.trades.filter(t => !t.won).length} losses
                  </span>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/[0.04]">
                      <th className="py-2.5 px-4 text-left text-xs text-gray-700">Date</th>
                      <th className="py-2.5 px-4 text-left text-xs text-gray-700">Stock</th>
                      <th className="py-2.5 px-4 text-left text-xs text-gray-700">Dir</th>
                      <th className="py-2.5 px-4 text-right text-xs text-gray-700">Rank</th>
                      <th className="py-2.5 px-4 text-right text-xs text-gray-700">Actual</th>
                      <th className="py-2.5 px-4 text-right text-xs text-gray-700">P&L</th>
                      <th className="py-2.5 px-4 text-center text-xs text-gray-700">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...result.trades].reverse().map((t, i) => (
                      <tr
                        key={i}
                        className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
                      >
                        <td className="py-2.5 px-4 text-xs text-gray-600 mono">{t.date}</td>
                        <td className="py-2.5 px-4 text-sm font-medium text-white">{t.ticker}</td>
                        <td className="py-2.5 px-4">
                          <span className={`text-xs font-semibold ${t.direction === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {t.direction === 'LONG' ? '▲ L' : '▼ S'}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-xs text-right mono text-gray-400">{t.confidence.toFixed(1)}%</td>
                        <td className={`py-2.5 px-4 text-xs text-right mono ${t.next_ret >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {t.next_ret >= 0 ? '+' : ''}{t.next_ret.toFixed(2)}%
                        </td>
                        <td className={`py-2.5 px-4 text-xs text-right mono font-semibold ${t.trade_ret >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {t.trade_ret >= 0 ? '+' : ''}{t.trade_ret.toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-4 text-center">
                          <span className={`text-xs px-2 py-0.5 rounded-full ${t.won ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                            {t.won ? 'WIN' : 'LOSS'}
                          </span>
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

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="card text-center py-16 border-dashed border-white/10">
          <Activity size={32} className="text-gray-700 mx-auto mb-3" />
          <p className="text-gray-500 text-sm">Click <span className="text-white font-medium">Run Backtest</span> to simulate the strategy</p>
        </div>
      )}
    </div>
  )
}
