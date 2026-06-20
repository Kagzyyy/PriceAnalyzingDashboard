import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AreaChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell, ReferenceLine,
} from 'recharts'
import {
  fetchGapBacktest, trainGapModel,
  type GapResult, type GapParams,
} from '../api/client'
import { AlertTriangle, Globe, RefreshCw, ChevronDown } from 'lucide-react'

// ── Helpers ───────────────────────────────────────────────────────────────────
const pct   = (v: number, d = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`
const poCol = (v: number) => v >= 0 ? 'var(--green)' : 'var(--red)'

// Month label: "2022-06-30" → "Jun '22"
const fmtMonth = (d: string) => {
  const parts = d.split('-')
  const mon   = new Date(0, parseInt(parts[1]) - 1).toLocaleString('en', { month: 'short' })
  return `${mon} '${parts[0].slice(2)}`
}

// Capital: "Jan 2021" → "Jan '21"
const fmtMonthLabel = (m: string) => m.slice(0, 3) + " '" + m.slice(-2)

const EXIT_COLOR: Record<string, string> = {
  target:       'var(--green)',
  stop_loss:    'var(--red)',
  close_of_day: 'var(--text-3)',
}
const EXIT_LABEL: Record<string, string> = {
  target:       'Target Hit',
  stop_loss:    'Stop Loss',
  close_of_day: 'Day Close',
}

// ── Tooltip — equity curve ────────────────────────────────────────────────────
function EqTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(18,18,20,0.97)',
      border: '1px solid var(--border-2)',
      borderRadius: 3,
      padding: '8px 12px',
      fontSize: 11,
      lineHeight: 1.8,
    }}>
      <p style={{ color: 'var(--text-3)', marginBottom: 3, fontSize: 10 }}>{fmtMonth(label)}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.dataKey} style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontWeight: 600,
          color: p.dataKey === 'value' ? 'var(--text-1)' : 'var(--text-3)',
          fontSize: 12,
        }}>
          <span style={{ color: 'var(--text-3)', fontSize: 10, marginRight: 6, fontWeight: 400 }}>
            {p.dataKey === 'value' ? 'Strategy' : 'Nifty 50'}
          </span>
          ₹{Number(p.value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </p>
      ))}
    </div>
  )
}

// ── Filter toggle (borderless, dot indicator) ──────────────────────────────
function FilterToggle({
  label, active, onClick, accent,
}: {
  label: string; active: boolean; onClick: () => void; accent?: string
}) {
  const accentColor = accent ?? 'var(--accent)'
  return (
    <button
      className={`filter-toggle ${active ? 'active' : ''}`}
      style={active && accent ? { color: 'var(--text-1)' } : undefined}
      onClick={onClick}
    >
      <span
        className="filter-toggle-dot"
        style={active ? { background: accentColor, borderColor: accentColor } : undefined}
      />
      {label}
    </button>
  )
}

// ── Param select ──────────────────────────────────────────────────────────────
function ParamSelect({
  label, value, opts, onChange,
}: {
  label: string
  value: number
  opts: [number, string][]
  onChange: (v: number) => void
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <span className="label">{label}</span>
      <select
        className="input-base"
        style={{ fontSize: 11, padding: '4px 8px', minWidth: 82 }}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
      >
        {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </label>
  )
}

// ── Section label ─────────────────────────────────────────────────────────────
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontSize: 9,
      fontWeight: 600,
      color: 'var(--text-3)',
      letterSpacing: '0.1em',
      textTransform: 'uppercase' as const,
      marginRight: 4,
      userSelect: 'none',
    }}>
      {children}
    </span>
  )
}

// ── Tab button ────────────────────────────────────────────────────────────────
function TabBtn({
  label, active, onClick,
}: {
  label: string; active: boolean; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '10px 0',
        marginRight: 28,
        fontSize: 11,
        fontWeight: active ? 600 : 400,
        color: active ? 'var(--text-1)' : 'var(--text-3)',
        background: 'none',
        borderTop: 'none',
        borderLeft: 'none',
        borderRight: 'none',
        borderBottom: `1.5px solid ${active ? 'var(--text-1)' : 'transparent'}`,
        marginBottom: -1,
        cursor: 'pointer',
        transition: 'color 0.12s',
        letterSpacing: '0.01em',
        whiteSpace: 'nowrap' as const,
        fontFamily: 'inherit',
      }}
    >
      {label}
    </button>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function GapAndGo() {
  const navigate = useNavigate()

  const [params, setParams] = useState<GapParams>({
    gap_threshold:      1.5,
    target_pct:         2.0,
    stop_pct:           0.5,
    ml_threshold:       0.50,
    sent_threshold:    -0.10,
    max_positions:      5,
    gap_max_pct:        4.0,
    use_ml_confirm:     true,
    use_sentiment:      true,
    use_sector_filter:  true,
    use_volume_filter:  true,
    volume_min_ratio:   1.5,
    rsi_low:            35.0,
    rsi_high:           68.0,
    use_rsi_filter:     false,
    use_nifty_filter:   true,
    nifty_min_ret:     -0.3,
    use_trend_filter:   true,
    use_gap_ml_confirm: false,
    gap_ml_threshold:   0.50,
    use_walkforward:    false,
    slippage_pct:       0.10,
    capital:            100_000,
  })

  const [result,     setResult]     = useState<GapResult | null>(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState<string | null>(null)
  const [tab,        setTab]        = useState<'monthly' | 'trades' | 'exit' | 'funnel'>('monthly')
  const [trainSt,    setTrainSt]    = useState<string | null>(null)
  const [showParams, setShowParams] = useState(false)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      const d = await fetchGapBacktest(params)
      setResult(d)
      setTab('monthly')
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  const trainGap = async () => {
    setTrainSt('training')
    try {
      const r = await trainGapModel()
      setTrainSt(r.status === 'training' ? 'started' : 'retraining')
      setTimeout(() => setTrainSt(null), 8000)
    } catch {
      setTrainSt('error')
      setTimeout(() => setTrainSt(null), 4000)
    }
  }

  const navigateToTrade = (trade: {
    ticker: string; date: string; gap_pct: number; ret_pct: number; exit_reason: string
  }) => {
    navigate('/', { state: { tradeIntel: trade } })
  }

  const m = result?.metrics

  // Merge benchmark into equity data
  const equityData = (() => {
    if (!result) return []
    const bMap = new Map(result.benchmark_curve.map(b => [b.date, b.value]))
    return result.equity_curve.map(e => ({
      date:      e.date,
      value:     e.value,
      benchmark: bMap.get(e.date) ?? null,
    }))
  })()

  // ── Derived metric helpers ─────────────────────────────────────────────────
  const sharpeColor = !m ? 'var(--text-1)'
    : m.sharpe_ratio >= 1.5 ? 'var(--green)'
    : m.sharpe_ratio >= 0.8 ? 'var(--amber)'
    : 'var(--red)'

  const pfColor = !m ? 'var(--text-1)'
    : m.profit_factor >= 1.5 ? 'var(--green)'
    : m.profit_factor >= 1.0 ? 'var(--amber)'
    : 'var(--red)'

  const rrValue = m && m.avg_win_pct > 0 && m.avg_loss_pct < 0
    ? `${Math.abs(m.avg_win_pct / m.avg_loss_pct).toFixed(2)}×`
    : '—'

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ padding: '32px 36px', paddingBottom: 72, minWidth: 0 }}>

      {/* ════════════════════════════════════════════════════════════════════
          HEADER
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: 24,
        marginBottom: 28,
        paddingBottom: 22,
        borderBottom: '1px solid var(--border)',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 7 }}>
            <h1 style={{
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--text-1)',
              letterSpacing: '-0.02em',
              margin: 0,
            }}>
              Gap &amp; Go Backtest
            </h1>
            <span style={{ fontSize: 11, color: 'var(--text-3)', letterSpacing: '0.03em' }}>
              Nifty 100 · NSE · 5-year
            </span>
          </div>
          <p style={{
            fontSize: 11,
            color: 'var(--text-2)',
            lineHeight: 1.65,
            margin: 0,
            maxWidth: 680,
          }}>
            Intraday gap-up momentum strategy. Target&nbsp;+{params.target_pct ?? 2.0}% /
            Stop&nbsp;−{params.stop_pct ?? 0.5}% · {((params.target_pct ?? 2.0) / (params.stop_pct ?? 0.5)).toFixed(0)}:1 R:R ·
            Prior-day features only — zero look-ahead bias · 0.20% round-trip execution cost.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          {result && !loading && (
            <span style={{ fontSize: 10, color: 'var(--text-3)', letterSpacing: '0.02em' }}>
              {m!.backtest_start} — {m!.backtest_end}
            </span>
          )}
          <button
            className="btn-primary"
            onClick={run}
            disabled={loading}
          >
            {loading && <RefreshCw size={10} className="animate-spin" />}
            {loading ? 'Running…' : 'Run Backtest'}
          </button>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          SIGNAL FILTERS
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
          <SectionLabel>Filters</SectionLabel>

          <FilterToggle
            label="ML Direction Model"
            active={!!params.use_ml_confirm}
            onClick={() => setParams(p => ({ ...p, use_ml_confirm: !p.use_ml_confirm }))}
          />
          <FilterToggle
            label="Sentiment"
            active={!!params.use_sentiment}
            onClick={() => setParams(p => ({ ...p, use_sentiment: !p.use_sentiment }))}
          />
          <FilterToggle
            label="Sector Momentum"
            active={!!params.use_sector_filter}
            onClick={() => setParams(p => ({ ...p, use_sector_filter: !p.use_sector_filter }))}
          />
          <FilterToggle
            label="Relative Volume"
            active={!!params.use_volume_filter}
            onClick={() => setParams(p => ({ ...p, use_volume_filter: !p.use_volume_filter }))}
          />
          <FilterToggle
            label="Market Direction"
            active={!!params.use_nifty_filter}
            onClick={() => setParams(p => ({ ...p, use_nifty_filter: !p.use_nifty_filter }))}
          />
          <FilterToggle
            label="Higher Timeframe Bias"
            active={!!params.use_trend_filter}
            onClick={() => setParams(p => ({ ...p, use_trend_filter: !p.use_trend_filter }))}
          />
          <FilterToggle
            label="RSI Range"
            active={!!params.use_rsi_filter}
            onClick={() => setParams(p => ({ ...p, use_rsi_filter: !p.use_rsi_filter }))}
          />
          <FilterToggle
            label="Signal Confidence Model"
            active={!!params.use_gap_ml_confirm}
            onClick={() => setParams(p => ({ ...p, use_gap_ml_confirm: !p.use_gap_ml_confirm }))}
          />

          {/* Separator */}
          <span style={{
            display: 'inline-block',
            width: 1, height: 14,
            background: 'var(--border-2)',
            margin: '0 6px',
            verticalAlign: 'middle',
          }} />

          <FilterToggle
            label="Walk-Forward OOS"
            active={!!params.use_walkforward}
            accent="var(--amber)"
            onClick={() => setParams(p => ({ ...p, use_walkforward: !p.use_walkforward }))}
          />
        </div>

        {/* Walk-forward note */}
        {params.use_walkforward && (
          <p style={{
            marginTop: 8,
            fontSize: 10,
            color: 'var(--amber)',
            lineHeight: 1.6,
            opacity: 0.8,
            paddingLeft: 2,
          }}>
            Walk-Forward enabled — retrains per year (2022→2023→2024→2025) for out-of-sample predictions only.
            Expect Sharpe ~1.2–1.8 OOS; above 1.0 is genuinely profitable. Runtime ~3 min.
          </p>
        )}
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          PARAMETERS (collapsible)
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ marginTop: 14, marginBottom: 24 }}>
        <button
          onClick={() => setShowParams(p => !p)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '4px 0',
            color: 'var(--text-3)',
            fontFamily: 'inherit',
          }}
        >
          <ChevronDown
            size={11}
            style={{
              transform: showParams ? 'rotate(0deg)' : 'rotate(-90deg)',
              transition: 'transform 0.15s ease',
              color: 'var(--text-3)',
            }}
          />
          <SectionLabel>Parameters</SectionLabel>
          {!showParams && (
            <span style={{ fontSize: 10, color: 'var(--text-3)', fontWeight: 400, marginLeft: 4 }}>
              Gap {params.gap_threshold}% · Target +{params.target_pct}% · Stop −{params.stop_pct}% · Slippage {params.slippage_pct}%
            </span>
          )}
        </button>

        {showParams && (
          <div style={{
            display: 'flex',
            alignItems: 'flex-end',
            flexWrap: 'wrap',
            gap: 18,
            marginTop: 14,
            paddingTop: 14,
            borderTop: '1px solid var(--border)',
          }}>
            <ParamSelect label="Gap Min"
              value={params.gap_threshold ?? 1.5}
              opts={[[1.0,'1.0%'],[1.5,'1.5%'],[2.0,'2.0%'],[2.5,'2.5%'],[3.0,'3.0%']]}
              onChange={v => setParams(p => ({ ...p, gap_threshold: v }))}
            />
            <ParamSelect label="Target"
              value={params.target_pct ?? 2.0}
              opts={[[1.5,'1.5%'],[2.0,'2.0%'],[2.5,'2.5%'],[3.0,'3.0%']]}
              onChange={v => setParams(p => ({ ...p, target_pct: v }))}
            />
            <ParamSelect label="Stop"
              value={params.stop_pct ?? 0.5}
              opts={[[0.5,'0.5%'],[0.75,'0.75%'],[1.0,'1.0%'],[1.5,'1.5%']]}
              onChange={v => setParams(p => ({ ...p, stop_pct: v }))}
            />
            <ParamSelect label="Signal Threshold"
              value={params.ml_threshold ?? 0.50}
              opts={[[0.50,'0.50'],[0.51,'0.51'],[0.52,'0.52'],[0.55,'0.55']]}
              onChange={v => setParams(p => ({ ...p, ml_threshold: v }))}
            />
            <ParamSelect label="Max Positions"
              value={params.max_positions ?? 5}
              opts={[[3,'3'],[5,'5'],[10,'10'],[15,'15']]}
              onChange={v => setParams(p => ({ ...p, max_positions: v }))}
            />
            <ParamSelect label="Slippage"
              value={params.slippage_pct ?? 0.10}
              opts={[[0,'0%'],[0.05,'0.05%'],[0.10,'0.10%'],[0.25,'0.25%'],[0.50,'0.50%']]}
              onChange={v => setParams(p => ({ ...p, slippage_pct: v }))}
            />
            <ParamSelect label="Capital"
              value={params.capital ?? 100000}
              opts={[[100000,'₹1L'],[500000,'₹5L'],[1000000,'₹10L']]}
              onChange={v => setParams(p => ({ ...p, capital: v }))}
            />

            {/* Divider */}
            <div style={{ width: 1, height: 32, background: 'var(--border)', alignSelf: 'flex-end', marginBottom: 4 }} />

            {/* Train Signal Confidence Model */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <span className="label">Signal Confidence Model</span>
              <button
                className="btn-ghost"
                onClick={trainGap}
                disabled={trainSt === 'training'}
                style={{ fontSize: 11, padding: '4px 12px', height: 29 }}
              >
                {trainSt === 'training' ? 'Sending…'
                : (trainSt === 'started' || trainSt === 'retraining') ? '✓ Queued'
                : trainSt === 'error' ? '✗ Error'
                : 'Train Model'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          LOADING
      ════════════════════════════════════════════════════════════════════ */}
      {loading && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '40px 0',
          color: 'var(--text-3)',
        }}>
          <div style={{
            width: 14, height: 14,
            border: '1.5px solid var(--border-2)',
            borderTopColor: 'var(--text-2)',
            borderRadius: '50%',
            animation: 'spin 0.7s linear infinite',
          }} />
          <span style={{ fontSize: 12 }}>Scanning Nifty 100 · simulating trades…</span>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════
          ERROR
      ════════════════════════════════════════════════════════════════════ */}
      {error && !loading && (
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 8,
          padding: '10px 14px',
          borderLeft: '2px solid var(--red)',
          background: 'var(--red-dim)',
          borderRadius: '0 3px 3px 0',
        }}>
          <AlertTriangle size={12} style={{ color: 'var(--red)', flexShrink: 0, marginTop: 1 }} />
          <p style={{ fontSize: 11, color: 'var(--red)', lineHeight: 1.5, margin: 0 }}>{error}</p>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════
          RESULTS
      ════════════════════════════════════════════════════════════════════ */}
      {result && !loading && (
        <div className="animate-fade-up">

          {/* Warnings */}
          {result.warnings.map((w, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 8,
              padding: '8px 0',
              marginBottom: 4,
              borderBottom: '1px solid var(--border)',
            }}>
              <AlertTriangle size={11} style={{ color: 'var(--amber)', flexShrink: 0, marginTop: 1 }} />
              <p style={{ fontSize: 11, color: 'var(--amber)', lineHeight: 1.5, margin: 0 }}>{w}</p>
            </div>
          ))}

          {/* ── KPI STRIP ──────────────────────────────────────────────────── */}
          <div style={{
            display: 'flex',
            gap: 0,
            paddingTop: result.warnings.length ? 20 : 0,
            paddingBottom: 28,
            borderBottom: '1px solid var(--border)',
            marginBottom: 32,
            overflowX: 'auto',
          }}>
            {[
              {
                label: 'Total Return',
                value: pct(m!.total_return_pct, 1),
                sub: `₹${(m!.initial_capital/1000).toFixed(0)}k → ₹${(m!.final_capital/1000).toFixed(0)}k`,
                color: poCol(m!.total_return_pct),
              },
              {
                label: 'CAGR',
                value: pct(m!.annualized_return, 1),
                sub: 'Annualized',
                color: poCol(m!.annualized_return),
              },
              {
                label: 'Sharpe Ratio',
                value: m!.sharpe_ratio.toFixed(2),
                sub: m!.sharpe_ratio >= 1.5 ? 'Strong' : m!.sharpe_ratio >= 0.8 ? 'Adequate' : 'Weak',
                color: sharpeColor,
              },
              {
                label: 'Max Drawdown',
                value: `${m!.max_drawdown_pct.toFixed(1)}%`,
                sub: 'Peak-to-trough',
                color: 'var(--red)',
              },
              {
                label: 'Win Rate',
                value: `${m!.win_rate_pct.toFixed(1)}%`,
                sub: `${m!.total_trades} trades`,
                color: m!.win_rate_pct >= 45 ? 'var(--green)' : 'var(--red)',
              },
              {
                label: 'Profit Factor',
                value: m!.profit_factor.toFixed(2),
                sub: 'Gross W / L',
                color: pfColor,
              },
              {
                label: 'Alpha vs Nifty',
                value: pct(m!.alpha, 1),
                sub: `Nifty ${pct(m!.benchmark_return, 1)}`,
                color: poCol(m!.alpha),
              },
              {
                label: 'Avg R:R',
                value: rrValue,
                sub: `+${m!.avg_win_pct.toFixed(2)}% / ${m!.avg_loss_pct.toFixed(2)}%`,
                color: 'var(--text-1)',
              },
            ].map(({ label, value, sub, color }, i) => (
              <div key={label} style={{
                paddingLeft:  i === 0 ? 0 : 28,
                paddingRight: 28,
                borderLeft:   i === 0 ? 'none' : '1px solid var(--border)',
                flexShrink: 0,
              }}>
                <p style={{
                  fontSize: 9,
                  fontWeight: 600,
                  color: 'var(--text-3)',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  marginBottom: 8,
                  margin: '0 0 8px',
                }}>
                  {label}
                </p>
                <p style={{
                  fontSize: 22,
                  fontWeight: 700,
                  fontFamily: 'JetBrains Mono, monospace',
                  letterSpacing: '-0.03em',
                  lineHeight: 1,
                  color,
                  margin: '0 0 5px',
                  fontVariantNumeric: 'tabular-nums',
                }}>
                  {value}
                </p>
                <p style={{
                  fontSize: 10,
                  color: 'var(--text-3)',
                  margin: 0,
                }}>
                  {sub}
                </p>
              </div>
            ))}
          </div>

          {/* ── EQUITY CURVE (hero) ─────────────────────────────────────────── */}
          <div style={{ marginBottom: 36 }}>

            {/* Chart header */}
            <div style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              marginBottom: 18,
            }}>
              <div>
                <span style={{
                  fontSize: 9,
                  fontWeight: 600,
                  color: 'var(--text-3)',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                }}>
                  Portfolio Value
                </span>
              </div>
              <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
                {[
                  { label: 'Strategy', color: 'var(--text-1)', dashed: false },
                  { label: 'Nifty 50', color: 'var(--text-3)', dashed: true },
                ].map(l => (
                  <div key={l.label} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 7,
                    fontSize: 10,
                    color: 'var(--text-3)',
                  }}>
                    <svg width="20" height="2">
                      {l.dashed
                        ? <line x1="0" y1="1" x2="20" y2="1" stroke={l.color} strokeWidth="1.5" strokeDasharray="3 3" />
                        : <line x1="0" y1="1" x2="20" y2="1" stroke={l.color} strokeWidth="1.5" />
                      }
                    </svg>
                    {l.label}
                  </div>
                ))}
              </div>
            </div>

            {/* Area chart */}
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart
                data={equityData}
                margin={{ top: 4, right: 4, bottom: 0, left: 0 }}
              >
                <defs>
                  <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="var(--text-1)" stopOpacity={0.10} />
                    <stop offset="100%" stopColor="var(--text-1)" stopOpacity={0}    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="0"
                  stroke="rgba(255,255,255,0.04)"
                  horizontal={true}
                  vertical={false}
                />
                <XAxis
                  dataKey="date"
                  tick={{ fill: 'var(--text-3)', fontSize: 9 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={fmtMonth}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: 'var(--text-3)', fontSize: 9 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={v => `₹${(v / 1000).toFixed(0)}k`}
                  width={44}
                />
                <Tooltip content={<EqTooltip />} />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="var(--text-1)"
                  strokeWidth={1.5}
                  fill="url(#stratGrad)"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="benchmark"
                  stroke="var(--text-3)"
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="4 4"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>

            {/* Period note */}
            <p style={{
              marginTop: 10,
              fontSize: 10,
              color: 'var(--text-3)',
              lineHeight: 1.6,
            }}>
              {m!.backtest_start} → {m!.backtest_end} &nbsp;·&nbsp; {m!.backtest_days} trading days
              &nbsp;·&nbsp; 0.20% round-trip cost (STT + exchange)
              {(params.slippage_pct ?? 0) > 0 ? ` · ${params.slippage_pct}% fill slippage` : ''}
            </p>
          </div>

          {/* ── TAB NAV ─────────────────────────────────────────────────────── */}
          <div style={{
            display: 'flex',
            borderBottom: '1px solid var(--border)',
            marginBottom: 28,
          }}>
            <TabBtn label="Monthly Returns"  active={tab === 'monthly'} onClick={() => setTab('monthly')} />
            <TabBtn label={`Trade Log (${result.trades.length})`} active={tab === 'trades'} onClick={() => setTab('trades')} />
            <TabBtn label="Exit Analysis"    active={tab === 'exit'}    onClick={() => setTab('exit')} />
            <TabBtn label="Signal Funnel"    active={tab === 'funnel'}  onClick={() => setTab('funnel')} />
          </div>

          {/* ── MONTHLY RETURNS ─────────────────────────────────────────────── */}
          {tab === 'monthly' && (
            <div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={result.monthly}
                  margin={{ top: 4, right: 4, bottom: 0, left: 0 }}
                  barCategoryGap="18%"
                >
                  <CartesianGrid
                    strokeDasharray="0"
                    stroke="rgba(255,255,255,0.04)"
                    horizontal={true}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="month"
                    tick={{ fill: 'var(--text-3)', fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={fmtMonthLabel}
                    interval={2}
                  />
                  <YAxis
                    tick={{ fill: 'var(--text-3)', fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`}
                    width={40}
                  />
                  <Tooltip
                    formatter={(v: number) => [`${v >= 0 ? '+' : ''}${v.toFixed(2)}%`, 'Return']}
                    contentStyle={{
                      background: 'rgba(18,18,20,0.97)',
                      border: '1px solid var(--border-2)',
                      borderRadius: 3,
                      fontSize: 11,
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                    labelStyle={{ color: 'var(--text-3)', fontFamily: 'Inter, sans-serif', fontSize: 10 }}
                    cursor={{ fill: 'rgba(255,255,255,0.025)' }}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.10)" strokeWidth={1} />
                  <Bar dataKey="return_pct" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                    {result.monthly.map((mo, i) => (
                      <Cell
                        key={i}
                        fill={mo.return_pct >= 0
                          ? 'rgba(52,199,88,0.72)'
                          : 'rgba(229,82,74,0.72)'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 10 }}>
                Zero-return months indicate no qualifying setups — strategy sat in cash.
                Not a profit/loss cancellation.
              </p>
            </div>
          )}

          {/* ── TRADE LOG ───────────────────────────────────────────────────── */}
          {tab === 'trades' && (
            <div>
              <p style={{
                fontSize: 10,
                color: 'var(--text-3)',
                marginBottom: 16,
                letterSpacing: '0.02em',
              }}>
                Showing last {result.trades.length} trades · newest first · click{' '}
                <Globe size={10} style={{ display: 'inline', verticalAlign: 'middle', marginBottom: 1 }} />{' '}
                to open Trade Intelligence
              </p>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Ticker</th>
                      <th style={{ textAlign: 'right' as const }}>Gap</th>
                      <th style={{ textAlign: 'right' as const }}>Entry</th>
                      <th style={{ textAlign: 'right' as const }}>Return</th>
                      <th>Exit</th>
                      <th style={{ width: 28 }} />
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.slice().reverse().map((t, i) => (
                      <tr key={i}>
                        <td style={{
                          fontFamily: 'JetBrains Mono, monospace',
                          fontSize: 11,
                          color: 'var(--text-3)',
                          fontVariantNumeric: 'tabular-nums',
                        }}>
                          {t.date}
                        </td>
                        <td style={{ fontWeight: 600, color: 'var(--text-1)', fontSize: 12 }}>
                          {t.ticker.replace('.NS', '').replace('.BO', '')}
                        </td>
                        <td style={{
                          textAlign: 'right',
                          fontFamily: 'JetBrains Mono, monospace',
                          fontSize: 11,
                          color: 'var(--green)',
                          fontVariantNumeric: 'tabular-nums',
                        }}>
                          +{t.gap_pct.toFixed(2)}%
                        </td>
                        <td style={{
                          textAlign: 'right',
                          fontFamily: 'JetBrains Mono, monospace',
                          fontSize: 11,
                          color: 'var(--text-3)',
                          fontVariantNumeric: 'tabular-nums',
                        }}>
                          ₹{t.entry.toLocaleString('en-IN')}
                        </td>
                        <td style={{
                          textAlign: 'right',
                          fontFamily: 'JetBrains Mono, monospace',
                          fontWeight: 600,
                          fontSize: 11,
                          color: t.ret_pct >= 0 ? 'var(--green)' : 'var(--red)',
                          fontVariantNumeric: 'tabular-nums',
                        }}>
                          {t.ret_pct >= 0 ? '+' : ''}{t.ret_pct.toFixed(3)}%
                        </td>
                        <td style={{
                          fontSize: 11,
                          color: EXIT_COLOR[t.exit_reason] ?? 'var(--text-3)',
                        }}>
                          {EXIT_LABEL[t.exit_reason] ?? t.exit_reason}
                        </td>
                        <td>
                          <button
                            title="Trade Intelligence"
                            onClick={() => navigateToTrade({
                              ticker:      t.ticker,
                              date:        t.date,
                              gap_pct:     t.gap_pct,
                              ret_pct:     t.ret_pct,
                              exit_reason: t.exit_reason,
                            })}
                            style={{
                              color: 'var(--text-3)',
                              padding: '3px 4px',
                              background: 'none',
                              border: 'none',
                              cursor: 'pointer',
                              borderRadius: 2,
                              display: 'flex',
                              alignItems: 'center',
                              transition: 'color 0.12s',
                            }}
                            onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = 'var(--text-1)')}
                            onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = 'var(--text-3)')}
                          >
                            <Globe size={12} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── EXIT ANALYSIS ───────────────────────────────────────────────── */}
          {tab === 'exit' && (() => {
            const exitEntries = Object.entries(result.exit_breakdown)
            const total = exitEntries.reduce((s, [, v]) => s + v, 0)
            return (
              <div style={{ maxWidth: 520 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  {exitEntries.map(([name, count]) => {
                    const pctVal = ((count / total) * 100)
                    return (
                      <div key={name}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'baseline',
                          marginBottom: 8,
                        }}>
                          <span style={{
                            fontSize: 12,
                            fontWeight: 500,
                            color: EXIT_COLOR[name] ?? 'var(--text-2)',
                          }}>
                            {EXIT_LABEL[name] ?? name}
                          </span>
                          <div style={{ display: 'flex', gap: 12, alignItems: 'baseline' }}>
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace',
                              fontSize: 11,
                              color: 'var(--text-3)',
                            }}>
                              {count} trades
                            </span>
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace',
                              fontSize: 13,
                              fontWeight: 600,
                              color: EXIT_COLOR[name] ?? 'var(--text-1)',
                            }}>
                              {pctVal.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <div style={{
                          height: 3,
                          background: 'var(--border)',
                          borderRadius: 2,
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            height: '100%',
                            width: `${pctVal}%`,
                            background: name === 'target'    ? 'rgba(52,199,88,0.65)'
                                      : name === 'stop_loss' ? 'rgba(229,82,74,0.65)'
                                      : 'rgba(100,100,109,0.45)',
                            borderRadius: 2,
                            transition: 'width 0.6s ease',
                          }} />
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Summary row */}
                <div style={{
                  marginTop: 28,
                  paddingTop: 16,
                  borderTop: '1px solid var(--border)',
                  display: 'flex',
                  gap: 28,
                }}>
                  {[
                    { label: 'Avg Win',      value: `+${m!.avg_win_pct.toFixed(3)}%`,  color: 'var(--green)' },
                    { label: 'Avg Loss',     value: `${m!.avg_loss_pct.toFixed(3)}%`,   color: 'var(--red)'   },
                    { label: 'Profit Factor',value: m!.profit_factor.toFixed(2),        color: pfColor        },
                  ].map(({ label, value, color }) => (
                    <div key={label}>
                      <p style={{ fontSize: 9, fontWeight: 600, color: 'var(--text-3)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 5 }}>
                        {label}
                      </p>
                      <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, fontWeight: 700, color, letterSpacing: '-0.02em' }}>
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}

          {/* ── SIGNAL FUNNEL ───────────────────────────────────────────────── */}
          {tab === 'funnel' && (() => {
            const fs = result.filter_stats
            const rawTotal = (fs.total_gaps || 0) + (fs.nifty_filtered || 0)
            const steps = [
              { label: 'Raw gaps detected',    val: rawTotal,                                                                     note: `≥${params.gap_threshold}% gap-up` },
              { label: 'After market filter',  val: rawTotal  - (fs.nifty_filtered   || 0),                                       note: 'Nifty direction filter' },
              { label: 'After gap cap',        val: fs.total_gaps,                                                                note: `≤${params.gap_max_pct}% cap` },
              { label: 'After ML direction',   val: fs.total_gaps - (fs.ml_filtered  || 0),                                       note: `prob ≥ ${params.ml_threshold}` },
              { label: 'After sector rank',    val: fs.total_gaps - (fs.ml_filtered  || 0) - (fs.sector_filtered  || 0),          note: 'Sector momentum rank > 0.5' },
              { label: 'After volume',         val: fs.total_gaps - (fs.ml_filtered  || 0) - (fs.sector_filtered  || 0) - (fs.volume_filtered || 0), note: `Vol ratio ≥ ${params.volume_min_ratio}×` },
              { label: 'Trades executed',      val: fs.traded,                                                                    note: `max ${params.max_positions} per day` },
            ]
            return (
              <div style={{ maxWidth: 600 }}>
                {steps.map((s, i) => (
                  <div key={s.label} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 16,
                    marginBottom: i < steps.length - 1 ? 12 : 0,
                  }}>
                    <div style={{ width: 160, flexShrink: 0 }}>
                      <p style={{ fontSize: 11, color: 'var(--text-2)', margin: 0, lineHeight: 1.4 }}>
                        {s.label}
                      </p>
                      <p style={{ fontSize: 9, color: 'var(--text-3)', margin: 0, letterSpacing: '0.02em' }}>
                        {s.note}
                      </p>
                    </div>
                    <div style={{
                      flex: 1,
                      height: 3,
                      background: 'var(--border)',
                      borderRadius: 2,
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${rawTotal > 0 ? Math.min(100, (s.val / rawTotal) * 100) : 0}%`,
                        background: i === steps.length - 1
                          ? 'var(--accent)'
                          : 'rgba(255,255,255,0.18)',
                        borderRadius: 2,
                        transition: 'width 0.6s ease',
                      }} />
                    </div>
                    <span style={{
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: 11,
                      color: i === steps.length - 1 ? 'var(--text-1)' : 'var(--text-3)',
                      fontWeight: i === steps.length - 1 ? 600 : 400,
                      width: 48,
                      textAlign: 'right',
                      fontVariantNumeric: 'tabular-nums',
                      flexShrink: 0,
                    }}>
                      {s.val.toLocaleString('en-IN')}
                    </span>
                  </div>
                ))}
                <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 20 }}>
                  Pass rate: <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                    {rawTotal > 0 ? ((fs.traded / rawTotal) * 100).toFixed(1) : '—'}%
                  </span> of raw gaps reach execution
                </p>
              </div>
            )
          })()}

        </div>
      )}

      {/* Inline spin keyframe */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
