/**
 * God's Eye View — Two modes:
 *
 *  1. MACRO MODE (default)
 *     Full-viewport globe with AI-generated forward macro predictions.
 *     Left panel: prediction list, key risks, India outlook.
 *     Overlays: regime status (top), sector outlook (bottom), detail card (right).
 *
 *  2. TRADE INTEL MODE (when navigated from Gap & Go with location.state.tradeIntel)
 *     Globe focused on the trade's exchange location.
 *     Left panel: back button, trade summary, Groq+RAG reasoning, news article list.
 *     Overlays: trade header (top), none (bottom).
 */
import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { RefreshCw, X, ArrowLeft } from 'lucide-react'
import {
  fetchMarketPulse, fetchTradeIntel,
  type MarketPulseResult, type MacroPrediction,
} from '../api/client'
import TradeGlobe from '../components/TradeGlobe'
import {
  CATEGORY_COLORS, CATEGORY_LABELS,
  type GeoEvent, type GlobeCategory, type TradeIntelResult,
} from '../types/globe'

// ── Constants ─────────────────────────────────────────────────────────────────
const DIRECTION_COLOR: Record<string, string> = {
  bullish: 'var(--green)',
  bearish: 'var(--red)',
  neutral: 'var(--accent)',
  risk:    'var(--amber)',
}
const DIRECTION_LABEL: Record<string, string> = {
  bullish: '↑ BULLISH',
  bearish: '↓ BEARISH',
  neutral: '→ NEUTRAL',
  risk:    '⚠ RISK',
}
const REGIME_LABEL: Record<string, string> = {
  risk_on:      'RISK ON',
  risk_off:     'RISK OFF',
  neutral:      'NEUTRAL',
  transitional: 'TRANSITIONAL',
}
const REGIME_COLOR: Record<string, string> = {
  risk_on:      'var(--green)',
  risk_off:     'var(--red)',
  neutral:      'var(--accent)',
  transitional: 'var(--amber)',
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const confPct = (v: number) => `${Math.round(v * 100)}%`

// ── Types ─────────────────────────────────────────────────────────────────────
interface TradeState {
  ticker:      string
  date:        string
  gap_pct:     number
  ret_pct:     number
  exit_reason: string
}

// ── PredCard (macro mode) ─────────────────────────────────────────────────────
function PredCard({
  pred, index, active, onClick,
}: { pred: MacroPrediction; index: number; active: boolean; onClick: () => void }) {
  const col = DIRECTION_COLOR[pred.direction] ?? 'var(--accent)'
  return (
    <button
      onClick={onClick}
      className={`pred-card w-full text-left ${active ? 'active' : ''}`}
      style={{ display: 'block' }}
    >
      <div className="conf-bar mb-2">
        <div className="conf-bar-fill" style={{ width: confPct(pred.confidence), background: col }} />
      </div>
      <div className="flex items-start gap-2 mb-1.5">
        <span className="num" style={{ fontSize: 10, color: 'var(--text-3)', flexShrink: 0, marginTop: 1 }}>
          {String(index + 1).padStart(2, '0')}
        </span>
        <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-1)', lineHeight: 1.4 }}>
          {pred.title}
        </p>
      </div>
      <div className="flex items-center gap-2" style={{ paddingLeft: 18 }}>
        <span style={{ fontSize: 10, color: CATEGORY_COLORS[pred.category as GlobeCategory] ?? 'var(--text-2)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {CATEGORY_LABELS[pred.category as GlobeCategory] ?? pred.category}
        </span>
        <span style={{ color: 'var(--text-3)', fontSize: 10 }}>·</span>
        <span style={{ fontSize: 10, color: 'var(--text-2)' }}>{pred.timeframe}</span>
        <span style={{ color: 'var(--text-3)', fontSize: 10 }}>·</span>
        <span style={{ fontSize: 10, color: col, fontWeight: 700 }}>
          {DIRECTION_LABEL[pred.direction] ?? pred.direction}
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 'auto', fontFamily: 'JetBrains Mono', letterSpacing: '-0.02em' }}>
          {confPct(pred.confidence)}
        </span>
      </div>
    </button>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function GodEyeView() {
  const location = useLocation()
  const navigate = useNavigate()

  // ── Macro mode state ───────────────────────────────────────────────────────
  const [pulseData,  setPulseData]  = useState<MarketPulseResult | null>(null)
  const [pulseLoad,  setPulseLoad]  = useState(false)
  const [pulseErr,   setPulseErr]   = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detailEv,   setDetailEv]   = useState<GeoEvent | null>(null)

  // ── Trade intel mode state ─────────────────────────────────────────────────
  const tradeState = (location.state as any)?.tradeIntel as TradeState | null
  const isTradeMode = !!tradeState
  const [tradeData,  setTradeData]  = useState<TradeIntelResult | null>(null)
  const [tradeLoad,  setTradeLoad]  = useState(false)
  const [tradeErr,   setTradeErr]   = useState<string | null>(null)

  // ── Load on mount / mode switch ────────────────────────────────────────────
  const loadPulse = async () => {
    setPulseLoad(true); setPulseErr(null)
    try   { setPulseData(await fetchMarketPulse()) }
    catch (e: any) { setPulseErr(e?.response?.data?.detail ?? e?.message ?? 'Failed') }
    finally { setPulseLoad(false) }
  }

  const loadTrade = async () => {
    if (!tradeState) return
    setTradeData(null); setTradeLoad(true); setTradeErr(null)
    try {
      const d = await fetchTradeIntel(
        tradeState.ticker, tradeState.date,
        { gap_pct: tradeState.gap_pct },
      )
      setTradeData(d as TradeIntelResult)
    } catch (e: any) {
      setTradeErr(e?.response?.data?.detail ?? e?.message ?? 'Failed to load trade intel')
    } finally { setTradeLoad(false) }
  }

  useEffect(() => {
    if (isTradeMode) {
      loadTrade()
    } else {
      loadPulse()
    }
    // Reset selection when mode changes
    setSelectedId(null)
    setDetailEv(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTradeMode, tradeState?.ticker, tradeState?.date])

  // ── Handlers (macro mode) ──────────────────────────────────────────────────
  const handlePredClick = (pred: MacroPrediction, index: number) => {
    const id = `pred_${index}`
    setSelectedId(prev => prev === id ? null : id)
    setDetailEv(null)
  }
  const handleGlobeClick = (ev: GeoEvent) => {
    if (ev.id === 'macro_hub') return
    setDetailEv(ev)
    setSelectedId(ev.id)
  }

  // ── Derived ────────────────────────────────────────────────────────────────
  const regime      = pulseData?.macro_regime ?? 'neutral'
  const regimeColor = REGIME_COLOR[regime] ?? 'var(--accent)'
  const exitCol     = tradeState
    ? tradeState.exit_reason === 'target'    ? 'var(--green)'
    : tradeState.exit_reason === 'stop_loss' ? 'var(--red)'
    : 'var(--text-2)'
    : 'var(--text-2)'

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0, overflow: 'hidden' }}>

      {/* ══════════════════════════════════════════════════════════════════════
          LEFT PANEL
      ══════════════════════════════════════════════════════════════════════ */}
      <div style={{
        width: 310, flexShrink: 0,
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
        background: 'rgba(3,8,14,0.92)',
      }}>

        {/* ── TRADE MODE left panel ──────────────────────────────────────── */}
        {isTradeMode ? (
          <>
            {/* Back button + trade header */}
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border)',
              flexShrink: 0,
            }}>
              <button
                onClick={() => navigate(-1)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--text-2)', fontSize: 11, marginBottom: 12,
                  padding: 0,
                  transition: 'color 0.1s',
                }}
                onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = 'var(--accent)')}
                onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = 'var(--text-2)')}
              >
                <ArrowLeft size={12} />
                Back to Gap &amp; Go
              </button>

              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
                  {tradeState!.ticker.replace('.NS', '').replace('.BO', '')}
                </span>
                <span className="label">{tradeState!.date}</span>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <span className="num" style={{ fontSize: 11, color: 'var(--green)' }}>
                  +{tradeState!.gap_pct.toFixed(2)}% gap
                </span>
                <span className="num" style={{ fontSize: 11, color: tradeState!.ret_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {tradeState!.ret_pct >= 0 ? '+' : ''}{tradeState!.ret_pct.toFixed(3)}% return
                </span>
                <span style={{ fontSize: 11, textTransform: 'capitalize', color: exitCol }}>
                  {tradeState!.exit_reason.replace('_', ' ')}
                </span>
              </div>
            </div>

            {/* Scrollable body: reasoning + news */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 20 }}>

              {/* AI Reasoning */}
              <div>
                <p className="label" style={{ marginBottom: 10 }}>
                  AI Reasoning
                  {tradeData && (
                    <span style={{ color: 'var(--text-3)', marginLeft: 6, textTransform: 'none', letterSpacing: 0 }}>
                      · via Groq + RAG ({tradeData.rag_count ?? 0} docs)
                    </span>
                  )}
                </p>
                {tradeLoad ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {[100, 90, 75, 60].map(w => (
                      <div key={w} className="skeleton" style={{ height: 11, width: `${w}%` }} />
                    ))}
                    <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6 }}>
                      Querying RAG store · Groq LLaMA 3.3 generating…
                    </p>
                  </div>
                ) : tradeErr ? (
                  <p style={{ fontSize: 11, color: 'var(--red)', lineHeight: 1.6 }}>{tradeErr}</p>
                ) : (
                  <p style={{ fontSize: 12, color: 'var(--text-1)', lineHeight: 1.75 }}>
                    {tradeData?.reasoning ?? 'No reasoning available.'}
                  </p>
                )}
              </div>

              {/* News articles */}
              {tradeData && tradeData.n_articles > 0 && (
                <div>
                  <p className="label" style={{ marginBottom: 10 }}>
                    Related News
                    <span style={{ color: 'var(--text-3)', marginLeft: 6, textTransform: 'none', letterSpacing: 0 }}>
                      · {tradeData.n_articles} sources
                    </span>
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {tradeData.events
                      .filter(ev => !ev.id.startsWith('hub'))
                      .map(ev => (
                        <a
                          key={ev.id}
                          href={ev.source_url || '#'}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            display: 'block',
                            padding: '9px 11px',
                            background: 'var(--surface-2)',
                            border: '1px solid var(--border)',
                            borderRadius: 2,
                            textDecoration: 'none',
                            transition: 'border-color 0.1s',
                          }}
                          onMouseEnter={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--border-2)')}
                          onMouseLeave={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--border)')}
                        >
                          <p style={{ fontSize: 11, color: 'var(--text-1)', lineHeight: 1.4, marginBottom: 3 }}>
                            {ev.title}
                          </p>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{
                              fontSize: 9, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
                              color: CATEGORY_COLORS[ev.category as GlobeCategory] ?? 'var(--text-3)',
                            }}>
                              {CATEGORY_LABELS[ev.category as GlobeCategory] ?? ev.category}
                            </span>
                            <span style={{ fontSize: 10, color: 'var(--text-3)' }}>·</span>
                            <span style={{ fontSize: 10, color: 'var(--text-2)' }}>{ev.source_name}</span>
                            {ev.published_at && !ev.published_at.startsWith('1970') && (
                              <>
                                <span style={{ fontSize: 10, color: 'var(--text-3)' }}>·</span>
                                <span className="num" style={{ fontSize: 10, color: 'var(--text-3)' }}>
                                  {ev.published_at.slice(0, 10)}
                                </span>
                              </>
                            )}
                          </div>
                        </a>
                      ))
                    }
                  </div>
                </div>
              )}

              {/* No data fallback */}
              {!tradeLoad && !tradeErr && tradeData && tradeData.n_articles === 0 && (
                <p style={{ fontSize: 11, color: 'var(--text-3)' }}>
                  No news found in RAG store for this trade. Load the God's Eye View first to populate the news store.
                </p>
              )}
            </div>
          </>

        ) : (
        /* ── MACRO MODE left panel ────────────────────────────────────────── */
        <>
          {/* Panel header: regime */}
          <div style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexShrink: 0,
          }}>
            <div>
              <p className="label" style={{ marginBottom: 4 }}>Macro Regime</p>
              {pulseData ? (
                <div className="flex items-center gap-2">
                  <span className="num" style={{ fontSize: 15, fontWeight: 700, color: regimeColor, letterSpacing: '-0.02em' }}>
                    {REGIME_LABEL[regime] ?? regime.toUpperCase()}
                  </span>
                  <span className="num" style={{ fontSize: 11, color: 'var(--text-2)' }}>
                    {confPct(pulseData.regime_confidence)}
                  </span>
                </div>
              ) : (
                <div className="skeleton" style={{ height: 18, width: 120 }} />
              )}
              {pulseData?.regime_reasoning && (
                <p style={{ fontSize: 10, color: 'var(--text-2)', marginTop: 4, lineHeight: 1.5 }}>
                  {pulseData.regime_reasoning}
                </p>
              )}
            </div>
          </div>

          {/* Predictions list */}
          <div style={{ flex: 1, overflowY: 'auto', borderBottom: '1px solid var(--border)' }}>
            <div style={{ padding: '10px 16px 6px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
              <p className="label">
                Predictions
                {pulseData && <span style={{ color: 'var(--text-3)', marginLeft: 6 }}>— {pulseData.predictions.length} generated</span>}
              </p>
            </div>

            {pulseLoad ? (
              <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
                {[1, 2, 3, 4, 5].map(i => (
                  <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div className="skeleton" style={{ height: 2, width: '100%' }} />
                    <div className="skeleton" style={{ height: 12, width: '90%' }} />
                    <div className="skeleton" style={{ height: 10, width: '60%' }} />
                  </div>
                ))}
              </div>
            ) : pulseErr ? (
              <div style={{ padding: 16 }}>
                <p style={{ fontSize: 11, color: 'var(--red)' }}>{pulseErr}</p>
                <button className="btn-ghost" style={{ marginTop: 10, fontSize: 11 }} onClick={loadPulse}>
                  Retry
                </button>
              </div>
            ) : pulseData?.predictions.length === 0 ? (
              <div style={{ padding: 16 }}>
                <p style={{ fontSize: 11, color: 'var(--text-2)' }}>
                  No predictions available. Check API keys.
                </p>
              </div>
            ) : (
              pulseData?.predictions.map((pred, i) => (
                <PredCard
                  key={i}
                  pred={pred}
                  index={i}
                  active={selectedId === `pred_${i}`}
                  onClick={() => handlePredClick(pred, i)}
                />
              ))
            )}
          </div>

          {/* Key risks */}
          {pulseData?.key_risks && pulseData.key_risks.length > 0 && (
            <div style={{ flexShrink: 0, padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <p className="label" style={{ marginBottom: 10 }}>Key Risks</p>
              <ul style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {pulseData.key_risks.map((risk, i) => (
                  <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                    <span style={{ color: 'var(--red)', fontSize: 9, marginTop: 3, flexShrink: 0, fontWeight: 700 }}>▲</span>
                    <span style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.45 }}>{risk}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* India impact */}
          {pulseData?.india_impact && (
            <div style={{ flexShrink: 0, padding: '12px 16px' }}>
              <p className="label" style={{ marginBottom: 8 }}>India Outlook</p>
              <p style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>{pulseData.india_impact}</p>
            </div>
          )}
        </>
        )}
      </div>

      {/* ══════════════════════════════════════════════════════════════════════
          GLOBE AREA
      ══════════════════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, minWidth: 0, position: 'relative', overflow: 'hidden' }}>

        {/* ── Globe canvas (absolute fill) ───────────────────────────────── */}
        <div style={{ position: 'absolute', inset: 0 }}>
          {(isTradeMode ? tradeLoad : pulseLoad) ? (
            <div style={{
              width: '100%', height: '100%', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 14,
              background: 'var(--bg)',
            }}>
              <div className="w-7 h-7 border-2 rounded-full animate-spin"
                   style={{ borderColor: 'var(--surface-3)', borderTopColor: 'var(--accent)' }} />
              <p className="label">
                {isTradeMode
                  ? 'Querying RAG store · generating Groq reasoning…'
                  : 'Analyzing 30-day headlines via Groq LLaMA 3.3…'
                }
              </p>
              {!isTradeMode && (
                <p style={{ fontSize: 10, color: 'var(--text-3)' }}>
                  Polygon.io → RAG store → LLM → Predictions · ~10 seconds
                </p>
              )}
            </div>
          ) : isTradeMode && tradeData ? (
            <TradeGlobe
              events={tradeData.events}
              arcs={tradeData.arcs}
              focusLat={tradeData.focus_lat}
              focusLng={tradeData.focus_lng}
              focusAltitude={1.8}
              onEventClick={handleGlobeClick}
              selectedEventId={selectedId}
            />
          ) : !isTradeMode && pulseData ? (
            <TradeGlobe
              events={pulseData.events}
              arcs={pulseData.arcs}
              focusLat={20}
              focusLng={10}
              focusAltitude={2.2}
              onEventClick={handleGlobeClick}
              selectedEventId={selectedId}
            />
          ) : (
            <div style={{
              width: '100%', height: '100%', display: 'flex',
              flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
              background: 'var(--bg)',
            }}>
              <p style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-3)', letterSpacing: '-0.02em' }}>
                {isTradeMode ? 'TRADE INTELLIGENCE' : "GOD'S EYE VIEW"}
              </p>
              <p className="label">
                {(isTradeMode ? tradeErr : pulseErr) ?? 'No data available'}
              </p>
              {!isTradeMode && (
                <button className="btn-ghost" onClick={loadPulse}>Initialize</button>
              )}
            </div>
          )}
        </div>

        {/* ── TOP overlay ────────────────────────────────────────────────── */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
          background: 'linear-gradient(to bottom, rgba(0,5,8,0.88) 0%, rgba(0,5,8,0.4) 70%, transparent 100%)',
          padding: '12px 20px 28px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>

            {isTradeMode ? (
              /* Trade mode top bar */
              <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
                <div>
                  <p className="label" style={{ marginBottom: 2 }}>Trade Intelligence</p>
                  <p style={{ fontSize: 11, color: 'var(--text-2)' }}>
                    RAG retrieval · Groq LLaMA 3.3 reasoning
                  </p>
                </div>
                {tradeData && (
                  <>
                    <div style={{ width: 1, height: 28, background: 'var(--border)' }} />
                    <div>
                      <p className="label" style={{ marginBottom: 2 }}>RAG Store</p>
                      <p className="num" style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)' }}>
                        {tradeData.rag_count ?? 0}
                        <span style={{ fontSize: 10, color: 'var(--text-2)', fontWeight: 400, marginLeft: 4 }}>docs</span>
                      </p>
                    </div>
                    <div style={{ width: 1, height: 28, background: 'var(--border)' }} />
                    <div>
                      <p className="label" style={{ marginBottom: 2 }}>News Sources</p>
                      <p className="num" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>
                        {tradeData.n_articles}
                      </p>
                    </div>
                  </>
                )}
              </div>
            ) : (
              /* Macro mode top bar */
              <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
                <div>
                  <p className="label" style={{ marginBottom: 2 }}>Intelligence Mode</p>
                  <p style={{ fontSize: 11, color: 'var(--text-2)' }}>
                    30-day headline synthesis → 2–6 week predictions
                  </p>
                </div>
                {pulseData && (
                  <>
                    <div style={{ width: 1, height: 28, background: 'var(--border)' }} />
                    <div>
                      <p className="label" style={{ marginBottom: 2 }}>Articles Analyzed</p>
                      <p className="num" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-1)' }}>
                        {pulseData.n_articles}
                      </p>
                    </div>
                    <div style={{ width: 1, height: 28, background: 'var(--border)' }} />
                    <div>
                      <p className="label" style={{ marginBottom: 2 }}>Generated</p>
                      <p style={{ fontSize: 11, color: 'var(--text-2)' }}>{pulseData.generated_at}</p>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Refresh (only in macro mode) */}
            {!isTradeMode && (
              <button
                className="btn-ghost flex items-center gap-2"
                onClick={loadPulse}
                disabled={pulseLoad}
                style={{ fontSize: 11, background: 'rgba(3,8,14,0.8)' }}
              >
                <RefreshCw size={11} className={pulseLoad ? 'animate-spin' : ''} />
                {pulseLoad ? 'Analyzing…' : 'Refresh'}
              </button>
            )}
          </div>
        </div>

        {/* ── BOTTOM sector overlay (macro mode only) ─────────────────────── */}
        {!isTradeMode && pulseData && Object.keys(pulseData.sector_outlook).length > 0 && (
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 10,
            background: 'linear-gradient(to top, rgba(0,5,8,0.95) 0%, rgba(0,5,8,0.5) 75%, transparent 100%)',
            padding: '28px 20px 14px',
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 0 }}>
              <div style={{ display: 'flex', gap: 1, flex: 1 }}>
                {Object.entries(pulseData.sector_outlook).map(([sector, outlook]) => {
                  const col = DIRECTION_COLOR[outlook.direction] ?? 'var(--text-2)'
                  return (
                    <div
                      key={sector}
                      title={outlook.reasoning}
                      style={{
                        flex: 1, padding: '8px 10px',
                        background: 'rgba(3,8,14,0.85)',
                        borderTop: `2px solid ${col}`,
                      }}
                    >
                      <p style={{ fontSize: 10, color: 'var(--text-2)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 2 }}>
                        {sector}
                      </p>
                      <p style={{ fontSize: 10, fontWeight: 700, color: col, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                        {outlook.direction}
                      </p>
                    </div>
                  )
                })}
              </div>
              {pulseData.india_impact && (
                <>
                  <div style={{ width: 1, background: 'var(--border)', alignSelf: 'stretch', margin: '0 16px' }} />
                  <div style={{ width: 260, flexShrink: 0 }}>
                    <p className="label" style={{ marginBottom: 4, color: 'var(--accent)' }}>India Outlook</p>
                    <p style={{ fontSize: 10, color: 'var(--text-2)', lineHeight: 1.55 }}>
                      {pulseData.india_impact.length > 200
                        ? pulseData.india_impact.slice(0, 200) + '…'
                        : pulseData.india_impact}
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── Detail overlay (top-right, on globe pin click) ─────────────── */}
        {detailEv && detailEv.id !== 'macro_hub' && !detailEv.id.startsWith('hub_') && (
          <div
            className="overlay-panel animate-slide-right"
            style={{ position: 'absolute', top: 64, right: 16, zIndex: 20, width: 296, padding: '16px 18px' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <span style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase',
                  color: CATEGORY_COLORS[detailEv.category as GlobeCategory] ?? 'var(--text-2)',
                }}>
                  {CATEGORY_LABELS[detailEv.category as GlobeCategory] ?? detailEv.category}
                </span>
                {detailEv.direction && (
                  <>
                    <span style={{ color: 'var(--text-3)', fontSize: 10 }}>·</span>
                    <span style={{ fontSize: 10, fontWeight: 700, color: DIRECTION_COLOR[detailEv.direction] ?? 'var(--text-2)' }}>
                      {DIRECTION_LABEL[detailEv.direction] ?? detailEv.direction}
                    </span>
                  </>
                )}
                {detailEv.timeframe && (
                  <>
                    <span style={{ color: 'var(--text-3)', fontSize: 10 }}>·</span>
                    <span style={{ fontSize: 10, color: 'var(--text-2)' }}>{detailEv.timeframe}</span>
                  </>
                )}
              </div>
              <button
                onClick={() => { setDetailEv(null); setSelectedId(null) }}
                style={{ color: 'var(--text-3)', fontSize: 14, lineHeight: 1, flexShrink: 0, marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer' }}
              >
                <X size={13} />
              </button>
            </div>

            <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)', lineHeight: 1.4, marginBottom: 8 }}>
              {detailEv.title}
            </p>

            {(detailEv.detail || detailEv.summary) && (
              <p style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6, marginBottom: 10 }}>
                {detailEv.detail || detailEv.summary}
              </p>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <p className="label">Confidence</p>
              <div className="conf-bar" style={{ flex: 1 }}>
                <div className="conf-bar-fill"
                     style={{ width: confPct(detailEv.severity), background: DIRECTION_COLOR[detailEv.direction ?? 'neutral'] ?? 'var(--accent)' }} />
              </div>
              <span className="num" style={{ fontSize: 11, color: 'var(--text-2)' }}>
                {confPct(detailEv.severity)}
              </span>
            </div>

            {detailEv.affected_markets && detailEv.affected_markets.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <p className="label" style={{ marginBottom: 6 }}>Affected Markets</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {detailEv.affected_markets.map((m, i) => (
                    <span key={i} style={{
                      fontSize: 10, padding: '2px 7px',
                      border: '1px solid var(--border)',
                      color: 'var(--text-2)', borderRadius: 1,
                    }}>{m}</span>
                  ))}
                </div>
              </div>
            )}

            {detailEv.evidence && (
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                <p className="label" style={{ marginBottom: 4 }}>Evidence</p>
                <p style={{ fontSize: 10, color: 'var(--text-3)', lineHeight: 1.5, fontStyle: 'italic' }}>
                  "{detailEv.evidence}"
                </p>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
