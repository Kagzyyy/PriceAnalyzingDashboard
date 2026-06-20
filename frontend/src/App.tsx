import { BrowserRouter, NavLink, useLocation } from 'react-router-dom'
import { Globe2, BarChart2 } from 'lucide-react'
import GodEyeView from './pages/GodEyeView'
import GapAndGo from './pages/GapAndGo'

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar() {
  const loc = useLocation()

  return (
    <aside style={{
      width: 196,
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      borderRight: '1px solid var(--border)',
      background: 'var(--surface)',
      height: '100vh',
      position: 'sticky',
      top: 0,
    }}>
      {/* ── Branding ── */}
      <div style={{
        padding: '20px 18px 17px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <span style={{
          fontSize: 12,
          fontWeight: 600,
          color: 'var(--text-1)',
          letterSpacing: '-0.01em',
          display: 'block',
          marginBottom: 3,
        }}>
          Alpha Edge
        </span>
        <p style={{
          fontSize: 10,
          color: 'var(--text-3)',
          letterSpacing: '0.04em',
          margin: 0,
        }}>
          NSE · Nifty 100
        </p>
      </div>

      {/* ── Navigation ── */}
      <nav style={{ flexShrink: 0, paddingTop: 4, paddingBottom: 4 }}>
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <Globe2 size={13} />
          God's Eye View
        </NavLink>
        <NavLink
          to="/gap"
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <BarChart2 size={13} />
          Gap &amp; Go
        </NavLink>
      </nav>

      <div style={{ height: 1, background: 'var(--border)', margin: '0 0' }} />

      {/* ── Context description ── */}
      {/* ── Context description ── */}
      <div style={{ padding: '16px 18px', flex: 1 }}>
        {loc.pathname === '/' || loc.pathname === '' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <p className="label" style={{ marginBottom: 7 }}>About</p>
              <p style={{ fontSize: 10, color: 'var(--text-2)', lineHeight: 1.65, margin: 0 }}>
                30-day macro synthesis via Groq LLaMA 3.3. Forward predictions with confidence scores and geographic mapping.
              </p>
            </div>
            <div>
              <p className="label" style={{ marginBottom: 7 }}>Prediction signals</p>
              {[
                { color: 'var(--green)', label: 'Bullish' },
                { color: 'var(--red)',   label: 'Bearish' },
                { color: 'var(--accent)',label: 'Neutral / structural' },
                { color: 'var(--amber)', label: 'Tail risk' },
              ].map(({ color, label }) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <div style={{ width: 5, height: 5, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 10, color: 'var(--text-3)' }}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <p className="label" style={{ marginBottom: 7 }}>Strategy</p>
              {[
                ['Gap',    '≥ 1.5% open'],
                ['Target', '+2.0%'],
                ['Stop',   '−0.5%'],
                ['R:R',    '4 : 1'],
                ['Cost',   '0.20% r/t'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-3)', letterSpacing: '0.03em' }}>{k}</span>
                  <span style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'JetBrains Mono, monospace' }}>{v}</span>
                </div>
              ))}
            </div>
            <div>
              <p className="label" style={{ marginBottom: 7 }}>Trade Intel</p>
              <p style={{ fontSize: 10, color: 'var(--text-3)', lineHeight: 1.6, margin: 0 }}>
                Click the globe icon on any trade row for AI reasoning and related news via RAG + Groq.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── Status footer ── */}
      <div style={{
        padding: '10px 18px',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <span className="status-dot live" style={{ fontSize: 9 }}>Live</span>
        <span style={{
          fontSize: 9,
          color: 'var(--text-3)',
          fontFamily: 'JetBrains Mono, monospace',
          letterSpacing: '0.03em',
        }}>
          v2.1
        </span>
      </div>
    </aside>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────
function AppInner() {
  const loc      = useLocation()
  const isGlobe  = loc.pathname === '/' || loc.pathname === ''

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)' }}>
      <Sidebar />
      <main
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          overflow: 'hidden',
          position: 'relative',   // anchor for absolute children
        }}
      >
        {/*
          Both pages stay MOUNTED at all times.
          We use position:absolute + visibility instead of display:none so
          each pane always fills the full area and keeps its layout dimensions.
          This is critical for the ResizeObserver inside TradeGlobe — when a
          parent has display:none its children report 0×0 and the WebGL canvas
          never recovers. With visibility:hidden the dimensions stay intact and
          the globe renders correctly the moment the tab becomes visible again.
        */}

        {/* God's Eye View — always laid out, hidden via visibility */}
        <div style={{
          position:      'absolute',
          inset:         0,
          display:       'flex',
          overflow:      'hidden',
          visibility:    isGlobe ? 'visible' : 'hidden',
          pointerEvents: isGlobe ? 'auto'    : 'none',
          zIndex:        isGlobe ? 1         : 0,
        }}>
          <GodEyeView />
        </div>

        {/* Gap & Go — always laid out, hidden via visibility */}
        <div style={{
          position:      'absolute',
          inset:         0,
          display:       'flex',
          flexDirection: 'column',
          overflowY:     'auto',
          overflowX:     'hidden',
          visibility:    isGlobe ? 'hidden' : 'visible',
          pointerEvents: isGlobe ? 'none'   : 'auto',
          zIndex:        isGlobe ? 0        : 1,
        }}>
          <GapAndGo />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  )
}
