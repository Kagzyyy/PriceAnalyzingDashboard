import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { fetchStatus, fetchSectors, fetchLongs, fetchShorts, type MarketStatus, type SectorSummary, type Prediction } from '../api/client'

function StatCard({ label, value, sub, delay = 0, accent }: {
  label: string; value: string | number; sub?: string; delay?: number; accent?: string
}) {
  return (
    <div
      className="card-hover opacity-0 animate-fade-up"
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-3">{label}</p>
      <p className={`text-3xl font-bold mono ${accent ?? 'text-white'}`}>{value ?? '—'}</p>
      {sub && <p className="text-xs text-gray-600 mt-2">{sub}</p>}
    </div>
  )
}

function SectorCard({ s, index }: { s: SectorSummary; index: number }) {
  const navigate = useNavigate()
  const pct = s.bullish_pct
  const isBull = pct >= 55
  const isBear = pct <= 45

  return (
    <button
      onClick={() => navigate(`/rankings?sector=${s.sector}`)}
      className="opacity-0 animate-fade-up group p-4 rounded-xl border border-white/[0.06] bg-white/[0.02]
                 hover:bg-white/[0.05] hover:border-white/10 transition-all duration-300 text-left"
      style={{ animationDelay: `${index * 40}ms`, animationFillMode: 'forwards' }}
    >
      <p className="text-xs text-gray-500 font-medium truncate mb-2">{s.sector}</p>
      <p className={`text-2xl font-bold mono ${isBull ? 'text-emerald-400' : isBear ? 'text-red-400' : 'text-gray-300'}`}>
        {pct.toFixed(0)}%
      </p>
      <div className="mt-3 h-0.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full animate-bar-grow ${isBull ? 'bg-emerald-500' : isBear ? 'bg-red-500' : 'bg-gray-600'}`}
          style={{ width: `${pct}%`, animationDelay: `${index * 40 + 200}ms` }}
        />
      </div>
      <p className="text-[10px] text-gray-700 mt-2">{s.longs}↑ {s.shorts}↓ of {s.count}</p>
    </button>
  )
}

function StockRow({ stock, type, index }: { stock: Prediction; type: 'long' | 'short'; index: number }) {
  const navigate = useNavigate()
  const isBull = type === 'long'
  return (
    <tr
      className="opacity-0 animate-fade-up border-b border-white/[0.04] hover:bg-white/[0.03] cursor-pointer transition-colors duration-150 group"
      style={{ animationDelay: `${index * 40}ms`, animationFillMode: 'forwards' }}
      onClick={() => navigate(`/stocks/${stock.symbol}`)}
    >
      <td className="py-3 px-3 text-xs text-gray-700 mono">{stock.rank}</td>
      <td className="py-3 px-3">
        <p className="text-sm font-semibold text-white group-hover:text-gray-100">{stock.symbol}</p>
        <p className="text-xs text-gray-600 truncate max-w-[120px]">{stock.name}</p>
      </td>
      <td className="py-3 px-3">
        <span className="text-xs text-gray-600 bg-white/[0.04] px-2 py-0.5 rounded-md">{stock.sector}</span>
      </td>
      <td className="py-3 px-3">
        <span className={isBull ? 'badge-up' : 'badge-down'}>
          {isBull ? '▲' : '▼'} {Number(stock.confidence_pct).toFixed(1)}%
        </span>
      </td>
      <td className="py-3 px-3 text-sm text-right mono text-gray-300">₹{stock.price?.toFixed(2)}</td>
      <td className={`py-3 px-3 text-sm text-right mono ${stock.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {stock.change_pct >= 0 ? '+' : ''}{stock.change_pct?.toFixed(2)}%
      </td>
    </tr>
  )
}

export default function Dashboard() {
  const [status, setStatus] = useState<MarketStatus | null>(null)
  const [sectors, setSectors] = useState<SectorSummary[]>([])
  const [longs, setLongs] = useState<Prediction[]>([])
  const [shorts, setShorts] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([fetchStatus(), fetchSectors(), fetchLongs(), fetchShorts()])
      .then(([s, sec, l, sh]) => {
        setStatus(s)
        setSectors(sec.data || [])
        setLongs((l.data || []).slice(0, 8))
        setShorts((sh.data || []).slice(0, 8))
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card">
              <div className="skeleton h-3 w-16 mb-4" />
              <div className="skeleton h-8 w-24" />
            </div>
          ))}
        </div>
        <div className="card">
          <div className="skeleton h-3 w-24 mb-5" />
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
            {[...Array(10)].map((_, i) => <div key={i} className="skeleton h-20 rounded-xl" />)}
          </div>
        </div>
      </div>
    )
  }

  const noData = !status?.last_refresh

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="opacity-0 animate-fade-up flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Market Overview</h1>
          <p className="text-sm text-gray-600 mt-1">
            {status?.last_refresh ? `Updated ${status.last_refresh}` : 'Click Refresh to load predictions'}
          </p>
        </div>
        {status?.market_sentiment && (
          <span className={`text-xs font-bold tracking-widest uppercase px-3 py-1.5 rounded-lg border mono ${
            status.market_sentiment === 'BULLISH'
              ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5'
              : status.market_sentiment === 'BEARISH'
              ? 'text-red-400 border-red-500/20 bg-red-500/5'
              : 'text-gray-500 border-white/10 bg-white/5'
          }`}>
            {status.market_sentiment}
          </span>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Stocks Analyzed" value={status?.total_predictions ?? '—'} delay={0} />
        <StatCard label="Long Signals" value={status?.longs ?? '—'} sub="Top 20% by rank" delay={60} accent="text-emerald-400" />
        <StatCard label="Short Signals" value={status?.shorts ?? '—'} sub="Bottom 20% by rank" delay={120} accent="text-red-400" />
        <StatCard
          label="Model Accuracy"
          value={status?.model_accuracy ? `${(status.model_accuracy * 100).toFixed(1)}%` : '—'}
          sub={`${status?.n_samples?.toLocaleString() ?? '—'} samples`}
          delay={180}
          accent="text-indigo-400"
        />
      </div>

      {/* No data */}
      {noData && (
        <div className="opacity-0 animate-fade-up card text-center py-12 border-dashed border-white/10" style={{ animationDelay: '240ms', animationFillMode: 'forwards' }}>
          <p className="text-gray-500 text-sm">No prediction data — click <span className="text-white font-medium">Refresh</span> to fetch NSE data</p>
          <p className="text-gray-700 text-xs mt-1">First run takes 10–30 minutes</p>
        </div>
      )}

      {/* Sector Heatmap */}
      {sectors.length > 0 && (
        <div className="opacity-0 animate-fade-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
          <p className="text-xs text-gray-600 uppercase tracking-widest font-medium mb-4">Sector Heatmap</p>
          <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-2">
            {sectors.map((s, i) => <SectorCard key={s.sector} s={s} index={i} />)}
          </div>
        </div>
      )}

      {/* Longs + Shorts */}
      {(longs.length > 0 || shorts.length > 0) && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {[{ data: longs, type: 'long' as const, label: 'Top Longs', color: 'text-emerald-400' },
            { data: shorts, type: 'short' as const, label: 'Top Shorts', color: 'text-red-400' }].map(({ data, type, label, color }) => (
            <div key={type} className="opacity-0 animate-fade-up card p-0 overflow-hidden" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.05]">
                <div className="flex items-center gap-2">
                  {type === 'long' ? <TrendingUp size={14} className="text-emerald-400" /> : <TrendingDown size={14} className="text-red-400" />}
                  <span className={`text-sm font-semibold ${color}`}>{label}</span>
                </div>
                <a href={`/rankings?direction=${type === 'long' ? 'UP' : 'DOWN'}`} className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
                  View all →
                </a>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/[0.04]">
                    <th className="py-2.5 px-3 text-left text-xs text-gray-700">#</th>
                    <th className="py-2.5 px-3 text-left text-xs text-gray-700">Stock</th>
                    <th className="py-2.5 px-3 text-left text-xs text-gray-700">Sector</th>
                    <th className="py-2.5 px-3 text-left text-xs text-gray-700">Conf.</th>
                    <th className="py-2.5 px-3 text-right text-xs text-gray-700">Price</th>
                    <th className="py-2.5 px-3 text-right text-xs text-gray-700">Chg%</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((s, i) => <StockRow key={s.ticker} stock={s} type={type} index={i} />)}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
