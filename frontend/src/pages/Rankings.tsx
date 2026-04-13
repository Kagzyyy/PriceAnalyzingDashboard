import { useEffect, useState, useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { TrendingUp, TrendingDown, Download, ChevronUp, ChevronDown } from 'lucide-react'
import { fetchLongs, fetchShorts, fetchSectors, type Prediction, type SectorSummary } from '../api/client'

type SortKey = 'rank' | 'confidence_pct' | 'price' | 'change_pct' | 'volume' | 'rsi'
type SortDir = 'asc' | 'desc'

export default function Rankings() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const initDir = searchParams.get('direction')?.toUpperCase() === 'DOWN' ? 'short' : 'long'
  const initSector = searchParams.get('sector') || ''

  const [tab, setTab] = useState<'long' | 'short'>(initDir as 'long' | 'short')
  const [longs, setLongs] = useState<Prediction[]>([])
  const [shorts, setShorts] = useState<Prediction[]>([])
  const [sectors, setSectors] = useState<SectorSummary[]>([])
  const [sectorFilter, setSectorFilter] = useState(initSector)
  const [minConf, setMinConf] = useState(50)
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([fetchLongs(), fetchShorts(), fetchSectors()])
      .then(([l, s, sec]) => {
        setLongs(l.data || [])
        setShorts(s.data || [])
        setSectors(sec.data || [])
      })
      .finally(() => setLoading(false))
  }, [])

  const activeData = tab === 'long' ? longs : shorts

  const filtered = useMemo(() => {
    let d = activeData.filter(s => s.confidence_pct >= minConf)
    if (sectorFilter) d = d.filter(s => s.sector.toLowerCase() === sectorFilter.toLowerCase())
    return [...d].sort((a, b) => {
      const av = (a[sortKey] as number) ?? 0
      const bv = (b[sortKey] as number) ?? 0
      return sortDir === 'asc' ? av - bv : bv - av
    })
  }, [activeData, sectorFilter, minConf, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('asc') }
  }

  const exportCSV = () => {
    const header = 'Rank,Symbol,Name,Sector,Confidence%,Direction,Price,Change%,Volume,RSI'
    const rows = filtered.map(s =>
      `${s.rank},${s.symbol},"${s.name}",${s.sector},${s.confidence_pct},${s.direction},${s.price},${s.change_pct},${s.volume},${s.rsi?.toFixed(1) ?? ''}`
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `nse_${tab}s_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
  }

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? (sortDir === 'asc' ? <ChevronUp size={11} /> : <ChevronDown size={11} />) : null

  const ColHead = ({ k, label, right }: { k: SortKey; label: string; right?: boolean }) => (
    <th
      className={`pb-3 px-3 text-xs text-gray-600 uppercase tracking-wider cursor-pointer hover:text-gray-400 select-none transition-colors ${right ? 'text-right' : 'text-left'}`}
      onClick={() => handleSort(k)}
    >
      <span className="inline-flex items-center gap-1">{label}<SortIcon k={k} /></span>
    </th>
  )

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="opacity-0 animate-fade-up flex items-center justify-between" style={{ animationFillMode: 'forwards' }}>
        <h1 className="text-2xl font-bold text-white tracking-tight">Rankings</h1>
        <button onClick={exportCSV} className="btn-ghost flex items-center gap-2 text-xs">
          <Download size={13} /> Export CSV
        </button>
      </div>

      {/* Controls */}
      <div className="opacity-0 animate-fade-up flex flex-wrap items-center gap-4" style={{ animationDelay: '60ms', animationFillMode: 'forwards' }}>
        {/* Tab toggle */}
        <div className="flex rounded-xl overflow-hidden border border-white/[0.08] p-0.5 bg-white/[0.02]">
          <button
            onClick={() => setTab('long')}
            className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
              tab === 'long' ? 'bg-emerald-500/15 text-emerald-400' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <TrendingUp size={13} /> Longs <span className="mono text-xs opacity-60">({longs.length})</span>
          </button>
          <button
            onClick={() => setTab('short')}
            className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
              tab === 'short' ? 'bg-red-500/15 text-red-400' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <TrendingDown size={13} /> Shorts <span className="mono text-xs opacity-60">({shorts.length})</span>
          </button>
        </div>

        {/* Sector filter */}
        <select
          value={sectorFilter}
          onChange={e => setSectorFilter(e.target.value)}
          className="input-base text-xs py-2"
        >
          <option value="">All Sectors</option>
          {sectors.map(s => <option key={s.sector} value={s.sector}>{s.sector}</option>)}
        </select>

        {/* Min confidence */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-600">Min confidence</span>
          <input
            type="range" min={50} max={90} step={5} value={minConf}
            onChange={e => setMinConf(+e.target.value)}
            className="w-20 accent-white"
          />
          <span className="text-sm text-white font-medium mono w-10">{minConf}%</span>
        </div>

        <span className="ml-auto text-xs text-gray-600 mono">{filtered.length} stocks</span>
      </div>

      {/* Table */}
      <div className="opacity-0 animate-fade-up card p-0 overflow-hidden" style={{ animationDelay: '120ms', animationFillMode: 'forwards' }}>
        {loading ? (
          <div className="py-20 text-center text-gray-600 text-sm">Loading rankings…</div>
        ) : filtered.length === 0 ? (
          <div className="py-20 text-center">
            <p className="text-gray-600 text-sm">No {tab} signals match your filters</p>
            <p className="text-gray-700 text-xs mt-1">Try lowering the confidence threshold</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-white/[0.05]">
                <tr>
                  <ColHead k="rank" label="#" />
                  <th className="pb-3 px-3 text-xs text-gray-600 uppercase tracking-wider text-left">Stock</th>
                  <th className="pb-3 px-3 text-xs text-gray-600 uppercase tracking-wider text-left">Sector</th>
                  <ColHead k="confidence_pct" label="Confidence" />
                  <ColHead k="price" label="Price" right />
                  <ColHead k="change_pct" label="Chg%" right />
                  <ColHead k="volume" label="Volume" right />
                  <ColHead k="rsi" label="RSI" right />
                </tr>
              </thead>
              <tbody>
                {filtered.map((s, i) => (
                  <tr
                    key={s.ticker}
                    className="opacity-0 animate-fade-up border-b border-white/[0.03] hover:bg-white/[0.03] cursor-pointer transition-colors duration-150 group"
                    style={{ animationDelay: `${i * 30}ms`, animationFillMode: 'forwards' }}
                    onClick={() => navigate(`/stocks/${s.symbol}`)}
                  >
                    <td className="py-3 px-3 text-xs text-gray-700 mono">{s.rank}</td>
                    <td className="py-3 px-3">
                      <p className="text-sm font-semibold text-white">{s.symbol}</p>
                      <p className="text-xs text-gray-600 truncate max-w-[160px]">{s.name}</p>
                    </td>
                    <td className="py-3 px-3">
                      <span className="text-xs text-gray-600 bg-white/[0.04] px-2 py-0.5 rounded-md">{s.sector}</span>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-white/5 rounded-full h-1">
                          <div
                            className={`h-1 rounded-full progress-bar ${tab === 'long' ? 'bg-emerald-500' : 'bg-red-500'}`}
                            style={{ width: `${Math.min(100, s.confidence_pct)}%` }}
                          />
                        </div>
                        <span className={tab === 'long' ? 'badge-up' : 'badge-down'}>
                          {Number(s.confidence_pct).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-3 text-sm text-right mono text-gray-300">₹{s.price?.toFixed(2)}</td>
                    <td className={`py-3 px-3 text-sm text-right mono ${s.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {s.change_pct >= 0 ? '+' : ''}{s.change_pct?.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-sm text-right mono text-gray-500">
                      {s.volume ? (s.volume >= 1e6 ? `${(s.volume / 1e6).toFixed(1)}M` : `${(s.volume / 1e3).toFixed(0)}K`) : '—'}
                    </td>
                    <td className={`py-3 px-3 text-sm text-right mono font-medium ${
                      s.rsi && s.rsi > 70 ? 'text-red-400' : s.rsi && s.rsi < 30 ? 'text-emerald-400' : 'text-gray-500'
                    }`}>
                      {s.rsi?.toFixed(0) ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
