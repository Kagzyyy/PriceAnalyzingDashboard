import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts'
import { ArrowLeft, TrendingUp, TrendingDown, Activity } from 'lucide-react'
import { fetchStockDetail, type StockDetail as StockDetailType } from '../api/client'

function GaugeBar({ label, value, min = 0, max = 100, low, high, unit = '' }: {
  label: string; value?: number; min?: number; max?: number; low?: number; high?: number; unit?: string
}) {
  if (value == null) return null
  const pct = ((value - min) / (max - min)) * 100
  const isLow = low !== undefined && value <= low
  const isHigh = high !== undefined && value >= high
  const color = isLow ? 'bg-emerald-500' : isHigh ? 'bg-red-500' : 'bg-indigo-500'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className={`font-medium ${isLow ? 'text-emerald-400' : isHigh ? 'text-red-400' : 'text-white'}`}>
          {value.toFixed(2)}{unit}
        </span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-gray-400 mb-1">{label}</p>
      <p className="text-white">O: ₹{d?.open?.toFixed(2)}</p>
      <p className="text-emerald-400">H: ₹{d?.high?.toFixed(2)}</p>
      <p className="text-red-400">L: ₹{d?.low?.toFixed(2)}</p>
      <p className="text-white font-bold">C: ₹{d?.close?.toFixed(2)}</p>
      <p className="text-gray-400 mt-1">Vol: {d?.volume?.toLocaleString()}</p>
    </div>
  )
}

export default function StockDetail() {
  const { symbol } = useParams<{ symbol: string }>()
  const navigate = useNavigate()
  const [data, setData] = useState<StockDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [range, setRange] = useState<string>('1M')
  const [error, setError] = useState('')

  const rangeToDays = (r: string): number => {
    const now = new Date()
    switch (r) {
      case '1M':  return 30
      case '3M':  return 90
      case '6M':  return 180
      case 'YTD': return Math.max(5, Math.ceil((now.getTime() - new Date(now.getFullYear(), 0, 1).getTime()) / 86400000))
      case 'Max': return 730
      default:    return 90
    }
  }
  const days = rangeToDays(range)

  useEffect(() => {
    if (!symbol) return
    setLoading(true)
    setError('')
    fetchStockDetail(symbol, rangeToDays(range))
      .then(setData)
      .catch(e => setError(e.response?.data?.detail || 'Stock not found'))
      .finally(() => setLoading(false))
  }, [symbol, range])

  if (loading) return <div className="py-24 text-center text-gray-500"><Activity size={28} className="mx-auto mb-3 animate-pulse" /><p>Loading {symbol}…</p></div>
  if (error) return (
    <div className="py-24 text-center">
      <p className="text-red-400 text-lg">{error}</p>
      <button onClick={() => navigate(-1)} className="btn-ghost mt-4">Go Back</button>
    </div>
  )
  if (!data) return null

  const isBull = data.direction === 'UP'
  const chartData = data.ohlcv.map(d => ({
    ...d,
    date: d.date.slice(5), // MM-DD
    change: d.close >= d.open ? d.close - d.open : 0,
    drop: d.close < d.open ? d.open - d.close : 0,
  }))

  return (
    <div className="space-y-6">
      {/* Back + Header */}
      <div>
        <button onClick={() => navigate(-1)} className="flex items-center gap-1.5 text-gray-500 hover:text-white text-sm mb-3 transition-colors">
          <ArrowLeft size={14} /> Back
        </button>
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white">{data.symbol}
              <span className="text-gray-500 text-lg font-normal ml-3">{data.name}</span>
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded">{data.sector}</span>
              <span className="text-xs text-gray-600">{data.industry}</span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-white">₹{data.price?.toFixed(2)}</p>
            <p className={`text-sm mt-0.5 ${data.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.change_pct >= 0 ? '+' : ''}{data.change_pct?.toFixed(2)}% today
            </p>
          </div>
        </div>
      </div>

      {/* Prediction card */}
      <div className={`card border ${isBull ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div className={`p-4 rounded-xl ${isBull ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
              {isBull ? <TrendingUp size={28} className="text-emerald-400" /> : <TrendingDown size={28} className="text-red-400" />}
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider">Prediction for tomorrow</p>
              <p className={`text-4xl font-black ${isBull ? 'text-emerald-400' : 'text-red-400'}`}>
                {isBull ? '▲ LONG' : '▼ SHORT'}
              </p>
              <p className="text-gray-400 text-sm mt-1">{data.confidence_pct}% confidence</p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Prob UP</p>
              <p className="text-emerald-400 font-bold text-lg">{(data.prob_up * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Prob DOWN</p>
              <p className="text-red-400 font-bold text-lg">{(data.prob_down * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart + Indicators */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="card xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Price Chart</h2>
            <div className="flex gap-0.5 bg-white/[0.03] rounded-lg p-0.5 border border-white/[0.06]">
              {['1M', '3M', '6M', 'YTD', 'Max'].map(r => (
                <button
                  key={r}
                  onClick={() => setRange(r)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all duration-150 ${
                    range === r
                      ? 'bg-white text-black'
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
          {chartData.length >= 2 ? (
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} interval={Math.max(0, Math.floor(chartData.length / 8))} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `₹${v.toFixed(0)}`} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="change" fill="#10b981" stackId="candle" opacity={0.7} />
                <Bar dataKey="drop" fill="#ef4444" stackId="candle" opacity={0.7} />
                <Line type="monotone" dataKey="close" stroke="#6366f1" strokeWidth={1.5} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
              Not enough data for this range — try 1M or longer
            </div>
          )}
        </div>

        {/* Technical Indicators */}
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Technical Indicators</h2>
          <GaugeBar label="RSI (14)" value={data.rsi} min={0} max={100} low={30} high={70} />
          <GaugeBar label="Bollinger %B" value={data.bb_pct} min={0} max={1} low={0.2} high={0.8} />
          <GaugeBar label="Volume Ratio" value={data.volume_ratio} min={0} max={5} high={2} />
          {data.macd != null && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500">MACD</span>
                <span className={`font-medium ${data.macd >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {data.macd?.toFixed(3)}
                </span>
              </div>
              <div className={`text-xs px-2 py-1 rounded ${data.macd >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                {data.macd >= 0 ? '↑ Bullish crossover' : '↓ Bearish crossover'}
              </div>
            </div>
          )}

          <div className="pt-3 border-t border-gray-800 space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">Prediction</span>
              <span className={isBull ? 'badge-up' : 'badge-down'}>{data.direction} {data.confidence_pct}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sector</span>
              <span className="text-gray-300">{data.sector}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Industry</span>
              <span className="text-gray-300">{data.industry}</span>
            </div>
          </div>

          <p className="text-xs text-gray-600 pt-2 border-t border-gray-800">
            ⚠ Predictions are probabilistic. Always use risk management.
          </p>
        </div>
      </div>
    </div>
  )
}
