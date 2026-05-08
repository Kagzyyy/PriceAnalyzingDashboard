import { BrowserRouter, Routes, Route, NavLink, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { TrendingUp, RefreshCw, Search } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Rankings from './pages/Rankings'
import StockDetail from './pages/StockDetail'
import Chat from './pages/Chat'
import Backtest from './pages/Backtest'
import PairsBacktest from './pages/PairsBacktest'
import { triggerRefresh } from './api/client'

function Navbar() {
  const [refreshing, setRefreshing] = useState(false)
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  const handleRefresh = async () => {
    setRefreshing(true)
    try { await triggerRefresh() }
    finally { setTimeout(() => setRefreshing(false), 2000) }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) navigate(`/stocks/${query.trim().toUpperCase()}`)
  }

  const navCls = ({ isActive }: { isActive: boolean }) =>
    `relative py-4 px-1 text-sm font-medium transition-colors duration-200 ${
      isActive ? 'text-white nav-active' : 'text-gray-500 hover:text-gray-300'
    }`

  return (
    <nav className="sticky top-0 z-50 bg-black/90 backdrop-blur-xl border-b border-white/[0.06]">
      <div className="max-w-screen-xl mx-auto px-6 flex items-center gap-8 h-14">
        {/* Logo */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="w-7 h-7 bg-white rounded-lg flex items-center justify-center">
            <TrendingUp className="text-black" size={14} strokeWidth={2.5} />
          </div>
          <span className="font-semibold text-white tracking-tight">NSE Predict</span>
        </div>

        {/* Nav */}
        <div className="flex items-center gap-6">
          <NavLink to="/" end className={navCls}>Dashboard</NavLink>
          <NavLink to="/rankings" className={navCls}>Rankings</NavLink>
          <NavLink to="/backtest" className={navCls}>Backtest</NavLink>
          <NavLink to="/pairs" className={navCls}>Pairs</NavLink>
          <NavLink to="/chat" className={navCls}>Research AI</NavLink>
        </div>

        {/* Search */}
        <form onSubmit={handleSearch} className="flex-1 max-w-[220px] ml-auto">
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search symbol…"
              className="input-base w-full pl-8 py-2 text-xs"
            />
          </div>
        </form>

        {/* Refresh */}
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 text-sm text-gray-500 hover:text-white transition-colors duration-200 disabled:opacity-40"
        >
          <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main className="max-w-screen-xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/rankings" element={<Rankings />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/pairs" element={<PairsBacktest />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
