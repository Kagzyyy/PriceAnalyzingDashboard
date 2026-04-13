import { useEffect, useState, useRef } from 'react'
import { Send, Sparkles, Loader2 } from 'lucide-react'
import { askAssistant, fetchSuggestions, type AssistantResponse } from '../api/client'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: AssistantResponse['sources']
}

function SourceBadge({ source }: { source: Record<string, unknown> }) {
  if (source.type === 'stock') {
    const isBull = source.direction === 'UP'
    return (
      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-md border mono ${
        isBull ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-400' : 'border-red-500/20 bg-red-500/5 text-red-400'
      }`}>
        {isBull ? '▲' : '▼'} {source.symbol as string} {source.confidence as number}%
      </span>
    )
  }
  if (source.type === 'sector') {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-md border border-white/10 bg-white/[0.04] text-gray-400 mono">
        {source.sector as string} {source.bullish_pct as number}%↑
      </span>
    )
  }
  if (source.type === 'market_overview') {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-md border border-indigo-500/20 bg-indigo-500/5 text-indigo-400">
        Market: {source.sentiment as string}
      </span>
    )
  }
  return null
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-1">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce"
          style={{ animationDelay: `${i * 150}ms`, animationDuration: '1s' }}
        />
      ))}
    </div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hello! I'm your NSE market research assistant. Ask me about today's predictions, sector performance, or trading signals.",
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchSuggestions().then(s => setSuggestions(s.questions)).catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async (q: string) => {
    if (!q.trim() || loading) return
    const question = q.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)
    try {
      const res = await askAssistant(question)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Connection error. Please check that the backend is running.',
      }])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-7.5rem)] animate-fade-in">
      {/* Header */}
      <div className="opacity-0 animate-fade-up flex items-center gap-3 mb-6" style={{ animationFillMode: 'forwards' }}>
        <div className="w-8 h-8 bg-indigo-500/10 border border-indigo-500/20 rounded-xl flex items-center justify-center">
          <Sparkles size={15} className="text-indigo-400" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-white">Research Assistant</h1>
          <p className="text-xs text-gray-600">Groq · Llama 3.3 70B · NSE market data</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-5 pb-4 pr-1">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex gap-3 opacity-0 animate-fade-up ${m.role === 'user' ? 'flex-row-reverse animate-slide-right' : 'animate-slide-left'}`}
            style={{ animationDelay: '0ms', animationFillMode: 'forwards' }}
          >
            {/* Avatar */}
            <div className={`flex-shrink-0 w-7 h-7 rounded-xl flex items-center justify-center text-xs font-bold ${
              m.role === 'assistant' ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' : 'bg-white/10 text-white'
            }`}>
              {m.role === 'assistant' ? 'AI' : 'You'}
            </div>

            <div className={`max-w-[78%] space-y-2 ${m.role === 'user' ? 'items-end flex flex-col' : ''}`}>
              <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                m.role === 'assistant'
                  ? 'bg-white/[0.03] border border-white/[0.07] text-gray-200'
                  : 'bg-white text-black font-medium'
              }`}>
                {m.content.split('\n').map((line, li) => (
                  <p key={li} className={li > 0 ? 'mt-1.5' : ''}>{line}</p>
                ))}
              </div>
              {m.sources && m.sources.length > 0 && (
                <div className="flex flex-wrap gap-1.5 px-1">
                  {m.sources.map((s, si) => <SourceBadge key={si} source={s} />)}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Thinking */}
        {loading && (
          <div className="flex gap-3 animate-fade-in">
            <div className="w-7 h-7 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-xs font-bold text-indigo-400">
              AI
            </div>
            <div className="bg-white/[0.03] border border-white/[0.07] rounded-2xl px-4 py-3">
              <ThinkingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && suggestions.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2 animate-fade-in">
          {suggestions.slice(0, 6).map(q => (
            <button
              key={q}
              onClick={() => send(q)}
              className="text-xs text-gray-500 hover:text-gray-200 bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.07] px-3 py-1.5 rounded-full transition-all duration-200"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="opacity-0 animate-fade-up flex gap-3 pt-2" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
          placeholder="Ask about sectors, stocks, signals…"
          disabled={loading}
          className="input-base flex-1 disabled:opacity-40"
        />
        <button
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          className="w-11 h-11 rounded-xl bg-white text-black flex items-center justify-center transition-all duration-200 hover:bg-gray-100 active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed flex-shrink-0"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} strokeWidth={2.5} />}
        </button>
      </div>
    </div>
  )
}
