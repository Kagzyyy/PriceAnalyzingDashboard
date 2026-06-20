export type GlobeCategory =
  | 'monetary_policy'
  | 'energy'
  | 'geopolitical'
  | 'trade'
  | 'inflation'
  | 'markets'

export interface GeoEvent {
  id:             string
  title:          string
  summary?:       string
  source_url?:    string
  source_name?:   string
  published_at:   string
  category:       GlobeCategory
  lat:            number
  lng:            number
  location_name?: string
  severity:       number
  is_analyzed:    boolean
  sentiment?:        'bullish' | 'bearish' | 'neutral'
  affected?:         string
  direction?:        'bullish' | 'bearish' | 'neutral' | 'risk'
  direction_color?:  string
  timeframe?:        string
  affected_markets?: string[]
  evidence?:         string
  detail?:           string
}

export interface GlobeArc {
  id:               string
  source_event_id:  string
  target_event_id:  string
  source_lat:       number
  source_lng:       number
  target_lat:       number
  target_lng:       number
  relationship_type:'causal' | 'correlated'
  strength:         number
  category:         string
}

export interface TradeIntelResult {
  ticker:     string
  date:       string
  gap_pct:    number
  reasoning:  string
  events:     GeoEvent[]
  arcs:       GlobeArc[]
  focus_lat:  number
  focus_lng:  number
  n_articles: number
  rag_count?: number   // total docs in ChromaDB at time of query
}

export const CATEGORY_COLORS: Record<GlobeCategory, string> = {
  monetary_policy: '#0a84ff',
  energy:          '#ff8c00',
  geopolitical:    '#ff2d55',
  trade:           '#ffd60a',
  inflation:       '#bf5af2',
  markets:         '#00ff88',
}

export const CATEGORY_LABELS: Record<GlobeCategory, string> = {
  monetary_policy: 'Monetary Policy',
  energy:          'Energy',
  geopolitical:    'Geopolitical',
  trade:           'Trade',
  inflation:       'Inflation',
  markets:         'Markets',
}
