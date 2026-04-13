from pydantic import BaseModel
from typing import Optional


class PredictionRecord(BaseModel):
    ticker: str
    symbol: str
    name: str
    sector: str
    industry: str
    prob_up: float
    prob_down: float
    confidence_pct: float
    direction: str
    price: float
    change_pct: float
    volume: int
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bb_pct: Optional[float] = None
    volume_ratio: Optional[float] = None
    rank: Optional[int] = None


class SectorSummary(BaseModel):
    sector: str
    avg_prob_up: float
    bullish_pct: float
    count: int
    longs: int
    shorts: int


class AssistantQuery(BaseModel):
    query: str
    top_k: Optional[int] = 12


class AssistantResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class MarketStatus(BaseModel):
    last_refresh: Optional[str]
    model_accuracy: Optional[float]
    n_stocks: Optional[int]
    n_samples: Optional[int]
    trained_at: Optional[str]
    doc_count: Optional[int]
    market_sentiment: Optional[str]
    total_predictions: Optional[int]
    longs: Optional[int]
    shorts: Optional[int]


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockDetail(BaseModel):
    ticker: str
    symbol: str
    name: str
    sector: str
    industry: str
    prob_up: float
    prob_down: float
    direction: str
    confidence_pct: float
    price: float
    change_pct: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bb_pct: Optional[float] = None
    volume_ratio: Optional[float] = None
    ohlcv: list[OHLCVPoint] = []
