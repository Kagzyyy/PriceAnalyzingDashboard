import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env", override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DB_PATH = ROOT / os.getenv("DB_PATH", "data/market.duckdb")
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/xgb_universal.pkl")
CHROMA_PATH = str(ROOT / os.getenv("CHROMA_PATH", "data/chroma"))
SYMBOLS_PATH = ROOT / "data" / "symbols.json"

# Ensure dirs exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# Model settings
LOOKBACK_DAYS = 1825      # 5 years for training
REFRESH_DAYS = 60         # 60 days for daily refresh
BATCH_SIZE = 50           # yfinance batch download size
LONG_THRESHOLD = 0.60     # prob_up > 60% → long
SHORT_THRESHOLD = 0.40    # prob_up < 40% → short
MIN_VOLUME = 50_000       # minimum daily volume filter
MIN_PRICE = 10.0          # minimum stock price filter
TOP_N = 50                # top N longs/shorts to return

# ── Stock universe ─────────────────────────────────────────────────────────
# "all"     → full ~177 stock list (original)
# "nifty50" → Nifty 50 only (50 stocks, best data quality for earnings/analyst)
# "nifty100"→ Nifty 100 (100 stocks, used by pairs strategy for more pairs)
UNIVERSE = os.getenv("UNIVERSE", "nifty50")
