"""
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.data.storage import init_db
from backend.scheduler import create_scheduler
from backend.api.routes import predictions, stocks, sectors, assistant, system, backtest, pairs_backtest, gap_backtest, trade_intel, market_pulse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

scheduler = create_scheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    init_db()
    logger.info("Starting scheduler...")
    scheduler.start()
    logger.info("NSE Prediction Dashboard backend ready.")
    yield
    # Shutdown
    scheduler.shutdown()
    logger.info("Scheduler stopped.")


app = FastAPI(
    title="NSE Stock Prediction Dashboard",
    description="XGBoost-powered stock direction predictions for all NSE stocks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route groups
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(stocks.router, prefix="/api", tags=["Stocks"])
app.include_router(sectors.router, prefix="/api", tags=["Sectors"])
app.include_router(assistant.router, prefix="/api", tags=["Assistant"])
app.include_router(system.router,    prefix="/api", tags=["System"])
app.include_router(backtest.router,       prefix="/api", tags=["Backtest"])
app.include_router(pairs_backtest.router, prefix="/api", tags=["Pairs"])
app.include_router(gap_backtest.router,   prefix="/api", tags=["Gap and Go"])
app.include_router(trade_intel.router,    prefix="/api", tags=["Trade Intel"])
app.include_router(market_pulse.router,   prefix="/api", tags=["Market Pulse"])


@app.get("/")
def root():
    return {"message": "NSE Prediction Dashboard API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
