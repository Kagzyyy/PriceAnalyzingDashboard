import logging
from fastapi import APIRouter, BackgroundTasks
from backend.data.storage import get_model_meta, get_last_prediction_date, get_latest_predictions
from backend.ml.ranking import get_market_breadth
from backend.rag.embeddings import get_doc_count

router = APIRouter()
logger = logging.getLogger(__name__)

_pipeline_running = False


@router.get("/status")
def get_status():
    meta = get_model_meta()
    preds = get_latest_predictions()
    breadth = get_market_breadth(preds) if not preds.empty else {}

    try:
        # Only call get_doc_count if ChromaDB is already initialised
        # (avoids loading sentence_transformers on every /status poll)
        from backend.rag.embeddings import _collection
        doc_count = get_doc_count() if _collection is not None else 0
    except Exception:
        doc_count = 0

    return {
        "last_refresh": get_last_prediction_date(),
        "model_accuracy": meta.get("accuracy"),
        "n_stocks": meta.get("n_stocks"),
        "n_samples": meta.get("n_samples"),
        "trained_at": meta.get("trained_at"),
        "doc_count": doc_count,
        "market_sentiment": breadth.get("market_sentiment"),
        "total_predictions": breadth.get("total_stocks"),
        "longs": breadth.get("longs"),
        "shorts": breadth.get("shorts"),
        "pipeline_running": _pipeline_running,
    }


@router.post("/refresh")
async def manual_refresh(background_tasks: BackgroundTasks):
    global _pipeline_running
    if _pipeline_running:
        return {"message": "Pipeline already running", "status": "running"}
    background_tasks.add_task(_run_pipeline)
    return {"message": "Refresh started", "status": "started"}


async def _run_pipeline():
    global _pipeline_running
    _pipeline_running = True
    try:
        from backend.scheduler import run_full_pipeline
        await run_full_pipeline(force=True)   # force=True bypasses freshness cache
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        _pipeline_running = False
