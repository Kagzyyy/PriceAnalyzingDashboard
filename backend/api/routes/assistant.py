from fastapi import APIRouter
from backend.api.schemas import AssistantQuery
from backend.rag.assistant import query, SUGGESTED_QUESTIONS

router = APIRouter()


@router.post("/assistant/query")
def ask_assistant(body: AssistantQuery):
    result = query(body.query, top_k=body.top_k or 12)
    return result


@router.get("/assistant/suggestions")
def get_suggestions():
    return {"questions": SUGGESTED_QUESTIONS}
