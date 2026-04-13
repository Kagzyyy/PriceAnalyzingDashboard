"""
ChromaDB vector store with sentence-transformers embeddings.
Heavy imports (chromadb, sentence_transformers / torch) are lazy-loaded
inside functions so uvicorn startup stays fast.
"""
import logging
from backend.config import CHROMA_PATH

logger = logging.getLogger(__name__)

_model = None
_client = None
_collection = None

COLLECTION_NAME = "nse_market_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _get_embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_documents(docs: list[dict]) -> None:
    """
    Upsert documents into ChromaDB.
    Each doc: {id: str, text: str, metadata: dict}
    """
    if not docs:
        return

    collection = _get_collection()
    embed_model = _get_embed_model()

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    logger.info(f"Embedding {len(texts)} documents...")
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

    # Delete existing docs with same IDs first (upsert)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(docs)} documents to ChromaDB")


def search(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search over the document store.
    Returns list of {text, metadata, distance} dicts.
    """
    collection = _get_collection()
    embed_model = _get_embed_model()

    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return docs


def get_doc_count() -> int:
    return _get_collection().count()
