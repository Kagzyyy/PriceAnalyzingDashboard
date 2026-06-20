"""
RAG Engine — ChromaDB vector store for financial news articles.

Articles are embedded using all-MiniLM-L6-v2 (via ChromaDB's built-in
ONNX runtime — no GPU, no PyTorch required).

Usage:
    rag = RAGEngine.get()
    rag.ingest(articles)                         # store news
    chunks = rag.retrieve_for_trade(ticker, date) # semantic search
"""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGEngine:
    """Singleton ChromaDB-backed news store + retriever."""

    _instance: Optional["RAGEngine"] = None

    def __init__(self) -> None:
        try:
            import chromadb
            from backend.config import CHROMA_PATH
            self._client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._col = self._client.get_or_create_collection(
                name="financial_news",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"RAGEngine ready — collection has {self._col.count()} documents"
            )
            self._ok = True
        except Exception as e:
            logger.warning(f"RAGEngine init failed (chromadb not available?): {e}")
            self._ok = False
            self._col = None  # type: ignore

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "RAGEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Public: ingest ────────────────────────────────────────────────────────

    def ingest(self, articles: list[dict]) -> int:
        """
        Embed and upsert articles into ChromaDB.
        Uses the article URL as a stable dedup key.
        Returns number of articles upserted.
        """
        if not self._ok or not articles:
            return 0

        docs, metas, ids = [], [], []

        for a in articles:
            url = (a.get("article_url") or a.get("id") or "").strip()
            if not url:
                continue

            title    = (a.get("title",       "") or "")[:250]
            desc     = (a.get("description", "") or "")[:600]
            pub      = a.get("publisher", {})
            pub_name = pub.get("name", "") if isinstance(pub, dict) else str(pub or "")
            date_str = (a.get("published_utc", "") or "")[:10]

            # Embed title + description together for richer semantics
            text = f"{title}. {desc}".strip(". ").strip()
            if len(text) < 12:
                continue

            # SHA-256 of URL → stable, collision-free 24-char ID
            doc_id = "n_" + hashlib.sha256(url.encode()).hexdigest()[:24]

            docs.append(text)
            metas.append({
                "title":     title,
                "publisher": pub_name,
                "date":      date_str,
                "url":       url[:600],
            })
            ids.append(doc_id)

        if docs:
            try:
                # upsert: safe to call multiple times on same articles
                self._col.upsert(documents=docs, metadatas=metas, ids=ids)
                logger.info(
                    f"RAG ingested {len(docs)} articles "
                    f"(total in store: {self._col.count()})"
                )
            except Exception as e:
                logger.warning(f"RAG upsert error: {e}")
                return 0

        return len(docs)

    # ── Public: retrieve for a single trade ───────────────────────────────────

    def retrieve_for_trade(
        self,
        ticker: str,
        date_str: str,
        n: int = 12,
        min_score: float = 0.18,
    ) -> list[dict]:
        """
        Retrieve semantically relevant articles for a gap trade.

        min_score (default 0.18): cosine similarity threshold — articles below
        this are generic noise (e.g. "Nvidia earnings" when searching for
        "BRITANNIA") and are discarded before being passed to the LLM.

        Fires 5 queries targeted at the specific company + Indian market context.
        Returns up to `n` deduplicated results above the relevance threshold,
        sorted by similarity descending.
        """
        if not self._ok or not self._col:
            return []

        count = self._col.count()
        if count == 0:
            logger.info("RAG store is empty — no articles to retrieve")
            return []

        clean = ticker.replace(".NS", "").replace(".BO", "").upper()
        month = date_str[:7]  # "YYYY-MM"

        # Company-name lookup for better semantic queries
        COMPANY_NAMES: dict[str, str] = {
            "BRITANNIA":    "Britannia Industries FMCG biscuits",
            "HINDUNILVR":   "Hindustan Unilever FMCG consumer goods",
            "NESTLEIND":    "Nestle India food beverages",
            "DABUR":        "Dabur India healthcare FMCG",
            "MARICO":       "Marico consumer goods",
            "GODREJCP":     "Godrej Consumer Products",
            "RELIANCE":     "Reliance Industries oil telecom retail",
            "HDFCBANK":     "HDFC Bank India private bank",
            "ICICIBANK":    "ICICI Bank India private bank",
            "INFY":         "Infosys IT services software",
            "TCS":          "Tata Consultancy Services IT",
            "WIPRO":        "Wipro IT services",
            "TATAMOTORS":   "Tata Motors automobiles EV",
            "MARUTI":       "Maruti Suzuki cars automobiles",
            "SUNPHARMA":    "Sun Pharmaceutical drugs pharma",
            "DRREDDY":      "Dr Reddys Laboratories pharma",
            "BAJFINANCE":   "Bajaj Finance NBFC lending",
            "AXISBANK":     "Axis Bank India",
            "KOTAKBANK":    "Kotak Mahindra Bank",
            "LT":           "Larsen Toubro infrastructure construction",
            "ADANIENT":     "Adani Enterprises conglomerate",
            "ADANIPORTS":   "Adani Ports logistics",
            "ONGC":         "ONGC oil gas exploration India",
            "NTPC":         "NTPC power electricity India",
            "POWERGRID":    "Power Grid Corporation India",
            "TATASTEEL":    "Tata Steel metals",
            "JSWSTEEL":     "JSW Steel metals",
            "HINDALCO":     "Hindalco aluminium metals",
            "COALINDIA":    "Coal India mining",
        }
        company_hint = COMPANY_NAMES.get(clean, f"{clean} India NSE stock")

        queries = [
            f"{company_hint} earnings results quarterly profit revenue",
            f"{clean} NSE share price analyst upgrade downgrade target price",
            f"{company_hint} news announcement deal acquisition",
            f"India Nifty FMCG sector momentum market rally {month}",
            f"{company_hint} guidance outlook forecast management",
        ]

        k = min(6, count)
        seen: set[str] = set()
        results: list[dict] = []

        for q in queries:
            try:
                r = self._col.query(query_texts=[q], n_results=k)
                for doc, meta, dist in zip(
                    r["documents"][0],
                    r["metadatas"][0],
                    r["distances"][0],
                ):
                    url  = meta.get("url", "")
                    score = round(1.0 - float(dist), 4)  # cosine similarity 0-1
                    if url in seen or score < min_score:
                        continue
                    seen.add(url)
                    results.append({
                        "title":         meta.get("title", doc[:120]),
                        "description":   doc,
                        "published_utc": meta.get("date", ""),
                        "article_url":   url,
                        "publisher":     {"name": meta.get("publisher", "")},
                        "insights":      [],
                        "_rag_score":    score,
                    })
            except Exception as e:
                logger.warning(f"RAG query error for '{q}': {e}")

        results.sort(key=lambda x: x.get("_rag_score", 0.0), reverse=True)
        top = results[:n]

        logger.info(
            f"RAG retrieved {len(top)} relevant articles for {clean} / {date_str} "
            f"(min_score={min_score}, searched {count} docs)"
        )
        return top

    # ── Public: stats ─────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if not self._ok or not self._col:
            return 0
        try:
            return self._col.count()
        except Exception:
            return 0

    @property
    def is_ready(self) -> bool:
        return self._ok
