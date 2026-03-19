"""RAG client using Qdrant for financial context retrieval with HyDE.

Embeds news articles, research briefs, and fundamentals into Qdrant.
Uses HyDE (Hypothetical Document Embeddings) for improved retrieval:
  1. LLM generates a hypothetical ideal analysis
  2. That gets embedded and used to find similar real documents
  3. Retrieved docs are injected into Pass 1 for grounded analysis.

Embeddings via Ollama (nomic-embed-text) on the same server as Qdrant.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import requests
import structlog

logger = structlog.get_logger()

COLLECTION_NAME = "financial_context"
VECTOR_SIZE = 768  # nomic-embed-text dimension
OLLAMA_EMBED_MODEL = "nomic-embed-text"


class RagClient:
    """Qdrant-based RAG with HyDE for financial context."""

    def __init__(
        self,
        qdrant_url: str = "http://192.168.0.169:6333",
        ollama_url: str = "http://192.168.0.169:11434",
        llm=None,
    ):
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.llm = llm
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.qdrant_url, timeout=30)
                self._ensure_collection()
            except ImportError:
                logger.warning("qdrant_client_not_installed")
                return None
            except Exception as e:
                logger.warning("qdrant_connection_failed", error=str(e))
                return None
        return self._client

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams
        client = self._client
        if client is None:
            return
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", name=COLLECTION_NAME)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama nomic-embed-text."""
        results = []
        for text in texts:
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": text[:2000]},
                    timeout=30,
                )
                resp.raise_for_status()
                embedding = resp.json().get("embedding", [])
                if embedding:
                    results.append(embedding)
            except Exception as e:
                logger.warning("ollama_embed_failed", error=str(e))
                return []
        return results

    def _doc_id(self, text: str, source: str) -> int:
        """Deterministic integer ID for dedup (Qdrant needs int or UUID)."""
        h = hashlib.md5(f"{source}:{text[:200]}".encode()).hexdigest()
        return int(h[:16], 16)  # 64-bit int from first 16 hex chars

    def store_documents(
        self,
        documents: list[dict],
        source: str = "news",
    ) -> int:
        """Store documents in Qdrant.

        Each doc: {"text": str, "title": str, "date": str, ...metadata}
        Returns count of new documents stored.
        """
        client = self._get_client()
        if client is None:
            return 0

        texts = [d["text"] for d in documents if d.get("text")]
        if not texts:
            return 0

        embeddings = self._embed(texts)
        if not embeddings:
            return 0

        from qdrant_client.models import PointStruct

        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = self._doc_id(doc["text"], source)
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": doc["text"][:3000],  # limit payload size
                    "title": doc.get("title", ""),
                    "source": source,
                    "date": doc.get("date", datetime.now().isoformat()[:10]),
                    "symbols": doc.get("symbols", []),
                },
            ))

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("rag_documents_stored", count=len(points), source=source)
            return len(points)
        except Exception as e:
            logger.warning("rag_store_failed", error=str(e))
            return 0

    def store_news(self, news_items: list) -> int:
        """Store NewsItem objects."""
        docs = []
        for item in news_items:
            text = f"{item.title}\n{item.summary}" if hasattr(item, "title") else str(item)
            docs.append({
                "text": text,
                "title": getattr(item, "title", ""),
                "date": getattr(item, "published", "")[:10],
                "source": getattr(item, "source", "news"),
            })
        return self.store_documents(docs, source="news")

    def store_research_brief(self, brief: dict) -> int:
        """Store research brief sections."""
        docs = []
        # Store each top symbol as a separate document
        for sym in brief.get("top_symbols", []):
            text = (
                f"Research pick: {sym['symbol']}\n"
                f"Direction: {sym.get('direction', '')}\n"
                f"Conviction: {sym.get('conviction', '')}\n"
                f"Thesis: {sym.get('thesis', '')}\n"
                f"Catalyst: {sym.get('catalyst', '')}"
            )
            docs.append({
                "text": text,
                "title": f"Research: {sym['symbol']}",
                "date": brief.get("date", ""),
                "symbols": [sym["symbol"]],
            })
        # Store regime + themes as one doc
        if brief.get("key_themes"):
            text = (
                f"Market regime: {brief.get('market_regime', '')}\n"
                f"Key themes: {', '.join(brief.get('key_themes', []))}\n"
                f"Macro events: {brief.get('macro_events_today', '')}\n"
                f"Avoid: {', '.join(str(a) for a in brief.get('avoid_today', []))}"
            )
            docs.append({
                "text": text,
                "title": "Daily regime & themes",
                "date": brief.get("date", ""),
            })
        # Store geopolitical risks
        for risk in brief.get("geopolitical_risks", []):
            text = (
                f"Geopolitical risk: {risk.get('event', '')}\n"
                f"Impact: {risk.get('market_impact', '')}\n"
                f"Sectors: {', '.join(risk.get('affected_sectors', []))}"
            )
            docs.append({"text": text, "title": risk.get("event", ""), "date": brief.get("date", "")})

        return self.store_documents(docs, source="research")

    def hyde_retrieve(
        self,
        query_context: str,
        symbols: list[str] | None = None,
        top_k: int = 8,
        model: str = "QWEN3.5",
    ) -> list[dict]:
        """HyDE retrieval: generate hypothetical doc, embed, find similar real docs.

        Args:
            query_context: Current market context (portfolio + regime summary)
            symbols: Optional filter by relevant symbols
            top_k: Number of documents to retrieve
            model: LLM model for hypothesis generation
        """
        client = self._get_client()
        if client is None:
            return []

        # Step 1: Generate hypothetical ideal analysis (HyDE)
        if self.llm:
            try:
                hyde_prompt = (
                    "Based on the current market context below, write a SHORT (3-4 sentences) "
                    "ideal financial analysis that would be most useful for making trading decisions. "
                    "Focus on actionable insights, not generic commentary.\n\n"
                    f"{query_context[:2000]}"
                )
                hypothesis = self.llm.chat(
                    messages=[{"role": "user", "content": hyde_prompt}],
                    model=model,
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception:
                hypothesis = query_context[:500]
        else:
            hypothesis = query_context[:500]

        # Step 2: Embed the hypothesis
        embeddings = self._embed([hypothesis])
        if not embeddings:
            return []

        # Step 3: Search Qdrant
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            search_filter = None
            if symbols:
                search_filter = Filter(
                    should=[
                        FieldCondition(key="symbols", match=MatchAny(any=symbols)),
                        # Also match docs without symbol filter (regime/themes)
                        FieldCondition(key="source", match=MatchAny(any=["research"])),
                    ]
                )

            from qdrant_client.models import Query
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embeddings[0],
                query_filter=search_filter,
                limit=top_k,
            )
            docs = []
            for r in response.points:
                payload = r.payload or {}
                docs.append({
                    "text": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "date": payload.get("date", ""),
                    "score": round(r.score, 3),
                })
            logger.info("rag_hyde_retrieved", count=len(docs), top_score=docs[0]["score"] if docs else 0)
            return docs
        except Exception as e:
            logger.warning("rag_search_failed", error=str(e))
            return []

    def format_for_prompt(self, docs: list[dict], max_chars: int = 3000) -> str:
        """Format retrieved docs for LLM prompt injection."""
        if not docs:
            return ""
        lines = ["== RETRIEVED CONTEXT (from knowledge base) =="]
        total = 0
        for d in docs:
            entry = f"[{d.get('source', '?')}, {d.get('date', '?')}, relevance={d.get('score', 0)}] {d['text']}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)
