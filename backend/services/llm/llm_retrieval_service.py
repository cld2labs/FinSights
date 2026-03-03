from __future__ import annotations

import time
from typing import Callable, Dict, Any, List

from services.rag.vector_store import vector_store


class LLMRetrievalService:
    def __init__(
        self,
        doc_store: Dict[str, Dict[str, Any]],
        embed_query: Callable[[str], List[float]],
        rag_min_score: float = 0.15,
        rag_context_max_chars: int = 18000,
    ) -> None:
        self.doc_store = doc_store
        self.embed_query = embed_query
        self.rag_min_score = float(rag_min_score)
        self.rag_context_max_chars = int(rag_context_max_chars)

    def is_index_ready(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        obj = self.doc_store.get(doc_id)
        if not obj:
            return False
        obj["ts"] = time.time()
        return str(obj.get("index_status", "")).lower() == "ready"

    def retrieve_context(self, doc_id: str, query: str, top_k: int) -> str:
        if not doc_id or not self.is_index_ready(doc_id):
            return ""
        if vector_store.count(doc_id) <= 0:
            return ""

        emb = self.embed_query(query)
        if not emb:
            return ""

        results = vector_store.query(
            doc_id=doc_id,
            query_embedding=emb,
            top_k=max(1, int(top_k)),
            min_score=self.rag_min_score,
        )
        if not results:
            return ""

        parts: List[str] = []
        seen_ids = set()
        total_chars = 0
        for chunk, _score in results:
            if chunk.chunk_id in seen_ids:
                continue
            txt = (chunk.text or "").strip()
            if not txt:
                continue
            if total_chars + len(txt) > self.rag_context_max_chars and parts:
                break
            parts.append(txt)
            seen_ids.add(chunk.chunk_id)
            total_chars += len(txt)

        return "\n\n".join(parts).strip()
