# backend/services/vector_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import threading


@dataclass
class VectorChunk:
    chunk_id: str
    text: str
    embedding: List[float]
    meta: Dict[str, object]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class InMemoryVectorStore:
    """
    Stores vectors per doc_id in memory.
    NOTE: Works only for a single backend process/worker.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, List[VectorChunk]] = {}

    def clear_doc(self, doc_id: str) -> None:
        with self._lock:
            self._store.pop(doc_id, None)

    def upsert_chunks(self, doc_id: str, chunks: List[VectorChunk]) -> None:
        with self._lock:
            self._store[doc_id] = list(chunks)

    def count(self, doc_id: str) -> int:
        with self._lock:
            return len(self._store.get(doc_id, []))

    def list_chunks(self, doc_id: str) -> List[VectorChunk]:
        with self._lock:
            chunks = list(self._store.get(doc_id, []))
        chunks.sort(key=lambda c: int(c.meta.get("index", 0)) if isinstance(c.meta, dict) else 0)
        return chunks

    def query(
        self,
        doc_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> List[Tuple[VectorChunk, float]]:
        with self._lock:
            chunks = list(self._store.get(doc_id, []))

        scored: List[Tuple[VectorChunk, float]] = []
        for ch in chunks:
            s = _cosine_similarity(query_embedding, ch.embedding)
            if s >= min_score:
                scored.append((ch, s))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, int(top_k))]


# singleton store
vector_store = InMemoryVectorStore()
