# backend/services/rag_index_service.py
from __future__ import annotations

import os
import time
import uuid
import logging
from typing import List, Dict, Any

from services.llm.llm_service import llm_service
from .vector_store import vector_store, VectorChunk

logger = logging.getLogger(__name__)


class RAGIndexService:
    """
    Builds an in-memory vector index per doc_id.
    Trigger this in BackgroundTasks right after create_doc().
    """

    def __init__(self) -> None:
        # Chunking defaults (tune later)
        self.chunk_chars = int(os.getenv("RAG_CHUNK_CHARS", "1400"))
        self.chunk_overlap_chars = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "220"))

        # Embedding batching (avoid huge requests)
        self.embed_batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "64"))

    def _set_doc_index_state(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = 0,
        error: str = "",
        started_at: float | None = None,
        finished_at: float | None = None,
    ) -> None:
        obj = llm_service.doc_store.get(doc_id)
        if not obj:
            return
        obj["ts"] = time.time()
        obj["index_status"] = status
        obj["chunk_count"] = int(chunk_count)
        obj["index_error"] = str(error or "")
        if started_at is not None:
            obj["index_started_at"] = float(started_at)
        if finished_at is not None:
            obj["index_finished_at"] = float(finished_at)

    def _normalize_text(self, text: str) -> str:
        # Keep this conservative; avoid heavy transformations that could break anchors.
        t = (text or "").replace("\r\n", "\n")
        # collapse extreme whitespace
        while "\n\n\n" in t:
            t = t.replace("\n\n\n", "\n\n")
        return t.strip()

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Simple char-based chunking with overlap.
        Returns list of dicts {chunk_id, text, meta}.
        """
        t = self._normalize_text(text)
        if not t:
            return []

        size = max(400, self.chunk_chars)
        overlap = max(0, min(self.chunk_overlap_chars, size - 50))

        out: List[Dict[str, Any]] = []
        start = 0
        idx = 0

        n = len(t)
        while start < n:
            end = min(n, start + size)
            chunk_txt = t[start:end].strip()
            if chunk_txt:
                out.append(
                    {
                        "chunk_id": f"{idx}-{uuid.uuid4().hex[:8]}",
                        "text": chunk_txt,
                        "meta": {"start": start, "end": end, "index": idx},
                    }
                )
                idx += 1

            if end >= n:
                break
            start = max(0, end - overlap)

        return out

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return llm_service.embed_texts(texts)

    def index_doc(self, doc_id: str) -> None:
        """
        Build the vector index for a given doc_id.
        Intended to be called via FastAPI BackgroundTasks.
        """
        started = time.time()

        try:
            # Mark pending early
            self._set_doc_index_state(doc_id, status="pending", chunk_count=0, error="", started_at=started)

            text = llm_service.get_doc_text(doc_id)
            if not text or not text.strip():
                vector_store.clear_doc(doc_id)
                self._set_doc_index_state(
                    doc_id,
                    status="failed",
                    chunk_count=0,
                    error="No text found to index",
                    started_at=started,
                    finished_at=time.time(),
                )
                return

            chunks = self._chunk_text(text)
            if not chunks:
                vector_store.clear_doc(doc_id)
                self._set_doc_index_state(
                    doc_id,
                    status="failed",
                    chunk_count=0,
                    error="Chunking produced no chunks",
                    started_at=started,
                    finished_at=time.time(),
                )
                return

            # Embed in batches
            vectors: List[VectorChunk] = []
            batch_size = max(1, self.embed_batch_size)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                batch_texts = [c["text"] for c in batch]
                embs = self._embed_texts(batch_texts)

                for c, e in zip(batch, embs):
                    vectors.append(
                        VectorChunk(
                            chunk_id=c["chunk_id"],
                            text=c["text"],
                            embedding=e,
                            meta=c["meta"],
                        )
                    )

            # Upsert
            vector_store.upsert_chunks(doc_id, vectors)

            finished = time.time()
            self._set_doc_index_state(
                doc_id,
                status="ready",
                chunk_count=len(vectors),
                error="",
                started_at=started,
                finished_at=finished,
            )
            logger.info(f"RAG index ready doc_id={doc_id} chunks={len(vectors)} in {finished - started:.2f}s")

        except Exception as e:
            logger.exception(f"RAG indexing failed doc_id={doc_id}")
            self._set_doc_index_state(
                doc_id,
                status="failed",
                chunk_count=0,
                error=str(e),
                started_at=started,
                finished_at=time.time(),
            )

    def get_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get the RAG index status for a document.
        Returns dict with ready status and other metadata.
        """
        obj = llm_service.doc_store.get(doc_id)
        if not obj:
            return {
                "ready": False,
                "status": "not_found",
                "chunk_count": 0,
                "error": f"Document {doc_id} not found",
            }
        
        index_status = obj.get("index_status", "unknown")
        return {
            "ready": index_status == "ready",
            "status": index_status,
            "chunk_count": obj.get("chunk_count", 0),
            "error": obj.get("index_error", ""),
            "started_at": obj.get("index_started_at"),
            "finished_at": obj.get("index_finished_at"),
        }

    def query(self, doc_id: str, message: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Query the RAG index for a document with a user message.
        Returns list of relevant chunks with scores.
        """
        # Embed the user's message
        msg_embedding = self._embed_texts([message])[0]
        
        # Query the vector store
        results = vector_store.query(doc_id, msg_embedding, top_k=top_k)
        
        # Format results
        output = []
        for chunk, score in results:
            output.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": score,
                "meta": chunk.meta,
            })
        
        return output


# singleton indexer
rag_index_service = RAGIndexService()
