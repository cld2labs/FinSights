"""
LLM Service for Document Summarization (FinSights)
Uses an OpenAI-compatible provider (OpenAI, inference API/Xeon, or Ollama)

This version supports:
- Dynamic section chips (2 to 5) generated from the document at initial step.
- Section-wise summaries for ANY selected section title (no pre-defined section list).
- Readable section summaries WITHOUT showing quotes.
  Internally, we extract "facts" with short "anchors" that must exist in the document,
  validate anchors, then generate the final section from facts only.

Compatibility:
- initial_summary_first_chunk(doc_id) returns ONLY a summary string (same as before).
- summarize_financial(mode="financial_section") accepts any section title.
- doc_id flow stays fast (no file re-upload needed on chip clicks).

Anti-hallucination strategy:
- We do NOT require exact quotes in the final output.
- We do require each extracted fact to include at least one anchor that exists in the document text.
  Anchors are not shown to the user; they are only used for validation.
"""

from typing import Iterator, Dict, Any, Optional, Union, List
import logging
import re
import os
import time
import uuid
import hashlib
import math

import config
from services.rag.vector_store import vector_store
from .llm_text_utils import clean_text, normalize_money, dedupe_section_heading, extract_response_text
from services.rag.summarization_pipeline import SummarizationPipeline, SourceChunk
from .llm_provider import OpenAIProvider
from .llm_retrieval_service import LLMRetrievalService
from .llm_section_service import LLMSectionService
from services.observability_service import observability_service

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if raw is None:
        return int(default)
    txt = str(raw).strip()
    if txt == "":
        return int(default)
    try:
        return int(txt)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if raw is None:
        return float(default)
    txt = str(raw).strip()
    if txt == "":
        return float(default)
    try:
        return float(txt)
    except Exception:
        return float(default)


class LLMService:
    def __init__(self):
        self.provider = OpenAIProvider()
        self.model = self.provider.model
        self.embedding_model = self.provider.embedding_model
        self._initialized = False

        # Large default; override via env if needed.
        self.model_context_tokens = _env_int("MODEL_CONTEXT_TOKENS", 128000)
        self.min_model_context_tokens = _env_int("MIN_MODEL_CONTEXT_TOKENS", 4096)
        self.context_retry_shrink_ratio = _env_float("CONTEXT_RETRY_SHRINK_RATIO", 0.75)
        self.context_retry_margin_tokens = _env_int("CONTEXT_RETRY_MARGIN_TOKENS", 1200)

        # In-memory doc store for doc_id flow
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = _env_int("CACHE_TTL_SECONDS", 60 * 60)  # 1 hour
        self.cache_max_docs = _env_int("CACHE_MAX_DOCS", 25)

        # Dynamic section discovery bounds (frontend chips)
        self.dynamic_min_sections = _env_int("DYNAMIC_SECTIONS_MIN", 2)
        self.dynamic_max_sections = _env_int("DYNAMIC_SECTIONS_MAX", 5)
        self.dynamic_sections_use_llm = os.getenv("DYNAMIC_SECTIONS_USE_LLM", "false").strip().lower() == "true"

        # Evidence/facts extraction bounds
        self.facts_max_items = _env_int("FACTS_MAX_ITEMS", 10)
        self.anchor_max_items = _env_int("ANCHOR_MAX_ITEMS", 3)
        self.anchor_max_chars_each = _env_int("ANCHOR_MAX_CHARS_EACH", 60)

        # Validation threshold: how many facts must be anchored to proceed
        # Example: 0.6 means at least 60% of extracted facts must have >=1 valid anchor.
        self.min_anchored_fact_ratio = _env_float("MIN_ANCHORED_FACT_RATIO", 0.6)
        self.summary_retry_enable = os.getenv("SUMMARY_RETRY_ENABLE", "false").strip().lower() == "true"

        # RAG-assisted summarization defaults for large docs
        self.rag_summary_top_k = _env_int("RAG_SUMMARY_TOP_K", 8)
        self.rag_section_top_k = _env_int("RAG_SECTION_TOP_K", 10)
        self.rag_min_score = _env_float("RAG_MIN_SCORE", 0.15)
        self.rag_context_max_chars = _env_int("RAG_CONTEXT_MAX_CHARS", 18000)
        self.retrieval_first_enable = os.getenv("RETRIEVAL_FIRST_ENABLE", "true").strip().lower() == "true"
        self.retrieval_force_index_on_demand = os.getenv("RETRIEVAL_FORCE_INDEX_ON_DEMAND", "false").strip().lower() == "true"

        # Map-reduce summarization for large docs
        self.enable_map_reduce = os.getenv("MAP_REDUCE_ENABLE", "true").lower() == "true"
        self.map_reduce_min_chars = _env_int("MAP_REDUCE_MIN_CHARS", 18000)
        self.map_reduce_min_chunks = _env_int("MAP_REDUCE_MIN_CHUNKS", 8)
        self.map_reduce_max_chunks = _env_int("MAP_REDUCE_MAX_CHUNKS", 24)
        self.map_reduce_chunk_chars = _env_int("MAP_REDUCE_CHUNK_CHARS", 2800)
        self.map_reduce_chunk_overlap = _env_int("MAP_REDUCE_CHUNK_OVERLAP", 250)
        self.map_reduce_batch_size = _env_int("MAP_REDUCE_BATCH_SIZE", 6)
        self.map_reduce_overhead_tokens = _env_int("MAP_REDUCE_OVERHEAD_TOKENS", 900)
        self.map_reduce_target_reduce_groups = _env_int("MAP_REDUCE_TARGET_REDUCE_GROUPS", 6)
        self.summary_pipeline = SummarizationPipeline(
            max_chunks=self.map_reduce_max_chunks,
            chunk_chars=self.map_reduce_chunk_chars,
            chunk_overlap_chars=self.map_reduce_chunk_overlap,
            reduce_batch_size=self.map_reduce_batch_size,
        )
        self.retrieval_service = LLMRetrievalService(
            doc_store=self.doc_store,
            embed_query=self._embed_query,
            rag_min_score=self.rag_min_score,
            rag_context_max_chars=self.rag_context_max_chars,
        )
        self.section_service = LLMSectionService(
            call_chat=self._call_chat,
            base_system_prompt=self._base_system_prompt,
            normalize_section_title=self._normalize_section_title,
            facts_max_items=self.facts_max_items,
            anchor_max_items=self.anchor_max_items,
            anchor_max_chars_each=self.anchor_max_chars_each,
        )

    def _ensure_initialized(self):
        if self._initialized:
            return
        self.provider.ensure_initialized()
        self.model = self.provider.model
        self.embedding_model = self.provider.embedding_model
        try:
            resolved_ctx = int(self.provider.resolve_model_context_tokens(self.model_context_tokens))
            if resolved_ctx > 0:
                self.model_context_tokens = resolved_ctx
        except Exception:
            logger.exception("Failed to auto-resolve model context window; using configured value")
        self._initialized = True

        logger.info("LLM provider initialized successfully")
        logger.info(f"Provider: {self.provider.provider_name or 'unknown'}")
        logger.info(f"Model: {self.model}")
        logger.info(f"MODEL_CONTEXT_TOKENS: {self.model_context_tokens}")
        logger.info(f"MIN_MODEL_CONTEXT_TOKENS: {self.min_model_context_tokens}")
        logger.info(f"CONTEXT_RETRY_SHRINK_RATIO: {self.context_retry_shrink_ratio}")
        logger.info(f"CONTEXT_RETRY_MARGIN_TOKENS: {self.context_retry_margin_tokens}")
        logger.info(f"CACHE_MAX_DOCS: {self.cache_max_docs}")
        logger.info(f"CACHE_TTL_SECONDS: {self.cache_ttl_seconds}")
        logger.info(f"DYNAMIC_SECTIONS_MIN: {self.dynamic_min_sections}")
        logger.info(f"DYNAMIC_SECTIONS_MAX: {self.dynamic_max_sections}")
        logger.info(f"DYNAMIC_SECTIONS_USE_LLM: {self.dynamic_sections_use_llm}")
        logger.info(f"FACTS_MAX_ITEMS: {self.facts_max_items}")
        logger.info(f"ANCHOR_MAX_ITEMS: {self.anchor_max_items}")
        logger.info(f"MIN_ANCHORED_FACT_RATIO: {self.min_anchored_fact_ratio}")
        logger.info(f"SUMMARY_RETRY_ENABLE: {self.summary_retry_enable}")
        logger.info(f"EMBEDDING_MODEL: {self.embedding_model}")
        logger.info(f"RAG_SUMMARY_TOP_K: {self.rag_summary_top_k}")
        logger.info(f"RAG_SECTION_TOP_K: {self.rag_section_top_k}")
        logger.info(f"RAG_MIN_SCORE: {self.rag_min_score}")
        logger.info(f"RAG_CONTEXT_MAX_CHARS: {self.rag_context_max_chars}")
        logger.info(f"RETRIEVAL_FIRST_ENABLE: {self.retrieval_first_enable}")
        logger.info(f"RETRIEVAL_FORCE_INDEX_ON_DEMAND: {self.retrieval_force_index_on_demand}")
        logger.info(f"MAP_REDUCE_ENABLE: {self.enable_map_reduce}")
        logger.info(f"MAP_REDUCE_MIN_CHARS: {self.map_reduce_min_chars}")
        logger.info(f"MAP_REDUCE_MIN_CHUNKS: {self.map_reduce_min_chunks}")
        logger.info(f"MAP_REDUCE_MAX_CHUNKS: {self.map_reduce_max_chunks}")
        logger.info(f"MAP_REDUCE_CHUNK_CHARS: {self.map_reduce_chunk_chars}")
        logger.info(f"MAP_REDUCE_CHUNK_OVERLAP: {self.map_reduce_chunk_overlap}")
        logger.info(f"MAP_REDUCE_BATCH_SIZE: {self.map_reduce_batch_size}")
        logger.info(f"MAP_REDUCE_OVERHEAD_TOKENS: {self.map_reduce_overhead_tokens}")
        logger.info(f"MAP_REDUCE_TARGET_REDUCE_GROUPS: {self.map_reduce_target_reduce_groups}")

    def get_provider_name(self) -> str:
        try:
            self._ensure_initialized()
            return self.provider.provider_name or "unknown"
        except Exception:
            return self.provider.provider_name or "unknown"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_initialized()
        return self.provider.embed_texts(texts)

    # ----------------------------
    # Compatibility wrapper
    # ----------------------------
    def summarize(
        self,
        text: str,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = False,
        mode: str = "financial_initial",
        section: str = None,
    ) -> Union[str, Iterator[str]]:
        return self.summarize_financial(
            text=text,
            mode=mode,
            section=section,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    # ----------------------------
    # doc_id based flow (compat)
    # ----------------------------
    def create_doc(self, text: str) -> str:
        self._ensure_initialized()
        self._evict_cache_if_needed()

        if not text or not text.strip():
            raise ValueError("Empty text")

        doc_id = str(uuid.uuid4())
        dk = self._doc_key(text)

        self.doc_store[doc_id] = {
            "ts": time.time(),
            "text": text,
            "doc_key": dk,
            # list of {"title": str, "hint": str} created during initial summary
            "sections": None,
            "sections_status": "pending",  # pending | ready | failed

            # ---- RAG indexing state (for chat) ----
            "index_status": "pending",   # pending | ready | failed
            "chunk_count": 0,
            "index_error": "",
            "index_started_at": None,
            "index_finished_at": None,
        }
        return doc_id

    def get_doc_text(self, doc_id: str) -> str:
        obj = self.doc_store.get(doc_id)
        if not obj:
            raise ValueError("Invalid doc_id")
        obj["ts"] = time.time()
        return obj.get("text", "")

    def get_doc_key(self, doc_id: str) -> str:
        obj = self.doc_store.get(doc_id)
        if not obj:
            raise ValueError("Invalid doc_id")
        obj["ts"] = time.time()
        return obj.get("doc_key", "")

    def get_doc_sections(self, doc_id: str) -> List[str]:
        """
        Returns discovered dynamic section titles for this doc_id.
        If not discovered yet, returns [].
        """
        obj = self.doc_store.get(doc_id)
        if not obj:
            raise ValueError("Invalid doc_id")
        obj["ts"] = time.time()

        secs = obj.get("sections")
        if not isinstance(secs, list):
            return []

        titles: List[str] = []
        for s in secs:
            if isinstance(s, dict) and s.get("title"):
                titles.append(str(s["title"]).strip())
            elif isinstance(s, str):
                titles.append(str(s).strip())
        return [t for t in titles if t]

    def get_doc_sections_payload(self, doc_id: str) -> Dict[str, Any]:
        obj = self.doc_store.get(doc_id)
        if not obj:
            raise ValueError("Invalid doc_id")
        obj["ts"] = time.time()

        status = str(obj.get("sections_status", "pending")).lower()
        sections_raw = obj.get("sections")
        sections: List[Dict[str, str]] = []
        if isinstance(sections_raw, list):
            for s in sections_raw:
                if isinstance(s, dict):
                    title = str(s.get("title", "")).strip()
                    hint = str(s.get("hint", "")).strip()
                    if title:
                        sections.append({"title": title, "hint": hint})
                elif isinstance(s, str):
                    t = str(s).strip()
                    if t:
                        sections.append({"title": t, "hint": ""})

        return {
            "doc_id": doc_id,
            "status": status,
            "ready": status == "ready",
            "count": len(sections),
            "sections": sections,
        }

    def generate_doc_sections(self, doc_id: str, force: bool = False) -> None:
        """
        Background-safe section discovery.
        """
        self._ensure_initialized()
        obj = self.doc_store.get(doc_id)
        if not obj:
            return

        obj["ts"] = time.time()
        if not force and isinstance(obj.get("sections"), list) and obj.get("sections"):
            obj["sections_status"] = "ready"
            return

        obj["sections_status"] = "pending"
        try:
            text = str(obj.get("text", "") or "")
            fitted = self._context_for_mode(
                text=text,
                mode="financial_initial",
                max_output_tokens=240,
                doc_id=doc_id,
            )
            sections = self._discover_dynamic_sections(
                fitted_text=fitted,
                min_sections=self.dynamic_min_sections,
                max_sections=self.dynamic_max_sections,
                temperature=0.2,
            )
            obj["sections"] = sections or []
            obj["sections_status"] = "ready"
            obj["ts"] = time.time()
        except Exception:
            obj["sections_status"] = "failed"
            obj["sections"] = []
            logger.exception("Background section discovery failed doc_id=%s", doc_id)

    def get_doc_section_hint(self, doc_id: str, section_title: str) -> str:
        """
        Returns a stored hint for a discovered section (optional), else "".
        Hints are short descriptions like "Totals, taxes, payment status".
        """
        obj = self.doc_store.get(doc_id)
        if not obj:
            raise ValueError("Invalid doc_id")
        obj["ts"] = time.time()

        secs = obj.get("sections")
        if not isinstance(secs, list):
            return ""

        target = self._normalize_section_title(section_title).lower()
        if not target:
            return ""

        for s in secs:
            if isinstance(s, dict):
                t = self._normalize_section_title(str(s.get("title", ""))).lower()
                if t == target:
                    return str(s.get("hint", "") or "").strip()
        return ""

    def prefetch_doc(self, doc_id: str) -> None:
        # No-op (kept so existing routes won't break)
        obj = self.doc_store.get(doc_id)
        if obj:
            obj["ts"] = time.time()
        return

    def summarize_by_doc_id(
        self,
        doc_id: str,
        mode: str = "financial_initial",
        section: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        text = self.get_doc_text(doc_id)
        return self.summarize_financial(
            text=text,
            mode=mode,
            section=section,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            doc_id=doc_id,
        )

    def initial_summary_first_chunk(self, doc_id: str, max_tokens: int = 240, temperature: float = 0.25) -> str:
        """
        Compatibility method expected by routes/frontend.
        Uses full doc text (fitted to context) to produce 4-5 sentences.

        Updated behavior:
        - Also runs dynamic section discovery and stores it in doc_store[doc_id]["sections"].
        - Still returns ONLY the summary string (so existing routes don't break).
        """
        self._ensure_initialized()
        text = self.get_doc_text(doc_id)
        fitted = self._context_for_mode(
            text=text,
            mode="financial_initial",
            max_output_tokens=max_tokens,
            doc_id=doc_id,
        )

        # Dynamic sections discovery (best effort; never breaks summary)
        try:
            sections = self._discover_dynamic_sections(
                fitted_text=fitted,
                min_sections=self.dynamic_min_sections,
                max_sections=self.dynamic_max_sections,
                temperature=0.2,
            )
            obj = self.doc_store.get(doc_id)
            if obj is not None:
                obj["sections"] = sections
                obj["sections_status"] = "ready"
                obj["ts"] = time.time()
        except Exception:
            logger.exception("Dynamic section discovery failed (continuing with summary only).")

        # Use map-reduce for large docs when enabled.
        # Keep section discovery above so frontend chips still work.
        if (not self.retrieval_first_enable) and self._should_use_map_reduce(text=text, doc_id=doc_id):
            mr = self._map_reduce_summary(
                text=text,
                doc_id=doc_id,
                max_tokens=max_tokens,
                temperature=temperature,
                style="initial",
            )
            if mr:
                return normalize_money(clean_text(mr))

        # Normal initial summary (non map-reduce path)
        system_prompt = self._base_system_prompt() + (
            "\nWrite a short generalized summary that tells the user what this document is about.\n"
            "Focus on: document type, company/entity, reporting period/date, and purpose ONLY if explicitly stated.\n"
            "Rules:\n"
            "- Plain text only.\n"
            "- 4 to 5 sentences.\n"
            "- Prefer exact names, dates, and periods when present.\n"
            "- Include key numeric highlights ONLY if explicitly present.\n"
            "- Do not invent any details.\n"
        )

        user_prompt = f"""Write the generalized summary for this document.

Document:
{fitted}
"""
        resp = self._call_chat(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=max(0.0, min(float(temperature), 0.35)),
            stream=False,
        )
        out = self._clean_or_retry_summary(
            draft=extract_response_text(resp),
            source_text=(fitted or text),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return normalize_money(out)

    # ----------------------------
    # Prompt building
    # ----------------------------
    @staticmethod
    def _base_system_prompt() -> str:
        return (
            "You are an analyst AI.\n"
            "You must use ONLY information supported by the provided document text.\n"
            "Do not invent facts, numbers, dates, names, or events.\n"
            "If you are uncertain, state it clearly.\n"
            "Output must be plain text only (no markdown emphasis: no **, *, _).\n"
        )
    def chat_with_context(
        self,
        question: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:
        """
        Answer a user question using ONLY the retrieved context.
        If the answer is not supported by the context, say so.
        """
        self._ensure_initialized()

        q = (question or "").strip()
        ctx = (context or "").strip()

        if not q:
            return "Please enter a question."

        system_prompt = (
            self._base_system_prompt()
            + "\nYou are answering questions about an uploaded document.\n"
              "Use ONLY the provided CONTEXT.\n"
              "If the answer is not in the context, say you cannot find it.\n"
              "Keep the response concise and readable.\n"
        )

        user_prompt = f"""CONTEXT:
{ctx}

QUESTION:
{q}

Answer using only the context.
"""

        resp = self._call_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=max(0.0, min(float(temperature), 0.5)),
            stream=False,
        )
        return normalize_money(clean_text(extract_response_text(resp)))


    # ----------------------------
    # Core helpers
    # ----------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4)) if text else 0

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        needles = (
            "context length",
            "maximum context length",
            "context_window_exceeded",
            "context_length_exceeded",
            "too many tokens",
            "prompt is too long",
            "token limit exceeded",
            "max context",
        )
        return any(n in msg for n in needles)

    def _shrink_context_window(self, reason: str) -> int:
        current = int(self.model_context_tokens)
        shrunk = int(max(self.min_model_context_tokens, current * self.context_retry_shrink_ratio))
        if shrunk >= current:
            shrunk = max(self.min_model_context_tokens, current - 2048)
        if shrunk < current:
            logger.warning(
                "Shrinking model context budget (%s): %d -> %d",
                reason,
                current,
                shrunk,
            )
            self.model_context_tokens = shrunk
        return int(self.model_context_tokens)

    def _fit_prompts_to_context(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        margin_tokens: int = 900,
    ) -> tuple[str, str, int]:
        """
        Ensure chat prompt + completion target fits into current model_context_tokens.
        Trims only the user prompt when needed.
        """
        sys_txt = system_prompt or ""
        usr_txt = user_prompt or ""
        out_tokens = max(64, int(max_tokens))
        margin = max(256, int(margin_tokens))

        available_prompt_tokens = max(256, int(self.model_context_tokens - out_tokens - margin))
        sys_tokens = self._estimate_tokens(sys_txt)
        user_tokens = self._estimate_tokens(usr_txt)
        total_prompt_tokens = sys_tokens + user_tokens
        if total_prompt_tokens <= available_prompt_tokens:
            return sys_txt, usr_txt, out_tokens

        available_user_tokens = max(64, available_prompt_tokens - sys_tokens)
        if user_tokens <= available_user_tokens:
            return sys_txt, usr_txt, out_tokens

        ratio = max(0.05, min(1.0, available_user_tokens / float(max(1, user_tokens))))
        keep_chars = max(600, int(len(usr_txt) * ratio))
        head = usr_txt[: int(keep_chars * 0.75)]
        tail = usr_txt[-int(keep_chars * 0.25) :] if keep_chars > 1800 else ""
        trimmed = head
        if tail and tail not in head:
            trimmed = head + "\n\n[...TRUNCATED FOR CONTEXT LIMIT...]\n\n" + tail
        return sys_txt, trimmed, out_tokens

    @staticmethod
    def _strip_summary_leadin(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        patterns = [
            r"^\s*here is (a|the) (short )?(generalized )?summary[^:\n]*:\s*",
            r"^\s*here'?s (a|the) summary[^:\n]*:\s*",
            r"^\s*summary\s*:\s*",
        ]
        for p in patterns:
            t = re.sub(p, "", t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def _extract_num_tokens(text: str) -> List[str]:
        if not text:
            return []
        vals = re.findall(r"\b\d[\d,./:-]*\b", text)
        out: List[str] = []
        seen = set()
        for v in vals:
            k = v.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(v.strip())
            if len(out) >= 24:
                break
        return out

    def _looks_generic_summary(self, summary: str, source_text: str) -> bool:
        s = (summary or "").strip()
        if not s:
            return True
        low = s.lower()
        generic_phrases = (
            "can be evaluated using",
            "provide insights",
            "stakeholders can assess",
            "ability to generate revenue",
            "manage costs",
            "invest in growth opportunities",
        )
        if any(g in low for g in generic_phrases):
            return True

        src_nums = self._extract_num_tokens(source_text)
        if src_nums:
            # If source has many numbers, summary should preserve at least one.
            if not any(n in s for n in src_nums[:12]):
                return True
        return False

    def _clean_or_retry_summary(
        self,
        draft: str,
        source_text: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        cleaned = clean_text(self._strip_summary_leadin(draft))
        if cleaned and not self._looks_generic_summary(cleaned, source_text):
            return cleaned
        if not self.summary_retry_enable:
            if cleaned:
                return cleaned
            return self._fallback_extractive_summary(source_text)

        retry_user = (
            user_prompt
            + "\n\nIMPORTANT:\n"
              "- Do NOT start with phrases like 'Here is a summary'.\n"
              "- Use only concrete facts from the document.\n"
              "- Include at least one exact date/period or numeric value if present in document.\n"
              "- Output plain text only.\n"
        )
        retry_resp = self._call_chat(
            system_prompt,
            retry_user,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        retry_text = clean_text(self._strip_summary_leadin(extract_response_text(retry_resp)))
        if retry_text and not self._looks_generic_summary(retry_text, source_text):
            return retry_text
        if cleaned and not self._looks_generic_summary(cleaned, source_text):
            return cleaned
        return self._fallback_extractive_summary(source_text)

    @staticmethod
    def _fallback_extractive_summary(source_text: str) -> str:
        """
        Deterministic fallback for weak models: pull concrete sentences from source.
        """
        t = (source_text or "").strip()
        if not t:
            return ""

        # Normalize whitespace then split into sentence-like units.
        norm = re.sub(r"\s+", " ", t)
        raw_sentences = re.split(r"(?<=[.!?])\s+|\s*\n+\s*", norm)
        sentences = [s.strip() for s in raw_sentences if s and len(s.strip()) >= 35]
        if not sentences:
            return t[:600].strip()

        def score(s: str) -> int:
            low = s.lower()
            sc = 0
            if re.search(r"\b\d[\d,./:-]*\b", s):
                sc += 3
            for k in ("ended", "year", "period", "balance sheet", "income statement", "cash flow", "revenue", "expense", "net", "total", "assets", "liabilities"):
                if k in low:
                    sc += 1
            if 60 <= len(s) <= 220:
                sc += 1
            return sc

        ranked = sorted(sentences, key=score, reverse=True)
        picked: List[str] = []
        seen = set()
        for s in ranked:
            key = s[:120].lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= 4:
                break

        if not picked:
            picked = sentences[:3]

        return " ".join(picked).strip()

    @staticmethod
    def _doc_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _normalize_section_title(title: str) -> str:
        t = (title or "").strip()
        t = re.sub(r"\s{2,}", " ", t)
        t = t.strip(" -:\t\r\n")
        if len(t) > 56:
            t = t[:56].rstrip()
        return t

    def _evict_cache_if_needed(self):
        now = time.time()

        expired_docs = [k for k, v in self.doc_store.items() if now - v.get("ts", now) > self.cache_ttl_seconds]
        for k in expired_docs:
            self.doc_store.pop(k, None)

        if len(self.doc_store) <= self.cache_max_docs:
            return

        items = sorted(self.doc_store.items(), key=lambda kv: kv[1].get("ts", 0))
        while len(items) > self.cache_max_docs:
            k, _ = items.pop(0)
            self.doc_store.pop(k, None)

    def _fit_text_to_context(self, text: str, max_output_tokens: int) -> str:
        """
        Truncate input so input + output stays within model_context_tokens.
        Keeps head + tail when truncating.
        """
        if not text:
            return ""

        overhead_tokens = 900
        available_input_tokens = max(500, self.model_context_tokens - int(max_output_tokens) - overhead_tokens)

        est = self._estimate_tokens(text)
        if est <= available_input_tokens:
            return text

        ratio = available_input_tokens / float(est)
        keep_chars = max(2500, int(len(text) * ratio))

        head = text[: int(keep_chars * 0.72)]
        tail = text[-int(keep_chars * 0.28) :] if keep_chars > 4500 else ""

        truncated = head
        if tail and tail not in head:
            truncated = head + "\n\n[...TRUNCATED...]\n\n" + tail

        return truncated

    def _is_index_ready(self, doc_id: Optional[str]) -> bool:
        return self.retrieval_service.is_index_ready(doc_id or "")

    def _embed_query(self, query: str) -> List[float]:
        self._ensure_initialized()
        try:
            return self.provider.embed_query(query)
        except Exception:
            logger.exception("Failed to embed query for retrieval")
            return []

    def _retrieve_context(self, doc_id: str, query: str, top_k: int) -> str:
        try:
            return self.retrieval_service.retrieve_context(doc_id, query, top_k)
        except Exception:
            logger.exception("Vector retrieval failed")
            return ""

    def _ensure_doc_index_ready(self, doc_id: Optional[str]) -> bool:
        if not doc_id:
            return False
        if self._is_index_ready(doc_id):
            return True
        if not self.retrieval_force_index_on_demand:
            return False
        try:
            from services.rag.rag_index_service import rag_index_service
            rag_index_service.index_doc(doc_id)
        except Exception:
            logger.exception("Failed to build RAG index on-demand doc_id=%s", doc_id)
            return False
        return self._is_index_ready(doc_id)

    @staticmethod
    def _dedupe_text_blocks(raw: str) -> List[str]:
        parts = [p.strip() for p in (raw or "").split("\n\n") if p and p.strip()]
        out: List[str] = []
        seen = set()
        for p in parts:
            key = hashlib.sha1(p.encode("utf-8", errors="ignore")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _retrieve_multi_query_context(self, doc_id: Optional[str], queries: List[str], top_k: int, max_chars: Optional[int] = None) -> str:
        if not doc_id or not queries:
            return ""
        if not self._is_index_ready(doc_id) and not self._ensure_doc_index_ready(doc_id):
            return ""

        limit = int(max_chars or max(4000, self.rag_context_max_chars))
        merged: List[str] = []
        seen = set()
        total = 0
        for q in queries:
            ctx = self._retrieve_context(doc_id, query=q, top_k=top_k)
            for block in self._dedupe_text_blocks(ctx):
                h = hashlib.sha1(block.encode("utf-8", errors="ignore")).hexdigest()
                if h in seen:
                    continue
                if total + len(block) > limit and merged:
                    return "\n\n".join(merged).strip()
                merged.append(block)
                seen.add(h)
                total += len(block)
                if total >= limit:
                    return "\n\n".join(merged).strip()
        return "\n\n".join(merged).strip()

    def _context_for_mode(
        self,
        text: str,
        mode: str,
        max_output_tokens: int,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
        section_hint: str = "",
    ) -> str:
        """
        Prefer retrieval context for large docs when index is ready.
        Falls back to legacy truncation path when retrieval is unavailable.
        """
        fitted = self._fit_text_to_context(text, max_output_tokens=max_output_tokens)
        if not doc_id:
            return fitted
        if mode == "financial_section":
            sec = self._normalize_section_title(section or "")
            query = f"Find facts for section: {sec}. {section_hint}".strip()
            rag_text = self._retrieve_context(doc_id, query=query, top_k=self.rag_section_top_k)
            return rag_text or fitted

        if mode == "financial_initial":
            queries = [
                "What is this document about? Include type, parties/entities, and purpose.",
                "What reporting period/date does this document cover?",
                "What are the key numeric amounts, totals, balances, or highlights?",
            ]
            rag_text = self._retrieve_multi_query_context(
                doc_id=doc_id,
                queries=queries,
                top_k=self.rag_summary_top_k,
                max_chars=max(8000, self.rag_context_max_chars),
            )
            return rag_text or fitted

        if mode == "financial_overall":
            query = "Summarize the document with key claims, names, dates, and numbers."
            rag_text = self._retrieve_context(doc_id, query=query, top_k=self.rag_summary_top_k)
            return rag_text or fitted

        if mode == "financial_sectionwise":
            query = "Identify major themes and section-level highlights for this document."
            rag_text = self._retrieve_context(doc_id, query=query, top_k=self.rag_summary_top_k)
            return rag_text or fitted

        return fitted

    def _source_chunks_for_map_reduce(self, text: str, doc_id: Optional[str], max_chunks: Optional[int] = None) -> List[SourceChunk]:
        if doc_id and self._is_index_ready(doc_id) and vector_store.count(doc_id) > 0:
            chunks = vector_store.list_chunks(doc_id)
            out: List[SourceChunk] = []
            for c in chunks:
                chunk_text = (c.text or "").strip()
                if not chunk_text:
                    continue
                idx = 0
                if isinstance(c.meta, dict):
                    try:
                        idx = int(c.meta.get("index", 0))
                    except Exception:
                        idx = 0
                out.append(SourceChunk(text=chunk_text, order=idx))
            if out:
                ordered = sorted(out, key=lambda x: x.order)
                if max_chunks is not None:
                    return self.summary_pipeline.limit_chunks_with_limit(ordered, max_chunks=max_chunks)
                return ordered

        raw = (text or "").strip()
        if not raw:
            return []
        if max_chunks is None:
            return [SourceChunk(text=raw, order=0)]
        return self.summary_pipeline.build_chunks_from_text_with_limit(text=raw, max_chunks=max_chunks)

    @staticmethod
    def _split_text_even(text: str, target_chunks: int, overlap_chars: int = 0) -> List[SourceChunk]:
        t = (text or "").strip()
        if not t:
            return []
        n = len(t)
        k = max(1, int(target_chunks))
        if k <= 1:
            return [SourceChunk(text=t, order=0)]

        overlap = max(0, int(overlap_chars))
        out: List[SourceChunk] = []
        for i in range(k):
            start = int(i * n / k)
            end = int((i + 1) * n / k)
            if i > 0:
                start = max(0, start - overlap)
            if i < k - 1:
                end = min(n, end + overlap)
            part = t[start:end].strip()
            if part:
                out.append(SourceChunk(text=part, order=i))
        return out

    @staticmethod
    def _merge_source_chunks_to_target(chunks: List[SourceChunk], target_chunks: int) -> List[SourceChunk]:
        ordered = sorted(chunks, key=lambda c: c.order)
        n = len(ordered)
        k = max(1, min(int(target_chunks), n))
        if k >= n:
            return ordered

        out: List[SourceChunk] = []
        for i in range(k):
            start = int(i * n / k)
            end = int((i + 1) * n / k)
            group = ordered[start:end]
            merged_text = "\n\n".join([g.text for g in group if (g.text or "").strip()]).strip()
            if not merged_text:
                continue
            base_order = group[0].order if group else i
            out.append(SourceChunk(text=merged_text, order=base_order))
        return out

    def _adaptive_map_reduce_plan(self, text: str, final_tokens: int) -> Dict[str, int]:
        est_doc_tokens = max(1, self._estimate_tokens(text))
        safe_output_tokens = max(128, int(final_tokens))
        overhead_tokens = max(256, int(self.map_reduce_overhead_tokens))
        input_budget_tokens = max(700, int(self.model_context_tokens - safe_output_tokens - overhead_tokens))

        chunks_needed = max(1, math.ceil(est_doc_tokens / float(input_budget_tokens)))
        target_chunks = max(1, min(chunks_needed, int(self.map_reduce_max_chunks)))

        # Prefer fewer reduce calls by choosing a larger merge batch when safe.
        target_groups = max(2, min(int(self.map_reduce_target_reduce_groups), target_chunks))
        adaptive_batch = math.ceil(target_chunks / float(target_groups)) if target_chunks > 0 else 2
        reduce_batch_size = max(2, min(target_chunks if target_chunks > 0 else 2, max(int(self.map_reduce_batch_size), adaptive_batch)))

        estimated_reduce_rounds = 0
        current = max(1, target_chunks)
        while current > reduce_batch_size and estimated_reduce_rounds < 12:
            current = math.ceil(current / float(reduce_batch_size))
            estimated_reduce_rounds += 1

        return {
            "estimated_doc_tokens": est_doc_tokens,
            "input_budget_tokens": input_budget_tokens,
            "target_chunks": int(target_chunks),
            "reduce_batch_size": int(reduce_batch_size),
            "estimated_reduce_rounds": int(estimated_reduce_rounds),
        }

    def _should_use_map_reduce(self, text: str, doc_id: Optional[str]) -> bool:
        if not self.enable_map_reduce:
            return False
        if not text or not text.strip():
            return False
        if len(text) >= self.map_reduce_min_chars:
            return True
        if doc_id and self._is_index_ready(doc_id) and vector_store.count(doc_id) >= self.map_reduce_min_chunks:
            return True
        return False

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self._call_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return clean_text(extract_response_text(resp))

    def _map_reduce_summary(self, text: str, doc_id: Optional[str], max_tokens: int, temperature: float, style: str) -> str:
        if style == "initial":
            map_instruction = (
                "Extract only factual points from this chunk: document type, parties/entities, reporting periods, dates, "
                "and key numeric values if explicitly present."
            )
            reduce_instruction = "Merge chunk summaries, remove duplicates, and keep only supported facts."
            final_instruction = (
                "Write a short generalized summary (3 to 5 sentences) for a non-expert user. "
                "Include exact names/dates only when present. Do not invent."
            )
            final_temp = max(0.0, min(float(temperature), 0.25))
            final_tokens = max(160, min(int(max_tokens), 320))
        else:
            map_instruction = (
                "Extract the most important factual points from this chunk. "
                "Preserve numbers, dates, and entity names when present."
            )
            reduce_instruction = "Merge summaries into a concise de-duplicated brief with key facts prioritized."
            final_instruction = (
                "Write a concise overall summary as 8 to 14 bullets max. "
                "Use only supported information and keep it readable for non-experts."
            )
            final_temp = max(0.0, min(float(temperature), 0.35))
            final_tokens = int(max_tokens)

        source_chunks = self._source_chunks_for_map_reduce(text=text, doc_id=doc_id, max_chunks=None)
        if not source_chunks:
            return ""
        last_error: Optional[Exception] = None
        for attempt in range(2):
            plan = self._adaptive_map_reduce_plan(
                text=text,
                final_tokens=final_tokens,
            )
            target_chunks = max(1, int(plan.get("target_chunks", 1)))
            reduce_batch_size = max(2, int(plan.get("reduce_batch_size", self.map_reduce_batch_size)))

            # Build large context-rich map chunks:
            # - If we have indexed chunks, merge them into target-sized groups.
            # - Otherwise split raw text evenly into target chunks.
            if doc_id and self._is_index_ready(doc_id) and source_chunks:
                chunks = self._merge_source_chunks_to_target(source_chunks, target_chunks=target_chunks)
            else:
                chunks = self._split_text_even(
                    text=text,
                    target_chunks=target_chunks,
                    overlap_chars=self.map_reduce_chunk_overlap,
                )

            if not chunks:
                return ""
            observability_service.record_map_reduce_plan(
                style=style,
                estimated_doc_tokens=int(plan.get("estimated_doc_tokens", 0)),
                input_budget_tokens=int(plan.get("input_budget_tokens", 0)),
                planned_chunks=target_chunks,
                planned_reduce_batch=reduce_batch_size,
                planned_reduce_rounds=int(plan.get("estimated_reduce_rounds", 0)),
            )

            try:
                logger.info(
                    "Map-reduce summary start style=%s chunks=%d est_doc_tokens=%d input_budget_tokens=%d reduce_batch=%d est_reduce_rounds=%d attempt=%d",
                    style,
                    len(chunks),
                    int(plan.get("estimated_doc_tokens", 0)),
                    int(plan.get("input_budget_tokens", 0)),
                    reduce_batch_size,
                    int(plan.get("estimated_reduce_rounds", 0)),
                    attempt + 1,
                )
                out = self.summary_pipeline.summarize(
                    chunks=chunks,
                    call_chat=self._chat_text,
                    base_system_prompt=self._base_system_prompt(),
                    map_instruction=map_instruction,
                    reduce_instruction=reduce_instruction,
                    final_instruction=final_instruction,
                    max_tokens=final_tokens,
                    temperature=final_temp,
                    max_chunks_override=target_chunks,
                    reduce_batch_size_override=reduce_batch_size,
                )
                logger.info("Map-reduce summary done style=%s output_chars=%d", style, len(out or ""))
                return out
            except Exception as e:
                last_error = e
                if attempt == 0 and self._is_context_limit_error(e):
                    self._shrink_context_window(reason="map-reduce-replan")
                    logger.warning("Retrying map-reduce with tighter context budget")
                    continue
                logger.exception("Map-reduce summary failed style=%s", style)
                return ""
        if last_error:
            logger.exception("Map-reduce summary failed after retries style=%s error=%s", style, str(last_error))
        return ""

    def _call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, stream: bool = False):
        sys_txt, usr_txt, out_tokens = self._fit_prompts_to_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            margin_tokens=900,
        )
        try:
            return self.provider.call_chat(sys_txt, usr_txt, out_tokens, temperature, stream=stream)
        except Exception as e:
            if self._is_context_limit_error(e):
                self._shrink_context_window(reason="context-limit-error")
                sys_txt2, usr_txt2, out_tokens2 = self._fit_prompts_to_context(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    margin_tokens=self.context_retry_margin_tokens,
                )
                try:
                    logger.warning("Retrying LLM call after context-limit adjustment")
                    return self.provider.call_chat(sys_txt2, usr_txt2, out_tokens2, temperature, stream=stream)
                except Exception as retry_error:
                    logger.exception("LLM retry failed after context-limit adjustment")
                    raise RuntimeError(f"LLM connection/request failed: {str(retry_error)}")
            logger.exception("LLM request failed")
            raise RuntimeError(f"LLM connection/request failed: {str(e)}")

    # ----------------------------
    # Dynamic section discovery (2 to 5)
    # ----------------------------
    def _discover_dynamic_sections(
        self,
        fitted_text: str,
        min_sections: int = 2,
        max_sections: int = 5,
        temperature: float = 0.2,
    ) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        min_sections = max(1, int(min_sections))
        max_sections = max(min_sections, int(max_sections))
        if not self.dynamic_sections_use_llm:
            try:
                heuristic = self.section_service._heuristic_sections_from_text(
                    fitted_text,
                    max_sections=max_sections,
                )
                if len(heuristic) >= min_sections:
                    return heuristic[:max_sections]
            except Exception:
                logger.exception("Heuristic section discovery failed; using fallback.")
            return [
                {"title": "General Summary", "hint": "What the document is about"},
                {"title": "Key Extracts", "hint": "Important names, dates, numbers"},
            ][:max_sections]
        try:
            return self.section_service.discover_dynamic_sections(
                fitted_text=fitted_text,
                min_sections=min_sections,
                max_sections=max_sections,
                temperature=temperature,
            )
        except Exception:
            logger.exception("Dynamic section discovery failed; using fallback.")
            return [
                {"title": "General Summary", "hint": "What the document is about"},
                {"title": "Key Extracts", "hint": "Important names, dates, numbers"},
            ][: max(1, int(max_sections))]

    # ----------------------------
    # Fact extraction with anchors (internal)
    # ----------------------------
    def _extract_facts_with_anchors(self, section_title: str, section_hint: str, fitted_text: str) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        try:
            return self.section_service.extract_facts_with_anchors(section_title, section_hint, fitted_text)
        except Exception:
            logger.exception("Failed to extract facts with anchors")
            return []

    def _validate_anchored_facts(self, facts: List[Dict[str, Any]], fitted_text: str) -> List[Dict[str, Any]]:
        return self.section_service.validate_anchored_facts(facts, fitted_text)

    # ----------------------------
    # Final section writing (user-facing, no anchors shown)
    # ----------------------------
    def _write_section_from_facts(self, section_title: str, facts: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        return self.section_service.write_section_from_facts(section_title, facts, max_tokens, temperature)

    # ----------------------------
    # Public API
    # ----------------------------
    def summarize_financial(
        self,
        text: str,
        mode: str = "financial_initial",
        section: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = False,
        doc_id: str = None,
    ) -> Union[str, Iterator[str]]:
        self._ensure_initialized()
        self._evict_cache_if_needed()

        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

        if mode not in ("financial_initial", "financial_section", "financial_overall", "financial_sectionwise"):
            raise ValueError(
                "mode must be one of: financial_initial, financial_section, financial_overall, financial_sectionwise"
            )

        if stream and mode != "financial_overall":
            raise ValueError("stream=True is only supported for mode='financial_overall'")

        if not text or not text.strip():
            return "No text found to summarize."

        if mode == "financial_initial":
            if (not (self.retrieval_first_enable and doc_id)) and self._should_use_map_reduce(text=text, doc_id=doc_id):
                mr = self._map_reduce_summary(
                    text=text,
                    doc_id=doc_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    style="initial",
                )
                if mr:
                    return normalize_money(clean_text(mr))
            mode_text = self._context_for_mode(
                text=text,
                mode=mode,
                max_output_tokens=max_tokens,
                doc_id=doc_id,
            )
            out = self._financial_initial(mode_text, max_tokens=max_tokens, temperature=temperature)
            return normalize_money(clean_text(out))

        if mode == "financial_overall":
            if not stream and (not (self.retrieval_first_enable and doc_id)) and self._should_use_map_reduce(text=text, doc_id=doc_id):
                mr = self._map_reduce_summary(
                    text=text,
                    doc_id=doc_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    style="overall",
                )
                if mr:
                    return normalize_money(clean_text(mr))
            mode_text = self._context_for_mode(
                text=text,
                mode=mode,
                max_output_tokens=max_tokens,
                doc_id=doc_id,
            )
            out = self._financial_overall(mode_text, max_tokens=max_tokens, temperature=temperature, stream=stream)
            if isinstance(out, str):
                return normalize_money(clean_text(out))
            return out

        if mode == "financial_section":
            # Dynamic: accept ANY section title.
            sec = self._normalize_section_title(section or "")
            if not sec:
                raise ValueError("section must be provided for mode='financial_section'")

            # Optional hint from discovery (helps retrieval)
            hint = ""
            try:
                if doc_id:
                    hint = self.get_doc_section_hint(doc_id, sec)
            except Exception:
                hint = ""

            mode_text = self._context_for_mode(
                text=text,
                mode=mode,
                max_output_tokens=max_tokens,
                doc_id=doc_id,
                section=sec,
                section_hint=hint,
            )

            # 1) Extract facts + anchors (internal)
            facts = self._extract_facts_with_anchors(sec, hint, mode_text)

            # 2) Validate anchors exist in document
            valid_facts = self._validate_anchored_facts(facts, mode_text)

            # If too few facts are anchored, treat as unsupported
            if facts:
                ratio = (len(valid_facts) / float(len(facts))) if len(facts) > 0 else 0.0
            else:
                ratio = 0.0

            if not valid_facts or ratio < self.min_anchored_fact_ratio:
                out = f"{sec}\n- No supported information found in the text for this section.\n"
                return normalize_money(clean_text(out))

            # 3) Write final section from validated facts (anchors NOT shown)
            out = self._write_section_from_facts(sec, valid_facts, max_tokens=max_tokens, temperature=temperature)
            out = normalize_money(clean_text(out))
            out = dedupe_section_heading(out, sec)
            return out

        # financial_sectionwise: keep a generic structured brief (still useful for some flows)
        mode_text = self._context_for_mode(
            text=text,
            mode=mode,
            max_output_tokens=max_tokens,
            doc_id=doc_id,
        )
        out = self._financial_sectionwise(mode_text, max_tokens=max_tokens, temperature=temperature)
        return normalize_money(clean_text(out))

    # ----------------------------
    # Implementations
    # ----------------------------
    def _financial_initial(self, text: str, max_tokens: int, temperature: float) -> str:
        init_max = max(160, min(int(max_tokens), 280))
        temperature_init = max(0.0, min(float(temperature), 0.25))

        system_prompt = self._base_system_prompt() + (
            "\nWrite a short generalized summary that tells the user what this document is about.\n"
            "Focus on: document type, company/entity, reporting period/date, and purpose ONLY if explicitly stated.\n"
            "Rules:\n"
            "- Plain text only.\n"
            "- 3 to 5 sentences.\n"
            "- Prefer exact names, dates, and periods when present.\n"
            "- Include key numeric highlights ONLY if explicitly present.\n"
        )

        user_prompt = f"""Write the generalized summary for this document.

Document:
{text}
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=init_max, temperature=temperature_init, stream=False)
        return self._clean_or_retry_summary(
            draft=extract_response_text(resp),
            source_text=text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=init_max,
            temperature=temperature_init,
        )

    def _financial_overall(self, text: str, max_tokens: int, temperature: float, stream: bool) -> Union[str, Iterator[str]]:
        temperature_overall = max(0.0, min(float(temperature), 0.35))

        system_prompt = self._base_system_prompt() + (
            "\nCreate a concise summary for a non-expert user.\n"
            "Prefer concrete numbers and dates when present.\n"
            "Keep it readable.\n"
            "\nRules:\n"
            "- Do not invent.\n"
            "- If the document is an article or non-financial, summarize its key claims and any numbers.\n"
        )

        user_prompt = f"""Create a neat summary of the document below.

Document:
{text}

Write 8 to 14 bullet points max.
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature_overall, stream=stream)
        if stream:
            return self._stream_response(resp)
        return self._clean_or_retry_summary(
            draft=extract_response_text(resp),
            source_text=text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature_overall,
        )

    def _financial_sectionwise(self, text: str, max_tokens: int, temperature: float) -> str:
        temperature_structured = max(0.0, min(float(temperature), 0.35))

        system_prompt = self._base_system_prompt() + (
            "\nCreate a short section-wise brief.\n"
            "Since the document type may be unknown, choose sensible headings that match the content.\n"
            "Do not invent.\n"
            "Plain text only.\n"
        )

        user_prompt = f"""Create a short section-wise brief for this document.

Document:
{text}

Return the section-wise brief now.
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature_structured, stream=False)
        return extract_response_text(resp)

    def _stream_response(self, response) -> Iterator[str]:
        accumulated = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                accumulated += delta.content
                if accumulated.endswith((".", "!", "?", "\n")):
                    yield normalize_money(clean_text(accumulated))
                    accumulated = ""
        if accumulated:
            yield normalize_money(clean_text(accumulated))

    def health_check(self) -> Dict[str, Any]:
        try:
            self._ensure_initialized()
            out = self.provider.health_check()
            if out.get("status") == "healthy":
                out["model_context_tokens"] = self.model_context_tokens
            return out
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "provider": self.get_provider_name(), "error": str(e)}


llm_service = LLMService()
