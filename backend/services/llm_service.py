"""
LLM Service for Document Summarization (FinSights)
Uses OpenAI Chat Completions API (gpt-4o-mini)

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
import json

import config
from openai import OpenAI

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _normalize_money(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\bRs\.\s*", "₹ ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bINR\s*", "₹ ", text, flags=re.IGNORECASE)
    return text


def _dedupe_section_heading(text: str, section: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    body = [ln for ln in lines if ln.lower() != section.lower()]
    out = section + "\n" + "\n".join(body)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


class LLMService:
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.model = os.getenv("OPENAI_MODEL", None) or getattr(config, "OPENAI_MODEL", None) or "gpt-4o-mini"
        self._initialized = False

        # Large default; override via env if needed.
        self.model_context_tokens = int(os.getenv("MODEL_CONTEXT_TOKENS", "128000"))

        # In-memory doc store for doc_id flow
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60)))  # 1 hour
        self.cache_max_docs = int(os.getenv("CACHE_MAX_DOCS", "25"))

        # Dynamic section discovery bounds (frontend chips)
        self.dynamic_min_sections = int(os.getenv("DYNAMIC_SECTIONS_MIN", "2"))
        self.dynamic_max_sections = int(os.getenv("DYNAMIC_SECTIONS_MAX", "5"))

        # Evidence/facts extraction bounds
        self.facts_max_items = int(os.getenv("FACTS_MAX_ITEMS", "10"))
        self.anchor_max_items = int(os.getenv("ANCHOR_MAX_ITEMS", "3"))
        self.anchor_max_chars_each = int(os.getenv("ANCHOR_MAX_CHARS_EACH", "60"))

        # Validation threshold: how many facts must be anchored to proceed
        # Example: 0.6 means at least 60% of extracted facts must have >=1 valid anchor.
        self.min_anchored_fact_ratio = float(os.getenv("MIN_ANCHORED_FACT_RATIO", "0.6"))

    def _ensure_initialized(self):
        if self._initialized:
            return
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )
        self._initialized = True

        logger.info("OpenAI client initialized successfully")
        logger.info(f"Model: {self.model}")
        logger.info(f"MODEL_CONTEXT_TOKENS: {self.model_context_tokens}")
        logger.info(f"CACHE_MAX_DOCS: {self.cache_max_docs}")
        logger.info(f"CACHE_TTL_SECONDS: {self.cache_ttl_seconds}")
        logger.info(f"DYNAMIC_SECTIONS_MIN: {self.dynamic_min_sections}")
        logger.info(f"DYNAMIC_SECTIONS_MAX: {self.dynamic_max_sections}")
        logger.info(f"FACTS_MAX_ITEMS: {self.facts_max_items}")
        logger.info(f"ANCHOR_MAX_ITEMS: {self.anchor_max_items}")
        logger.info(f"MIN_ANCHORED_FACT_RATIO: {self.min_anchored_fact_ratio}")

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

    def validate_finance_document(self, text: str) -> dict:
        """
        Validate if the document contains finance-related information.
        
        Returns:
            {
                "is_finance_related": bool,
                "confidence": float (0.0 to 1.0),
                "message": str
            }
        """
        self._ensure_initialized()
        
        if not text or not text.strip():
            return {
                "is_finance_related": False,
                "confidence": 1.0,
                "message": "Document is empty"
            }
        
        # Use first 2000 characters for quick validation
        sample_text = text[:2000]
        
        system_prompt = (
            "You are a document classifier.\n"
            "Your task is to determine if a document contains finance-related information.\n"
            "Finance-related documents include: invoices, receipts, bank statements, financial reports, "
            "tax documents, expense reports, budgets, balance sheets, income statements, payroll records, "
            "loan documents, investment statements, insurance documents, accounting records, audit reports, "
            "financial forecasts, quarterly/annual reports, expense tracking, and similar financial documents.\n"
            "Respond with ONLY valid JSON (no markdown, no code blocks):\n"
            '{"is_finance": true/false, "confidence": 0.0-1.0, "reason": "brief reason"}\n'
        )
        
        user_prompt = f"""Analyze this document sample and determine if it contains finance-related information:

{sample_text}

Respond with ONLY the JSON object."""
        
        try:
            resp = self._call_chat(
                system_prompt,
                user_prompt,
                max_tokens=200,
                temperature=0.0,
                stream=False,
            )
            
            response_text = resp.choices[0].message.content or ""
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            result = json.loads(response_text)
            
            is_finance = result.get("is_finance", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")
            
            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "is_finance_related": bool(is_finance),
                "confidence": confidence,
                "message": reason
            }
            
        except Exception as e:
            logger.error(f"Finance document validation error: {str(e)}")
            # On error, assume not finance to be safe
            return {
                "is_finance_related": False,
                "confidence": 0.0,
                "message": f"Validation error: {str(e)}"
            }

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
        fitted = self._fit_text_to_context(text, max_output_tokens=max_tokens)

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
                obj["ts"] = time.time()
        except Exception:
            logger.exception("Dynamic section discovery failed (continuing with summary only).")

        # Normal initial summary
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
        return _normalize_money(clean_text(resp.choices[0].message.content or ""))

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
        return _normalize_money(clean_text(resp.choices[0].message.content or ""))


    # ----------------------------
    # Core helpers
    # ----------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4)) if text else 0

    @staticmethod
    def _doc_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

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

    def _call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, stream: bool = False):
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
        except Exception as e:
            logger.exception("OpenAI request failed")
            raise RuntimeError(f"OpenAI connection/request failed: {str(e)}")

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
        """
        Returns 2 to 5 section items derived from the document content.
        This is for UI chips. We avoid quotes here; we keep short hints only.

        Output item shape:
        - {"title": "Invoice Totals", "hint": "Totals, taxes, payment status"}
        """
        self._ensure_initialized()

        min_sections = max(1, min(int(min_sections), 6))
        max_sections = max(min_sections, min(int(max_sections), 8))

        system_prompt = (
            self._base_system_prompt()
            + "\nYou propose section options (chips) for a summarization UI.\n"
              "The user can upload ANY document: invoices, payroll, tax returns, audit reports, loan documents, or general articles.\n"
              "Propose only sections that are likely supported by the text.\n"
              "\nCRITICAL RULES:\n"
              f"- Return between {min_sections} and {max_sections} sections.\n"
              "- Use short titles (2 to 5 words). No numbering.\n"
              "- Provide a short hint (6 to 12 words) describing what to expect.\n"
              "- Do not invent specific numbers.\n"
              "- Output MUST be strict JSON only.\n"
        )

        user_prompt = f"""Read the document and propose section chips.

Return JSON:
{{
  "sections": [
    {{
      "title": "Short Title",
      "hint": "Short description"
    }}
  ]
}}

Document:
{fitted_text}
"""

        resp = self._call_chat(
            system_prompt,
            user_prompt,
            max_tokens=350,
            temperature=max(0.0, min(float(temperature), 0.35)),
            stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()

        items: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
            secs = data.get("sections", [])
            if isinstance(secs, list):
                for s in secs:
                    if not isinstance(s, dict):
                        continue
                    title = self._normalize_section_title(str(s.get("title", "")).strip())
                    hint = str(s.get("hint", "") or "").strip()
                    hint = re.sub(r"\s{2,}", " ", hint)
                    if not title:
                        continue
                    if len(hint) > 90:
                        hint = hint[:90].rstrip()
                    items.append({"title": title, "hint": hint})
        except Exception:
            logger.exception("Failed to parse dynamic sections JSON; using fallback.")

        # de-dupe titles
        out: List[Dict[str, Any]] = []
        seen = set()
        for it in items:
            key = it["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(it)

        if len(out) >= min_sections:
            return out[:max_sections]

        # Fallback for low-signal docs
        fallback = [
            {"title": "General Summary", "hint": "What the document is about"},
            {"title": "Key Extracts", "hint": "Important names, dates, numbers"},
        ]
        return fallback[:max_sections]

    # ----------------------------
    # Fact extraction with anchors (internal)
    # ----------------------------
    def _extract_facts_with_anchors(self, section_title: str, section_hint: str, fitted_text: str) -> List[Dict[str, Any]]:
        """
        Extract readable facts (paraphrased) relevant to a requested section.
        Each fact must include at least one short anchor that appears in the document text.
        Anchors are used ONLY for validation and are not shown in the final user output.
        """
        title = self._normalize_section_title(section_title)
        hint = (section_hint or "").strip()

        if not title:
            return []

        system_prompt = (
            self._base_system_prompt()
            + "\nTask: extract readable key facts for a requested section.\n"
              "Facts must be supported by the document.\n"
              "\nCRITICAL RULES:\n"
              "- Return STRICT JSON only.\n"
              "- Facts must be written in plain English (not copied verbatim).\n"
              "- Each fact MUST include at least one anchor.\n"
              "- Anchors must be SHORT strings that appear verbatim in the document (examples: a number, a date, an entity name, a short label).\n"
              "- Do NOT return long quotes.\n"
        )

        user_prompt = f"""Requested section: {title}
Section hint (if any): {hint}

Return JSON:
{{
  "facts": [
    {{
      "point": "Readable summarized point",
      "anchors": ["anchor1", "anchor2"]
    }}
  ]
}}

Rules:
- Provide up to {self.facts_max_items} facts.
- Each fact must include 1 to {self.anchor_max_items} anchors.
- Anchors must be <= {self.anchor_max_chars_each} characters.
- Prefer anchors that include numbers, dates, totals, ratios, names, account labels, or table row labels.
- Do not invent any numbers.

Document:
{fitted_text}
"""

        resp = self._call_chat(system_prompt, user_prompt, max_tokens=650, temperature=0.15, stream=False)
        raw = (resp.choices[0].message.content or "").strip()

        facts: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
            items = data.get("facts", [])
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    point = str(it.get("point", "") or "").strip()
                    anchors = it.get("anchors", [])
                    if not point or not isinstance(anchors, list):
                        continue

                    # normalize anchors
                    norm_anchors: List[str] = []
                    for a in anchors[: self.anchor_max_items]:
                        s = str(a or "").strip()
                        if not s:
                            continue
                        if len(s) > self.anchor_max_chars_each:
                            s = s[: self.anchor_max_chars_each].rstrip()
                        norm_anchors.append(s)

                    if not norm_anchors:
                        continue

                    facts.append({"point": point, "anchors": norm_anchors})
        except Exception:
            logger.exception("Failed to parse facts JSON from extractor")
            return []

        # de-dupe similar points
        out: List[Dict[str, Any]] = []
        seen = set()
        for f in facts:
            key = re.sub(r"\s+", " ", f["point"]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _validate_anchored_facts(self, facts: List[Dict[str, Any]], fitted_text: str) -> List[Dict[str, Any]]:
        """
        Keep only facts that have at least one anchor substring present in fitted_text.
        """
        if not facts:
            return []

        valid: List[Dict[str, Any]] = []
        for f in facts:
            anchors = f.get("anchors", [])
            if not isinstance(anchors, list):
                continue
            ok = False
            for a in anchors:
                if not a:
                    continue
                if str(a) in fitted_text:
                    ok = True
                    break
            if ok:
                valid.append(f)
        return valid

    # ----------------------------
    # Final section writing (user-facing, no anchors shown)
    # ----------------------------
    def _write_section_from_facts(self, section_title: str, facts: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        title = self._normalize_section_title(section_title)

        if not facts:
            return f"{title}\n- No supported information found in the text for this section.\n"

        # Build fact lines (anchors not shown)
        fact_lines: List[str] = []
        for f in facts:
            p = str(f.get("point", "") or "").strip()
            if p:
                fact_lines.append(f"- {p}")

        if not fact_lines:
            return f"{title}\n- No supported information found in the text for this section.\n"

        system_prompt = self._base_system_prompt() + (
            "\nWrite the requested section using ONLY the provided facts.\n"
            "Do not invent.\n"
            "\nOUTPUT RULES:\n"
            "- Start with the heading exactly as provided.\n"
            "- Keep it easy to read.\n"
            "- Use bullets (recommended) or short paragraphs.\n"
            "- Keep numbers when present in facts.\n"
            "- If facts are weak, explicitly say it is limited.\n"
        )

        user_prompt = f"""Section heading: {title}

Facts (use only these):
{chr(10).join(fact_lines)}

Write the section now.
"""

        resp = self._call_chat(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=max(0.0, min(float(temperature), 0.3)),
            stream=False,
        )
        out = clean_text(resp.choices[0].message.content or "")
        out = _dedupe_section_heading(out, title)
        return out

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

        fitted_text = self._fit_text_to_context(text, max_output_tokens=max_tokens)

        if mode == "financial_initial":
            out = self._financial_initial(fitted_text, max_tokens=max_tokens, temperature=temperature)
            return _normalize_money(clean_text(out))

        if mode == "financial_overall":
            out = self._financial_overall(fitted_text, max_tokens=max_tokens, temperature=temperature, stream=stream)
            if isinstance(out, str):
                return _normalize_money(clean_text(out))
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

            # 1) Extract facts + anchors (internal)
            facts = self._extract_facts_with_anchors(sec, hint, fitted_text)

            # 2) Validate anchors exist in document
            valid_facts = self._validate_anchored_facts(facts, fitted_text)

            # If too few facts are anchored, treat as unsupported
            if facts:
                ratio = (len(valid_facts) / float(len(facts))) if len(facts) > 0 else 0.0
            else:
                ratio = 0.0

            if not valid_facts or ratio < self.min_anchored_fact_ratio:
                out = f"{sec}\n- No supported information found in the text for this section.\n"
                return _normalize_money(clean_text(out))

            # 3) Write final section from validated facts (anchors NOT shown)
            out = self._write_section_from_facts(sec, valid_facts, max_tokens=max_tokens, temperature=temperature)
            out = _normalize_money(clean_text(out))
            out = _dedupe_section_heading(out, sec)
            return out

        # financial_sectionwise: keep a generic structured brief (still useful for some flows)
        out = self._financial_sectionwise(fitted_text, max_tokens=max_tokens, temperature=temperature)
        return _normalize_money(clean_text(out))

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
        return resp.choices[0].message.content or ""

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
        return resp.choices[0].message.content or ""

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
        return resp.choices[0].message.content or ""

    def _stream_response(self, response) -> Iterator[str]:
        accumulated = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                accumulated += delta.content
                if accumulated.endswith((".", "!", "?", "\n")):
                    yield _normalize_money(clean_text(accumulated))
                    accumulated = ""
        if accumulated:
            yield _normalize_money(clean_text(accumulated))

    def health_check(self) -> Dict[str, Any]:
        try:
            if not config.OPENAI_API_KEY:
                return {"status": "not_configured", "provider": "OpenAI", "message": "OPENAI_API_KEY not configured"}

            self._ensure_initialized()
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10,
                temperature=0,
            )
            return {
                "status": "healthy",
                "provider": "OpenAI",
                "model": self.model,
                "model_context_tokens": self.model_context_tokens,
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "provider": "OpenAI", "error": str(e)}


llm_service = LLMService()
