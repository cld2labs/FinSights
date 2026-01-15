"""
LLM Service for Financial Document Summarization (FinSights)
Uses OpenAI Chat Completions API (gpt-4o-mini)

Updated to improve section quality (context-aware, not keyword-dependent):

Key changes:
- Stronger, section-specific definitions (semantic mapping) so the model can classify content
  even when the document does not have matching headings.
- Output style aligned to your examples:
  - Financial Performance: short narrative + key numbers
  - Key Metrics: clean metric list
  - Risks / Opportunities: if not explicit, allow "potential" risks/opportunities ONLY when
    grounded in the document's numbers/structure; otherwise say not explicitly stated.
  - Outlook / Guidance: if no forward-looking info, state it clearly.
  - Other Important Highlights: notable balance sheet/cash/dividends/notes/auditor items.

Notes:
- Uses stored extracted text (doc_id flow) and does NOT re-run pdf_service on section clicks.
- If document exceeds context, it is truncated safely.
"""

from typing import Iterator, Dict, Any, Optional, Union
import logging
import re
import os
import time
import uuid
import hashlib

import config
from openai import OpenAI

logger = logging.getLogger(__name__)

SECTION_OPTIONS = [
    "Financial Performance",
    "Key Metrics",
    "Risks",
    "Opportunities",
    "Outlook / Guidance",
    "Other Important Highlights",
]
SECTION_TITLES = set(SECTION_OPTIONS)


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
    body = [ln for ln in lines if ln != section]
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

        self.doc_store[doc_id] = {"ts": time.time(), "text": text, "doc_key": dk}
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
        )

    def initial_summary_first_chunk(self, doc_id: str, max_tokens: int = 240, temperature: float = 0.25) -> str:
        """
        Compatibility method expected by your routes/frontend.
        Uses the full doc text (fitted to context) to produce 4-5 sentences.
        """
        self._ensure_initialized()
        text = self.get_doc_text(doc_id)
        fitted = self._fit_text_to_context(text, max_output_tokens=max_tokens)

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

        user_prompt = f"""Write the generalized summary for this financial document.

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
            "You are a financial analyst AI.\n"
            "You must use ONLY information supported by the provided document text.\n"
            "Do not invent facts, numbers, dates, names, or events.\n"
            "If you are uncertain, state it clearly.\n"
            "Output must be plain text only (no markdown emphasis: no **, *, _).\n"
        )

    @staticmethod
    def _section_definitions() -> str:
        # These are semantic "signals" (not strict keywords) to help the model map content.
        return (
            "SECTION DEFINITIONS (use as semantic mapping even if headings differ):\n"
            "1) Financial Performance:\n"
            "   - Revenue/turnover/income, expenses/costs, profit/loss (PBT/PAT), EBITDA, margins, cash flow, drivers.\n"
            "2) Key Metrics:\n"
            "   - Ratios and headline KPIs: current ratio, debt-to-equity, ROE/ROA, margins, EPS, cash balance, total assets,\n"
            "     borrowings, working capital, capex, dividend amounts, YoY growth rates.\n"
            "3) Risks:\n"
            "   - Explicit risks: litigation, contingent liabilities, going concern, defaults, liquidity/credit/market/FX risks,\n"
            "     regulatory issues, internal controls, audit qualifications.\n"
            "   - If NOT explicitly stated, you may list 'Potential risks' ONLY when grounded in the document (e.g., high leverage,\n"
            "     large borrowings, heavy capex, large receivables, concentration, material contractual obligations).\n"
            "4) Opportunities:\n"
            "   - Pipeline/backlog/order book, new contracts/wins, expansion plans, investments, partnerships, growth initiatives,\n"
            "     margin improvement levers, strong operating cash enabling reinvestment.\n"
            "   - If NOT explicitly stated, you may list 'Potential opportunities' ONLY when grounded in the document (e.g., strong cash\n"
            "     from operations, improving margins, strong equity base).\n"
            "5) Outlook / Guidance:\n"
            "   - Forward-looking statements: management expectations, forecasts/targets, budgets, next year plans, guidance.\n"
            "   - If absent, say: 'The document does not provide forward-looking guidance.'\n"
            "6) Other Important Highlights:\n"
            "   - Notable items from balance sheet/cash flows/notes: total assets, PPE, cash & equivalents, dividends,\n"
            "     auditor/notes/standards compliance, major expense movements, finance costs, retained earnings, significant accounting notes.\n"
        )

    # ----------------------------
    # Core helpers
    # ----------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4)) if text else 0

    @staticmethod
    def _doc_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

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
        Keeps head (context) + tail (notes/auditor) when truncating.
        """
        if not text:
            return ""

        overhead_tokens = 900  # section definitions + instructions
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
            if not section or section.strip() not in SECTION_TITLES:
                raise ValueError(f"section must be one of: {', '.join(SECTION_OPTIONS)}")
            sec = section.strip()
            out = self._financial_section(fitted_text, section=sec, max_tokens=max_tokens, temperature=temperature)
            out = _normalize_money(clean_text(out))
            out = _dedupe_section_heading(out, sec)
            return out

        out = self._financial_sectionwise(fitted_text, max_tokens=max_tokens, temperature=temperature)
        return _normalize_money(clean_text(out))

    # ----------------------------
    # Implementations (context-aware)
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

        user_prompt = f"""Write the generalized summary for this financial document.

Document:
{text}
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=init_max, temperature=temperature_init, stream=False)
        return resp.choices[0].message.content or ""

    def _financial_overall(self, text: str, max_tokens: int, temperature: float, stream: bool) -> Union[str, Iterator[str]]:
        temperature_overall = max(0.0, min(float(temperature), 0.35))

        system_prompt = self._base_system_prompt() + (
            "\nCreate a numbers-first financial summary for a non-expert user.\n"
            "Prefer concrete numbers and dates when present.\n"
            "Keep it readable and structured.\n"
            "\n"
            + self._section_definitions()
            + "\nRules:\n"
              "- If the document is mainly historical statements, say so.\n"
              "- Do not add speculative risks/opportunities unless explicitly labeled as potential and grounded.\n"
        )

        user_prompt = f"""Create a neat financial summary of the document below.

Document:
{text}

Write 8 to 14 bullet points max, mixing performance + balance sheet + cash flow + notable notes.
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature_overall, stream=stream)
        if stream:
            return self._stream_response(resp)
        return resp.choices[0].message.content or ""

    def _financial_section(self, text: str, section: str, max_tokens: int, temperature: float) -> str:
        temperature_section = max(0.0, min(float(temperature), 0.3))

        section_specific_rules = (
            "OUTPUT STYLE (match these patterns):\n"
            "- Financial Performance: 3 to 6 sentences narrative + include key numbers (revenue, profit, margins, cash flow) if present.\n"
            "- Key Metrics: list metrics each on its own line. Format: 'Metric: value (short meaning)'.\n"
            "- Risks: if explicit risks exist, list them. If not explicit, you may list up to 3 'Potential risks' ONLY if grounded in the document's numbers/structure.\n"
            "- Opportunities: same approach as Risks. If not explicit, list up to 3 'Potential opportunities' ONLY if grounded.\n"
            "- Outlook / Guidance: if absent, clearly state it in 1 to 2 lines.\n"
            "- Other Important Highlights: list 4 to 8 notable highlights with numbers where available.\n"
        )

        system_prompt = self._base_system_prompt() + "\n" + self._section_definitions() + "\n" + section_specific_rules + (
            "\nRules:\n"
            "- Use the heading exactly as requested.\n"
            "- Do not invent. Use only supported information from the document.\n"
            "- If you include 'Potential' items, they MUST be tied to specific figures or statements in the text.\n"
            "- If there is truly no supported info, output exactly:\n"
            f"{section}\n- No supported information found in the text for this section.\n"
        )

        user_prompt = f"""Requested section: {section}

Document:
{text}

Output format:
{section}
<content>
"""
        resp = self._call_chat(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature_section, stream=False)
        out = resp.choices[0].message.content or ""
        out = clean_text(out)
        out = _dedupe_section_heading(out, section)
        return out

    def _financial_sectionwise(self, text: str, max_tokens: int, temperature: float) -> str:
        temperature_structured = max(0.0, min(float(temperature), 0.35))

        system_prompt = self._base_system_prompt() + "\n" + self._section_definitions() + (
            "\nCreate a section-wise financial brief using EXACT headings below.\n"
            "Headings may not exist in the document; classify content semantically using the definitions.\n"
            "\nRules:\n"
            "- Use ONLY supported information from the document.\n"
            "- Do not invent.\n"
            "- Omit a section entirely if there is no supported content for it.\n"
            "- Plain text only.\n"
            "\nOutput headings (omit empty ones):\n"
            "Financial Performance\n"
            "Key Metrics\n"
            "Risks\n"
            "Opportunities\n"
            "Outlook / Guidance\n"
            "Other Important Highlights\n"
            "\nFormatting:\n"
            "- Financial Performance: short narrative.\n"
            "- Key Metrics: one metric per line.\n"
            "- Others: short lines or bullets.\n"
        )

        user_prompt = f"""Create a numbers-first section-wise brief for this document.

Document:
{text}

Return the final section-wise brief now.
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
