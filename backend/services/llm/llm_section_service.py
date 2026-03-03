from __future__ import annotations

import json
import re
from typing import Callable, Dict, Any, List

from .llm_text_utils import clean_text, dedupe_section_heading, extract_response_text


class LLMSectionService:
    def __init__(
        self,
        call_chat: Callable[[str, str, int, float, bool], Any],
        base_system_prompt: Callable[[], str],
        normalize_section_title: Callable[[str], str],
        facts_max_items: int,
        anchor_max_items: int,
        anchor_max_chars_each: int,
    ) -> None:
        self.call_chat = call_chat
        self.base_system_prompt = base_system_prompt
        self.normalize_section_title = normalize_section_title
        self.facts_max_items = int(facts_max_items)
        self.anchor_max_items = int(anchor_max_items)
        self.anchor_max_chars_each = int(anchor_max_chars_each)

    def discover_dynamic_sections(
        self,
        fitted_text: str,
        min_sections: int = 2,
        max_sections: int = 5,
        temperature: float = 0.2,
    ) -> List[Dict[str, Any]]:
        min_sections = max(1, min(int(min_sections), 6))
        max_sections = max(min_sections, min(int(max_sections), 8))

        system_prompt = (
            self.base_system_prompt()
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

        items: List[Dict[str, Any]] = []
        for attempt in range(2):
            prompt_to_use = user_prompt
            temp = max(0.0, min(float(temperature), 0.35)) if attempt == 0 else 0.0
            max_tok = 350 if attempt == 0 else 220
            if attempt == 1:
                prompt_to_use = (
                    "Return ONLY strict JSON. No prose. No markdown.\n"
                    "Schema:\n"
                    '{"sections":[{"title":"Short Title","hint":"Short description"}]}\n\n'
                    f"Document:\n{fitted_text}"
                )

            resp = self.call_chat(
                system_prompt,
                prompt_to_use,
                max_tokens=max_tok,
                temperature=temp,
                stream=False,
            )
            raw = extract_response_text(resp).strip()
            items = self._normalize_sections(self._parse_json_loose(raw).get("sections", []))
            if len(items) >= min_sections:
                break

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

        heuristic = self._heuristic_sections_from_text(fitted_text, max_sections=max_sections)
        if len(heuristic) >= min_sections:
            return heuristic[:max_sections]

        fallback = [
            {"title": "General Summary", "hint": "What the document is about"},
            {"title": "Key Extracts", "hint": "Important names, dates, numbers"},
        ]
        return fallback[:max_sections]

    def extract_facts_with_anchors(self, section_title: str, section_hint: str, fitted_text: str) -> List[Dict[str, Any]]:
        title = self.normalize_section_title(section_title)
        hint = (section_hint or "").strip()
        if not title:
            return []

        system_prompt = (
            self.base_system_prompt()
            + "\nTask: extract readable key facts for a requested section.\n"
              "Facts must be supported by the document.\n"
              "\nCRITICAL RULES:\n"
              "- Return STRICT JSON only.\n"
              "- Facts must be written in plain English (not copied verbatim).\n"
              "- Each fact MUST include at least one anchor.\n"
              "- Anchors must be SHORT strings that appear verbatim in the document.\n"
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
        facts: List[Dict[str, Any]] = []
        for attempt in range(2):
            prompt_to_use = user_prompt
            temp = 0.15 if attempt == 0 else 0.0
            max_tok = 650 if attempt == 0 else 420
            if attempt == 1:
                prompt_to_use = (
                    f"Requested section: {title}\n"
                    f"Section hint: {hint}\n"
                    "Return ONLY strict JSON. No prose. No markdown.\n"
                    'Schema: {"facts":[{"point":"Readable point","anchors":["anchor1","anchor2"]}]}\n'
                    f"Constraints: up to {self.facts_max_items} facts, 1 to {self.anchor_max_items} anchors each.\n\n"
                    f"Document:\n{fitted_text}"
                )

            resp = self.call_chat(system_prompt, prompt_to_use, max_tokens=max_tok, temperature=temp, stream=False)
            raw = extract_response_text(resp).strip()
            facts = self._normalize_facts(self._parse_json_loose(raw).get("facts", []))
            if facts:
                break

        out: List[Dict[str, Any]] = []
        seen = set()
        for f in facts:
            key = re.sub(r"\s+", " ", f["point"]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _normalize_sections(self, secs: Any) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if not isinstance(secs, list):
            return items
        for s in secs:
            if not isinstance(s, dict):
                continue
            title = self.normalize_section_title(str(s.get("title", "")).strip())
            hint = str(s.get("hint", "") or "").strip()
            hint = re.sub(r"\s{2,}", " ", hint)
            if not title:
                continue
            if len(hint) > 90:
                hint = hint[:90].rstrip()
            items.append({"title": title, "hint": hint})
        return items

    def _normalize_facts(self, items: Any) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return facts
        for it in items:
            if not isinstance(it, dict):
                continue
            point = str(it.get("point", "") or "").strip()
            anchors = it.get("anchors", [])
            if not point or not isinstance(anchors, list):
                continue
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
        return facts

    @staticmethod
    def validate_anchored_facts(facts: List[Dict[str, Any]], fitted_text: str) -> List[Dict[str, Any]]:
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

    def write_section_from_facts(self, section_title: str, facts: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        title = self.normalize_section_title(section_title)

        if not facts:
            return f"{title}\n- No supported information found in the text for this section.\n"

        fact_lines: List[str] = []
        for f in facts:
            p = str(f.get("point", "") or "").strip()
            if p:
                fact_lines.append(f"- {p}")
        if not fact_lines:
            return f"{title}\n- No supported information found in the text for this section.\n"

        system_prompt = self.base_system_prompt() + (
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
        resp = self.call_chat(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=max(0.0, min(float(temperature), 0.3)),
            stream=False,
        )
        out = clean_text(extract_response_text(resp))
        return dedupe_section_heading(out, title)

    @staticmethod
    def _parse_json_loose(raw: str) -> Dict[str, Any]:
        txt = (raw or "").strip()
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except Exception:
            pass
        # Remove common markdown fences
        txt2 = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE).strip()
        txt2 = re.sub(r"\s*```$", "", txt2).strip()
        try:
            return json.loads(txt2)
        except Exception:
            pass
        # Fallback: parse first JSON object span
        s = txt2.find("{")
        e = txt2.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(txt2[s : e + 1])
            except Exception:
                return {}
        return {}

    def _heuristic_sections_from_text(self, text: str, max_sections: int) -> List[Dict[str, str]]:
        t = (text or "").lower()
        out: List[Dict[str, str]] = []

        def add(title: str, hint: str) -> None:
            if any(x["title"].lower() == title.lower() for x in out):
                return
            out.append({"title": title, "hint": hint})

        # Prefer document-structure driven chips for weak models.
        if any(k in t for k in ("balance sheet", "statement of financial position")):
            add("Balance Sheet", "Assets, liabilities, and equity snapshot")
        if any(k in t for k in ("income statement", "profit and loss", "statement of operations")):
            add("Income Statement", "Revenue, expenses, and net income")
        if any(k in t for k in ("cash flow", "cash flows")):
            add("Cash Flow", "Operating, investing, and financing cash movement")
        if any(k in t for k in ("retained earnings", "shareholder", "equity")):
            add("Equity & Retained", "Changes in equity and retained earnings")
        if any(k in t for k in ("notes", "accounting policy", "policies")):
            add("Notes & Policies", "Important notes and accounting assumptions")
        if any(k in t for k in ("tax", "gst", "vat", "income tax")):
            add("Taxes", "Tax-related details and amounts")
        if any(k in t for k in ("debt", "loan", "borrow", "interest")):
            add("Debt & Financing", "Borrowings, interest, and funding details")
        if any(k in t for k in ("invoice", "payable", "receivable")):
            add("Billing & Payments", "Invoices, receivables, and payment status")

        if not out:
            out = [
                {"title": "General Summary", "hint": "What the document is about"},
                {"title": "Key Figures", "hint": "Important amounts, dates, and ratios"},
                {"title": "Entities & Dates", "hint": "Main names, periods, and timelines"},
            ]

        return out[: max(1, int(max_sections))]
