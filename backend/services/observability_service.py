from __future__ import annotations

import os
import time
import threading
from collections import deque
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

_endpoint_ctx: ContextVar[str] = ContextVar("obs_endpoint", default="-")
_method_ctx: ContextVar[str] = ContextVar("obs_method", default="-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            return {}
    out: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, key):
            out[key] = getattr(usage, key)
    return out


def _estimate_tokens_from_chars(char_count: int) -> int:
    # Practical rough estimate for observability dashboards.
    if char_count <= 0:
        return 0
    return max(1, char_count // 4)


class ObservabilityService:
    def __init__(self) -> None:
        self.max_rows = max(100, int(os.getenv("OBSERVABILITY_MAX_ROWS", "1000")))
        self.rows: Deque[Dict[str, Any]] = deque(maxlen=self.max_rows)
        self.lock = threading.Lock()

    def set_request_context(self, endpoint: str, method: str) -> Tuple[Token, Token]:
        ep = (endpoint or "-").strip() or "-"
        m = (method or "-").strip().upper() or "-"
        return _endpoint_ctx.set(ep), _method_ctx.set(m)

    def reset_request_context(self, tokens: Tuple[Token, Token]) -> None:
        ep_token, method_token = tokens
        _endpoint_ctx.reset(ep_token)
        _method_ctx.reset(method_token)

    def _current_endpoint(self) -> str:
        return f"{_method_ctx.get()} {_endpoint_ctx.get()}".strip()

    def _append(self, row: Dict[str, Any]) -> None:
        with self.lock:
            self.rows.appendleft(row)

    def record_request(self, status_code: int, duration_ms: float) -> None:
        self._append(
            {
                "time": _now_iso(),
                "endpoint": self._current_endpoint(),
                "event": "request",
                "model": "-",
                "planned_chunks": 0,
                "planned_reduce_batch": 0,
                "planned_reduce_rounds": 0,
                "input_budget_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "user_prompt_tokens": 0,
                "system_prompt_tokens": 0,
                "uploaded_document_tokens": 0,
                "user_input_tokens": 0,
                "duration_ms": round(float(duration_ms), 2),
                "status": str(status_code),
            }
        )

    def record_llm_call(
        self,
        *,
        event: str,
        model: str,
        provider: str,
        duration_ms: float,
        usage: Any = None,
        system_prompt_chars: int = 0,
        user_prompt_chars: int = 0,
        user_input_chars: int = 0,
        uploaded_document_chars: int = 0,
        success: bool = True,
        error: str = "",
    ) -> None:
        usage_dict = _usage_to_dict(usage)
        prompt_tokens = _to_int(usage_dict.get("prompt_tokens"), 0)
        completion_tokens = _to_int(usage_dict.get("completion_tokens"), 0)
        total_tokens = _to_int(usage_dict.get("total_tokens"), 0)

        est_system_tokens = _estimate_tokens_from_chars(system_prompt_chars)
        est_user_prompt_tokens = _estimate_tokens_from_chars(user_prompt_chars)
        est_user_input_tokens = _estimate_tokens_from_chars(user_input_chars)
        est_uploaded_document_tokens = _estimate_tokens_from_chars(uploaded_document_chars)
        est_input_tokens = est_system_tokens + est_user_prompt_tokens + est_user_input_tokens + est_uploaded_document_tokens

        if prompt_tokens <= 0:
            prompt_tokens = est_input_tokens
        if total_tokens <= 0:
            total_tokens = max(prompt_tokens + completion_tokens, 0)

        self._append(
            {
                "time": _now_iso(),
                "endpoint": self._current_endpoint(),
                "event": f"{event}:{provider or '-'}",
                "model": (model or "-").strip() or "-",
                "planned_chunks": 0,
                "planned_reduce_batch": 0,
                "planned_reduce_rounds": 0,
                "input_budget_tokens": 0,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "uploaded_document_tokens": est_uploaded_document_tokens,
                "user_prompt_tokens": est_user_prompt_tokens,
                "user_input_tokens": est_user_input_tokens,
                "system_prompt_tokens": est_system_tokens,
                "duration_ms": round(float(duration_ms), 2),
                "status": "ok" if success else f"error: {error[:120]}",
            }
        )

    def record_map_reduce_plan(
        self,
        *,
        style: str,
        estimated_doc_tokens: int,
        input_budget_tokens: int,
        planned_chunks: int,
        planned_reduce_batch: int,
        planned_reduce_rounds: int,
    ) -> None:
        self._append(
            {
                "time": _now_iso(),
                "endpoint": self._current_endpoint(),
                "event": "map_reduce_plan",
                "model": (style or "-").strip() or "-",
                "planned_chunks": int(planned_chunks),
                "planned_reduce_batch": int(planned_reduce_batch),
                "planned_reduce_rounds": int(planned_reduce_rounds),
                "input_budget_tokens": int(input_budget_tokens),
                "input_tokens": int(estimated_doc_tokens),
                "output_tokens": 0,
                "total_tokens": int(estimated_doc_tokens),
                "uploaded_document_tokens": int(estimated_doc_tokens),
                "user_prompt_tokens": 0,
                "user_input_tokens": 0,
                "system_prompt_tokens": 0,
                "duration_ms": 0.0,
                "status": "plan",
            }
        )

    def get_rows(self, limit: int = 100, llm_only: bool = False) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), self.max_rows))
        with self.lock:
            rows = list(self.rows)
        if llm_only:
            rows = [
                r
                for r in rows
                if str(r.get("event", "")).startswith(("chat:", "embedding:", "embedding_fallback:", "map_reduce_plan"))
            ]
        return rows[:lim]

    def render_table(self, limit: int = 100, llm_only: bool = False) -> str:
        rows = self.get_rows(limit=limit, llm_only=llm_only)
        cols = [
            "time",
            "endpoint",
            "event",
            "model",
            "plan",
            "budget",
            "doc_tok",
            "u_prompt",
            "u_input",
            "sys_tok",
            "out_tok",
            "total_tok",
            "ms",
            "status",
        ]
        numeric_cols = {
            "budget",
            "doc_tok",
            "u_prompt",
            "u_input",
            "sys_tok",
            "out_tok",
            "total_tok",
            "ms",
        }
        labels = {
            "time": "time",
            "endpoint": "endpoint",
            "event": "event",
            "model": "model",
            "plan": "plan(c/b/r)",
            "budget": "budget",
            "doc_tok": "doc_tok",
            "u_prompt": "u_prompt",
            "u_input": "u_input",
            "sys_tok": "sys_tok",
            "out_tok": "out_tok",
            "total_tok": "total_tok",
            "ms": "ms",
            "status": "status",
        }

        if not rows:
            return "No LLM observability data yet." if llm_only else "No observability data yet."

        def _short(s: str, max_len: int) -> str:
            t = (s or "").strip()
            if len(t) <= max_len:
                return t
            if max_len <= 3:
                return t[:max_len]
            return t[: max_len - 3] + "..."

        table_rows: List[Dict[str, Any]] = []
        for r in rows:
            planned_chunks = int(r.get("planned_chunks", 0) or 0)
            planned_batch = int(r.get("planned_reduce_batch", 0) or 0)
            planned_rounds = int(r.get("planned_reduce_rounds", 0) or 0)
            plan_val = "-" if (planned_chunks == 0 and planned_batch == 0 and planned_rounds == 0) else f"{planned_chunks}/{planned_batch}/{planned_rounds}"
            table_rows.append(
                {
                    "time": _short(str(r.get("time", "")), 26),
                    "endpoint": _short(str(r.get("endpoint", "")), 18),
                    "event": _short(str(r.get("event", "")), 18),
                    "model": _short(str(r.get("model", "")), 22),
                    "plan": plan_val,
                    "budget": int(r.get("input_budget_tokens", 0) or 0),
                    "doc_tok": int(r.get("uploaded_document_tokens", 0) or 0),
                    "u_prompt": int(r.get("user_prompt_tokens", 0) or 0),
                    "u_input": int(r.get("user_input_tokens", 0) or 0),
                    "sys_tok": int(r.get("system_prompt_tokens", 0) or 0),
                    "out_tok": int(r.get("output_tokens", 0) or 0),
                    "total_tok": int(r.get("total_tokens", 0) or 0),
                    "ms": f"{float(r.get('duration_ms', 0.0) or 0.0):.2f}",
                    "status": _short(str(r.get("status", "")), 12),
                }
            )

        widths: Dict[str, int] = {c: len(labels[c]) for c in cols}
        for r in table_rows:
            for c in cols:
                widths[c] = max(widths[c], len(str(r.get(c, ""))))

        def fmt_line(values: Dict[str, Any]) -> str:
            parts: List[str] = []
            for c in cols:
                v = str(values.get(c, ""))
                parts.append(v.rjust(widths[c]) if c in numeric_cols else v.ljust(widths[c]))
            return " | ".join(parts)

        header = fmt_line({c: labels[c] for c in cols})
        sep = "-+-".join("-" * widths[c] for c in cols)
        lines = [header, sep]
        for r in table_rows:
            lines.append(fmt_line(r))
        return "\n".join(lines)


observability_service = ObservabilityService()
