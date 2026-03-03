from __future__ import annotations

import os
import time
import json
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urlunparse

import httpx
import config
from openai import OpenAI
from services.observability_service import observability_service


def _as_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_base_url(endpoint: str) -> str:
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return ep
    if ep.endswith("/v1"):
        return ep
    return f"{ep}/v1"


def _rewrite_local_domain(endpoint: str, local_domain: str) -> str:
    ep = (endpoint or "").strip()
    d = (local_domain or "").strip()
    if not ep or not d or d.lower() == "not-needed":
        return ep

    parsed = urlparse(ep)
    if parsed.hostname and parsed.hostname.lower() == d.lower():
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal")
        return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return ep


def _infer_provider_name(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if not ep:
        return "openai"
    parsed = urlparse(ep if "://" in ep else f"https://{ep}")
    host = (parsed.hostname or "").lower()
    port = parsed.port

    if "openai.com" in host:
        return "openai"
    if "ollama" in host:
        return "ollama"
    if host in {"localhost", "127.0.0.1", "host.docker.internal"} and (port == 11434 or ":11434" in ep):
        return "ollama"
    return "inference_api"


class LLMProvider:
    def __init__(self) -> None:
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        self.embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", str(self.timeout)))
        self.embedding_max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", str(self.max_retries)))
        self.verify_ssl = _as_bool(os.getenv("VERIFY_SSL", "true"), default=True)
        self.local_url_endpoint = os.getenv("LOCAL_URL_ENDPOINT", "not-needed").strip()

        # Unified config (preferred)
        self.api_endpoint = os.getenv("API_ENDPOINT", "").strip()
        self.api_token = os.getenv("API_TOKEN", "").strip()
        self.model_name = os.getenv("MODEL_NAME", "").strip()
        self.provider_name_env = os.getenv("PROVIDER_NAME", "").strip().lower()
        self.embedding_model = (
            os.getenv("EMBEDDING_MODEL", "").strip()
            or os.getenv("EMBEDDING_MODEL_NAME", "").strip()
            or os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small"
        )

        # Legacy config (fallback)
        self.legacy_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "") or getattr(config, "OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", None) or getattr(config, "OPENAI_MODEL", None) or "gpt-4o-mini"

        self.inference_endpoint = os.getenv("INFERENCE_API_ENDPOINT", "").strip()
        self.inference_token = os.getenv("INFERENCE_API_TOKEN", "").strip()
        self.inference_model = os.getenv("INFERENCE_MODEL_NAME", "").strip()

        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").strip()
        self.ollama_token = os.getenv("OLLAMA_TOKEN", "ollama").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

        # Embedding behavior
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "same").strip().lower()  # same | openai
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or self.embedding_model
        self.inference_embedding_model = os.getenv("INFERENCE_EMBEDDING_MODEL_NAME", "").strip() or self.embedding_model
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "").strip() or self.embedding_model
        # Simple dedicated embedding config (preferred)
        self.embedding_api_endpoint = (
            os.getenv("EMBEDDING_ENDPOINT", "").strip()
            or os.getenv("EMBEDDING_API_ENDPOINT", "").strip()  # backward compatible
        )
        self.embedding_api_token = (
            os.getenv("EMBEDDING_API", "").strip()
            or os.getenv("EMBEDDING_API_TOKEN", "").strip()  # backward compatible
        )
        self.embedding_model_name = (
            os.getenv("EMBEDDING_MODEL", "").strip()
            or os.getenv("EMBEDDING_MODEL_NAME", "").strip()  # backward compatible
            or self.embedding_model
        )
        self.embedding_provider_name = os.getenv("EMBEDDING_PROVIDER_NAME", "").strip().lower()

        self.client: Optional[OpenAI] = None
        self.embedding_client: Optional[OpenAI] = None
        self.provider_embedding_client: Optional[OpenAI] = None
        self.api_key: str = ""
        self._context_tokens_cache: Optional[int] = None
        settings = self._resolve_settings()
        self.model = settings.get("model", "")
        self.provider_name = settings.get("name", "unknown")
        self.base_url = settings.get("base_url", "")
        self.initialized = False

    def _using_unified_config(self) -> bool:
        return bool(self.api_endpoint or self.api_token or self.model_name or self.provider_name_env)

    def _resolve_settings_legacy(self) -> Dict[str, str]:
        p = self.legacy_provider
        if p == "openai":
            return {"name": "openai", "base_url": "", "api_key": self.openai_api_key, "model": self.openai_model}
        if p in {"inference_api", "xeon"}:
            ep = _normalize_base_url(_rewrite_local_domain(self.inference_endpoint, self.local_url_endpoint))
            return {"name": "inference_api", "base_url": ep, "api_key": self.inference_token, "model": self.inference_model}
        if p == "ollama":
            ep = _normalize_base_url(_rewrite_local_domain(self.ollama_endpoint, self.local_url_endpoint))
            return {"name": "ollama", "base_url": ep, "api_key": self.ollama_token or "ollama", "model": self.ollama_model}
        raise ValueError("LLM_PROVIDER must be one of: openai, inference_api, xeon, ollama")

    def _resolve_settings(self) -> Dict[str, str]:
        if not self._using_unified_config():
            return self._resolve_settings_legacy()

        ep = _rewrite_local_domain(self.api_endpoint, self.local_url_endpoint)
        base = _normalize_base_url(ep) if ep else ""
        name = self.provider_name_env or _infer_provider_name(ep)
        token = self.api_token or self.openai_api_key
        model = self.model_name or self.openai_model

        return {
            "name": name,
            "base_url": base,
            "api_key": token,
            "model": model,
        }

    def ensure_initialized(self) -> None:
        if self.initialized:
            return

        settings = self._resolve_settings()
        api_key = (settings.get("api_key") or "").strip()
        model = (settings.get("model") or "").strip()
        base_url = (settings.get("base_url") or "").strip()

        if not api_key:
            raise ValueError("Missing API token/key. Set API_TOKEN (or OPENAI_API_KEY for OpenAI).")
        if not model:
            raise ValueError("Missing model name. Set MODEL_NAME (or OPENAI_MODEL).")

        http_client = httpx.Client(verify=self.verify_ssl, timeout=self.timeout)
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "http_client": http_client,
        }
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider_name = settings.get("name", "unknown")
        self.initialized = True

    @staticmethod
    def _extract_int_candidate(payload: Any) -> int:
        """
        Best-effort extractor for context window fields from model metadata.
        """
        keys = {
            "context_length",
            "max_context_length",
            "context_window",
            "max_input_tokens",
            "max_prompt_tokens",
            "input_token_limit",
            "num_ctx",
            "n_ctx",
        }

        def walk(obj: Any) -> int:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = str(k).strip().lower()
                    if lk in keys:
                        try:
                            iv = int(v)
                            if 512 <= iv <= 10_000_000:
                                return iv
                        except Exception:
                            pass
                    nested = walk(v)
                    if nested > 0:
                        return nested
            elif isinstance(obj, list):
                for item in obj:
                    nested = walk(item)
                    if nested > 0:
                        return nested
            return 0

        return walk(payload)

    def _context_override_from_env(self) -> int:
        direct = (os.getenv("MODEL_CONTEXT_TOKENS", "") or "").strip()
        if direct:
            try:
                v = int(direct)
                if v > 0:
                    return v
            except Exception:
                pass

        raw_map = (os.getenv("MODEL_CONTEXT_TOKENS_MAP", "") or "").strip()
        if not raw_map:
            return 0
        try:
            data = json.loads(raw_map)
            if not isinstance(data, dict):
                return 0
            model_key = (self.model or "").strip().lower()
            for k, v in data.items():
                if str(k).strip().lower() == model_key:
                    iv = int(v)
                    if iv > 0:
                        return iv
        except Exception:
            return 0
        return 0

    def _discover_context_tokens_from_models_endpoint(self) -> int:
        self.ensure_initialized()
        assert self.client is not None
        target_model = (self.model or "").strip().lower()

        try:
            listing = self.client.models.list()
            items = getattr(listing, "data", []) or []
            fallback_item: Dict[str, Any] = {}
            for it in items:
                if hasattr(it, "model_dump"):
                    obj = it.model_dump()
                elif isinstance(it, dict):
                    obj = it
                else:
                    obj = {}
                if not isinstance(obj, dict):
                    continue
                item_id = str(obj.get("id", "")).strip().lower()
                if item_id == target_model and obj:
                    found = self._extract_int_candidate(obj)
                    if found > 0:
                        return found
                if not fallback_item and obj:
                    fallback_item = obj

            if fallback_item:
                found = self._extract_int_candidate(fallback_item)
                if found > 0:
                    return found
        except Exception:
            pass
        return 0

    def _discover_context_tokens_from_raw_http(self) -> int:
        """
        For OpenAI-compatible gateways that expose extra model metadata on /v1/models.
        """
        base = (self.base_url or "").strip().rstrip("/")
        if not base or not self.api_key:
            return 0

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as c:
                resp = c.get(f"{base}/models", headers=headers)
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data", []) if isinstance(body, dict) else []
                if not isinstance(data, list):
                    return 0
                target_model = (self.model or "").strip().lower()
                fallback: Dict[str, Any] = {}
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    mid = str(item.get("id", "")).strip().lower()
                    if mid == target_model:
                        found = self._extract_int_candidate(item)
                        if found > 0:
                            return found
                    if not fallback:
                        fallback = item
                if fallback:
                    found = self._extract_int_candidate(fallback)
                    if found > 0:
                        return found
        except Exception:
            pass
        return 0

    def resolve_model_context_tokens(self, default_tokens: int) -> int:
        """
        Resolve model context window dynamically:
        1) explicit env override
        2) cached discovered value
        3) provider metadata (/models)
        4) fallback default
        """
        env_override = self._context_override_from_env()
        if env_override > 0:
            self._context_tokens_cache = env_override
            return env_override

        if self._context_tokens_cache and self._context_tokens_cache > 0:
            return int(self._context_tokens_cache)

        discovered = self._discover_context_tokens_from_models_endpoint()
        if discovered <= 0:
            discovered = self._discover_context_tokens_from_raw_http()
        if discovered > 0:
            self._context_tokens_cache = int(discovered)
            return int(discovered)
        return int(default_tokens)

    @staticmethod
    def _extract_between(text: str, start_marker: str, end_marker: str = "") -> str:
        t = text or ""
        start_idx = t.find(start_marker)
        if start_idx < 0:
            return ""
        start_idx += len(start_marker)
        if not end_marker:
            return t[start_idx:].strip()
        end_idx = t.find(end_marker, start_idx)
        if end_idx < 0:
            return t[start_idx:].strip()
        return t[start_idx:end_idx].strip()

    def _split_chat_prompt_parts(self, user_prompt: str) -> Dict[str, str]:
        """
        Best-effort parsing of our prompt templates:
        - uploaded_document: large document/context block
        - user_input: direct user question/input
        - user_prompt: instruction wrapper around the above
        """
        up = user_prompt or ""
        parts = {
            "uploaded_document": "",
            "user_input": "",
            "user_prompt": "",
        }

        # RAG chat template
        if "CONTEXT:\n" in up and "\n\nQUESTION:\n" in up:
            ctx = self._extract_between(up, "CONTEXT:\n", "\n\nQUESTION:\n")
            question = self._extract_between(up, "\n\nQUESTION:\n", "\n\nAnswer using only the context.")
            parts["uploaded_document"] = ctx
            parts["user_input"] = question
            wrapper = up.replace(ctx, "").replace(question, "")
            parts["user_prompt"] = wrapper.strip()
            return parts

        # Summarization templates
        if "\n\nDocument:\n" in up:
            doc = self._extract_between(up, "\n\nDocument:\n")
            parts["uploaded_document"] = doc
            wrapper = up.replace(doc, "")
            parts["user_prompt"] = wrapper.strip()
            return parts

        # Map phase template
        if "Chunk text:\n" in up and "\n\nReturn concise bullet points from this chunk only." in up:
            chunk = self._extract_between(up, "Chunk text:\n", "\n\nReturn concise bullet points from this chunk only.")
            parts["uploaded_document"] = chunk
            wrapper = up.replace(chunk, "")
            parts["user_prompt"] = wrapper.strip()
            return parts

        # Reduce phase template
        if "Summaries to merge:\n" in up and "\n\nMerge these into a single de-duplicated brief." in up:
            merge_input = self._extract_between(up, "Summaries to merge:\n", "\n\nMerge these into a single de-duplicated brief.")
            parts["uploaded_document"] = merge_input
            wrapper = up.replace(merge_input, "")
            parts["user_prompt"] = wrapper.strip()
            return parts

        # Final phase template
        if "Consolidated notes:\n" in up and "\n\nWrite the final output now." in up:
            notes = self._extract_between(up, "Consolidated notes:\n", "\n\nWrite the final output now.")
            parts["uploaded_document"] = notes
            wrapper = up.replace(notes, "")
            parts["user_prompt"] = wrapper.strip()
            return parts

        # Fallback: treat full user message as direct user input
        parts["user_input"] = up.strip()
        return parts

    def call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, stream: bool = False):
        self.ensure_initialized()
        assert self.client is not None
        parts = self._split_chat_prompt_parts(user_prompt)
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            observability_service.record_llm_call(
                event="chat",
                model=self.model,
                provider=self.provider_name,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                usage=getattr(resp, "usage", None),
                system_prompt_chars=len(system_prompt or ""),
                user_prompt_chars=len(parts.get("user_prompt", "")),
                user_input_chars=len(parts.get("user_input", "")),
                uploaded_document_chars=len(parts.get("uploaded_document", "")),
                success=True,
            )
            return resp
        except Exception as e:
            observability_service.record_llm_call(
                event="chat",
                model=self.model,
                provider=self.provider_name,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                system_prompt_chars=len(system_prompt or ""),
                user_prompt_chars=len(parts.get("user_prompt", "")),
                user_input_chars=len(parts.get("user_input", "")),
                uploaded_document_chars=len(parts.get("uploaded_document", "")),
                success=False,
                error=str(e),
            )
            raise

    def _openai_embed_client(self) -> OpenAI:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        return OpenAI(
            api_key=self.openai_api_key,
            timeout=self.embedding_timeout,
            max_retries=self.embedding_max_retries,
            http_client=httpx.Client(verify=self.verify_ssl, timeout=self.embedding_timeout),
        )

    def _using_dedicated_embedding_config(self) -> bool:
        return bool(self.embedding_model_name and (self.embedding_api_endpoint or self.embedding_api_token))

    def _dedicated_embedding_client(self) -> OpenAI:
        if self.embedding_client is not None:
            return self.embedding_client

        endpoint_src = self.embedding_api_endpoint or self.api_endpoint
        endpoint = _normalize_base_url(_rewrite_local_domain(endpoint_src, self.local_url_endpoint))
        if not endpoint:
            raise ValueError("Embedding endpoint missing. Set EMBEDDING_ENDPOINT or API_ENDPOINT.")

        token = self.embedding_api_token or self.api_token or self.openai_api_key
        if not token:
            raise ValueError("EMBEDDING_API_TOKEN (or API_TOKEN/OPENAI_API_KEY fallback) is required for dedicated embeddings")

        self.embedding_client = OpenAI(
            api_key=token,
            base_url=endpoint,
            timeout=self.embedding_timeout,
            max_retries=self.embedding_max_retries,
            http_client=httpx.Client(verify=self.verify_ssl, timeout=self.embedding_timeout),
        )
        return self.embedding_client

    def _same_provider_embed_client(self) -> OpenAI:
        if self.provider_embedding_client is not None:
            return self.provider_embedding_client

        self.ensure_initialized()
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.embedding_timeout,
            "max_retries": self.embedding_max_retries,
            "http_client": httpx.Client(verify=self.verify_ssl, timeout=self.embedding_timeout),
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.provider_embedding_client = OpenAI(**kwargs)
        return self.provider_embedding_client

    def _embedding_model_for_active_provider(self) -> str:
        if self.provider_name in {"inference_api", "xeon"}:
            return self.inference_embedding_model
        if self.provider_name == "ollama":
            return self.ollama_embedding_model
        return self.openai_embedding_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.ensure_initialized()
        if not texts:
            return []
        started = time.perf_counter()
        input_chars = sum(len(t or "") for t in texts)

        if self.embedding_provider == "openai":
            emb_client = self._openai_embed_client()
            resp = emb_client.embeddings.create(model=self.openai_embedding_model, input=texts)
            observability_service.record_llm_call(
                event="embedding",
                model=self.openai_embedding_model,
                provider="openai",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                usage=getattr(resp, "usage", None),
                uploaded_document_chars=input_chars,
                success=True,
            )
            return [d.embedding for d in resp.data]

        if self._using_dedicated_embedding_config():
            emb_client = self._dedicated_embedding_client()
            model_for_custom = self.embedding_model_name
            provider_for_custom = self.embedding_provider_name or _infer_provider_name(self.embedding_api_endpoint or self.api_endpoint)
            resp = emb_client.embeddings.create(model=model_for_custom, input=texts)
            observability_service.record_llm_call(
                event="embedding",
                model=model_for_custom,
                provider=provider_for_custom,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                usage=getattr(resp, "usage", None),
                uploaded_document_chars=input_chars,
                success=True,
            )
            return [d.embedding for d in resp.data]

        emb_client = self._same_provider_embed_client()
        model_for_provider = self._embedding_model_for_active_provider()
        try:
            resp = emb_client.embeddings.create(model=model_for_provider, input=texts)
            observability_service.record_llm_call(
                event="embedding",
                model=model_for_provider,
                provider=self.provider_name,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                usage=getattr(resp, "usage", None),
                uploaded_document_chars=input_chars,
                success=True,
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            observability_service.record_llm_call(
                event="embedding",
                model=model_for_provider,
                provider=self.provider_name,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                uploaded_document_chars=input_chars,
                success=False,
                error=str(e),
            )
            if self.openai_api_key:
                emb_client = self._openai_embed_client()
                started_fallback = time.perf_counter()
                resp = emb_client.embeddings.create(model=self.openai_embedding_model, input=texts)
                observability_service.record_llm_call(
                    event="embedding_fallback",
                    model=self.openai_embedding_model,
                    provider="openai",
                    duration_ms=(time.perf_counter() - started_fallback) * 1000.0,
                    usage=getattr(resp, "usage", None),
                    uploaded_document_chars=input_chars,
                    success=True,
                )
                return [d.embedding for d in resp.data]
            raise

    def embed_query(self, query: str) -> List[float]:
        q = (query or "").strip()
        if not q:
            return []
        out = self.embed_texts([q])
        return out[0] if out else []

    def health_check(self) -> Dict[str, Any]:
        try:
            self.ensure_initialized()
            assert self.client is not None
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10,
                temperature=0,
            )
            return {"status": "healthy", "provider": self.provider_name, "model": self.model}
        except Exception as e:
            return {"status": "unhealthy", "provider": self.provider_name or "unknown", "error": str(e)}


OpenAIProvider = LLMProvider
