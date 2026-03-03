from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence


ChatCall = Callable[[str, str, int, float], str]


@dataclass
class SourceChunk:
    text: str
    order: int


class SummarizationPipeline:
    """
    Generic map-reduce summarization for large documents.
    """

    def __init__(
        self,
        max_chunks: int = 24,
        chunk_chars: int = 2800,
        chunk_overlap_chars: int = 250,
        reduce_batch_size: int = 6,
    ) -> None:
        self.max_chunks = max(4, int(max_chunks))
        self.chunk_chars = max(800, int(chunk_chars))
        self.chunk_overlap_chars = max(0, min(int(chunk_overlap_chars), self.chunk_chars - 50))
        self.reduce_batch_size = max(2, int(reduce_batch_size))
        self.max_reduce_rounds = 8
        self.max_final_notes_chars = 24000

    def build_chunks_from_text(self, text: str) -> List[SourceChunk]:
        return self.build_chunks_from_text_with_limit(text=text, max_chunks=self.max_chunks)

    def build_chunks_from_text_with_limit(self, text: str, max_chunks: int) -> List[SourceChunk]:
        t = (text or "").strip()
        if not t:
            return []

        out: List[SourceChunk] = []
        n = len(t)
        start = 0
        idx = 0
        limit = max(1, int(max_chunks))
        while start < n and len(out) < limit:
            end = min(n, start + self.chunk_chars)
            chunk = t[start:end].strip()
            if chunk:
                out.append(SourceChunk(text=chunk, order=idx))
                idx += 1
            if end >= n:
                break
            start = max(0, end - self.chunk_overlap_chars)
        return out

    def limit_chunks(self, chunks: Sequence[SourceChunk]) -> List[SourceChunk]:
        return self.limit_chunks_with_limit(chunks=chunks, max_chunks=self.max_chunks)

    def limit_chunks_with_limit(self, chunks: Sequence[SourceChunk], max_chunks: int) -> List[SourceChunk]:
        if not chunks:
            return []
        ordered = sorted(list(chunks), key=lambda c: c.order)
        return ordered[: max(1, int(max_chunks))]

    def summarize(
        self,
        chunks: Sequence[SourceChunk],
        call_chat: ChatCall,
        base_system_prompt: str,
        map_instruction: str,
        reduce_instruction: str,
        final_instruction: str,
        max_tokens: int,
        temperature: float,
        max_chunks_override: int | None = None,
        reduce_batch_size_override: int | None = None,
    ) -> str:
        max_chunks = int(max_chunks_override) if max_chunks_override is not None else self.max_chunks
        reduce_batch_size = int(reduce_batch_size_override) if reduce_batch_size_override is not None else self.reduce_batch_size
        reduce_batch_size = max(2, reduce_batch_size)

        src = self.limit_chunks_with_limit(chunks, max_chunks=max_chunks)
        if not src:
            return ""

        map_points: List[str] = []
        map_max_tokens = max(160, min(420, int(max_tokens // 3) if max_tokens else 260))
        map_temp = max(0.0, min(float(temperature), 0.25))

        for ch in src:
            system_prompt = base_system_prompt + "\n" + map_instruction
            user_prompt = (
                "Chunk text:\n"
                f"{ch.text}\n\n"
                "Return concise bullet points from this chunk only."
            )
            mapped = call_chat(system_prompt, user_prompt, map_max_tokens, map_temp).strip()
            if mapped:
                map_points.append(mapped)

        if not map_points:
            return ""

        current = map_points
        reduce_max_tokens = max(220, min(700, int(max_tokens // 2) if max_tokens else 420))
        rounds = 0
        while len(current) > reduce_batch_size and rounds < self.max_reduce_rounds:
            rounds += 1
            nxt: List[str] = []
            for i in range(0, len(current), reduce_batch_size):
                batch = current[i : i + reduce_batch_size]
                system_prompt = base_system_prompt + "\n" + reduce_instruction
                user_prompt = (
                    "Summaries to merge:\n"
                    f"{chr(10).join(batch)}\n\n"
                    "Merge these into a single de-duplicated brief."
                )
                merged = call_chat(system_prompt, user_prompt, reduce_max_tokens, map_temp).strip()
                if merged:
                    nxt.append(merged)
            # Prevent infinite loops when model returns empty reduce outputs.
            if not nxt:
                break
            current = nxt
            if len(current) <= 1:
                break

        final_system = base_system_prompt + "\n" + final_instruction
        final_notes = "\n".join(current).strip()
        if len(final_notes) > self.max_final_notes_chars:
            final_notes = final_notes[: self.max_final_notes_chars]
        final_user = (
            "Consolidated notes:\n"
            f"{final_notes}\n\n"
            "Write the final output now."
        )
        return call_chat(final_system, final_user, max_tokens, temperature).strip()
