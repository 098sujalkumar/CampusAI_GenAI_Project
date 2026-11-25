"""Dual-Mode LLM Client: Google Gemini API (SDK + REST fallback) & Intelligent Offline Demo Synthesizer."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.retriever import RetrievedChunk, tokenize


@dataclass
class LLMResponse:
    """Standardized response container from Gemini or Offline Demo engine."""

    text: str
    backend: str  # "Google Gemini API" | "Offline Demo Mode (Extractive RAG)"
    model: str
    latency_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    error: str | None = None


class CampusLLMClient:
    """Unified interface for Google Gemini generation, embeddings, and Offline Demo fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.embedding_model = embedding_model
        self._sdk_client = None

        if self.api_key:
            try:
                from google import genai  # type: ignore

                self._sdk_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._sdk_client = None

    @property
    def is_live_gemini(self) -> bool:
        return bool(self.api_key)

    @property
    def backend_label(self) -> str:
        if self.api_key:
            via = "google-genai SDK" if self._sdk_client else "Gemini REST API"
            return f"Google Gemini ({self.model} via {via})"
        return "Offline Demo Mode (Grounded Extractive RAG)"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings via Google Gemini when an API key is configured."""
        if not self.api_key:
            raise RuntimeError("Gemini API key required for dense embeddings.")

        if self._sdk_client is not None:
            result = self._sdk_client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
            )
            return [list(emb.values) for emb in result.embeddings]

        # Standard library HTTPS REST fallback (no external SDK required)
        vectors: list[list[float]] = []
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.embedding_model}:embedContent?key={self.api_key}"
        )
        for text in texts:
            payload = json.dumps({"content": {"parts": [{"text": text[:6000]}]}}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vectors.append(data["embedding"]["values"])
        return vectors

    def generate_text(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Call Google Gemini API (SDK or REST); raises or returns error metadata on failure."""
        start = time.perf_counter()
        prompt_tokens = max(1, int((len(system_instruction) + len(user_prompt)) / 4))

        if not self.api_key:
            raise RuntimeError("No Gemini API key configured.")

        try:
            if self._sdk_client is not None:
                from google.genai import types  # type: ignore

                cfg_kwargs = {
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                }
                if json_mode:
                    cfg_kwargs["response_mime_type"] = "application/json"

                response = self._sdk_client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                text = (response.text or "").strip()
            else:
                text = self._generate_via_rest(
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    json_mode=json_mode,
                )

            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text=text,
                backend=self.backend_label,
                model=self.model,
                latency_ms=round(latency_ms, 1),
                prompt_tokens_est=prompt_tokens,
                completion_tokens_est=max(1, int(len(text) / 4)),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                backend=self.backend_label,
                model=self.model,
                latency_ms=round(latency_ms, 1),
                prompt_tokens_est=prompt_tokens,
                completion_tokens_est=0,
                error=str(exc),
            )

    def _generate_via_rest(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        gen_cfg: dict[str, object] = {"temperature": temperature}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        body = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            details = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini HTTP {err.code}: {details[:300]}") from err

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    def synthesize_offline_rag_answer(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        strategy: str = "grounded_cot",
    ) -> LLMResponse:
        """Intelligent offline RAG synthesizer that ranks evidence bullets directly from retrieved chunks.

        Ensures 100% reproducible, strictly grounded, cited answers even without an API key.
        """
        start = time.perf_counter()
        query_terms = set(tokenize(query))

        # Collect clean factual lines/sentences along with their token sets
        candidates: list[tuple[float, int, str, str, set[str]]] = []
        for src_idx, item in enumerate(retrieved, start=1):
            raw_lines = [ln.strip() for ln in item.chunk.text.splitlines() if ln.strip()]
            for line in raw_lines:
                if line.startswith("#") or line.startswith("| :---"):
                    continue
                cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
                if len(cleaned) < 18:
                    continue
                line_terms = set(tokenize(cleaned))
                overlap = len(query_terms & line_terms)
                base_score = (overlap / max(1, len(query_terms))) * 0.70 + item.score * 0.30
                if any(tok.isdigit() and tok in line_terms for tok in query_terms):
                    base_score += 0.12
                candidates.append((base_score, src_idx, item.chunk.section, cleaned, line_terms))

        # Greedy MMR-style selection: prioritize lines covering *new* uncovered query terms
        selected: list[tuple[int, str, str]] = []
        covered_terms: set[str] = set()
        seen_prefixes: set[str] = set()
        remaining = list(candidates)

        while remaining and len(selected) < 6:
            best_idx = -1
            best_score = -1.0
            for idx, (base_score, src_idx, section, text, line_terms) in enumerate(remaining):
                key = text[:60].lower()
                if key in seen_prefixes:
                    continue
                uncovered_hits = len((query_terms - covered_terms) & line_terms)
                marginal_gain = uncovered_hits / max(1, len(query_terms))
                mmr_score = base_score + 0.65 * marginal_gain
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx == -1:
                break
            _, src_idx, section, text, line_terms = remaining.pop(best_idx)
            seen_prefixes.add(text[:60].lower())
            covered_terms.update(query_terms & line_terms)
            selected.append((src_idx, section, text))

        if not selected:
            # Fallback to first retrieved chunk summary
            first = retrieved[0]
            answer = (
                f"Based on **{first.chunk.title}** (*{first.chunk.section}*):\n\n"
                f"{first.chunk.text[:450]} [Source 1]"
            )
        else:
            lead_src, lead_section, lead_text = selected[0]
            header = f"Based on the official university records (**{retrieved[lead_src - 1].chunk.title}**):\n\n"
            bullets = [f"- {lead_text} [Source {lead_src}]"]
            for src_idx, _sec, txt in selected[1:]:
                bullets.append(f"- {txt} [Source {src_idx}]")

            footer = ""
            if strategy == "grounded_cot":
                cited_sources = sorted({s for s, _, _ in selected})
                src_list = ", ".join(f"[Source {s}]" for s in cited_sources)
                footer = (
                    f"\n\n**Grounded Verification ({src_list}):** All figures and rules above were "
                    "extracted directly from the retrieved university documents."
                )

            answer = header + "\n".join(bullets) + footer

        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = sum(r.chunk.token_estimate for r in retrieved) + max(1, len(query) // 4)
        return LLMResponse(
            text=answer,
            backend="Offline Demo Mode (Grounded Extractive RAG)",
            model="local-extractive-rag-v1",
            latency_ms=round(latency_ms, 1),
            prompt_tokens_est=prompt_tokens,
            completion_tokens_est=max(1, len(answer) // 4),
        )
