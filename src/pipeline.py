"""End-to-End CampusAI RAG Orchestration Pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import KNOWLEDGE_BASE_DIR, RAGConfig, get_gemini_api_key
from src.guardrails import (
    FaithfulnessReport,
    InputGuardrailResult,
    check_input_guardrails,
    evaluate_faithfulness,
)
from src.ingestion import Document, TextChunk, chunk_documents, load_knowledge_base
from src.llm_client import CampusLLMClient
from src.prompts import build_rag_prompt
from src.retriever import HybridRetriever, RetrievedChunk


@dataclass
class RAGResult:
    """Complete auditable output of a single CampusAI query."""

    query: str
    answer: str
    retrieved: list[RetrievedChunk]
    guardrail: InputGuardrailResult
    faithfulness: FaithfulnessReport
    is_refusal: bool
    is_blocked: bool
    backend_used: str
    model_used: str
    prompt_strategy: str
    system_prompt: str
    user_prompt: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    fallback_notice: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class CampusAIPipeline:
    """Orchestrates document ingestion, hybrid indexing, guardrails, prompting, and LLM generation."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        api_key: str | None = None,
        knowledge_base_dir: Path | None = None,
        documents: list[Document] | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env()
        self.api_key = get_gemini_api_key(api_key)
        self.knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR

        self.llm = CampusLLMClient(
            api_key=self.api_key,
            model=self.config.gemini_model,
            embedding_model=self.config.embedding_model,
        )

        self.documents: list[Document] = (
            list(documents) if documents is not None else load_knowledge_base(self.knowledge_base_dir)
        )
        self.chunks: list[TextChunk] = []
        self.retriever: HybridRetriever | None = None
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Re-chunk all loaded documents and rebuild the Hybrid Vector/Lexical index."""
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        embed_fn = (
            self.llm.embed_texts
            if (self.llm.is_live_gemini and self.config.use_gemini_embeddings)
            else None
        )
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            alpha=self.config.hybrid_alpha,
            embed_fn=embed_fn,
        )

    def add_document(self, doc: Document) -> int:
        """Add or replace a document in the live session knowledge base and re-index."""
        self.documents = [d for d in self.documents if d.source != doc.source]
        self.documents.append(doc)
        self.rebuild_index()
        return len([c for c in self.chunks if c.source == doc.source])

    def update_credentials(self, api_key: str | None, model: str | None = None) -> None:
        """Switch between Google Gemini live API and Offline Demo Mode on the fly."""
        self.api_key = get_gemini_api_key(api_key)
        if model:
            self.config.gemini_model = model
        self.llm = CampusLLMClient(
            api_key=self.api_key,
            model=self.config.gemini_model,
            embedding_model=self.config.embedding_model,
        )
        if self.config.use_gemini_embeddings:
            self.rebuild_index()

    def available_categories(self) -> list[str]:
        cats = sorted({d.category for d in self.documents})
        return ["All Categories"] + cats

    def answer_query(
        self,
        query: str,
        category_filter: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        """Execute full Guardrails -> Hybrid Retrieval -> Prompting -> Generation -> Faithfulness pipeline."""
        total_start = time.perf_counter()

        # 1. Input Safety Guardrails
        if self.config.enable_guardrails:
            guardrail = check_input_guardrails(query)
        else:
            guardrail = InputGuardrailResult(allowed=True, sanitized_query=query.strip())

        if not guardrail.allowed:
            total_ms = round((time.perf_counter() - total_start) * 1000.0, 1)
            return RAGResult(
                query=query,
                answer=guardrail.blocked_reason or "Request blocked by safety guardrails.",
                retrieved=[],
                guardrail=guardrail,
                faithfulness=evaluate_faithfulness("", [], is_refusal=True),
                is_refusal=False,
                is_blocked=True,
                backend_used="CampusAI Safety Guardrail",
                model_used="rule-based-guardrail-v1",
                prompt_strategy=self.config.prompt_strategy,
                system_prompt="",
                user_prompt="",
                retrieval_ms=0.0,
                generation_ms=0.0,
                total_ms=total_ms,
                prompt_tokens_est=0,
                completion_tokens_est=0,
            )

        effective_query = guardrail.sanitized_query

        # 2. Hybrid Retrieval (TF-IDF Cosine + BM25 + optional Gemini Embeddings + RRF)
        ret_start = time.perf_counter()
        assert self.retriever is not None
        retrieved = self.retriever.retrieve(
            query=effective_query,
            top_k=self.config.top_k,
            category_filter=category_filter,
        )
        retrieval_ms = round((time.perf_counter() - ret_start) * 1000.0, 1)

        # 3. Relevance Thresholding & Hallucination Prevention (Safe Refusal)
        top_score = retrieved[0].score if retrieved else 0.0
        if not retrieved or top_score < self.config.min_relevance_score:
            refusal_msg = (
                "I could not find relevant information about that topic in the indexed university "
                "regulations, syllabi, or campus handbooks. To avoid guessing or fabricating official "
                "rules, I cannot answer this question from the current knowledge base.\n\n"
                "**Suggestion:** Upload the relevant policy/course document in the **Knowledge Base** tab "
                "or contact the Office of Academic Affairs / Student Services directly."
            )
            total_ms = round((time.perf_counter() - total_start) * 1000.0, 1)
            return RAGResult(
                query=query,
                answer=refusal_msg,
                retrieved=retrieved,
                guardrail=guardrail,
                faithfulness=evaluate_faithfulness(refusal_msg, [], is_refusal=True),
                is_refusal=True,
                is_blocked=False,
                backend_used=self.llm.backend_label,
                model_used=self.llm.model if self.llm.is_live_gemini else "local-extractive-rag-v1",
                prompt_strategy=self.config.prompt_strategy,
                system_prompt="",
                user_prompt="",
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
                prompt_tokens_est=0,
                completion_tokens_est=len(refusal_msg) // 4,
            )

        # Filter out low-noise tail chunks while keeping at least the top match
        relevant_chunks = [
            r for r in retrieved if r.score >= max(0.08, self.config.min_relevance_score * 0.75)
        ] or retrieved[:1]

        # 4. Prompt Construction
        system_prompt, user_prompt = build_rag_prompt(
            query=effective_query,
            retrieved=relevant_chunks,
            strategy=self.config.prompt_strategy,
            chat_history=chat_history,
        )

        # 5. Generation (Google Gemini Live API with automatic Offline Demo fallback)
        fallback_notice: str | None = None
        if self.llm.is_live_gemini:
            llm_resp = self.llm.generate_text(
                system_instruction=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
            )
            if llm_resp.error or not llm_resp.text:
                fallback_notice = (
                    f"Gemini API unavailable ({llm_resp.error}); seamlessly fell back to Offline Demo Mode."
                )
                llm_resp = self.llm.synthesize_offline_rag_answer(
                    query=effective_query,
                    retrieved=relevant_chunks,
                    strategy=self.config.prompt_strategy,
                )
        else:
            llm_resp = self.llm.synthesize_offline_rag_answer(
                query=effective_query,
                retrieved=relevant_chunks,
                strategy=self.config.prompt_strategy,
            )

        # 6. Post-Generation Faithfulness / Hallucination Audit
        is_model_refusal = any(
            phrase in llm_resp.text.lower()
            for phrase in (
                "could not find",
                "do not cover",
                "not mentioned in the provided",
                "not in the indexed university",
            )
        )
        faithfulness = evaluate_faithfulness(
            answer=llm_resp.text,
            context_texts=[f"{r.chunk.title}\n{r.chunk.section}\n{r.chunk.text}" for r in relevant_chunks],
            is_refusal=is_model_refusal,
        )

        total_ms = round((time.perf_counter() - total_start) * 1000.0, 1)
        return RAGResult(
            query=query,
            answer=llm_resp.text,
            retrieved=relevant_chunks,
            guardrail=guardrail,
            faithfulness=faithfulness,
            is_refusal=is_model_refusal,
            is_blocked=False,
            backend_used=llm_resp.backend,
            model_used=llm_resp.model,
            prompt_strategy=self.config.prompt_strategy,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieval_ms=retrieval_ms,
            generation_ms=llm_resp.latency_ms,
            total_ms=total_ms,
            prompt_tokens_est=llm_resp.prompt_tokens_est,
            completion_tokens_est=llm_resp.completion_tokens_est,
            fallback_notice=fallback_notice,
        )
