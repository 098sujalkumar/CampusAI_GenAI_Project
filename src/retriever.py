"""Hybrid Retrieval Engine: TF-IDF Cosine + BM25 Lexical + Optional Gemini Dense Embeddings with RRF."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.ingestion import TextChunk


TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "our", "out", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "why", "will", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Normalize text into informative unigram tokens preserving numbers and course codes."""
    raw = re.findall(r"[a-zA-Z0-9]+(?:[.%+-][a-zA-Z0-9]+)*%?", text.lower())
    tokens: list[str] = []
    for tok in raw:
        if tok in TOKEN_STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        # Light suffix normalization so "exams"/"exam", "passing"/"pass", "scholarships"/"scholarship" match well
        if len(tok) > 5 and tok.endswith("ing"):
            tok = tok[:-3]
        elif len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def extract_features(text: str) -> list[str]:
    """Unigrams + bigrams for richer TF-IDF semantic phrase matching."""
    unigrams = tokenize(text)
    bigrams = [f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def cosine_similarity_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse L2-normalized dictionaries."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())


def cosine_similarity_dense(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two dense embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedChunk:
    """A retrieved chunk annotated with hybrid ranking breakdown."""

    chunk: TextChunk
    score: float
    tfidf_score: float
    bm25_score: float
    dense_score: float
    rrf_score: float
    rank: int


class HybridRetriever:
    """Hybrid Vector + Lexical Retriever with Reciprocal Rank Fusion (RRF).

    Combines:
      1. TF-IDF unigram+bigram cosine similarity (fast local semantic/phrase signal)
      2. BM25 Okapi ranking (exact keyword & numeric threshold matching)
      3. Optional Google Gemini Dense Embeddings (when enabled with API key)
    """

    def __init__(
        self,
        chunks: list[TextChunk],
        alpha: float = 0.55,
        embed_fn=None,
    ) -> None:
        self.chunks = chunks
        self.alpha = alpha
        self.embed_fn = embed_fn

        # BM25 statistics
        self._doc_tokens: list[list[str]] = []
        self._doc_tf: list[Counter[str]] = []
        self._df_bm25: Counter[str] = Counter()
        self._avgdl: float = 1.0

        # TF-IDF statistics
        self._tfidf_idf: dict[str, float] = {}
        self._tfidf_vectors: list[dict[str, float]] = []

        # Optional dense embeddings
        self._dense_vectors: list[list[float]] | None = None
        self.dense_enabled: bool = False

        self._build_index()

    def _build_index(self) -> None:
        n_docs = len(self.chunks)
        if n_docs == 0:
            return

        # Include title & section headings in searchable representation for higher recall
        searchable_texts = [
            f"{c.title} {c.section} {c.text}" for c in self.chunks
        ]

        # 1. BM25 index
        total_len = 0
        for text in searchable_texts:
            toks = tokenize(text)
            self._doc_tokens.append(toks)
            total_len += len(toks)
            tf = Counter(toks)
            self._doc_tf.append(tf)
            for term in tf:
                self._df_bm25[term] += 1
        self._avgdl = max(1.0, total_len / n_docs)

        # 2. TF-IDF (unigram + bigram) L2-normalized index
        df_feat: Counter[str] = Counter()
        doc_feat_counts: list[Counter[str]] = []
        for text in searchable_texts:
            feats = extract_features(text)
            counts = Counter(feats)
            doc_feat_counts.append(counts)
            for feat in counts:
                df_feat[feat] += 1

        self._tfidf_idf = {
            feat: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            for feat, df in df_feat.items()
        }

        for counts in doc_feat_counts:
            vec: dict[str, float] = {}
            for feat, cnt in counts.items():
                tf_weight = 1.0 + math.log(cnt)
                vec[feat] = tf_weight * self._tfidf_idf[feat]
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._tfidf_vectors.append({k: v / norm for k, v in vec.items()})

        # 3. Optional Gemini dense embeddings
        if self.embed_fn is not None:
            try:
                vectors = self.embed_fn(searchable_texts)
                if vectors and len(vectors) == n_docs:
                    self._dense_vectors = vectors
                    self.dense_enabled = True
            except Exception:
                self._dense_vectors = None
                self.dense_enabled = False

    def _query_tfidf(self, query: str) -> dict[str, float]:
        counts = Counter(extract_features(query))
        vec: dict[str, float] = {}
        for feat, cnt in counts.items():
            idf = self._tfidf_idf.get(feat)
            if idf is not None:
                vec[feat] = (1.0 + math.log(cnt)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _score_bm25(self, query_tokens: list[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        n_docs = len(self.chunks)
        tf_map = self._doc_tf[doc_idx]
        dl = len(self._doc_tokens[doc_idx])
        score = 0.0
        for term in query_tokens:
            f = tf_map.get(term, 0)
            if f == 0:
                continue
            df = self._df_bm25.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = f + k1 * (1.0 - b + b * (dl / self._avgdl))
            score += idf * ((f * (k1 + 1.0)) / max(denom, 1e-9))
        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        category_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks ranked by Hybrid Similarity + Reciprocal Rank Fusion."""
        if not self.chunks or not query.strip():
            return []

        query_tokens = tokenize(query)
        q_tfidf = self._query_tfidf(query)

        q_dense: list[float] | None = None
        if self.dense_enabled and self.embed_fn is not None:
            try:
                res = self.embed_fn([query])
                if res:
                    q_dense = res[0]
            except Exception:
                q_dense = None

        candidate_indices = [
            i
            for i, c in enumerate(self.chunks)
            if not category_filter or category_filter == "All Categories" or c.category == category_filter
        ]
        if not candidate_indices:
            return []

        tfidf_scores: dict[int, float] = {}
        bm25_raw: dict[int, float] = {}
        dense_scores: dict[int, float] = {}

        for idx in candidate_indices:
            tfidf_scores[idx] = cosine_similarity_sparse(q_tfidf, self._tfidf_vectors[idx])
            bm25_raw[idx] = self._score_bm25(query_tokens, idx)
            if q_dense is not None and self._dense_vectors is not None:
                dense_scores[idx] = max(0.0, cosine_similarity_dense(q_dense, self._dense_vectors[idx]))
            else:
                dense_scores[idx] = 0.0

        # Normalize BM25 to [0, 1] damped by query-term coverage to prevent single-word false positives
        unique_q_terms = set(query_tokens)
        bm25_norm: dict[int, float] = {}
        for idx, raw in bm25_raw.items():
            if raw <= 0:
                bm25_norm[idx] = 0.0
                continue
            matched = sum(1 for t in unique_q_terms if self._doc_tf[idx].get(t, 0) > 0)
            coverage = matched / max(1, len(unique_q_terms))
            bm25_norm[idx] = (1.0 - math.exp(-raw / 5.0)) * (0.35 + 0.65 * coverage)

        # Course-code / specific entity alignment boost (e.g. CS401, CS302)
        course_codes = {t for t in unique_q_terms if re.match(r"^[a-z]{2,4}\d{3}$", t)}
        code_boost: dict[int, float] = {}
        for idx in candidate_indices:
            if course_codes and any(self._doc_tf[idx].get(code, 0) > 0 for code in course_codes):
                code_boost[idx] = 0.18
            else:
                code_boost[idx] = 0.0

        # Reciprocal Rank Fusion (RRF) across ranking signals
        rrf_k = 60
        tfidf_order = sorted(candidate_indices, key=lambda i: tfidf_scores[i], reverse=True)
        bm25_order = sorted(candidate_indices, key=lambda i: bm25_raw[i], reverse=True)
        dense_order = (
            sorted(candidate_indices, key=lambda i: dense_scores[i], reverse=True)
            if q_dense is not None
            else []
        )

        rrf_scores: dict[int, float] = {idx: 0.0 for idx in candidate_indices}
        for rank, idx in enumerate(tfidf_order, start=1):
            if tfidf_scores[idx] > 0:
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(bm25_order, start=1):
            if bm25_raw[idx] > 0:
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(dense_order, start=1):
            if dense_scores[idx] > 0:
                rrf_scores[idx] += 1.2 / (rrf_k + rank)

        scored: list[RetrievedChunk] = []
        for idx in candidate_indices:
            if q_dense is not None:
                combined = min(
                    1.0,
                    0.50 * dense_scores[idx] + 0.25 * tfidf_scores[idx] + 0.25 * bm25_norm[idx] + code_boost[idx],
                )
            else:
                # Calibrated hybrid confidence score in [0, 1]
                combined = min(
                    1.0,
                    self.alpha * tfidf_scores[idx] + (1.0 - self.alpha) * bm25_norm[idx] + code_boost[idx],
                )
                # Require lexical or phrase alignment to avoid spurious matches
                if tfidf_scores[idx] == 0.0 and bm25_raw[idx] == 0.0:
                    combined = 0.0

            scored.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=round(combined, 4),
                    tfidf_score=round(tfidf_scores[idx], 4),
                    bm25_score=round(bm25_norm[idx], 4),
                    dense_score=round(dense_scores[idx], 4),
                    rrf_score=round(rrf_scores[idx], 5),
                    rank=0,
                )
            )

        scored.sort(key=lambda item: (item.score, item.rrf_score), reverse=True)
        top_results = scored[:top_k]
        for rank, item in enumerate(top_results, start=1):
            item.rank = rank
        return top_results

    def index_stats(self) -> dict[str, int | str | bool]:
        """Return diagnostic stats about the vector & lexical index."""
        unique_docs = {c.doc_id for c in self.chunks}
        return {
            "documents_indexed": len(unique_docs),
            "total_chunks": len(self.chunks),
            "vocabulary_features": len(self._tfidf_idf),
            "dense_embeddings_active": self.dense_enabled,
            "retrieval_mode": (
                "Hybrid (Gemini Dense + TF-IDF + BM25 + RRF)"
                if self.dense_enabled
                else "Hybrid (TF-IDF Cosine + BM25 + RRF)"
            ),
        }
