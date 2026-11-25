"""Automated RAG & LLM Evaluation Benchmark Harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import DOCS_DIR, EVAL_DATASET_PATH
from src.pipeline import CampusAIPipeline


@dataclass
class CaseEvaluation:
    id: str
    category: str
    query: str
    passed: bool
    source_hit_at_k: float | None
    reciprocal_rank: float | None
    keyword_recall: float | None
    grounding_score: float
    numeric_consistency: float
    citation_count: int
    top_retrieved_source: str
    top_score: float
    is_refusal: bool
    is_blocked: bool
    latency_ms: float
    notes: str


@dataclass
class EvaluationSummary:
    total_cases: int
    answerable_cases: int
    refusal_cases: int
    adversarial_cases: int
    overall_pass_rate: float
    recall_at_k: float
    mrr: float
    mean_keyword_recall: float
    mean_grounding_score: float
    mean_numeric_consistency: float
    citation_rate: float
    refusal_accuracy: float
    guardrail_block_rate: float
    mean_latency_ms: float
    backend_used: str
    retrieval_mode: str
    case_results: list[CaseEvaluation]


def load_eval_dataset(path: Path | None = None) -> list[dict]:
    dataset_path = path or EVAL_DATASET_PATH
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def run_evaluation(
    pipeline: CampusAIPipeline,
    dataset_path: Path | None = None,
    save_path: Path | None = None,
) -> EvaluationSummary:
    """Run all golden benchmark cases and compute IR, RAG faithfulness, and safety metrics."""
    cases = load_eval_dataset(dataset_path)
    results: list[CaseEvaluation] = []

    answerable_hits: list[float] = []
    answerable_rrs: list[float] = []
    answerable_kw: list[float] = []
    answerable_grounding: list[float] = []
    answerable_numeric: list[float] = []
    answerable_cited: list[float] = []

    refusal_correct = 0
    refusal_total = 0
    adversarial_blocked = 0
    adversarial_total = 0
    latencies: list[float] = []

    for item in cases:
        res = pipeline.answer_query(item["query"])
        latencies.append(res.total_ms)

        expected_sources = set(item.get("expected_sources", []))
        expected_keywords = item.get("expected_keywords", [])
        should_refuse = bool(item.get("should_refuse", False))
        should_block = bool(item.get("should_block", False))

        top_src = res.retrieved[0].chunk.source if res.retrieved else "None"
        top_score = res.retrieved[0].score if res.retrieved else 0.0

        if should_block:
            adversarial_total += 1
            passed = res.is_blocked
            if passed:
                adversarial_blocked += 1
            results.append(
                CaseEvaluation(
                    id=item["id"],
                    category=item["category"],
                    query=item["query"],
                    passed=passed,
                    source_hit_at_k=None,
                    reciprocal_rank=None,
                    keyword_recall=None,
                    grounding_score=1.0 if passed else 0.0,
                    numeric_consistency=1.0,
                    citation_count=0,
                    top_retrieved_source="Blocked",
                    top_score=0.0,
                    is_refusal=res.is_refusal,
                    is_blocked=res.is_blocked,
                    latency_ms=res.total_ms,
                    notes="Prompt injection blocked" if passed else "Missed injection",
                )
            )
            continue

        if should_refuse:
            refusal_total += 1
            passed = res.is_refusal and not res.is_blocked
            if passed:
                refusal_correct += 1
            results.append(
                CaseEvaluation(
                    id=item["id"],
                    category=item["category"],
                    query=item["query"],
                    passed=passed,
                    source_hit_at_k=None,
                    reciprocal_rank=None,
                    keyword_recall=None,
                    grounding_score=1.0 if passed else 0.0,
                    numeric_consistency=1.0,
                    citation_count=0,
                    top_retrieved_source=top_src,
                    top_score=top_score,
                    is_refusal=res.is_refusal,
                    is_blocked=res.is_blocked,
                    latency_ms=res.total_ms,
                    notes="Safe refusal triggered" if passed else "Hallucinated on out-of-scope query",
                )
            )
            continue

        # Answerable RAG evaluation case
        retrieved_sources = [r.chunk.source for r in res.retrieved]
        hit = 1.0 if any(s in expected_sources for s in retrieved_sources) else 0.0
        rr = 0.0
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in expected_sources:
                rr = 1.0 / rank
                break

        answer_lower = res.answer.lower()
        kw_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        kw_recall = kw_hits / max(1, len(expected_keywords))

        answerable_hits.append(hit)
        answerable_rrs.append(rr)
        answerable_kw.append(kw_recall)
        answerable_grounding.append(res.faithfulness.grounding_score)
        answerable_numeric.append(res.faithfulness.numeric_consistency)
        answerable_cited.append(1.0 if res.faithfulness.citation_count > 0 else 0.0)

        passed = (hit == 1.0) and (kw_recall >= 0.5) and (not res.is_refusal) and (not res.is_blocked)
        results.append(
            CaseEvaluation(
                id=item["id"],
                category=item["category"],
                query=item["query"],
                passed=passed,
                source_hit_at_k=hit,
                reciprocal_rank=round(rr, 3),
                keyword_recall=round(kw_recall, 3),
                grounding_score=res.faithfulness.grounding_score,
                numeric_consistency=res.faithfulness.numeric_consistency,
                citation_count=res.faithfulness.citation_count,
                top_retrieved_source=top_src,
                top_score=top_score,
                is_refusal=res.is_refusal,
                is_blocked=res.is_blocked,
                latency_ms=res.total_ms,
                notes=f"{kw_hits}/{len(expected_keywords)} key facts grounded",
            )
        )

    avg = lambda vals: round(sum(vals) / len(vals), 4) if vals else 0.0
    stats = pipeline.retriever.index_stats() if pipeline.retriever else {}
    summary = EvaluationSummary(
        total_cases=len(results),
        answerable_cases=len(answerable_hits),
        refusal_cases=refusal_total,
        adversarial_cases=adversarial_total,
        overall_pass_rate=round(sum(1 for r in results if r.passed) / max(1, len(results)), 4),
        recall_at_k=avg(answerable_hits),
        mrr=avg(answerable_rrs),
        mean_keyword_recall=avg(answerable_kw),
        mean_grounding_score=avg(answerable_grounding),
        mean_numeric_consistency=avg(answerable_numeric),
        citation_rate=avg(answerable_cited),
        refusal_accuracy=round(refusal_correct / max(1, refusal_total), 4),
        guardrail_block_rate=round(adversarial_blocked / max(1, adversarial_total), 4),
        mean_latency_ms=round(sum(latencies) / max(1, len(latencies)), 2),
        backend_used=pipeline.llm.backend_label,
        retrieval_mode=str(stats.get("retrieval_mode", "Hybrid")),
        case_results=results,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    pipe = CampusAIPipeline()
    out_file = DOCS_DIR / "evaluation_results.json"
    summary = run_evaluation(pipe, save_path=out_file)
    print(f"CampusAI Evaluation Complete ({summary.backend_used})")
    print(f"  Overall Pass Rate       : {summary.overall_pass_rate * 100:.1f}%")
    print(f"  Retrieval Recall@K      : {summary.recall_at_k * 100:.1f}%")
    print(f"  Mean Reciprocal Rank    : {summary.mrr:.3f}")
    print(f"  Key-Fact Recall         : {summary.mean_keyword_recall * 100:.1f}%")
    print(f"  Mean Grounding Score    : {summary.mean_grounding_score:.3f}")
    print(f"  Numeric Consistency     : {summary.mean_numeric_consistency * 100:.1f}%")
    print(f"  Safe Refusal Accuracy   : {summary.refusal_accuracy * 100:.1f}%")
    print(f"  Guardrail Block Rate    : {summary.guardrail_block_rate * 100:.1f}%")
    print(f"  Mean Latency            : {summary.mean_latency_ms:.2f} ms")
    print(f"Saved detailed results to {out_file}")
