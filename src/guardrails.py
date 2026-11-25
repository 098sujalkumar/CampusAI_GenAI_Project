"""Responsible AI Guardrails: Prompt Injection Defense, PII Scanning & Faithfulness Scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "may",
    "me", "my", "of", "on", "or", "our", "should", "so", "source", "that", "the",
    "their", "there", "these", "this", "to", "was", "we", "what", "when", "where",
    "which", "who", "will", "with", "you", "your", "based", "provided", "context",
    "according", "document", "university", "students", "student", "must", "only",
}

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "Instruction Override Attempt"),
    (r"(reveal|show|print|leak|dump)\s+(your\s+)?(hidden\s+)?(system\s+prompt|instructions|rules)", "System Prompt Extraction"),
    (r"you\s+are\s+now\s+(dan|unrestricted|jailbroken|in\s+developer\s+mode)", "Persona Jailbreak"),
    (r"bypass\s+(safety|guardrails|policy|filters|rules)", "Guardrail Bypass Attempt"),
    (r"(pretend|act\s+as\s+if)\s+.*(no\s+rules|no\s+restrictions|attendance\s+is\s+waived)", "Policy Override Manipulation"),
    (r"override\s+(system|academic\s+policy|admin)", "Privilege Escalation Prompt"),
]

PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "Potential 12-digit ID / Aadhaar Number"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "Potential PAN Number"),
    (r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", "Phone Number"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "Email Address"),
]


@dataclass
class InputGuardrailResult:
    """Outcome of pre-generation safety & integrity checks."""

    allowed: bool
    blocked_reason: str | None = None
    matched_rules: list[str] = field(default_factory=list)
    pii_warnings: list[str] = field(default_factory=list)
    sanitized_query: str = ""


@dataclass
class FaithfulnessReport:
    """Post-generation grounding & hallucination verification metrics."""

    grounding_score: float  # 0.0 - 1.0 overlap of factual tokens/numbers with evidence
    numeric_consistency: float  # 0.0 - 1.0 match rate of numbers/percentages cited
    citation_count: int
    verdict: str  # "High Grounding" | "Moderate Grounding" | "Low Grounding / Review Needed" | "Safe Refusal"
    unsupported_numbers: list[str] = field(default_factory=list)


def check_input_guardrails(query: str) -> InputGuardrailResult:
    """Detect prompt injection / jailbreak attempts and flag PII before calling the LLM."""
    cleaned = query.strip()
    if not cleaned:
        return InputGuardrailResult(
            allowed=False,
            blocked_reason="Query is empty. Please enter a question about university policies, courses, or study topics.",
            sanitized_query="",
        )

    matched_rules: list[str] = []
    lowered = cleaned.lower()
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            matched_rules.append(label)

    if matched_rules:
        return InputGuardrailResult(
            allowed=False,
            blocked_reason=(
                "Blocked by CampusAI Safety Guardrail: potential prompt injection or policy override detected "
                f"({', '.join(matched_rules)})."
            ),
            matched_rules=matched_rules,
            sanitized_query=cleaned,
        )

    pii_warnings: list[str] = []
    sanitized = cleaned
    for pattern, label in PII_PATTERNS:
        if re.search(pattern, sanitized):
            pii_warnings.append(label)
            sanitized = re.sub(pattern, "[REDACTED_PII]", sanitized)

    return InputGuardrailResult(
        allowed=True,
        pii_warnings=pii_warnings,
        sanitized_query=sanitized,
    )


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9%]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def _extract_numbers(text: str) -> set[str]:
    # Capture meaningful numeric values like 75%, 1,500, 9.2, 160, 40,000
    raw = re.findall(r"\b\d+(?:[.,]\d+)?%?", text)
    normalized = set()
    for num in raw:
        cleaned = num.replace(",", "")
        # Ignore small citation indices like [1], [2], [3], [4]
        if cleaned in {"1", "2", "3", "4", "5"}:
            continue
        normalized.add(cleaned)
    return normalized


def evaluate_faithfulness(
    answer: str,
    context_texts: list[str],
    is_refusal: bool = False,
) -> FaithfulnessReport:
    """Measure how strongly the generated answer is grounded in retrieved evidence."""
    if is_refusal:
        return FaithfulnessReport(
            grounding_score=1.0,
            numeric_consistency=1.0,
            citation_count=0,
            verdict="Safe Refusal",
        )

    combined_context = "\n".join(context_texts)
    context_tokens = _content_tokens(combined_context)
    answer_tokens = _content_tokens(answer)

    if not answer_tokens or not context_tokens:
        grounding_score = 0.0
    else:
        overlap = answer_tokens.intersection(context_tokens)
        grounding_score = min(1.0, len(overlap) / max(1, len(answer_tokens)))

    # Numeric consistency check (critical for university rules: % attendance, CGPA, fees, deadlines)
    ctx_numbers = _extract_numbers(combined_context)
    ans_numbers = _extract_numbers(answer)
    unsupported = sorted(ans_numbers - ctx_numbers)
    if not ans_numbers:
        numeric_consistency = 1.0
    else:
        numeric_consistency = len(ans_numbers & ctx_numbers) / len(ans_numbers)

    # Combine lexical grounding and numeric fidelity
    composite = round(0.65 * grounding_score + 0.35 * numeric_consistency, 3)
    citations = re.findall(r"\[(?:Source\s*)?\d+\]", answer, flags=re.IGNORECASE)

    if composite >= 0.65 and numeric_consistency >= 0.8:
        verdict = "High Grounding"
    elif composite >= 0.42:
        verdict = "Moderate Grounding"
    else:
        verdict = "Low Grounding / Review Needed"

    return FaithfulnessReport(
        grounding_score=composite,
        numeric_consistency=round(numeric_consistency, 3),
        citation_count=len(citations),
        verdict=verdict,
        unsupported_numbers=unsupported,
    )
