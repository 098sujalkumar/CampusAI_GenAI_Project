"""Structured GenAI Study Studio: MCQ Quiz Generator, Flashcards & Adaptive Study Planner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.ingestion import Document
from src.llm_client import CampusLLMClient
from src.prompts import build_quiz_prompt, build_study_plan_prompt


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    unit: str


@dataclass
class Flashcard:
    unit: str
    front: str
    back: str


@dataclass
class StudyPhase:
    phase: str
    focus_units: str
    topics: list[str]
    active_recall_task: str
    hours: float


@dataclass
class StudyPlan:
    course_title: str
    summary: str
    schedule: list[StudyPhase]
    exam_tips: list[str]
    backend_used: str


# Curated high-quality syllabus-grounded question banks for instant Offline Demo Mode
OFFLINE_QUIZ_BANK: dict[str, list[dict]] = {
    "cs401": [
        {
            "question": "In the Transformer Scaled Dot-Product Attention formula Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, why is QK^T divided by sqrt(d_k)?",
            "options": [
                "To prevent large dot-product magnitudes from pushing softmax into regions with vanishingly small gradients",
                "To reduce the sequence length before tokenization",
                "To convert decoder-only models into encoder-only architectures",
                "To eliminate the need for positional encodings",
            ],
            "correct_index": 0,
            "explanation": "Scaling by sqrt(d_k) keeps the variance of dot products stable as key dimension d_k grows, ensuring healthy softmax gradients during training (CS401 Unit 1).",
            "unit": "Unit 1: Foundations of Generative Models & Transformers",
        },
        {
            "question": "How does LoRA (Low-Rank Adaptation) achieve Parameter-Efficient Fine-Tuning (PEFT)?",
            "options": [
                "By freezing base weights and decomposing weight updates into trainable low-rank matrices Delta W = B * A",
                "By removing all attention heads except the first layer",
                "By replacing softmax with TF-IDF cosine similarity",
                "By retraining the entire tokenizer vocabulary from scratch",
            ],
            "correct_index": 0,
            "explanation": "LoRA keeps pre-trained weights frozen and learns compact low-rank update matrices B and A, drastically cutting memory and trainable parameters (CS401 Unit 2).",
            "unit": "Unit 2: Pre-training, Adaptation & Alignment",
        },
        {
            "question": "Which combination best characterizes Hybrid Search in Retrieval-Augmented Generation (RAG)?",
            "options": [
                "Combining lexical ranking (BM25/TF-IDF) with dense vector embeddings via Reciprocal Rank Fusion (RRF)",
                "Using only Byte-Pair Encoding without any similarity metric",
                "Running RLHF at inference time on every user query",
                "Increasing sampling temperature above 2.0 to reduce hallucinations",
            ],
            "correct_index": 0,
            "explanation": "Hybrid Search pairs exact lexical matching (BM25/TF-IDF) with semantic dense embeddings and merges ranked lists with RRF for higher recall and precision (CS401 Unit 4).",
            "unit": "Unit 4: Retrieval-Augmented Generation (RAG) & Vector Search",
        },
        {
            "question": "What are the three dimensions of the RAG Triad used to evaluate RAG pipelines?",
            "options": [
                "Context Relevance, Groundedness (Faithfulness), and Answer Relevance",
                "Perplexity, Batch Size, and Learning Rate",
                "Precision, Dropout, and Weight Decay",
                "Tokenization, Quantization, and Pruning",
            ],
            "correct_index": 0,
            "explanation": "The RAG Triad measures whether retrieved context is relevant, whether the answer stays faithful to that context, and whether the response answers the user query (CS401 Unit 5).",
            "unit": "Unit 5: LLM Evaluation, Safety & Responsible AI",
        },
        {
            "question": "Which prompting paradigm interleaves step-by-step reasoning traces with external tool/action execution?",
            "options": [
                "ReAct (Reason + Act) prompting",
                "WordPiece tokenization",
                "Greedy decoding with temperature 0",
                "Static Word2Vec embedding lookup",
            ],
            "correct_index": 0,
            "explanation": "ReAct combines explicit reasoning steps with tool calls/actions so an LLM can fetch external observations before finalizing an answer (CS401 Unit 3).",
            "unit": "Unit 3: Prompt Engineering & Structured Generation",
        },
        {
            "question": "Why does RAG reduce hallucinations compared to standalone pre-trained LLMs?",
            "options": [
                "It grounds generation in up-to-date retrieved external evidence without requiring costly full retraining",
                "It doubles the number of hidden layers inside the Transformer",
                "It disables system instructions during inference",
                "It removes stop words from the output vocabulary",
            ],
            "correct_index": 0,
            "explanation": "RAG injects verified domain documents at query time, addressing both knowledge cutoffs and factual fabrication (CS401 Unit 4).",
            "unit": "Unit 4: Retrieval-Augmented Generation (RAG) & Vector Search",
        },
    ],
    "cs302": [
        {
            "question": "What is the primary difference between L1 (Lasso) and L2 (Ridge) regularization?",
            "options": [
                "L1 encourages sparse weights (feature selection), whereas L2 shrinks weights smoothly via weight decay",
                "L1 is only used for clustering, whereas L2 is only used for tokenization",
                "L2 sets all weights strictly to zero while L1 increases variance",
                "Neither affects the Bias-Variance Tradeoff",
            ],
            "correct_index": 0,
            "explanation": "L1 penalty produces sparse solutions that zero out irrelevant features, while L2 penalizes squared magnitudes to prevent overfitting (CS302 Unit 1).",
            "unit": "Unit 1: Supervised Learning & Model Generalization",
        },
        {
            "question": "Which dimensionality reduction technique is specifically designed for visualizing high-dimensional embedding clusters non-linearly?",
            "options": [
                "t-SNE or UMAP",
                "Stratified k-fold cross-validation",
                "BM25 Okapi ranking",
                "Batch Normalization",
            ],
            "correct_index": 0,
            "explanation": "While PCA is linear, t-SNE and UMAP preserve local neighborhood geometry to visualize complex high-dimensional embeddings (CS302 Unit 2).",
            "unit": "Unit 2: Unsupervised Learning & Representation Learning",
        },
        {
            "question": "How do contextualized embeddings (e.g., BERT) improve upon static embeddings (Word2Vec, GloVe)?",
            "options": [
                "They generate dynamic token representations conditioned on surrounding sentence context",
                "They ignore word order completely like Bag-of-Words",
                "They only work on numerical tabular CSV datasets",
                "They eliminate the need for any neural network layers",
            ],
            "correct_index": 0,
            "explanation": "Contextualized embeddings assign different vectors to polysemous words (like 'bank') depending on context, unlike static lookup tables (CS302 Unit 4).",
            "unit": "Unit 4: Classical & Neural Natural Language Processing",
        },
        {
            "question": "In production MLOps monitoring, what is the difference between data drift and concept drift?",
            "options": [
                "Data drift is a shift in input feature distributions; concept drift is a change in the relationship between inputs and target labels",
                "Data drift only occurs during Docker builds; concept drift only occurs during tokenization",
                "Concept drift means losing GPU drivers; data drift means missing CSV headers",
                "They are identical synonyms for training loss convergence",
            ],
            "correct_index": 0,
            "explanation": "Monitoring both input distribution shifts (data drift) and evolving ground-truth mappings (concept drift) prevents silent model degradation (CS302 Unit 5).",
            "unit": "Unit 5: MLOps, Deployment & Monitoring",
        },
        {
            "question": "Which set of metrics is most informative when evaluating a classifier on a heavily imbalanced dataset?",
            "options": [
                "Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix analysis",
                "Raw Accuracy alone",
                "Silhouette Score alone",
                "Perplexity alone",
            ],
            "correct_index": 0,
            "explanation": "High raw accuracy can be misleading on imbalanced classes; Precision, Recall, F1-Score, and ROC-AUC capture minority-class performance accurately (CS302 Unit 1).",
            "unit": "Unit 1: Supervised Learning & Model Generalization",
        },
    ],
    "policy": [
        {
            "question": "What is the minimum attendance required in every theory and lab course to sit for End-Semester Examinations?",
            "options": [
                "75% independently in every registered theory and laboratory course",
                "60% averaged across all courses",
                "85% for theory and 50% for labs",
                "There is no attendance requirement",
            ],
            "correct_index": 0,
            "explanation": "Academic Regulations Section 2 mandates at least 75% attendance per course, with condonation possible between 65% and 74.9%.",
            "unit": "Academic Regulations: Attendance Policy",
        },
        {
            "question": "Under the University Generative AI Policy, when is LLM usage permitted in coursework?",
            "options": [
                "For brainstorming, concept clarification, and debugging with a required AI Disclosure Statement",
                "For submitting verbatim AI-generated exam answers without attribution",
                "Only after paying a INR 1,500 condonation fee",
                "Never under any circumstances",
            ],
            "correct_index": 0,
            "explanation": "Students may use LLMs responsibly for learning and debugging when accompanied by an AI Disclosure Statement citing tools and prompts.",
            "unit": "Academic Regulations: Academic Integrity & AI Policy",
        },
        {
            "question": "What are the passing requirements for a standard university theory course?",
            "options": [
                "At least 40% overall AND at least 35% independently in the End-Semester Examination",
                "At least 50% overall with no ESE minimum",
                "At least 30% overall and 30% in quizzes",
                "Only passing the Mid-Semester Exam is required",
            ],
            "correct_index": 0,
            "explanation": "The Examination Handbook specifies dual passing criteria: >= 40% combined score and >= 35% in the ESE.",
            "unit": "Examination & Grading Handbook",
        },
        {
            "question": "What CGPA threshold qualifies a student for the Chancellor's Excellence Scholarship (100% tuition waiver)?",
            "options": [
                "Top 5% of the department with a CGPA of 9.2 or higher and zero backlogs",
                "CGPA of 7.5 or higher",
                "CGPA of 8.0 with at least 60% attendance",
                "Any student who completes 14 credits",
            ],
            "correct_index": 0,
            "explanation": "The Chancellor's Excellence Scholarship covers 100% of tuition for top 5% departmental scholars maintaining >= 9.2 CGPA.",
            "unit": "Scholarships & Financial Aid Guide",
        },
        {
            "question": "Within how many days of result publication can a student apply for answer paper re-evaluation, and when is the INR 1,000 fee refunded?",
            "options": [
                "Within 7 days; refunded in full if the score increases by 5% or more",
                "Within 30 days; never refunded",
                "Within 24 hours; refunded only if the grade changes to O",
                "Within 14 days; half is refunded automatically",
            ],
            "correct_index": 0,
            "explanation": "Re-evaluation requests are due within 7 days (INR 1,000 fee), and the full fee is refunded when marks increase by at least 5%.",
            "unit": "Examination & Grading Handbook",
        },
    ],
}


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_units_from_doc(doc: Document) -> list[tuple[str, list[str]]]:
    """Parse markdown sections and bullet topics from a syllabus or policy document."""
    units: list[tuple[str, list[str]]] = []
    current_heading = doc.title
    topics: list[str] = []

    for line in doc.content.splitlines():
        stripped = line.strip()
        heading = re.match(r"^#{2,3}\s+(.+)$", stripped)
        if heading:
            if topics:
                units.append((current_heading, topics))
            current_heading = heading.group(1).strip()
            topics = []
        elif stripped.startswith(("- ", "* ")):
            item = re.sub(r"^[-*]\s+", "", stripped)
            item = re.sub(r"\*\*", "", item).strip()
            if len(item) > 12:
                topics.append(item)

    if topics:
        units.append((current_heading, topics))
    return units


def generate_quiz(
    doc: Document,
    llm_client: CampusLLMClient,
    num_questions: int = 5,
    difficulty: str = "Medium",
) -> tuple[list[QuizQuestion], str]:
    """Generate structured MCQ questions using Gemini JSON mode or Offline Syllabus Bank."""
    if llm_client.is_live_gemini:
        sys_p, user_p = build_quiz_prompt(
            course_title=doc.title,
            syllabus_excerpt=doc.content[:4500],
            num_questions=num_questions,
            difficulty=difficulty,
        )
        resp = llm_client.generate_text(
            system_instruction=sys_p,
            user_prompt=user_p,
            temperature=0.3,
            json_mode=True,
        )
        if not resp.error and resp.text:
            try:
                parsed = json.loads(_strip_json_fences(resp.text))
                if isinstance(parsed, dict) and "questions" in parsed:
                    parsed = parsed["questions"]
                questions: list[QuizQuestion] = []
                for item in parsed[:num_questions]:
                    opts = item.get("options", [])
                    if len(opts) == 4:
                        questions.append(
                            QuizQuestion(
                                question=str(item["question"]),
                                options=[str(o) for o in opts],
                                correct_index=int(item.get("correct_index", 0)) % 4,
                                explanation=str(item.get("explanation", "")),
                                unit=str(item.get("unit", doc.title)),
                            )
                        )
                if questions:
                    return questions, resp.backend
            except Exception:
                pass

    # Offline grounded fallback: pick matching curated bank or dynamically derive from uploaded doc
    key_text = f"{doc.source} {doc.title}".lower()
    if "cs401" in key_text or "generative ai" in key_text:
        bank = OFFLINE_QUIZ_BANK["cs401"]
    elif "cs302" in key_text or "machine learning" in key_text:
        bank = OFFLINE_QUIZ_BANK["cs302"]
    elif any(k in key_text for k in ("regulation", "exam", "scholarship", "handbook", "service")):
        bank = OFFLINE_QUIZ_BANK["policy"]
    else:
        bank = _build_dynamic_offline_quiz(doc, num_questions)

    selected = bank[:num_questions]
    return (
        [
            QuizQuestion(
                question=q["question"],
                options=q["options"],
                correct_index=q["correct_index"],
                explanation=q["explanation"],
                unit=q["unit"],
            )
            for q in selected
        ],
        "Offline Demo Mode (Syllabus-Grounded Quiz Engine)",
    )


def _build_dynamic_offline_quiz(doc: Document, num_questions: int) -> list[dict]:
    """Dynamically synthesize conceptual MCQs from any user-uploaded document."""
    units = _extract_units_from_doc(doc)
    questions: list[dict] = []
    all_units = [u for u, _ in units] or [doc.title]

    for idx, (unit_name, topics) in enumerate(units):
        if len(questions) >= num_questions:
            break
        if not topics:
            continue
        fact = topics[0]
        distractors = [u for u in all_units if u != unit_name][:3]
        while len(distractors) < 3:
            distractors.append(f"Unrelated Administrative Topic {len(distractors) + 1}")
        options = [unit_name] + distractors[:3]
        questions.append(
            {
                "question": f"According to '{doc.title}', which section covers: \"{fact[:115]}\"?",
                "options": options,
                "correct_index": 0,
                "explanation": f"This concept appears directly under '{unit_name}' in {doc.source}.",
                "unit": unit_name,
            }
        )

    if not questions:
        questions = OFFLINE_QUIZ_BANK["cs401"][:num_questions]
    return questions


def generate_flashcards(doc: Document, max_cards: int = 8) -> list[Flashcard]:
    """Extract structured Active-Recall flashcards directly from document units."""
    units = _extract_units_from_doc(doc)
    cards: list[Flashcard] = []
    for unit_name, topics in units:
        for topic in topics:
            if len(cards) >= max_cards:
                return cards
            if ":" in topic:
                term, definition = topic.split(":", 1)
                if len(term.strip()) > 3 and len(definition.strip()) > 10:
                    cards.append(
                        Flashcard(
                            unit=unit_name,
                            front=f"Explain {term.strip()} ({unit_name})",
                            back=definition.strip(),
                        )
                    )
                    continue
            cards.append(
                Flashcard(
                    unit=unit_name,
                    front=f"Key concept in {unit_name}: What should you know about '{topic[:55]}...'?",
                    back=topic,
                )
            )
    return cards


def generate_study_plan(
    doc: Document,
    llm_client: CampusLLMClient,
    days_until_exam: int = 10,
    hours_per_day: float = 2.5,
    weak_topics: str = "",
) -> StudyPlan:
    """Generate a personalized multi-phase study plan via Gemini JSON or Offline Planner."""
    if llm_client.is_live_gemini:
        sys_p, user_p = build_study_plan_prompt(
            course_title=doc.title,
            syllabus_text=doc.content,
            days_until_exam=days_until_exam,
            hours_per_day=hours_per_day,
            weak_topics=weak_topics,
        )
        resp = llm_client.generate_text(
            system_instruction=sys_p,
            user_prompt=user_p,
            temperature=0.35,
            json_mode=True,
        )
        if not resp.error and resp.text:
            try:
                data = json.loads(_strip_json_fences(resp.text))
                phases = [
                    StudyPhase(
                        phase=str(p.get("phase", f"Phase {i + 1}")),
                        focus_units=str(p.get("focus_units", "")),
                        topics=[str(t) for t in p.get("topics", [])],
                        active_recall_task=str(p.get("active_recall_task", "")),
                        hours=float(p.get("hours", hours_per_day)),
                    )
                    for i, p in enumerate(data.get("schedule", []))
                ]
                if phases:
                    return StudyPlan(
                        course_title=doc.title,
                        summary=str(data.get("summary", "")),
                        schedule=phases,
                        exam_tips=[str(t) for t in data.get("exam_tips", [])],
                        backend_used=resp.backend,
                    )
            except Exception:
                pass

    # Deterministic, syllabus-grounded spaced-repetition planner (Offline Demo Mode)
    units = _extract_units_from_doc(doc)
    study_units = [u for u in units if "objective" not in u[0].lower() and "assessment" not in u[0].lower()]
    if not study_units:
        study_units = units or [(doc.title, ["Review full document notes", "Practice core problems"])]

    weak_lower = weak_topics.lower().strip()
    # Prioritize units matching student's weak topics
    weighted_units = sorted(
        study_units,
        key=lambda u: 0 if (weak_lower and any(w in (u[0] + " " + " ".join(u[1])).lower() for w in weak_lower.split(","))) else 1,
    )

    total_hours = round(days_until_exam * hours_per_day, 1)
    core_days = max(1, days_until_exam - 2) if days_until_exam >= 4 else max(1, days_until_exam - 1)
    num_phases = min(len(weighted_units), max(2, min(5, core_days)))
    days_per_phase = max(1, core_days // num_phases)

    schedule: list[StudyPhase] = []
    day_cursor = 1
    for i in range(num_phases):
        unit_name, topics = weighted_units[i]
        end_day = min(core_days, day_cursor + days_per_phase - 1)
        span_days = max(1, end_day - day_cursor + 1)
        phase_label = f"Day {day_cursor}" if day_cursor == end_day else f"Days {day_cursor}–{end_day}"
        priority_tag = " (High-Priority Focus)" if (weak_lower and i == 0) else ""
        schedule.append(
            StudyPhase(
                phase=phase_label,
                focus_units=f"{unit_name}{priority_tag}",
                topics=topics[:3],
                active_recall_task=(
                    f"Close notes and write a 1-page concept sheet for {unit_name.split(':')[0]}, "
                    "then complete a 5-question timed quiz in the Quiz Studio."
                ),
                hours=round(span_days * hours_per_day, 1),
            )
        )
        day_cursor = end_day + 1
        if day_cursor > core_days:
            break

    # Final spaced-repetition & mock exam phase
    if day_cursor <= days_until_exam:
        final_span = days_until_exam - day_cursor + 1
        final_label = (
            f"Day {day_cursor}" if day_cursor == days_until_exam else f"Days {day_cursor}–{days_until_exam}"
        )
        schedule.append(
            StudyPhase(
                phase=final_label,
                focus_units="Full-Syllabus Revision & Timed Mock Exam",
                topics=[
                    "Spaced-repetition review of flashcards & missed quiz questions",
                    "Solve one full-length 3-hour End-Semester style practice paper under exam rules",
                    "Formula, theorem, and architecture quick-reference sweep",
                ],
                active_recall_task="Take a closed-book mock exam and audit every incorrect answer against the syllabus.",
                hours=round(final_span * hours_per_day, 1),
            )
        )

    focus_note = f" with extra emphasis on '{weak_topics}'" if weak_topics.strip() else ""
    return StudyPlan(
        course_title=doc.title,
        summary=(
            f"A {days_until_exam}-day ({total_hours} total hours) active-recall study roadmap for "
            f"{doc.title}{focus_note}, balancing deep conceptual mastery with final timed mock testing."
        ),
        schedule=schedule,
        exam_tips=[
            "Use 50-minute focused study blocks followed by 10-minute active recall summaries without peeking at notes.",
            "Remember the university passing criteria: aim well above the 40% overall and 35% ESE thresholds.",
            "Bring your University Photo ID Card and Exam Hall Ticket, and review past lab/project architectures.",
        ],
        backend_used="Offline Demo Mode (Spaced-Repetition Syllabus Planner)",
    )
