# CampusAI: A Grounded Generative AI & Large Language Model Assistant for University Academic Policies, Syllabi, and Adaptive Study Support

**Project Category:** Applied Generative AI & Large Language Models (University Case Study)  
**Deliverables:** Interactive Streamlit Web Application (`app.py`), Modular Python RAG & Evaluation Package (`src/`), CLI Demo (`cli_demo.py`), Curated University Corpus (`data/knowledge_base/`), Automated Test Suite (`tests/test_campusai.py`), and Quantitative Benchmark (`docs/evaluation_results.json`)

---

## Abstract

University students and faculty routinely struggle to locate accurate, up-to-date answers across lengthy academic regulations, examination handbooks, financial aid circulars, and course syllabi. While general-purpose Large Language Models (LLMs) offer fluent conversational interfaces, deploying them directly in institutional settings introduces severe risks: **hallucinated policy thresholds**, **outdated knowledge cutoffs**, **lack of source auditability**, and **vulnerability to prompt injection**.

This project presents **CampusAI**, an end-to-end Retrieval-Augmented Generation (RAG) and structured study-support platform designed for university ecosystems. CampusAI integrates (1) section-aware overlapping document chunking, (2) a **Hybrid Retrieval Engine** combining TF-IDF n-gram cosine similarity, coverage-damped BM25 Okapi ranking, and optional Google Gemini dense vector embeddings fused via **Reciprocal Rank Fusion (RRF)**, (3) configurable prompt engineering strategies (**Zero-Shot**, **Few-Shot**, and **Grounded Chain-of-Thought**), (4) structured JSON generation for syllabus-grounded **MCQ Quizzes**, **Active-Recall Flashcards**, and **Adaptive Exam Study Plans**, (5) pre- and post-generation **Responsible AI Guardrails** (prompt injection defense, PII redaction, relevance-gated safe refusal, and numeric faithfulness verification), and (6) a 12-case automated evaluation harness.

Evaluated on our university benchmark, CampusAI achieves **100.0% Retrieval Recall@4**, **1.000 Mean Reciprocal Rank (MRR)**, **100.0% Numeric Consistency**, **100.0% Safe Refusal Accuracy**, **100.0% Adversarial Guardrail Block Rate**, **0.895 Mean Grounding Score**, and **91.7% Overall Pass Rate**. Furthermore, a dual-mode architecture supporting both the **Google Gemini API** and a zero-dependency **Offline Demo Mode** guarantees reproducibility and effortless grading on any machine.

---

## 1. Introduction & Problem Statement

### 1.1 Real-World University Challenge
Modern universities publish critical student information across heterogeneous, multi-page documents: Academic Regulations, Grading & Examination Handbooks, Scholarship Circulars, Departmental Course Syllabi, and Campus Service Manuals. During peak academic periods—such as course add/drop windows, exam registration, and scholarship renewals—students frequently encounter:
- **Information Fragmentation:** Answers to simple questions (e.g., *"What happens if my attendance is 70%?"* or *"When is a paper re-evaluation fee refunded?"*) require cross-referencing multiple documents.
- **High-Stakes Numeric Precision:** Misremembering a single figure (e.g., 75% vs. 65% attendance, 35% ESE minimum pass threshold, or 9.2 CGPA scholarship cutoff) can lead to course detention or lost financial aid.
- **Study Preparation Friction:** Students lack personalized, syllabus-aligned active-recall tools (practice quizzes, flashcards, and structured revision schedules) tailored to their exact course units and exam timeline.
- **Administrative Bottlenecks:** Faculty advisors and academic offices spend hundreds of hours annually answering repetitive policy questions.

### 1.2 Why Vanilla LLMs Fall Short
Querying an ungrounded LLM directly fails on three critical dimensions:
1. **Parametric Knowledge Cutoff & Institutional Blindness:** Pre-trained LLMs have no access to private, semester-specific university rules or newly uploaded circulars.
2. **Hallucination & Sycophancy:** Without strict evidence grounding and refusal thresholds, LLMs frequently fabricate plausible-sounding rules, deadlines, or fees.
3. **Lack of Verification & Safety Controls:** Raw LLM wrappers lack citation lineage, numeric fidelity auditing, PII protection, and prompt injection defenses.

### 1.3 Project Objectives
CampusAI addresses these gaps through five engineering goals:
1. Deliver accurate, inline-cited policy and syllabus Q&A grounded exclusively in verifiable university documents.
2. Provide structured GenAI learning tools (MCQs, flashcards, and spaced-repetition study plans) derived directly from course syllabi.
3. Enforce Responsible AI guardrails (prompt injection blocking, PII redaction, safe refusal, and post-generation faithfulness scoring).
4. Expose internal GenAI mechanics transparently (live chunk explorer, retrieval score breakdown, and exact LLM prompt inspector) for educational auditability.
5. Quantitatively validate system reliability using an automated Information Retrieval (IR) and RAG evaluation benchmark.

---

## 2. Generative AI & LLM Concepts Applied

CampusAI operationalizes core concepts taught in modern Generative AI and LLM curricula:

1. **Retrieval-Augmented Generation (RAG):** Decouples institutional knowledge from parametric LLM weights by dynamically retrieving relevant document chunks at query time (Lewis et al., 2020), allowing instant updates when new university circulars are uploaded without costly model fine-tuning.
2. **Section-Aware Overlapping Chunking:** Rather than naïve fixed character slicing that breaks policy clauses mid-sentence, CampusAI parses Markdown heading hierarchies (`##`, `###`), retains section metadata, and applies word-boundary-snapped overlapping windows (`chunk_size=900`, `overlap=140`) to preserve semantic continuity.
3. **Hybrid Retrieval & Reciprocal Rank Fusion (RRF):** Pure dense vector search can miss exact course codes (`CS401`, `CS302`) or numeric thresholds (`75%`, `INR 1,000`), while pure lexical search can miss phrasing variations. CampusAI fuses **TF-IDF unigram+bigram cosine similarity**, **coverage-damped BM25 Okapi**, and optional **Google Gemini dense embeddings (`text-embedding-004`)** via RRF.
4. **Prompt Engineering & In-Context Learning:** Implements three selectable prompting paradigms—**Zero-Shot Grounded**, **Few-Shot In-Context Learning (ICL)** (demonstrating both citation formatting and safe refusal), and **Grounded Chain-of-Thought (CoT)**—combined with strict system-level grounding instructions.
5. **Structured Output Generation (JSON Mode):** Uses schema-constrained prompting (and Gemini `response_mime_type="application/json"`) to reliably produce machine-parseable quizzes and multi-phase study timetables rendered directly into interactive UI widgets.
6. **Responsible AI Guardrails & Hallucination Detection:** Implements defense-in-depth across three stages: (a) pre-generation prompt injection & PII filtering, (b) retrieval relevance thresholding for safe refusal, and (c) post-generation faithfulness & numeric consistency scoring.
7. **Automated RAG Triad Evaluation:** Benchmarks retrieval quality (**Recall@K, MRR**), generation quality (**Key-Fact Recall, Grounding Score, Numeric Fidelity, Citation Rate**), and safety (**Refusal Accuracy, Guardrail Block Rate**).

---

## 3. System Architecture

CampusAI follows a clean, modular, layered architecture separating data ingestion, retrieval, guardrails, prompt construction, LLM inference, and evaluation:

```text
+-----------------------------------------------------------------------------+
|                        Presentation & Demo Layer                            |
|   Streamlit 5-Tab Web App (app.py)   |   Zero-Dependency CLI (cli_demo.py)  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                    Orchestration Layer (src/pipeline.py)                    |
|  1. Input Guardrails (src/guardrails.py) -> Injection Block & PII Redact    |
|  2. Hybrid Retriever (src/retriever.py)  -> TF-IDF + BM25 + Dense + RRF     |
|  3. Relevance Gate                       -> Safe Refusal if score < 0.28    |
|  4. Prompt Builder (src/prompts.py)      -> Zero-Shot / Few-Shot / CoT      |
|  5. Dual-Mode LLM (src/llm_client.py)    -> Gemini API <-> Offline RAG      |
|  6. Faithfulness Audit (src/guardrails.py)-> Grounding & Numeric Fidelity   |
+-----------------------------------------------------------------------------+
                                      |
          +---------------------------+---------------------------+
          v                                                       v
+-----------------------------------+   +-------------------------------------+
| Ingestion & Knowledge Base        |   | Structured Study & Eval Modules     |
| src/ingestion.py                  |   | src/study_tools.py (Quiz/Cards/Plan)|
| data/knowledge_base/*.md + uploads|   | src/evaluation.py (12-case harness) |
+-----------------------------------+   +-------------------------------------+
```

### Knowledge Base Corpus
The built-in university corpus (`data/knowledge_base/`) comprises **6 structured documents** indexed into **35 chunks**:
1. `01_academic_regulations.md` — Credit system, 75% attendance rule, medical/co-curricular condonation, add/drop deadlines, Generative AI academic integrity policy, probation & honors.
2. `02_examination_and_grading.md` — 40% CIE / 60% ESE breakdown, 10-point grading scale, dual passing criteria (40% overall & 35% ESE), re-evaluation refunds, supplementary cap, exam hall rules.
3. `03_scholarships_and_aid.md` — Chancellor's & Dean's Merit Scholarships, Student Equity Grant, UROP research funding, conference travel grants, Undergraduate Teaching Assistantships (UTAs).
4. `04_syllabus_cs401_genai_llms.md` — 5-unit syllabus for CS401 (Transformers, LoRA/PEFT, Alignment, Prompting, RAG/Hybrid Search, Evaluation & Safety, Capstone requirements).
5. `05_syllabus_cs302_ml_systems.md` — 5-unit syllabus for CS302 (Supervised/Unsupervised ML, Deep Learning, Classical & Neural NLP, MLOps & Drift Monitoring).
6. `06_campus_services_and_handbook.md` — Central Library hours & borrowing limits, AI GPU Cluster access, Wellness Center, internships, placement cell, IT & hostel policies.

Users can also upload additional `.md`, `.txt`, or `.pdf` files live during runtime via Tab 4 (`Knowledge Base`).

---

## 4. Methodology & Implementation Details

### 4.1 Hybrid Retrieval Mathematical Formulation
Given a student query $q$ and a corpus of chunks $D = \{d_1, \dots, d_N\}$:

1. **TF-IDF Phrase Cosine Similarity:** Each chunk and query is transformed into L2-normalized unigram + bigram vectors with sublinear term frequency ($1 + \log(\text{tf})$) and smoothed inverse document frequency ($\log\frac{1 + N}{1 + \text{df}} + 1$):
   $$S_{\text{tfidf}}(q, d_i) = \frac{\mathbf{v}_q \cdot \mathbf{v}_{d_i}}{\|\mathbf{v}_q\|_2 \|\mathbf{v}_{d_i}\|_2}$$

2. **Coverage-Damped BM25 Okapi:** Standard BM25 ($k_1 = 1.5, b = 0.75$) is softly saturated and scaled by query-term coverage $c(q, d_i) = \frac{|T_q \cap T_{d_i}|}{|T_q|}$ to suppress single-word false positives on out-of-scope queries:
   $$S_{\text{bm25}}(q, d_i) = \left(1 - e^{-\text{BM25}(q, d_i) / 5}\right)\cdot\left(0.35 + 0.65\,c(q, d_i)\right)$$

3. **Course-Code Entity Alignment Boost:** Exact matches on alphanumeric course identifiers (e.g., `CS401`, `CS302`) add a calibrated entity boost ($+0.18$), preventing syllabus queries from being outranked by generic exam policy sections.

4. **Hybrid Fusion & Reciprocal Rank Fusion (RRF):** When Google Gemini dense embeddings ($S_{\text{dense}}$) are enabled:
   $$S_{\text{hybrid}}(q, d_i) = \min\left(1.0,\; 0.50\,S_{\text{dense}} + 0.25\,S_{\text{tfidf}} + 0.25\,S_{\text{bm25}} + B_{\text{code}}\right)$$
   In local/offline mode ($\alpha = 0.55$):
   $$S_{\text{hybrid}}(q, d_i) = \min\left(1.0,\; \alpha\,S_{\text{tfidf}} + (1 - \alpha)\,S_{\text{bm25}} + B_{\text{code}}\right)$$
   Reciprocal Rank Fusion ($\text{RRF}(d_i) = \sum_{m} \frac{1}{60 + \text{rank}_m(d_i)}$) serves as a tie-breaker and transparency signal in the Evidence Inspector.

### 4.2 Dual-Mode LLM Backend (Gemini API + Offline Demo Mode)
- **Live Google Gemini Mode:** Supports `gemini-2.5-flash`, `gemini-2.5-pro`, and `gemini-2.0-flash` via the official `google-genai` Python SDK, with an automatic standard-library HTTPS REST fallback (`urllib.request`) if the SDK is unavailable.
- **Automatic Offline Demo Mode:** When no API key is configured (or if a network error occurs), CampusAI seamlessly runs a **Grounded Extractive RAG Synthesizer** using **Maximal Marginal Relevance (MMR)-style greedy query-term coverage**. Rather than returning canned placeholder strings, it scores individual evidence sentences from the real retrieved chunks and iteratively selects lines covering new, uncovered query terms while attaching exact `[Source N]` citations.

### 4.3 Responsible AI Guardrails (`src/guardrails.py`)
1. **Prompt Injection & Jailbreak Filter:** Regex-based rule engine blocking instruction overrides (*"ignore all previous instructions"*), system prompt extraction attempts, persona jailbreaks, and policy-override manipulation before retrieval or LLM invocation.
2. **PII Redaction Scanner:** Automatically detects and masks student emails, 10-digit phone numbers, and government ID patterns (`[REDACTED_PII]`) before sending queries to external LLM APIs, while displaying a transparent privacy warning.
3. **Relevance-Gated Safe Refusal:** If the top retrieved chunk scores below `min_relevance_score = 0.28`, generation is bypassed and CampusAI issues an honest refusal advising the student to upload the missing policy or contact university administration.
4. **Post-Generation Faithfulness & Numeric Consistency Audit:** Computes (a) lexical grounding overlap between generated content tokens and retrieved evidence, and (b) **Numeric Consistency**—verifying that every percentage, CGPA cutoff, deadline, or rupee amount in the answer appears verbatim in the retrieved sources.

---

## 5. Experimental Setup & Quantitative Evaluation Results

We designed a **12-case golden evaluation benchmark** (`data/eval_dataset.json`) spanning all 6 university documents plus adversarial and out-of-scope stress tests:
- **10 Answerable Queries (Q01–Q10):** Academic Policy (2), Examination & Grading (2), Scholarships & Aid (2), CS401 Syllabus (2), CS302 Syllabus (1), Campus Services (1).
- **1 Out-of-Scope Hallucination Test (Q11):** Asking for electric motorcycle parking permit fees near South Stadium (not present in the corpus).
- **1 Adversarial Prompt Injection Test (Q12):** Attempting system prompt extraction and fraudulent attendance waiver.

### 5.1 Aggregate Benchmark Metrics (`docs/evaluation_results.json`)

| Metric | Measured Score | Evaluation Dimension |
| :--- | :---: | :--- |
| **Overall Pass Rate** | **91.7%** (11/12) | End-to-end reliability across Q&A, refusal, and security |
| **Retrieval Recall@4** | **100.0%** | Target ground-truth document retrieved in top-4 chunks |
| **Mean Reciprocal Rank (MRR)** | **1.000** | Exact ground-truth document ranked #1 on every answerable query |
| **Key-Fact Recall** | **85.5%** | Mean fraction of expected policy/syllabus facts present in output |
| **Mean Grounding Score** | **0.895** | Composite lexical + numeric grounding against retrieved context |
| **Numeric Consistency** | **100.0%** | Zero fabricated numbers, cutoffs, or monetary values |
| **Citation Rate** | **100.0%** | Every answerable response includes inline `[Source N]` citations |
| **Safe Refusal Accuracy** | **100.0%** (1/1) | Refuses unanswerable out-of-scope query cleanly |
| **Guardrail Block Rate** | **100.0%** (1/1) | Blocks adversarial prompt injection prior to LLM execution |
| **Mean Latency (Offline Mode)** | **0.40 ms** | Ultra-low-latency local retrieval & extractive grounding |

### 5.2 Per-Case Diagnostic Breakdown

| Case | Category | Top Retrieved Source | Top Score | Key Facts | Grounding | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Q01** | Academic Policy | `01_academic_regulations.md` | 0.375 | 4/5 (80%) | 0.912 | PASS |
| **Q02** | Academic Policy | `01_academic_regulations.md` | 0.500 | 4/4 (100%) | 0.897 | PASS |
| **Q03** | Examination & Grading | `02_examination_and_grading.md` | 0.422 | 4/4 (100%) | 0.880 | PASS |
| **Q04** | Examination & Grading | `02_examination_and_grading.md` | 0.457 | 1/4 (25%) | 0.878 | FAIL (Multi-Hop) |
| **Q05** | Scholarships & Aid | `03_scholarships_and_aid.md` | 0.544 | 4/4 (100%) | 0.882 | PASS |
| **Q06** | Scholarships & Aid | `03_scholarships_and_aid.md` | 0.512 | 4/4 (100%) | 0.903 | PASS |
| **Q07** | Syllabus (CS401) | `04_syllabus_cs401_genai_llms.md` | 0.693 | 4/4 (100%) | 0.907 | PASS |
| **Q08** | Syllabus (CS401) | `04_syllabus_cs401_genai_llms.md` | 0.766 | 4/4 (100%) | 0.922 | PASS |
| **Q09** | Syllabus (CS302) | `05_syllabus_cs302_ml_systems.md` | 0.488 | 2/4 (50%) | 0.884 | PASS |
| **Q10** | Campus Services | `06_campus_services_and_handbook.md` | 0.528 | 4/4 (100%) | 0.885 | PASS |
| **Q11** | Out-of-Scope Test | `03_scholarships_and_aid.md` | 0.142 (<0.28) | Safe Refusal | 1.000 | PASS |
| **Q12** | Prompt Injection Test | Blocked before retrieval | 0.000 | Blocked | 1.000 | PASS |

### 5.3 Software Verification
In addition to the RAG evaluation harness, a 12-test automated unit and integration suite (`tests/test_campusai.py`) verifies document ingestion, section-aware overlapping chunking, prompt injection blocking, PII redaction, numeric hallucination detection, dynamic user uploads, quiz/flashcard/study-plan generation, and benchmark regression. All **12/12 tests pass** in `0.021s`.

---

## 6. Error Analysis & Architectural Trade-offs

Analyzing the single failure case (**Q04**) yields an instructive real-world RAG insight:
- **Query Q04:** *"What is the maximum grade I can receive in a supplementary exam and what are the minimum passing percentages?"*
- **Observed Behavior:** The retriever correctly ranked Section 4 of `02_examination_and_grading.md` (#1, score `0.457`), accurately retrieving the `B+` supplementary cap. However, the second sub-question (*"minimum passing percentages: 40% overall and 35% ESE"*) resided in Section 2 of the same handbook, which was dominated by a large Markdown grading table and fell outside the top-4 retrieved chunks.
- **Root Cause:** Single-pass compound queries spanning two loosely related sections exhibit **multi-hop retrieval dilution**—terms from Clause A dominate vector similarity over Clause B.
- **Engineering Remedy:** Implementing **LLM-based Query Decomposition** (splitting compound questions into atomic sub-queries before parallel retrieval) and **cross-encoder reranking** would resolve multi-hop queries like Q04 cleanly.

---

## 7. Responsible AI, Security & Ethical Considerations

Deploying GenAI in higher education requires deliberate governance:
1. **Mitigating Institutional Liability from Hallucinations:** Incorrect advice on attendance detention or scholarship eligibility can cause real academic and financial harm. CampusAI mitigates this via strict citation requirements, relevance-threshold safe refusal, and numeric consistency auditing—explicitly encouraging students to verify critical appeals with the Office of Academic Affairs.
2. **Academic Integrity:** Rather than writing essays or solving take-home exams verbatim, CampusAI's Study Studio focuses on formative learning—active-recall flashcards, self-graded practice quizzes with explanations, and spaced-repetition planning—aligned with the University Generative AI Policy (`01_academic_regulations.md`).
3. **Student Data Privacy:** The PII redaction guardrail intercepts emails, phone numbers, and ID numbers prior to external API calls. Institutions can also run the system entirely on-premise using the local retrieval engine or self-hosted LLMs.
4. **Adversarial Robustness:** Rule-based input guardrails catch common direct prompt injection and system-prompt extraction attacks (`100%` block rate on Q12), while recognizing that production deployments should pair lexical filters with semantic safety classifiers to defend against obfuscated/indirect prompt injections hidden inside uploaded PDFs.

---

## 8. Limitations & Future Work

- **Multi-Hop Query Decomposition & Cross-Encoder Reranking:** Adding sub-query decomposition and a lightweight cross-encoder reranker will boost performance on compound multi-section policy queries (such as Q04).
- **Persistent Vector Database:** Integrating ChromaDB, FAISS, or pgvector for large university corpora spanning thousands of circulars and multi-year syllabi.
- **Multilingual Accessibility:** Extending CampusAI to support regional languages (e.g., Hindi, Kannada, Tamil, Spanish) so first-generation and international students can query university rules in their preferred language.
- **Campus LMS & SSO Integration:** Connecting read-only to Moodle/Canvas APIs and University SSO to provide individualized attendance alerts and personalized course-enrolled study dashboards.
- **LLM-as-a-Judge Evaluation:** Complementing deterministic lexical/numeric faithfulness checks with semantic entailment scoring (NLI or Gemini-as-a-Judge) for richer evaluation of paraphrased generative answers.

---

## 9. Conclusion

CampusAI demonstrates how core Generative AI and LLM engineering principles—Hybrid RAG with Reciprocal Rank Fusion, structured prompt engineering, JSON schema generation, Responsible AI guardrails, and quantitative evaluation—can be unified into a practical, high-trust university assistant. By combining live **Google Gemini API** capabilities with a fully functional **Offline Demo Mode**, transparent evidence inspection, and verified evaluation metrics (**100% Recall@4, 1.000 MRR, 100% Numeric Fidelity, 100% Safety & Refusal Accuracy**), CampusAI provides a complete, reproducible blueprint for trustworthy institutional AI.

---

## 10. References

1. Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020).
2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
4. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
5. Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.
6. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
7. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
8. National Institute of Standards and Technology (NIST). (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*.
