# CampusAI: Trustworthy Hybrid Retrieval-Augmented Generation with Multi-Stage Guardrails and Structured Study Synthesis for Higher Education

**Author(s):** *[Student Name(s)]*  
**Affiliation:** *Department of Computer Science & Engineering / Artificial Intelligence, [University Name]*  
**Contact:** *[student@university.edu]*

---

## Abstract

Large Language Models (LLMs) offer compelling conversational interfaces for higher education, yet their direct deployment in university administration and academic advising remains hindered by factual hallucinations, institutional knowledge cutoffs, numerical imprecision, and vulnerability to adversarial prompt injection. In this paper, we present **CampusAI**, a lightweight, auditable Retrieval-Augmented Generation (RAG) and structured study-support architecture tailored to university ecosystems. CampusAI introduces three complementary design contributions: (1) a **coverage-damped Hybrid Retrieval Engine** fusing unigram/bigram TF-IDF cosine similarity, BM25 Okapi lexical ranking, alphanumeric course-code entity alignment, and optional dense neural embeddings via Reciprocal Rank Fusion (RRF); (2) a **multi-stage Responsible AI guardrail pipeline** combining pre-generation prompt injection filtering, Personally Identifiable Information (PII) redaction, relevance-thresholded safe refusal, and post-generation **numeric consistency verification**; and (3) a **dual-mode inference architecture** pairing Google Gemini API structured generation with a deterministic, zero-dependency Maximal Marginal Relevance (MMR) extractive fallback for reproducible offline evaluation. On a 12-case golden university benchmark spanning academic regulations, examination policies, scholarships, technical course syllabi, out-of-scope queries, and prompt injection attacks, CampusAI achieves **100.0% Retrieval Recall@4**, **1.000 Mean Reciprocal Rank (MRR)**, **100.0% Numeric Consistency**, **100.0% Safe Refusal Accuracy**, **100.0% Adversarial Block Rate**, **0.895 Mean Grounding Score**, and **91.7% Overall Pass Rate** with sub-millisecond local latency (**0.33–0.40 ms**). Controlled ablation studies empirically demonstrate that omitting lexical BM25 fusion reduces pass rate from 91.7% to 58.3%, while removing relevance gating or safety guardrails collapses refusal and injection defense to 0.0%.

**Keywords:** Retrieval-Augmented Generation (RAG), Large Language Models, Hybrid Information Retrieval, Reciprocal Rank Fusion, Hallucination Mitigation, Prompt Injection Defense, Educational Technology, Responsible AI.

---

## I. Introduction

Higher education institutions govern student progression through dense, frequently updated textual policies covering degree credit rules, attendance thresholds, examination grading scales, financial aid eligibility, and multi-unit course syllabi. Although modern Large Language Models (LLMs) excel at open-domain language synthesis [1], [2], institutional question answering constitutes a *high-precision, closed-world* domain where fabricating a single figure—such as reporting a 65% attendance cutoff instead of 75%, or misstating an examination re-evaluation deadline—can inflict tangible academic or financial harm on students.

Retrieval-Augmented Generation (RAG) [3] addresses parametric knowledge cutoffs by conditioning LLM generation on dynamically retrieved institutional passages. Nevertheless, practical campus RAG deployments still encounter four recurring failure modes:
1. **Lexical-Semantic Retrieval Mismatch:** Pure dense embeddings often under-weight exact course codes (`CS401`, `CS302`) and numeric rules, whereas pure TF-IDF cosine similarity suffers score compression on multi-clause natural language queries.
2. **Spurious Matching on Out-of-Scope Queries:** Standard BM25 scoring can assign artificially high scores when an unanswerable query shares a single incidental token (e.g., *"monthly"*) with an unrelated document, prompting the LLM to hallucinate.
3. **Unverified Numeric Claims:** Even with retrieved context, generative models may conflate percentages, CGPA thresholds, or currency amounts across paragraphs without automated post-hoc verification.
4. **Security & Reproducibility Barriers:** Student-facing systems are exposed to prompt injection attacks [4] and accidental PII disclosure, while academic research prototypes frequently break when commercial API keys expire or rate-limit.

To address these challenges simultaneously, we design, implement, and benchmark **CampusAI**, an open, modular university assistant providing grounded policy Q&A, structured syllabus-based study tools (MCQ quizzes, active-recall flashcards, and spaced-repetition study plans), transparent evidence/prompt inspection, and quantitative evaluation.

---

## II. Related Work

### A. Retrieval-Augmented Generation and Hybrid Search
Lewis et al. [3] established RAG by pairing dense passage retrieval with sequence-to-sequence generation. Subsequent production systems demonstrated that combining sparse lexical retrievers such as BM25 [5] with dense vector search via Reciprocal Rank Fusion (RRF) [6] consistently improves robustness on domain-specific entities, acronyms, and exact identifiers. CampusAI extends this paradigm with **query-term coverage damping** and **course-code entity boosting** specifically calibrated for university corpora.

### B. Prompt Engineering and Structured Generation
Few-Shot In-Context Learning [1] and Chain-of-Thought (CoT) prompting [7] improve multi-step reasoning and citation compliance. Modern LLM APIs further support constrained JSON schema decoding, enabling reliable generation of educational artifacts such as self-graded multiple-choice assessments and structured study schedules.

### C. RAG Evaluation and Responsible AI Guardrails
Automated RAG evaluation frameworks such as RAGAS [8] formalize the *RAG Triad*—Context Relevance, Faithfulness (Groundedness), and Answer Relevance. Concurrently, the NIST AI Risk Management Framework [9] and OWASP Top 10 for LLMs [4] emphasize layered defenses against prompt injection, sensitive data disclosure, and hallucination. CampusAI integrates these principles into a lightweight, auditable pipeline featuring explicit numeric consistency verification.

---

## III. System Architecture and Methodology

CampusAI orchestrates six modular stages: (1) section-aware ingestion, (2) pre-generation safety guardrails, (3) hybrid retrieval with RRF, (4) relevance-gated refusal, (5) configurable prompt engineering with dual-mode inference, and (6) post-generation faithfulness auditing.

### A. Section-Aware Overlapping Document Chunking
Given a corpus of institutional documents $\mathcal{C} = \{D_1, \dots, D_M\}$ (`.md`, `.txt`, `.pdf`), each document is parsed along Markdown heading boundaries (`##`, `###`) so policy sections remain semantically intact. Sections exceeding `chunk_size = 900` characters are segmented via a sliding window with `chunk_overlap = 140` characters, snapping chunk ends to paragraph or sentence boundaries and snapping overlap starts forward to word boundaries to prevent mid-token fragmentation. Each resulting chunk $d_i$ retains structural metadata `(doc_id, source, title, section, category, token_estimate)`.

### B. Coverage-Damped Hybrid Retrieval with RRF
Every indexed chunk representation concatenates document title, section heading, and chunk text to preserve hierarchical context. Tokens undergo lightweight morphological normalization (handling plurals and `-ing` inflections while preserving course codes and numeric tokens).

1. **TF-IDF Phrase Vector Similarity:** Documents and queries are projected into L2-normalized unigram + bigram TF-IDF space using sublinear term weighting $w_{t,d} = (1 + \log \text{tf}_{t,d}) \cdot \left(\log\frac{1+N}{1+\text{df}_t} + 1\right)$, yielding cosine similarity $S_{\text{tfidf}}(q, d_i) \in [0, 1]$.

2. **Coverage-Damped BM25 Okapi:** Let $\text{BM25}(q, d_i)$ denote standard Okapi BM25 ($k_1 = 1.5, b = 0.75$) and $c(q, d_i) = \frac{|T_q \cap T_{d_i}|}{|T_q|}$ denote distinct query-term coverage. We compute:
   $$S_{\text{bm25}}(q, d_i) = \left(1 - e^{-\text{BM25}(q, d_i) / \tau}\right) \cdot \left(0.35 + 0.65\,c(q, d_i)\right), \quad \tau = 5.0$$
   The coverage factor suppresses false-positive scores when an out-of-scope query overlaps a chunk on only a single generic token.

3. **Course-Code Entity Alignment Boost:** If query tokens contain an institutional course code matching `^[a-z]{2,4}\d{3}$` (e.g., `CS401`, `CS302`) that appears in chunk $d_i$, an entity boost $B_{\text{code}}(q, d_i) = 0.18$ is added.

4. **Hybrid Scoring & RRF:** When Google Gemini dense embeddings (`text-embedding-004`) are enabled, dense cosine similarity $S_{\text{dense}}$ is fused as:
   $$S_{\text{hybrid}}(q, d_i) = \min\!\Big(1.0,\; 0.50\,S_{\text{dense}} + 0.25\,S_{\text{tfidf}} + 0.25\,S_{\text{bm25}} + B_{\text{code}}\Big)$$
   In local/offline mode ($\alpha = 0.55$):
   $$S_{\text{hybrid}}(q, d_i) = \min\!\Big(1.0,\; \alpha\,S_{\text{tfidf}} + (1 - \alpha)\,S_{\text{bm25}} + B_{\text{code}}\Big)$$
   Reciprocal Rank Fusion $\text{RRF}(d_i) = \sum_{m \in \mathcal{M}} \frac{1}{60 + \text{rank}_m(d_i)}$ serves as a principled tie-breaker and explainability metric.

### C. Multi-Stage Responsible AI Guardrails
1. **Input Injection & PII Filter:** Before retrieval, queries are screened against six adversarial attack signatures (instruction overrides, system prompt extraction, persona jailbreaks, guardrail bypasses, policy overrides, and privilege escalation). Simultaneously, regex PII detectors mask emails, phone numbers, and government ID patterns to `[REDACTED_PII]`.
2. **Relevance-Gated Safe Refusal:** If $\max_i S_{\text{hybrid}}(q, d_i) < \theta_{\text{rel}}$ ($\theta_{\text{rel}} = 0.28$), CampusAI halts generation and returns a transparent safe refusal advising the user to upload the missing policy or contact university administration.
3. **Post-Generation Faithfulness & Numeric Consistency Audit:** Let $A_{\text{tok}}$ and $C_{\text{tok}}$ denote content tokens in the generated answer and retrieved evidence, and $A_{\text{num}}, C_{\text{num}}$ denote extracted numerical quantities (percentages, CGPA values, fees, deadlines). We compute:
   $$\text{LexicalGrounding} = \frac{|A_{\text{tok}} \cap C_{\text{tok}}|}{|A_{\text{tok}}|}, \qquad \text{NumericConsistency} = \frac{|A_{\text{num}} \cap C_{\text{num}}|}{|A_{\text{num}}|}$$
   $$\text{GroundingScore} = 0.65 \cdot \text{LexicalGrounding} + 0.35 \cdot \text{NumericConsistency}$$
   Any element in $A_{\text{num}} \setminus C_{\text{num}}$ is flagged as an unsupported numeric claim.

### D. Dual-Mode Inference & Structured Study Synthesis
When a Google Gemini API key is provided, CampusAI invokes `gemini-2.5-flash` (or `gemini-2.5-pro`) via the `google-genai` SDK with an HTTPS REST fallback, supporting toggleable **Zero-Shot**, **Few-Shot**, and **Grounded CoT** prompts as well as strict JSON schema outputs for MCQs and study plans. When operating offline, CampusAI automatically falls back to a **Grounded Extractive RAG Synthesizer** using **MMR-style greedy query-term coverage**: candidate factual sentences from retrieved chunks are iteratively selected by maximizing $\text{BaseScore}(s) + 0.65 \cdot \frac{|(T_q \setminus T_{\text{cov}}) \cap T_s|}{|T_q|}$, encouraging multi-aspect answer completeness with exact `[Source N]` provenance.

---

## IV. Experimental Setup

### A. University Corpus
We curated six realistic university documents (`35` indexed chunks, `~3,100` vocabulary features): Academic Regulations (`01`), Examination & Grading Handbook (`02`), Scholarships & Research Grants (`03`), CS401 Generative AI & LLMs Syllabus (`04`), CS302 Machine Learning Systems Syllabus (`05`), and Campus Services & Library Handbook (`06`).

### B. Golden Evaluation Benchmark
Our automated evaluation harness (`src/evaluation.py`) tests **12 curated benchmark cases** (`data/eval_dataset.json`):
- **10 Answerable Queries (Q01–Q10)** spanning single-policy lookups, numerical thresholds, syllabus topics, and compound multi-clause questions.
- **1 Out-of-Scope Hallucination Probe (Q11):** Asking for monthly electric motorcycle parking permit fees near South Stadium (absent from corpus).
- **1 Adversarial Prompt Injection Probe (Q12):** Attempting system prompt extraction and attendance policy override.

### C. Metrics
We report **Retrieval Recall@K**, **Mean Reciprocal Rank (MRR)**, **Key-Fact Recall** (fraction of ground-truth facts surfaced), **Mean Grounding Score**, **Numeric Consistency**, **Citation Rate**, **Safe Refusal Accuracy**, **Guardrail Block Rate**, **Overall Pass Rate**, and **Mean End-to-End Latency (ms)**.

---

## V. Results and Quantitative Analysis

### A. Primary Benchmark Performance
Table I summarizes CampusAI's verified performance in its reproducible default configuration (`alpha = 0.55`, `top_k = 4`, `chunk_size = 900`, `min_relevance = 0.28`, Offline Grounded RAG baseline).

**Table I: Primary Benchmark Results on the 12-Case University Evaluation Suite**

| Evaluation Metric | Score |
| :--- | :---: |
| Retrieval Recall@4 | **100.0%** |
| Mean Reciprocal Rank (MRR) | **1.000** |
| Numeric Consistency | **100.0%** |
| Citation Rate | **100.0%** |
| Safe Refusal Accuracy (Q11) | **100.0%** |
| Guardrail Block Rate (Q12) | **100.0%** |
| Mean Grounding Score | **0.895** |
| Key-Fact Recall (Q01–Q10) | **85.5%** |
| Overall Benchmark Pass Rate | **91.7%** (11/12) |
| Mean End-to-End Latency | **0.33–0.40 ms** |

Every single answerable query retrieved its exact authoritative document at **Rank #1 (`MRR = 1.000`)**, achieved **100% numeric fidelity** with zero fabricated figures, and cited inline evidence. Q11 scored `0.142` (well below $\theta_{\text{rel}} = 0.28$), triggering Safe Refusal, while Q12 was intercepted at `0.0 ms` prior to retrieval.

### B. Controlled Ablation Study
Table II isolates the contribution of each architectural component across seven controlled experiments.

**Table II: Empirical Ablation Study Across Retrieval, Thresholding, and Safety Configurations**

| Variant | Pass Rate | Recall@K | MRR | Key-Fact Recall | Grounding | Refusal Acc. | Block Rate | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure TF-IDF (`alpha=1.0`) | 58.3% | 100.0% | 1.000 | 47.5% | 0.921 | 100.0% | 100.0% | 0.24 ms |
| Pure BM25 (`alpha=0.0`) | 91.7% | 100.0% | 1.000 | 85.5% | 0.910 | 100.0% | 100.0% | 0.47 ms |
| Without Input Guardrails | 83.3% | 100.0% | 1.000 | 85.5% | 0.895 | 100.0% | **0.0%** | 0.34 ms |
| Without Relevance Gate ($\theta=0$) | 83.3% | 100.0% | 1.000 | 85.5% | 0.914 | **0.0%** | 100.0% | 0.50 ms |
| Hybrid RRF (`k=2`) | 91.7% | 100.0% | 1.000 | 85.5% | 0.888 | 100.0% | 100.0% | 0.27 ms |
| Hybrid RRF (`k=6`) | 91.7% | 100.0% | 1.000 | 85.5% | 0.898 | 100.0% | 100.0% | 0.36 ms |
| **CampusAI Default (`alpha=0.55, k=4`)** | **91.7%** | **100.0%** | **1.000** | **85.5%** | **0.895** | **100.0%** | **100.0%** | **0.33 ms** |

Three insights emerge from Table II:
1. **Pure TF-IDF Underperforms Under Strict Refusal Thresholds:** Although Pure TF-IDF achieves `100%` Recall@4, its cosine magnitudes frequently fall below $\theta_{\text{rel}} = 0.28$ on multi-topic queries, inducing false refusals and dropping Key-Fact Recall by **38.0 percentage points** (`85.5%` $\to$ `47.5%`).
2. **Balanced Hybrid Fusion Improves Efficiency & Stability:** Combining TF-IDF with coverage-damped BM25 (`alpha=0.55`) matches BM25's peak accuracy while reducing mean latency by **29.8%** (`0.47 ms` $\to$ `0.33 ms`) and providing smooth integration with dense Gemini embeddings when live.
3. **Orthogonal Safety Guarantees:** Removing pre-generation guardrails drops injection defense from `100%` to `0%`, while removing the relevance gate drops out-of-scope refusal from `100%` to `0%`, proving both controls are non-redundant.

---

## VI. Error Analysis

Inspection of the sole benchmark failure (**Q04**: *"What is the maximum grade I can receive in a supplementary exam and what are the minimum passing percentages?"*) reveals a classic **multi-hop retrieval dilution** phenomenon. While CampusAI ranked `02_examination_and_grading.md` (Section 4: Supplementary Exams, `B+` cap) at #1 (`0.457`), the second clause (*40% overall and 35% ESE minimum passing thresholds*) resides in Section 2 beneath an extensive Markdown grade table, pushing that chunk outside the top-4 retrieval window. Single-vector or single-bag-of-words retrieval inherently favors the dominant clause in compound queries. Future integration of **sub-query decomposition** (breaking compound questions into atomic queries prior to parallel retrieval) and **cross-encoder reranking** directly addresses this bottleneck.

---

## VII. Responsible AI, Security, and Ethical Implications

Deploying LLM-driven assistants in universities introduces unique governance obligations:
- **Hallucination Accountability:** By combining relevance-gated safe refusals, inline `[Source N]` provenance, and automated numeric consistency scoring, CampusAI ensures students can audit every institutional policy figure before taking academic action.
- **Academic Integrity:** Consistent with institutional AI policies, CampusAI's study tools emphasize formative active recall (MCQs, flashcards, revision planning) rather than automated completion of graded deliverables.
- **Privacy Preservation:** Pre-generation PII redaction mitigates accidental leakage of student identifiers to external APIs, while the zero-dependency offline engine enables air-gapped on-premise deployment.
- **Adversarial Defense Limitations:** While rule-based guardrails block direct prompt injections (`100%` on Q12), production campus deployments should add semantic safety classifiers and sanitize user-uploaded PDFs against *indirect* prompt injections.

---

## VIII. Conclusion and Future Work

We presented **CampusAI**, a complete, empirically validated Generative AI and LLM framework for university academic policies, syllabi, and adaptive exam preparation. Through coverage-damped Hybrid Retrieval with RRF, multi-stage Responsible AI guardrails, structured JSON study synthesis, and a dual-mode Google Gemini / Offline architecture, CampusAI achieves **100% Recall@4**, **1.000 MRR**, **100% Numeric Consistency**, and **100% Refusal & Safety Accuracy** while remaining fully inspectable and reproducible.

Future work will focus on (1) LLM-driven query decomposition and neural cross-encoder reranking to solve multi-hop policy queries, (2) scaling to persistent vector databases (FAISS/ChromaDB) across multi-year institutional archives, (3) multilingual support for regional and international students, and (4) read-only LMS/SSO integration for individualized student advising.

---

## References

[1] T. Brown et al., "Language Models are Few-Shot Learners," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 1877–1901, 2020.  
[2] A. Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.  
[3] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.  
[4] OWASP Foundation, "OWASP Top 10 for Large Language Model Applications," v2025.  
[5] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.  
[6] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," *Proc. ACM SIGIR*, pp. 758–759, 2009.  
[7] J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.  
[8] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *arXiv preprint arXiv:2309.15217*, 2023.  
[9] National Institute of Standards and Technology (NIST), "Artificial Intelligence Risk Management Framework (AI RMF 1.0)," NIST AI 100-1, Jan. 2023.  
[10] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *Proc. International Conference on Learning Representations (ICLR)*, 2022.
