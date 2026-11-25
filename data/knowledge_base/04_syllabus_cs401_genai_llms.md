# Course Syllabus: CS401 — Generative AI & Large Language Models

**Credits:** 4 (3 Lectures + 1 Lab) | **Prerequisites:** CS302 Machine Learning, Probability & Linear Algebra
**Instructor:** Prof. Ananya Sharma | **Office Hours:** Tuesdays & Thursdays, 3:00 PM – 5:00 PM (Block C, Room 412)

## Course Objectives
This course provides theoretical foundations and hands-on engineering skills in Generative AI and Large Language Models (LLMs), covering Transformer architectures, pre-training, alignment, Retrieval-Augmented Generation (RAG), prompt engineering, AI agents, evaluation, and Responsible AI.

## Unit 1: Foundations of Generative Models & Transformers (Weeks 1–3)
- Discriminative vs. Generative modeling; Autoregressive language modeling and next-token prediction.
- Tokenization algorithms: Byte-Pair Encoding (BPE), WordPiece, and SentencePiece.
- Transformer architecture: Scaled Dot-Product Attention (`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`), Multi-Head Attention, Positional Encodings (Sinusoidal, RoPE), Layer Normalization, and Residual Connections.
- Encoder-only (BERT), Decoder-only (GPT, Gemini, Llama), and Encoder-Decoder (T5) architectures.

## Unit 2: Pre-training, Adaptation & Alignment (Weeks 4–6)
- Scaling laws (Chinchilla optimal compute), KV-caching, and inference optimization (quantization: INT8/INT4, FlashAttention).
- Parameter-Efficient Fine-Tuning (PEFT): LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices `Delta W = B * A` to dramatically reduce trainable parameters, QLoRA, and Prefix Tuning.
- Alignment techniques: Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), Reward Modeling, and Direct Preference Optimization (DPO).

## Unit 3: Prompt Engineering & Structured Generation (Weeks 7–8)
- Zero-shot, Few-shot (In-Context Learning), Chain-of-Thought (CoT), Tree-of-Thoughts, and ReAct (Reason + Act) prompting.
- System instructions, temperature, top-p (nucleus sampling), top-k sampling, and structured JSON schema outputs.
- Tool use / Function calling and agentic workflow orchestration.

## Unit 4: Retrieval-Augmented Generation (RAG) & Vector Search (Weeks 9–11)
- Why RAG mitigates LLM hallucinations and knowledge cutoffs without expensive retraining.
- Document ingestion, chunking strategies (fixed-size with overlap, recursive, semantic chunking).
- Dense vector embeddings, cosine similarity, Approximate Nearest Neighbor (ANN) indexes (HNSW, IVF), and Hybrid Search combining lexical BM25/TF-IDF with dense embeddings via Reciprocal Rank Fusion (RRF).
- Reranking, query expansion, and grounded citation generation.

## Unit 5: LLM Evaluation, Safety & Responsible AI (Weeks 12–14)
- Automated metrics: BLEU, ROUGE, BERTScore, Perplexity, Retrieval Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- RAG Triad evaluation: Context Relevance, Groundedness (Faithfulness), and Answer Relevance; LLM-as-a-Judge frameworks.
- Safety & Guardrails: Prompt injection defense, jailbreak mitigation, PII leakage prevention, bias auditing, and EU AI Act / NIST AI RMF principles.

## Capstone Project Requirements (30% of Total Grade)
- Teams of 1–3 students must design, implement, and evaluate a real-world GenAI/LLM application addressing a **university or societal challenge**.
- Deliverables: Working interactive prototype, automated evaluation benchmark, source code repository, and a technical report covering architecture, prompt design, guardrails, and ethical analysis.
- **Submission Deadline:** Week 14, Friday 11:59 PM.
