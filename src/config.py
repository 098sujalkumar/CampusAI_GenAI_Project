"""Configuration defaults and environment loading for CampusAI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
DOCS_DIR = PROJECT_ROOT / "docs"


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Lightweight .env loader (no external dependency required)."""
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()


@dataclass
class RAGConfig:
    """Hyperparameters controlling ingestion, retrieval, prompting, and generation."""

    chunk_size: int = 900
    chunk_overlap: int = 140
    top_k: int = 4
    min_relevance_score: float = 0.28
    hybrid_alpha: float = 0.55  # Weight for semantic/TF-IDF cosine vs BM25 lexical score
    prompt_strategy: str = "grounded_cot"  # zero_shot | few_shot | grounded_cot
    temperature: float = 0.2
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"
    use_gemini_embeddings: bool = False  # Opt-in dense embeddings when API key is configured
    enable_guardrails: bool = True

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
        )


def get_gemini_api_key(override: str | None = None) -> str | None:
    """Return an explicit API key override or read GEMINI_API_KEY / GOOGLE_API_KEY."""
    if override and override.strip():
        return override.strip()
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if val and not val.startswith("your_"):
            return val
    return None
