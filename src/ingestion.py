"""Document ingestion and section-aware overlapping text chunking."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Document:
    """Represents a loaded university policy, syllabus, or study document."""

    doc_id: str
    source: str
    title: str
    content: str
    category: str = "General"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class TextChunk:
    """A single indexed chunk derived from a Document with lineage metadata."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    category: str
    text: str
    token_estimate: int
    char_start: int
    char_end: int


def estimate_tokens(text: str) -> int:
    """Approximate LLM token count (~0.75 words per token for English text)."""
    words = len(re.findall(r"\S+", text))
    return max(1, int(round(words * 1.33)))


def infer_category(filename: str, title: str) -> str:
    """Categorize documents for filtering and analytics."""
    combined = f"{filename} {title}".lower()
    if "syllabus" in combined or "cs401" in combined or "cs302" in combined or "course" in combined:
        return "Course Syllabus"
    if "exam" in combined or "grading" in combined:
        return "Examination & Grading"
    if "scholarship" in combined or "aid" in combined or "grant" in combined:
        return "Scholarships & Aid"
    if "regulation" in combined or "attendance" in combined or "policy" in combined:
        return "Academic Policy"
    if "service" in combined or "library" in combined or "handbook" in combined:
        return "Campus Services"
    return "Uploaded Document"


def extract_title(text: str, fallback: str) -> str:
    """Extract first Markdown H1 heading or fallback to filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback.replace("_", " ").replace("-", " ").title()


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf if installed."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF upload requires the 'pypdf' package. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages.append(f"## Page {idx}\n{extracted}")
    return "\n\n".join(pages)


def load_document_from_bytes(filename: str, raw_bytes: bytes) -> Document:
    """Load an uploaded .md, .txt, or .pdf document into a Document object."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        content = extract_pdf_text(raw_bytes)
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    title = extract_title(content, Path(filename).stem)
    category = infer_category(filename, title)
    return Document(
        doc_id=re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem.lower()),
        source=filename,
        title=title,
        content=content.strip(),
        category=category,
        metadata={"origin": "user_upload"},
    )


def load_knowledge_base(directory: Path) -> list[Document]:
    """Load all .md and .txt files from the university knowledge base directory."""
    documents: list[Document] = []
    if not directory.exists():
        return documents

    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title = extract_title(content, path.stem)
        category = infer_category(path.name, title)
        documents.append(
            Document(
                doc_id=path.stem,
                source=path.name,
                title=title,
                content=content,
                category=category,
                metadata={"path": str(path), "origin": "built_in"},
            )
        )
    return documents


def _split_into_sections(markdown_text: str, default_section: str) -> list[tuple[str, str]]:
    """Split markdown text on ## / ### headings while keeping section context."""
    sections: list[tuple[str, str]] = []
    current_heading = default_section
    buffer: list[str] = []

    for line in markdown_text.splitlines():
        heading_match = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
        if heading_match:
            joined = "\n".join(buffer).strip()
            # Skip emitting a standalone H1-only preamble with no body lines
            non_title_lines = [ln for ln in joined.splitlines() if ln.strip() and not ln.strip().startswith("# ")]
            if joined and non_title_lines:
                sections.append((current_heading, joined))
            current_heading = heading_match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)

    if buffer and "\n".join(buffer).strip():
        sections.append((current_heading, "\n".join(buffer).strip()))

    return sections or [(default_section, markdown_text.strip())]


def chunk_document(
    doc: Document,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """Section-aware overlapping chunker that preserves semantic boundaries."""
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    sections = _split_into_sections(doc.content, doc.title)
    chunks: list[TextChunk] = []
    chunk_counter = 0
    cursor = 0

    for section_name, section_text in sections:
        text = section_text.strip()
        if not text:
            continue

        if len(text) <= chunk_size:
            chunk_counter += 1
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}::c{chunk_counter:02d}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    section=section_name,
                    category=doc.category,
                    text=text,
                    token_estimate=estimate_tokens(text),
                    char_start=cursor,
                    char_end=cursor + len(text),
                )
            )
            cursor += len(text) + 2
            continue

        # Sliding window with sentence/newline boundary alignment
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                # Prefer snapping to paragraph or sentence boundary
                newline_pos = text.rfind("\n", start + chunk_size // 2, end)
                sentence_pos = text.rfind(". ", start + chunk_size // 2, end)
                snap = max(newline_pos, sentence_pos + 1 if sentence_pos != -1 else -1)
                if snap > start + chunk_size // 3:
                    end = snap

            piece = text[start:end].strip()
            if piece:
                chunk_counter += 1
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc.doc_id}::c{chunk_counter:02d}",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        section=section_name,
                        category=doc.category,
                        text=piece,
                        token_estimate=estimate_tokens(piece),
                        char_start=cursor + start,
                        char_end=cursor + end,
                    )
                )

            if end >= len(text):
                break
            next_start = max(start + 1, end - chunk_overlap)
            # Snap overlap start forward to a clean word/newline boundary
            if next_start < len(text) and text[next_start - 1] not in " \n\t":
                ws_pos = text.find(" ", next_start, end)
                nl_pos = text.find("\n", next_start, end)
                candidates = [p + 1 for p in (ws_pos, nl_pos) if p != -1]
                if candidates:
                    next_start = min(candidates)
            start = next_start

        cursor += len(text) + 2

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """Chunk an entire collection of documents."""
    all_chunks: list[TextChunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
