"""Build crisp A4 PDFs for Synopsis, Report & Research Paper + <10MB Google Form ZIP bundle."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission_docs"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

ARCH_SVG = """
<div class="diagram-box">
<svg viewBox="0 0 820 275" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/>
    </marker>
  </defs>
  <!-- Top row -->
  <rect x="12" y="24" width="132" height="58" rx="8" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>
  <text x="78" y="48" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">Student / Faculty</text>
  <text x="78" y="65" text-anchor="middle" font-size="10" fill="#334155">Query or Doc Upload</text>

  <rect x="178" y="24" width="144" height="58" rx="8" fill="#fef2f2" stroke="#b91c1c" stroke-width="1.5"/>
  <text x="250" y="47" text-anchor="middle" font-size="11.5" font-weight="700" fill="#991b1b">Input Guardrails</text>
  <text x="250" y="64" text-anchor="middle" font-size="9.5" fill="#475569">Injection Block + PII Redact</text>

  <rect x="356" y="24" width="168" height="58" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="1.5"/>
  <text x="440" y="47" text-anchor="middle" font-size="11.5" font-weight="700" fill="#155e75">Hybrid Retriever (RRF)</text>
  <text x="440" y="64" text-anchor="middle" font-size="9.5" fill="#475569">TF-IDF + BM25 + Gemini Dense</text>

  <rect x="558" y="24" width="118" height="58" rx="8" fill="#fffbeb" stroke="#b45309" stroke-width="1.5"/>
  <text x="617" y="47" text-anchor="middle" font-size="11.5" font-weight="700" fill="#92400e">Relevance Gate</text>
  <text x="617" y="64" text-anchor="middle" font-size="9.5" fill="#475569">Score &ge; 0.28?</text>

  <rect x="706" y="24" width="102" height="58" rx="8" fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>
  <text x="757" y="47" text-anchor="middle" font-size="11" font-weight="700" fill="#1e293b">Safe Refusal</text>
  <text x="757" y="64" text-anchor="middle" font-size="9.5" fill="#475569">No Hallucination</text>

  <!-- Bottom row -->
  <rect x="12" y="168" width="170" height="62" rx="8" fill="#f8fafc" stroke="#475569" stroke-width="1.5"/>
  <text x="97" y="191" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">University Corpus</text>
  <text x="97" y="208" text-anchor="middle" font-size="9.5" fill="#475569">6 Built-in Docs + Live Uploads</text>
  <text x="97" y="221" text-anchor="middle" font-size="9" fill="#64748b">(35 Section-Aware Chunks)</text>

  <rect x="218" y="168" width="166" height="62" rx="8" fill="#f5f3ff" stroke="#6d28d9" stroke-width="1.5"/>
  <text x="301" y="191" text-anchor="middle" font-size="11.5" font-weight="700" fill="#5b21b6">Prompt Engineering</text>
  <text x="301" y="208" text-anchor="middle" font-size="9.5" fill="#475569">Zero-Shot / Few-Shot / CoT</text>
  <text x="301" y="221" text-anchor="middle" font-size="9" fill="#64748b">+ JSON Quiz &amp; Study Plans</text>

  <rect x="420" y="168" width="176" height="62" rx="8" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>
  <text x="508" y="191" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">Dual-Mode LLM Client</text>
  <text x="508" y="208" text-anchor="middle" font-size="9.5" fill="#475569">Google Gemini API (Live)</text>
  <text x="508" y="221" text-anchor="middle" font-size="9" fill="#64748b">or MMR Offline RAG Mode</text>

  <rect x="632" y="168" width="176" height="62" rx="8" fill="#f0fdf4" stroke="#15803d" stroke-width="1.5"/>
  <text x="720" y="191" text-anchor="middle" font-size="11.5" font-weight="700" fill="#166534">Faithfulness Audit</text>
  <text x="720" y="208" text-anchor="middle" font-size="9.5" fill="#475569">Grounding + Numeric Fidelity</text>
  <text x="720" y="221" text-anchor="middle" font-size="9" fill="#64748b">&rarr; Cited UI / CLI / Eval</text>

  <!-- Arrows -->
  <line x1="144" y1="53" x2="176" y2="53" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
  <line x1="322" y1="53" x2="354" y2="53" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
  <line x1="524" y1="53" x2="556" y2="53" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
  <line x1="676" y1="53" x2="704" y2="53" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
  <text x="690" y="45" text-anchor="middle" font-size="9" fill="#b91c1c" font-weight="700">No</text>

  <!-- Corpus to Hybrid Retriever -->
  <path d="M 97 168 L 97 125 L 440 125 L 440 84" fill="none" stroke="#0e7490" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text x="270" y="119" text-anchor="middle" font-size="9" fill="#0e7490" font-weight="600">Overlapping Section-Aware Chunks Indexed</text>

  <!-- Relevance Gate Yes -> Prompt Engineering -->
  <path d="M 617 82 L 617 142 L 301 142 L 301 166" fill="none" stroke="#15803d" stroke-width="1.6" marker-end="url(#arrow)"/>
  <text x="580" y="136" text-anchor="middle" font-size="9" fill="#15803d" font-weight="700">Yes (Top-K Chunks)</text>

  <!-- Bottom row connections -->
  <line x1="384" y1="199" x2="418" y2="199" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
  <line x1="596" y1="199" x2="630" y2="199" stroke="#334155" stroke-width="1.6" marker-end="url(#arrow)"/>
</svg>
<div class="diagram-caption">Figure 1: CampusAI End-to-End Hybrid RAG, Responsible AI Guardrails, and Dual-Mode Generation Architecture.</div>
</div>
"""


def _format_math_expr(expr: str) -> str:
    """Convert clean LaTeX math snippets into readable HTML mathematical typography."""
    s = expr.strip()
    # Handle nested \frac{...}{...} cleanly (up to 2 passes for simple braces)
    def _replace_fracs(txt: str) -> str:
        while r"\frac{" in txt:
            idx = txt.find(r"\frac{")
            # Find first balanced {...}
            depth = 0
            i = idx + 5
            start1 = i + 1
            while i < len(txt):
                if txt[i] == "{":
                    depth += 1
                elif txt[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end1 = i
                        break
                i += 1
            else:
                break
            num = txt[start1:end1]
            if end1 + 1 >= len(txt) or txt[end1 + 1] != "{":
                break
            depth = 0
            i = end1 + 1
            start2 = i + 1
            while i < len(txt):
                if txt[i] == "{":
                    depth += 1
                elif txt[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end2 = i
                        break
                i += 1
            else:
                break
            den = txt[start2:end2]
            txt = txt[:idx] + f"({num}) / ({den})" + txt[end2 + 1 :]
        return txt

    s = _replace_fracs(s)
    replacements = [
        (r"\\left\s*", ""),
        (r"\\right\s*", ""),
        (r"\\Big", ""),
        (r"\\text\{([^}]+)\}", r"\1"),
        (r"\\mathbf\{([^}]+)\}", r"<b>\1</b>"),
        (r"\\mathcal\{([^}]+)\}", r"<i>\1</i>"),
        (r"\\min", "min"),
        (r"\\max", "max"),
        (r"\\sum", "&Sigma;"),
        (r"\\log", "log"),
        (r"\\alpha", "&alpha;"),
        (r"\\tau", "&tau;"),
        (r"\\theta", "&theta;"),
        (r"\\cdot", "&middot;"),
        (r"\\cap", "&cap;"),
        (r"\\setminus", " &#8726; "),
        (r"\\in\b", "&isin;"),
        (r"\\to\b", "&rarr;"),
        (r"\\dots", "&hellip;"),
        (r"\\ge\b", "&ge;"),
        (r"\\le\b", "&le;"),
        (r"\\\|", "&#8214;"),
        (r"\\\{", "{"),
        (r"\\\}", "}"),
        (r"\\qquad", "&nbsp;&nbsp;&nbsp;&nbsp;"),
        (r"\\quad", "&nbsp;&nbsp;&nbsp;"),
        (r"\\;", "&nbsp;"),
        (r"\\,", "&thinsp;"),
        (r"\\!", ""),
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s)
    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", s)
    return s


def _inline_md(text: str) -> str:
    """Render inline markdown: code, bold, italic, links, and inline math."""
    # Protect inline code first
    code_tokens: list[str] = []

    def _stash_code(m: re.Match) -> str:
        code_tokens.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"@@CODE{len(code_tokens) - 1}@@"

    text = re.sub(r"`([^`]+)`", _stash_code, text)

    # Protect inline math $...$
    math_tokens: list[str] = []

    def _stash_math(m: re.Match) -> str:
        math_tokens.append(f"<span class='math-inline'>{_format_math_expr(m.group(1))}</span>")
        return f"@@MATH{len(math_tokens) - 1}@@"

    text = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", _stash_math, text)

    # Escape HTML safely
    text = html.escape(text)
    # Restore unescaped markdown links [label](href)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", text)
    # Bold + italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = text.replace("\\_", "_")

    for i, tok in enumerate(math_tokens):
        text = text.replace(f"@@MATH{i}@@", tok)
    for i, tok in enumerate(code_tokens):
        text = text.replace(f"@@CODE{i}@@", tok)
    return text


def markdown_to_html(md_text: str) -> str:
    """Clean, dependency-free Markdown-to-HTML converter supporting tables, math, code, and SVG diagrams."""
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block (or mermaid diagram)
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            i += 1
            block_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                block_lines.append(lines[i])
                i += 1
            i += 1
            if lang == "mermaid":
                out.append(ARCH_SVG)
            else:
                out.append(f"<pre><code>{html.escape(chr(10).join(block_lines))}</code></pre>")
            continue

        # Display math block $$...$$
        if stripped.startswith("$$"):
            expr = stripped[2:]
            if expr.endswith("$$") and len(expr) >= 2:
                expr = expr[:-2]
                i += 1
            else:
                parts = [expr]
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    parts.append(lines[i])
                    i += 1
                if i < n:
                    parts.append(lines[i].strip()[:-2])
                    i += 1
                expr = " ".join(parts)
            out.append(f"<div class='math-block'>{_format_math_expr(expr)}</div>")
            continue

        # Horizontal rule
        if stripped in {"---", "***", "___"}:
            out.append("<hr/>")
            i += 1
            continue

        # Headings
        h_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if h_match:
            lvl = len(h_match.group(1))
            out.append(f"<h{lvl}>{_inline_md(h_match.group(2))}</h{lvl}>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{_inline_md(' '.join(quote_lines))}</blockquote>")
            continue

        # Markdown Table
        if "|" in stripped and i + 1 < n and re.match(r"^\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1].strip()):
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < n and "|" in lines[i].strip() and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            tbl = ["<table><thead><tr>"]
            for h in headers:
                tbl.append(f"<th>{_inline_md(h)}</th>")
            tbl.append("</tr></thead><tbody>")
            for r in rows:
                tbl.append("<tr>")
                for c in r:
                    tbl.append(f"<td>{_inline_md(c)}</td>")
                tbl.append("</tr>")
            tbl.append("</tbody></table>")
            out.append("".join(tbl))
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline_md(it)}</li>" for it in items) + "</ul>")
            continue

        # Ordered list (preserve starting number so split lists keep numbering)
        ol_match = re.match(r"^(\d+)\.\s+", stripped)
        if ol_match:
            start_num = int(ol_match.group(1))
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            start_attr = f" start='{start_num}'" if start_num != 1 else ""
            out.append(f"<ol{start_attr}>" + "".join(f"<li>{_inline_md(it)}</li>" for it in items) + "</ol>")
            continue

        # Paragraph (respect Markdown two-space line breaks & bracketed [N] reference lines)
        if re.match(r"^\[\d+\]\s+", stripped):
            out.append(f"<p class='ref-item'>{_inline_md(stripped)}</p>")
            i += 1
            continue

        para = [stripped]
        hard_break = line.endswith("  ")
        i += 1
        while (
            not hard_break
            and i < n
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "```", "$$", "> ", "- ", "* ", "---"))
            and not re.match(r"^\d+\.\s+", lines[i].strip())
            and not re.match(r"^\[\d+\]\s+", lines[i].strip())
            and not ("|" in lines[i] and i + 1 < n and "---" in lines[i + 1])
        ):
            para.append(lines[i].strip())
            hard_break = lines[i].endswith("  ")
            i += 1
        out.append(f"<p>{_inline_md(' '.join(para))}</p>")

    return "\n".join(out)


ACADEMIC_CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10.2pt;
    line-height: 1.55;
    color: #1e293b;
}
h1 {
    font-size: 18pt;
    color: #0f172a;
    border-bottom: 2.2px solid #1d4ed8;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
}
h2 {
    font-size: 13pt;
    color: #1e3a8a;
    margin-top: 18px;
    margin-bottom: 7px;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    color: #0f172a;
    margin-top: 13px;
    margin-bottom: 5px;
    page-break-after: avoid;
}
p { margin: 0 0 8px 0; text-align: justify; }
ul, ol { margin: 4px 0 10px 0; padding-left: 20px; }
li { margin-bottom: 4px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th {
    background-color: #1e3a8a;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #1e3a8a;
}
td {
    padding: 5.5px 8px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
}
tr:nth-child(even) td { background-color: #f8fafc; }
code {
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
    font-size: 8.8pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
}
pre {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #1d4ed8;
    border-radius: 5px;
    padding: 9px 11px;
    font-size: 8.3pt;
    overflow-x: auto;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
blockquote {
    background: #eff6ff;
    border-left: 3.5px solid #1d4ed8;
    margin: 10px 0;
    padding: 8px 12px;
    color: #1e293b;
    font-style: italic;
}
hr { border: none; border-top: 1px solid #cbd5e1; margin: 14px 0; }
.math-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 8px 0 10px 0;
    text-align: center;
    font-family: "Times New Roman", Georgia, serif;
    font-size: 10.5pt;
    page-break-inside: avoid;
}
.math-inline {
    font-family: "Times New Roman", Georgia, serif;
    font-style: italic;
}
.diagram-box {
    margin: 12px 0 14px 0;
    padding: 8px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
    page-break-inside: avoid;
}
.diagram-caption {
    text-align: center;
    font-size: 8.6pt;
    color: #475569;
    margin-top: 4px;
    font-style: italic;
}
a { color: #1d4ed8; text-decoration: none; }
"""


def render_pdf_from_html(body_html: str, pdf_path: Path, footer_label: str) -> None:
    page_footer_css = f"""
    @page {{
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
        @bottom-left {{
            content: "{footer_label}";
            font-family: Helvetica, Arial, sans-serif;
            font-size: 8pt;
            color: #64748b;
        }}
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-family: Helvetica, Arial, sans-serif;
            font-size: 8pt;
            color: #64748b;
        }}
    }}
    .doc-break {{ page-break-before: always; break-before: page; }}
    """
    full_html = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        f"<title>{html.escape(footer_label)}</title>"
        f"<style>{ACADEMIC_CSS}\n{page_footer_css}</style></head><body>{body_html}</body></html>"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="campusai_pdf_"))
    temp_html = tmp_dir / "doc.html"
    profile_dir = tmp_dir / "chrome_profile"
    temp_html.write_text(full_html, encoding="utf-8")

    try:
        subprocess.run(
            [
                CHROME_BIN,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={profile_dir}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                temp_html.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_pdf(md_path: Path, pdf_path: Path, footer_label: str) -> None:
    body_html = markdown_to_html(md_path.read_text(encoding="utf-8"))
    render_pdf_from_html(body_html, pdf_path, footer_label)


def build_zip_bundle(zip_path: Path) -> None:
    excluded_parts = {"__pycache__", ".venv", ".git", ".pytest_cache", ".DS_Store"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file_path in sorted(PROJECT_ROOT.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.resolve() == zip_path.resolve():
                continue
            if any(part in excluded_parts for part in file_path.parts):
                continue
            if file_path.suffix == ".pyc":
                continue
            arcname = Path("CampusAI_GenAI_Project") / file_path.relative_to(PROJECT_ROOT)
            zf.write(file_path, arcname)


def main() -> None:
    targets = [
        ("01_Project_Synopsis.md", "01_Project_Synopsis.pdf", "CampusAI — Project Synopsis"),
        ("02_Project_Report.md", "02_Project_Report.pdf", "CampusAI — Project Report"),
        ("03_Research_Paper.md", "03_Research_Paper.pdf", "CampusAI — Research Paper"),
    ]
    combined_sections: list[str] = []
    for idx, (md_name, pdf_name, label) in enumerate(targets):
        md_file = SUBMISSION_DIR / md_name
        pdf_file = SUBMISSION_DIR / pdf_name
        section_html = markdown_to_html(md_file.read_text(encoding="utf-8"))
        render_pdf_from_html(section_html, pdf_file, label)
        size_kb = pdf_file.stat().st_size / 1024.0
        print(f"Created {pdf_file.name:44s} ({size_kb:6.1f} KB / {size_kb / 1024.0:.3f} MB)")
        if idx > 0:
            combined_sections.append(f"<div class='doc-break'></div>{section_html}")
        else:
            combined_sections.append(section_html)

    combined_pdf = SUBMISSION_DIR / "00_CampusAI_All_Documents_Combined.pdf"
    render_pdf_from_html(
        "\n".join(combined_sections),
        combined_pdf,
        "CampusAI — Complete Academic Submission (Synopsis, Report & Research Paper)",
    )
    comb_kb = combined_pdf.stat().st_size / 1024.0
    print(f"Created {combined_pdf.name:44s} ({comb_kb:6.1f} KB / {comb_kb / 1024.0:.3f} MB)")

    zip_file = SUBMISSION_DIR / "CampusAI_Complete_Submission_Under10MB.zip"
    build_zip_bundle(zip_file)
    zip_kb = zip_file.stat().st_size / 1024.0
    print(f"Created {zip_file.name:44s} ({zip_kb:6.1f} KB / {zip_kb / 1024.0:.3f} MB)")

    # Verify every file is strictly under 10 MB
    limit_bytes = 10 * 1024 * 1024
    for f in sorted(SUBMISSION_DIR.iterdir()):
        if f.is_file():
            assert f.stat().st_size < limit_bytes, f"{f.name} exceeds 10 MB!"
    print("VERIFIED: All submission files and ZIP bundle are strictly under 10 MB.")


if __name__ == "__main__":
    main()
