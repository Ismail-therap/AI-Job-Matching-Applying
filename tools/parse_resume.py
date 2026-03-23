"""
tools/parse_resume.py — Resume .tex Parser

Accepts raw bytes from a Streamlit file uploader for a .tex file.
Returns:
  - tex_str   : the raw LaTeX source as a string (used for tailoring)
  - plain_text: LaTeX-stripped plain text (used for semantic scoring)

strip_latex() is copied verbatim from AIApply/tools/semantic_match.py.
"""

from __future__ import annotations

import re


def parse(tex_bytes: bytes) -> tuple[str, str]:
    """
    Parse an uploaded .tex file.

    Args:
        tex_bytes: Raw bytes from st.file_uploader.

    Returns:
        (tex_str, plain_text) — raw source and plain text version.
    """
    tex_str = tex_bytes.decode("utf-8", errors="replace")
    plain_text = strip_latex(tex_str)
    return tex_str, plain_text


def strip_latex(tex: str) -> str:
    """
    Convert LaTeX source to plain text for embedding.
    Copied verbatim from AIApply/tools/semantic_match.py.
    """
    # Remove comments
    tex = re.sub(r"%.*$", "", tex, flags=re.MULTILINE)
    # Remove \begin{...} / \end{...}
    tex = re.sub(r"\\(begin|end)\{[^}]*\}", "", tex)
    # \href{url}{text} → text
    tex = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", tex)
    # \textbf{text}, \emph{text}, \textit{text} → text
    tex = re.sub(r"\\(textbf|emph|textit|underline)\{([^}]*)\}", r"\2", tex)
    # \section{title} → title
    tex = re.sub(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}", r"\2", tex)
    # Remove \\ (line break)
    tex = tex.replace("\\\\", " ")
    # Remove remaining \command[opts]{arg} — keep the arg content
    tex = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", tex)
    # Remove remaining \command tokens
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)
    # Remove braces
    tex = tex.replace("{", " ").replace("}", " ")
    # Remove LaTeX special chars
    tex = re.sub(r"[&$#_^~]", " ", tex)
    # Collapse whitespace
    tex = re.sub(r"\s+", " ", tex).strip()
    return tex
