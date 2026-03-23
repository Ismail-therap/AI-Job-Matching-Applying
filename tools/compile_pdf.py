"""
tools/compile_pdf.py — LaTeX → PDF via local MiKTeX

Uses pdflatex with -enable-installer so MiKTeX auto-downloads any missing
packages on first run. Runs pdflatex twice to resolve cross-references.

Requirements:
  - MiKTeX installed and `pdflatex` on PATH
  - Internet connection on first run (for package downloads)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def compile_from_string(
    tex_str: str,
    filename_hint: str,
    tmp_dir: str = ".tmp",
) -> bytes | None:
    """
    Compile a LaTeX string to PDF bytes via local pdflatex (MiKTeX).

    Args:
        tex_str:       Complete LaTeX source as a string.
        filename_hint: Name hint for the .tex file (e.g. "Stripe_Resume.tex").
        tmp_dir:       Not used (kept for API compatibility). A system temp dir
                       is always used for compilation to avoid path issues.

    Returns:
        PDF bytes on success, None on failure.
    """
    safe_hint = filename_hint if filename_hint.endswith(".tex") else f"{filename_hint}.tex"

    with tempfile.TemporaryDirectory() as work_dir:
        tex_path = Path(work_dir) / safe_hint
        tex_path.write_text(tex_str, encoding="utf-8")

        log_text, success = _run_pdflatex(tex_path, Path(work_dir))

        if not success:
            log.error(f"pdflatex failed for {safe_hint}:\n{log_text[-2000:]}")
            return None

        pdf_path = tex_path.with_suffix(".pdf")
        if not pdf_path.exists():
            log.error(f"pdflatex reported success but no PDF found at {pdf_path}")
            return None

        pdf_bytes = pdf_path.read_bytes()
        log.info(f"Compiled {safe_hint} ({len(pdf_bytes) // 1024} KB)")
        return pdf_bytes


def get_compile_log(tex_str: str, filename_hint: str, tmp_dir: str = ".tmp") -> str:
    """
    Attempt compilation and return the pdflatex log on failure (or success).
    Used by the UI to surface compile errors to the user.
    """
    safe_hint = filename_hint if filename_hint.endswith(".tex") else f"{filename_hint}.tex"

    with tempfile.TemporaryDirectory() as work_dir:
        tex_path = Path(work_dir) / safe_hint
        tex_path.write_text(tex_str, encoding="utf-8")

        log_text, _ = _run_pdflatex(tex_path, Path(work_dir))
        return log_text


# ── Internal ──────────────────────────────────────────────────────────────────

def _run_pdflatex(tex_path: Path, work_dir: Path) -> tuple[str, bool]:
    """
    Run pdflatex twice (for cross-references) with -enable-installer.
    Returns (log_text, success).
    """
    if not shutil.which("pdflatex"):
        msg = (
            "pdflatex not found on PATH. "
            "Ensure MiKTeX is installed and added to your system PATH."
        )
        log.error(msg)
        return msg, False

    # Use forward slashes for pdflatex paths (avoids Windows backslash issues)
    out_dir = str(work_dir).replace("\\", "/")
    tex_file = str(tex_path).replace("\\", "/")

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",   # don't pause on errors
        "-enable-installer",           # MiKTeX: auto-install missing packages
        f"-output-directory={out_dir}",
        tex_file,
    ]

    combined_log = ""
    for run in (1, 2):  # two passes for cross-references / TOC
        # Pass 1 gets a longer timeout to allow MiKTeX to download missing packages
        timeout = 300 if run == 1 else 120
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
            )
            combined_log = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            msg = f"pdflatex timed out on pass {run}."
            log.error(msg)
            return combined_log + f"\n{msg}", False
        except Exception as e:
            msg = f"pdflatex subprocess error on pass {run}: {e}"
            log.error(msg)
            return combined_log + f"\n{msg}", False

        # pdflatex exit code 0 = success, 1 = errors in document
        # We check for PDF existence rather than trusting exit code alone
        pdf_path = tex_path.with_suffix(".pdf")
        if run == 2 and not pdf_path.exists():
            return combined_log, False

    return combined_log, True
