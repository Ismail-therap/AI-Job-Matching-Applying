"""
tools/generate_cover.py — Cover Letter Generator (single API call, full rewrite)

Always produces a complete, concise, job-specific cover letter in one API call.

Behaviour:
  - If user uploads a .tex template: extracts the LaTeX preamble (documentclass,
    usepackage, etc.) from it and uses it as the formatting shell.
  - If no template: uses the default_cover_letter.tex skeleton's preamble.
  - The body is always fully rewritten by the model — targeted, concise, max 1 page
    (3 paragraphs: opening/fit, value/skills, closing call-to-action).

Output is always a complete, compilable .tex string.
"""

from __future__ import annotations

import logging
import os
import re
import time

import openai

log = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = 0.4


def generate(
    cover_tex: str | None,
    jd_text: str,
    meta: dict,
    resume_plain: str,
    client: openai.OpenAI,
    default_template_path: str = "default_cover_letter.tex",
) -> tuple[str, list[str]]:
    """
    Generate a tailored, concise cover letter .tex (single API call).

    Args:
        cover_tex:             User-uploaded .tex string, or None.
        jd_text:               Plain text of the job description.
        meta:                  {company_name, role_title, safe_company}.
        resume_plain:          Plain text resume for context.
        client:                Initialized OpenAI client.
        default_template_path: Path to fallback skeleton .tex file.

    Returns:
        (cover_tex_out, warnings)
    """
    warnings: list[str] = []
    company = meta.get("company_name", "the company")
    role = meta.get("role_title", "the position")

    # Determine which preamble to use
    preamble = _extract_preamble(cover_tex)
    if not preamble:
        try:
            with open(default_template_path, "r", encoding="utf-8") as f:
                default_tex = f.read()
            preamble = _extract_preamble(default_tex)
        except FileNotFoundError:
            pass
    if not preamble:
        preamble = _minimal_preamble()
        warnings.append(
            "Default cover letter template not found — using built-in preamble."
        )

    # Single API call — full body rewrite
    body = _generate_body(company, role, jd_text, resume_plain, client)
    if body is None:
        warnings.append("Cover letter generation failed — using minimal fallback.")
        body = _fallback_body(company, role)

    result = preamble + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"

    if not _is_valid_latex(result):
        warnings.append("Cover letter LaTeX validation warning: missing document tags.")

    return result, warnings


# ── Body generation ───────────────────────────────────────────────────────────

def _generate_body(
    company: str,
    role: str,
    jd_text: str,
    resume_plain: str,
    client: openai.OpenAI,
) -> str | None:
    system = (
        "You are a professional cover letter writer. "
        "Write the body of a cover letter in LaTeX — concise, targeted, and fitting on ONE page.\n\n"
        "Structure (exactly 3 paragraphs, each wrapped in \\noindent and separated by \\vspace{8pt}):\n"
        "  Para 1 — Opening: state the role, express genuine interest, one specific reason you are a fit.\n"
        "  Para 2 — Value: highlight 2–3 most relevant skills/experiences from the resume that match the JD. "
        "Be specific, reference real achievements. No fluff.\n"
        "  Para 3 — Closing: confident call-to-action, thank the reader, one sentence.\n\n"
        "Rules:\n"
        "- Output ONLY raw LaTeX for the document body (no \\documentclass, no \\begin{document}).\n"
        "- Include a date line, salutation (Dear Hiring Manager,), and a sign-off (Sincerely, / [Your Name]).\n"
        "- Use \\noindent before each paragraph. Separate blocks with \\vspace{8pt}.\n"
        "- Do NOT use bullet points, lists, or tables.\n"
        "- Keep total word count under 300 words.\n"
        "- No markdown, no code fences, no explanation."
    )
    user = (
        f"Company: {company}\n"
        f"Role: {role}\n\n"
        f"Job Description:\n{jd_text[:2500]}\n\n"
        f"Candidate Resume (plain text):\n{resume_plain[:1500]}\n\n"
        "Write the cover letter body now."
    )

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            body = (resp.choices[0].message.content or "").strip()
            body = re.sub(r"^```[a-z]*\n?", "", body, flags=re.MULTILINE)
            body = re.sub(r"\n?```$", "", body).strip()
            log.info("Cover letter body generated.")
            return body
        except openai.RateLimitError:
            log.warning("Rate limit — waiting 30s")
            time.sleep(30)
        except openai.APIError as e:
            log.error(f"Cover letter generation failed: {e}")
            break

    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

_RESUME_CLASSES = ("moderncv", "resume", "twentysecondscv", "altacv", "awesome-cv")

def _extract_preamble(tex: str | None) -> str:
    """Extract everything before \\begin{document} from a .tex string.

    Returns empty string if the document class looks like a resume template
    (moderncv etc.) — those classes are incompatible with cover letter body
    content and cause pdfTeX font-expansion errors.
    """
    if not tex:
        return ""
    match = re.search(r"\\begin\{document\}", tex)
    if not match:
        return ""
    preamble = tex[: match.start()].rstrip()
    # Reject resume document classes — fall back to minimal preamble
    if any(cls in preamble for cls in _RESUME_CLASSES):
        log.info("Cover letter template uses a resume document class — using minimal preamble instead.")
        return ""
    return preamble


def _is_valid_latex(tex: str) -> bool:
    return r"\begin{document}" in tex and r"\end{document}" in tex


def _fallback_body(company: str, role: str) -> str:
    return (
        f"\\noindent\\today\n\n"
        f"\\vspace{{12pt}}\n"
        f"\\noindent Dear Hiring Manager,\n\n"
        f"\\vspace{{8pt}}\n"
        f"\\noindent\n"
        f"I am writing to express my strong interest in the \\textbf{{{role}}} "
        f"position at \\textbf{{{company}}}.\n\n"
        f"\\vspace{{8pt}}\n"
        f"\\noindent\n"
        f"My background and experience make me a strong candidate for this role. "
        f"I am confident that my skills align closely with what your team is looking for.\n\n"
        f"\\vspace{{8pt}}\n"
        f"\\noindent\n"
        f"Thank you for your time and consideration. "
        f"I look forward to the opportunity to discuss how I can contribute to your team.\n\n"
        f"\\vspace{{16pt}}\n"
        f"\\noindent Sincerely,\\\\[8pt]\n"
        f"\\textbf{{[Your Name]}}"
    )


def _minimal_preamble() -> str:
    return (
        r"\documentclass[11pt,a4paper]{article}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[margin=1in, top=0.8in, bottom=0.8in]{geometry}" + "\n"
        r"\usepackage{hyperref}" + "\n"
        r"\usepackage{parskip}" + "\n"
        r"\hypersetup{colorlinks=true, urlcolor=blue}" + "\n"
        r"\pagestyle{empty}"
    )
