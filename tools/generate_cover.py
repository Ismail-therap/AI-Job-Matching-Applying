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
from tools import cost_tracker

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
    resume_tex: str = "",
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
        resume_tex:            Raw .tex source of the resume (for name extraction).

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
    body = _generate_body(company, role, jd_text, resume_plain, resume_tex, client)
    if body is None:
        warnings.append("Cover letter generation failed — using minimal fallback.")
        candidate_name = _extract_candidate_name(resume_tex)
        body = _fallback_body(company, role, candidate_name)

    result = preamble + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"

    if not _is_valid_latex(result):
        warnings.append("Cover letter LaTeX validation warning: missing document tags.")

    return result, warnings


# ── Body generation ────────────────────────────────────────────────────────────

def _experience_guidance(role: str, jd_text: str) -> str:
    """Return experience-selection rules based on the target role type."""
    combined = (role + " " + jd_text[:500]).lower()

    biostat_signals = [
        "biostatistician", "biostatistical", "biostatistics",
        "clinical statistician", "medical statistician", "health statistician",
        "clinical data analyst", "epidemiologist", "pharmacostatistician",
        "biometrics analyst", "clinical research analyst",
    ]
    ds_signals = [
        "data scientist", "senior data scientist", "principal data scientist",
        "machine learning engineer", "ml engineer", "ai engineer",
    ]
    analyst_signals = [
        "institutional analyst", "institutional data analyst", "data analyst",
        "research analyst", "business analyst", "analytics analyst",
    ]

    is_biostat = any(kw in combined for kw in biostat_signals)
    is_ds      = any(kw in combined for kw in ds_signals)
    is_analyst = any(kw in combined for kw in analyst_signals)

    if is_biostat:
        return (
            "\n\nCRITICAL EXPERIENCE SELECTION RULE (must follow exactly):\n"
            "- This is a biostatistician / medical analyst role.\n"
            "- You MUST use the LSU Health Shreveport Biostatistician experience as the primary credibility anchor.\n"
            "- You MUST NOT reference or mention the University of Wyoming position for this role.\n"
            "- Build all biostatistics and clinical-analysis talking points from the LSU Health Shreveport role only."
        )
    elif is_ds:
        return (
            "\n\nCRITICAL EXPERIENCE SELECTION RULE (must follow exactly):\n"
            "- This is a data scientist role.\n"
            "- You MUST draw talking points from BOTH the University of Wyoming Data Analyst experience "
            "AND the Insightin Senior Data Scientist experience.\n"
            "- Use University of Wyoming to demonstrate analytical rigor and the Insightin role to "
            "demonstrate advanced data science / ML capabilities."
        )
    elif is_analyst:
        return (
            "\n\nCRITICAL EXPERIENCE SELECTION RULE (must follow exactly):\n"
            "- This is an institutional / data analyst role.\n"
            "- You MUST use the University of Wyoming position as the primary credibility anchor.\n"
            "- Do NOT emphasise biostatistics-specific experience; focus on institutional data analysis work."
        )
    return ""


def _extract_candidate_name(resume_tex: str) -> str:
    """Extract candidate name from the raw LaTeX resume source.

    Handles the most common CV templates in order of specificity.
    Returns empty string if no name can be reliably identified.
    """
    if not resume_tex:
        return ""

    # Jake Gutierrez / sb2nov template (most common GitHub/Overleaf CV):
    # \textbf{\Huge \scshape Md. Ismail Hossain} or \textbf{\Huge Name}
    m = re.search(
        r"\\textbf\{\\Huge\s+(?:\\scshape\s+)?([A-Za-z][^}\\]*)\}",
        resume_tex,
    )
    if m:
        name = m.group(1).strip()
        if 1 <= len(name.split()) <= 6:
            return name

    # moderncv: \name{Firstname}{Lastname}
    m = re.search(r"\\name\{([^}]+)\}\{([^}]+)\}", resume_tex)
    if m:
        return f"{m.group(1).strip()} {m.group(2).strip()}"

    # AltaCV / single-arg: \name{Full Name} or \cvname{Full Name}
    for cmd in (r"\\name", r"\\cvname", r"\\Name"):
        m = re.search(cmd + r"\{([^}]+)\}", resume_tex)
        if m:
            name = m.group(1).strip()
            if 1 <= len(name.split()) <= 6:
                return name

    # Standard LaTeX: \author{Full Name}
    m = re.search(r"\\author\{([^}]+)\}", resume_tex)
    if m:
        name = m.group(1).strip()
        if 1 <= len(name.split()) <= 6:
            return name

    # {\Huge\textbf{Name}} or {\huge\textbf{Name}}
    m = re.search(
        r"\{\\(?:huge|Huge|LARGE|Large)\\textbf\{([A-Za-z][^}]*)\}\}",
        resume_tex,
    )
    if m:
        name = m.group(1).strip()
        if 1 <= len(name.split()) <= 6:
            return name

    return ""


def _generate_body(
    company: str,
    role: str,
    jd_text: str,
    resume_plain: str,
    resume_tex: str,
    client: openai.OpenAI,
) -> str | None:
    exp_rule = _experience_guidance(role, jd_text)
    candidate_name = _extract_candidate_name(resume_tex)
    name_instruction = (
        f"- The sign-off name is: {candidate_name}. "
        f"End the letter with: \\noindent Sincerely,\\\\[8pt]\\textbf{{{candidate_name}}}"
        if candidate_name
        else "- End with: \\noindent Sincerely,\\\\[8pt]\\textbf{{[Your Name]}}"
    )

    system = (
        "You are a senior professional writing a cover letter in first person on behalf of the candidate.\n\n"
        "Structure (exactly 3 paragraphs, each wrapped in \\noindent, separated by \\vspace{8pt}):\n\n"
        "  Para 1 — Opening: name the role, express genuine interest, and state one specific reason "
        "this candidate is a strong fit — grounded in a real credential or achievement.\n\n"
        "  Para 2 — Strength narrative: Write 3–4 sentences that naturally showcase the candidate's "
        "most relevant capabilities and accomplishments. Internally, use the job description to know "
        "what matters most to this employer — but do NOT quote, list, or echo job requirements. "
        "Instead, write confident prose about what the candidate has actually done, using concrete "
        "details (tools, outcomes, scale, impact). Let the achievements demonstrate fit implicitly. "
        "The reader should feel 'this person can do exactly what we need' without being told so.\n\n"
        "  Para 3 — Closing: one confident sentence inviting next steps, thank the reader.\n\n"
        "Formatting rules:\n"
        "- Output ONLY raw LaTeX for the document body (no \\documentclass, no \\begin{document}).\n"
        "- Start with a date line (\\noindent\\today), then \\vspace{12pt}, then salutation "
        "(\\noindent Dear Hiring Manager,).\n"
        "- Use \\noindent before each paragraph. Separate blocks with \\vspace{8pt}.\n"
        f"- {name_instruction}\n"
        "- Do NOT use bullet points, lists, or tables.\n"
        "- Keep total word count under 320 words.\n"
        "- CRITICAL: escape all LaTeX special characters in plain text — "
        "dollar amounts must use \\$ (e.g. \\$20,000 not $20,000), "
        "percent must use \\%, ampersand must use \\&, "
        "hash must use \\#. Unescaped $ will break the PDF.\n"
        "- No markdown, no code fences, no explanation — output LaTeX only."
        + exp_rule
    )

    user = (
        f"Company: {company}\n"
        f"Role: {role}\n\n"
        f"Job Description:\n{jd_text[:2500]}\n\n"
        f"Candidate Resume (plain text):\n{resume_plain[:2500]}\n\n"
        "Write the cover letter body now."
    )

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            cost_tracker.record(MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens, "Cover letter")
            body = (resp.choices[0].message.content or "").strip()
            body = re.sub(r"^```[a-z]*\n?", "", body, flags=re.MULTILINE)
            body = re.sub(r"\n?```$", "", body).strip()
            body = _sanitize_body(body)
            log.info("Cover letter body generated.")
            return body
        except openai.RateLimitError:
            log.warning("Rate limit — waiting 30s")
            time.sleep(30)
        except openai.APIError as e:
            log.error(f"Cover letter generation failed: {e}")
            break

    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_body(body: str) -> str:
    """Fix unescaped LaTeX special characters in generated body text.

    Only fixes characters that appear in plain-text contexts (not inside
    existing LaTeX commands or math environments).
    """
    # Escape bare $ that are NOT already preceded by a backslash
    # (catches $20,000 → \$20,000 while leaving \$ and math \(...\) alone)
    body = re.sub(r"(?<!\\)\$(?![^$]*\$)", r"\\$", body)

    # Escape bare % not already escaped (common in "reduced costs by 20%")
    body = re.sub(r"(?<!\\)%", r"\\%", body)

    # Escape bare & not already escaped and not inside tabular/align
    body = re.sub(r"(?<!\\)&", r"\\&", body)

    return body


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
    if any(cls in preamble for cls in _RESUME_CLASSES):
        log.info("Cover letter template uses a resume document class — using minimal preamble instead.")
        return ""
    return preamble


def _is_valid_latex(tex: str) -> bool:
    return r"\begin{document}" in tex and r"\end{document}" in tex


def _fallback_body(company: str, role: str, candidate_name: str = "") -> str:
    sign_off_name = candidate_name if candidate_name else "[Your Name]"
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
        f"\\textbf{{{sign_off_name}}}"
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


def refine(
    cover_tex_out: str,
    feedback: str,
    jd_text: str,
    meta: dict,
    resume_plain: str,
    client: openai.OpenAI,
) -> tuple[str, list[str]]:
    """Revise an existing cover letter based on user feedback.

    Args:
        cover_tex_out: The full .tex string from the previous generation.
        feedback:      Plain-text instructions from the user describing changes.
        jd_text:       Original job description (for context).
        meta:          {company_name, role_title, safe_company}.
        resume_plain:  Plain-text resume (for context).
        client:        Initialized OpenAI client.

    Returns:
        (new_cover_tex, warnings)
    """
    warnings: list[str] = []
    company = meta.get("company_name", "the company")
    role = meta.get("role_title", "the position")

    # Extract preamble and existing body from the prior output
    preamble = _extract_preamble(cover_tex_out)
    if not preamble:
        preamble = _minimal_preamble()

    # Extract the body between \begin{document} and \end{document}
    body_match = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}",
        cover_tex_out,
        re.DOTALL,
    )
    existing_body = body_match.group(1).strip() if body_match else cover_tex_out

    exp_rule = _experience_guidance(role, jd_text)
    candidate_name = _extract_candidate_name(cover_tex_out)
    name_instruction = (
        f"- The sign-off name is: {candidate_name}. "
        f"End the letter with: \\noindent Sincerely,\\\\[8pt]\\textbf{{{candidate_name}}}"
        if candidate_name
        else "- End with: \\noindent Sincerely,\\\\[8pt]\\textbf{{[Your Name]}}"
    )

    system = (
        "You are a senior professional revising a cover letter in first person on behalf of the candidate.\n\n"
        "Structure (exactly 3 paragraphs, each wrapped in \\noindent, separated by \\vspace{8pt}):\n\n"
        "  Para 1 — Opening: name the role, express genuine interest, and state one specific reason "
        "this candidate is a strong fit — grounded in a real credential or achievement.\n\n"
        "  Para 2 — Strength narrative: Write 3–4 sentences that naturally showcase the candidate's "
        "most relevant capabilities and accomplishments. Internally, use the job description to know "
        "what matters most to this employer — but do NOT quote, list, or echo job requirements. "
        "Instead, write confident prose about what the candidate has actually done, using concrete "
        "details (tools, outcomes, scale, impact).\n\n"
        "  Para 3 — Closing: one confident sentence inviting next steps, thank the reader.\n\n"
        "Formatting rules:\n"
        "- Output ONLY raw LaTeX for the document body (no \\documentclass, no \\begin{document}).\n"
        "- Start with a date line (\\noindent\\today), then \\vspace{12pt}, then salutation "
        "(\\noindent Dear Hiring Manager,).\n"
        "- Use \\noindent before each paragraph. Separate blocks with \\vspace{8pt}.\n"
        f"- {name_instruction}\n"
        "- Do NOT use bullet points, lists, or tables.\n"
        "- Keep total word count under 320 words.\n"
        "- CRITICAL: escape all LaTeX special characters in plain text — "
        "dollar amounts must use \\$ (e.g. \\$20,000 not $20,000), "
        "percent must use \\%, ampersand must use \\&, "
        "hash must use \\#. Unescaped $ will break the PDF.\n"
        "- No markdown, no code fences, no explanation — output LaTeX only."
        + exp_rule
    )

    user = (
        f"Company: {company}\n"
        f"Role: {role}\n\n"
        f"Job Description:\n{jd_text[:2500]}\n\n"
        f"Candidate Resume (plain text):\n{resume_plain[:2500]}\n\n"
        f"Current cover letter body:\n{existing_body}\n\n"
        f"User feedback (changes requested):\n{feedback}\n\n"
        "Revise the cover letter body incorporating the above feedback. "
        "Keep the same 3-paragraph structure and LaTeX formatting rules."
    )

    new_body = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            cost_tracker.record(MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens, "Cover letter (refined)")
            body = (resp.choices[0].message.content or "").strip()
            body = re.sub(r"^```[a-z]*\n?", "", body, flags=re.MULTILINE)
            body = re.sub(r"\n?```$", "", body).strip()
            new_body = _sanitize_body(body)
            log.info("Cover letter refined.")
            break
        except openai.RateLimitError:
            log.warning("Rate limit — waiting 30s")
            time.sleep(30)
        except openai.APIError as e:
            log.error(f"Cover letter refinement failed: {e}")
            break

    if new_body is None:
        warnings.append("Cover letter refinement failed — keeping previous version.")
        new_body = existing_body

    result = preamble + "\n\\begin{document}\n\n" + new_body + "\n\n\\end{document}\n"

    if not _is_valid_latex(result):
        warnings.append("Cover letter LaTeX validation warning: missing document tags.")

    return result, warnings
