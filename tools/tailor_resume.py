"""
tools/tailor_resume.py — ATS Keyword Injection (two-phase)

Phase 1 — LLM patch (natural weaving):
  GPT receives only the numbered \\item lines and returns a JSON dict of
  {line_number: new_line_content} for lines where a keyword fits naturally.

Phase 2 — Dynamic deterministic fallback (guaranteed injection):
  Step A: Extract all \\section headers from the CV.
  Step B: Ask the LLM which section is the best place to add ATS keywords
          (cheap call — just a list of names, returns one number).
  Step C: Within that section, auto-detect the format:
            - "structured"  →  \\textbf{Label} {: value}\\\\ lines inside }}
            - "itemize"     →  standard \\item lines inside \\end{itemize}
  Step D: Insert the keywords in the correct format and position.

This works for any CV/resume .tex structure without any hardcoded section names.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import openai

log = logging.getLogger(__name__)

MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = 0.2

NEW_SUBSECTION_LABEL = "Applied Research Terms"


# ── Public helpers ─────────────────────────────────────────────────────────────

def extract_sections_and_subsections(tex_str: str) -> list[tuple[int, str, str]]:
    """
    Return [(line_idx, display_name, level), ...] for every \\section and \\subsection.
    level is 'section' or 'subsection'.

    Handles nested-brace patterns like \\section{{} Section Name} which are common
    in Jake-Gutierrez-style LaTeX templates.
    """
    # Matches content that may contain one level of nested braces, e.g. {{} Name}
    _SEC_RE = re.compile(r'\\(sub)?section\*?\s*\{((?:[^{}]|\{[^}]*\})*)\}')
    results = []
    for i, line in enumerate(tex_str.splitlines()):
        m = _SEC_RE.search(line)
        if m:
            level = "subsection" if m.group(1) else "section"
            raw_name = m.group(2)
            # Strip \cmd{...} formatting, then remove remaining braces/backslashes
            clean = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', raw_name)
            clean = re.sub(r'[{}\\]', '', clean).strip()
            if clean:  # skip empty names that result from pure-formatting arguments
                results.append((i, clean, level))
    return results


def insert_keyword_line(
    tex_str: str,
    section_line_idx: int,
    keywords_str: str,
    label: str = NEW_SUBSECTION_LABEL,
) -> str:
    """
    Insert keywords as a single comma-separated line into the section/subsection
    that starts at section_line_idx.  Detects structured vs itemize format.
    Returns the modified tex_str.
    """
    lines = tex_str.splitlines()

    # Find section bounds: next \section or \subsection, or \end{document}
    end = len(lines)
    for i in range(section_line_idx + 1, len(lines)):
        if re.search(r'\\(?:sub)?section\*?\s*\{', lines[i]):
            end = i
            break
        if r'\end{document}' in lines[i]:
            end = i
            break

    start = section_line_idx
    block = "\n".join(lines[start:end])

    # Structured format: \textbf{Label} {: value}\\
    if re.search(r'\\textbf\{[^}]+\}\s*\{:', block):
        for i in range(end - 1, start, -1):
            if lines[i].strip() == "}}":
                # Find the last \textbf line before }} and ensure it ends with \\
                for j in range(i - 1, start, -1):
                    if re.search(r'\\textbf\{', lines[j]):
                        if not lines[j].rstrip().endswith(r'\\'):
                            lines[j] = lines[j].rstrip() + r'\\'
                        break
                new_lines = [
                    r"    \vspace{.5mm}",
                    f"    \\textbf{{{label}}} {{: {keywords_str}}}",
                ]
                lines = lines[:i] + new_lines + lines[i:]
                log.info(f"insert_keyword_line [structured]: inserted before '}}}}' at line {i}.")
                return "\n".join(lines)

    # Itemize format: insert before last \end{itemize} in section
    insert_at = None
    for i in range(end - 1, start, -1):
        if lines[i].strip() in (r"\end{itemize}", r"\end{enumerate}"):
            insert_at = i
            break

    if insert_at is not None:
        indent = "  "
        for j in range(max(start, insert_at - 10), insert_at):
            if lines[j].strip().startswith(r"\item"):
                indent = re.match(r"^(\s*)", lines[j]).group(1)
                break
        new_lines = [
            r"    \vspace{.5mm}",
            f"{indent}\\item \\textbf{{{label}}}: {keywords_str}",
        ]
        lines = lines[:insert_at] + new_lines + lines[insert_at:]
        log.info(f"insert_keyword_line [itemize]: inserted labeled subsection at line {insert_at}.")
        return "\n".join(lines)

    # Fallback: insert as a comment after the section header line
    lines = lines[:start + 1] + [f"% {label}: {keywords_str}"] + lines[start + 1:]
    log.warning("insert_keyword_line [fallback]: no itemize/structured block found — inserted as comment.")
    return "\n".join(lines)


def check_keyword_coverage(tex: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """Return (found, missing) based on case-insensitive scan of the tex."""
    tex_lower = tex.lower()
    found   = [k for k in keywords if k.lower() in tex_lower]
    missing = [k for k in keywords if k.lower() not in tex_lower]
    return found, missing


# ── Section detection ──────────────────────────────────────────────────────────

def _extract_sections(lines: list[str]) -> list[tuple[int, str]]:
    """
    Return [(line_index, section_name), ...] for every \\section command.
    Strips LaTeX formatting noise ({}, \\textbf, etc.) from the name.
    Handles nested-brace patterns like \\section{{} Name}.
    """
    _SEC_RE = re.compile(r'\\section\*?\s*\{((?:[^{}]|\{[^}]*\})*)\}')
    sections = []
    for i, line in enumerate(lines):
        m = _SEC_RE.search(line)
        if m:
            raw_name = m.group(1)
            clean = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', raw_name)
            clean = re.sub(r'[{}\\]', '', clean).strip()
            if clean:
                sections.append((i, clean))
    return sections


def _pick_best_section(sections: list[tuple[int, str]], client: openai.OpenAI) -> int:
    """
    Ask the LLM which section index (0-based into `sections`) is the best
    place to add ATS keywords. Returns the chosen index, defaulting to 0
    on any failure.
    """
    if not sections:
        return 0

    numbered = "\n".join(f"{i + 1}. {name}" for i, (_, name) in enumerate(sections))
    prompt = (
        "A candidate wants to add ATS job-matching keywords to their resume. "
        "Which section number below is the most appropriate place to add them? "
        "Choose the skills, tools, competencies, or research interest section. "
        "Reply with ONLY the section number — nothing else.\n\n"
        f"Sections:\n{numbered}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": "You are a resume expert. Reply with only a number."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        idx = int(re.search(r"\d+", raw).group()) - 1
        idx = max(0, min(idx, len(sections) - 1))
        log.info(f"Phase 2: LLM chose section [{idx}] → '{sections[idx][1]}'")
        return idx
    except Exception as e:
        log.warning(f"Phase 2: section picker failed ({e}) — defaulting to section 0.")
        return 0


def _section_bounds(lines: list[str], sections: list[tuple[int, str]], idx: int) -> tuple[int, int]:
    """
    Return (start_line, end_line) — exclusive end — for section at `idx`.
    End is the start of the next \\section or \\end{document}.
    """
    start = sections[idx][0]
    if idx + 1 < len(sections):
        end = sections[idx + 1][0]
    else:
        # Find \end{document}
        end = len(lines)
        for i in range(len(lines) - 1, start, -1):
            if r"\end{document}" in lines[i]:
                end = i
                break
    return start, end


# ── Format detection and insertion ─────────────────────────────────────────────

def _detect_format(lines: list[str], start: int, end: int) -> str:
    """
    Return 'structured' if the section uses \\textbf{Label} {: value} lines,
    or 'itemize' if it uses standard \\item bullets.
    """
    block = "\n".join(lines[start:end])
    if re.search(r'\\textbf\{[^}]+\}\s*\{:', block):
        return "structured"
    return "itemize"


def _insert_structured(lines: list[str], start: int, end: int, keywords: list[str]) -> list[str]:
    """
    Insert a new \\textbf{Applied Research Terms} {: kw1, kw2} line
    just before the '}}' that closes the \\small{\\item{ block.
    """
    for i in range(end - 1, start, -1):
        if lines[i].strip() == "}}":
            kw_string = ", ".join(keywords)
            # Find the last \textbf line before }} and ensure it ends with \\
            for j in range(i - 1, start, -1):
                if re.search(r'\\textbf\{', lines[j]):
                    if not lines[j].rstrip().endswith(r'\\'):
                        lines[j] = lines[j].rstrip() + r'\\'
                    break
            new_lines = [
                r"    \vspace{.5mm}",
                f"    \\textbf{{{NEW_SUBSECTION_LABEL}}} {{: {kw_string}}}",
            ]
            log.info(f"Phase 2 [structured]: inserting before '}}}}' at line {i}.")
            return lines[:i] + new_lines + lines[i:]

    # Fallback: insert before \end{itemize} within section
    log.warning("Phase 2 [structured]: no '}}}}' found — falling back to itemize insert.")
    return _insert_itemize(lines, start, end, keywords)


def _insert_itemize(lines: list[str], start: int, end: int, keywords: list[str]) -> list[str]:
    """
    Insert keywords as \\item lines just before the last \\end{itemize}
    within the section. Falls back to the very last \\end{itemize} in the file.
    """
    # Find last \end{itemize} or \end{enumerate} within the section
    insert_at = None
    for i in range(end - 1, start, -1):
        if lines[i].strip() in (r"\end{itemize}", r"\end{enumerate}"):
            insert_at = i
            break

    if insert_at is None:
        # Last resort: last \end{itemize} anywhere in the file
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() in (r"\end{itemize}", r"\end{enumerate}"):
                insert_at = i
                break

    if insert_at is None:
        log.warning("Phase 2 [itemize]: no itemize block found anywhere.")
        return lines

    # Match indentation of surrounding \item lines
    indent = "  "
    for j in range(max(start, insert_at - 10), insert_at):
        if lines[j].strip().startswith(r"\item"):
            indent = re.match(r"^(\s*)", lines[j]).group(1)
            break

    kw_string = ", ".join(keywords)
    new_lines = [
        r"    \vspace{.5mm}",
        f"{indent}\\item \\textbf{{{NEW_SUBSECTION_LABEL}}}: {kw_string}",
    ]
    log.info(f"Phase 2 [itemize]: inserting labeled subsection at line {insert_at}.")
    return lines[:insert_at] + new_lines + lines[insert_at:]


# ── Phase 2 orchestrator ───────────────────────────────────────────────────────

def _phase2_append(lines: list[str], keywords: list[str], client: openai.OpenAI) -> list[str]:
    """
    Dynamically find the best section and insert the missing keywords there.
    """
    sections = _extract_sections(lines)
    if not sections:
        log.warning("Phase 2: no \\section commands found — cannot insert keywords.")
        return lines

    idx          = _pick_best_section(sections, client)
    start, end   = _section_bounds(lines, sections, idx)
    fmt          = _detect_format(lines, start, end)

    log.info(f"Phase 2: target section '{sections[idx][1]}', format='{fmt}'.")

    if fmt == "structured":
        return _insert_structured(lines, start, end, keywords)
    else:
        return _insert_itemize(lines, start, end, keywords)


# ── Main entry point ───────────────────────────────────────────────────────────

def tailor(
    tex_str: str,
    jd_text: str,
    client: openai.OpenAI,
    keywords: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Inject ATS keywords into the resume.

    Phase 1: LLM weaves keywords naturally into existing \\item bullets.
    Phase 2: Any keyword still absent is dynamically inserted into the
             most appropriate section, in the correct format.

    Returns (tailored_tex, warnings).
    """
    warnings: list[str] = []
    lines = tex_str.splitlines()

    # ── Phase 1: LLM natural injection ────────────────────────────────────────

    item_lines = {
        i + 1: line
        for i, line in enumerate(lines)
        if line.strip().startswith(r"\item")
    }

    if not item_lines:
        warnings.append("No \\item bullet lines found — skipping LLM phase.")
    else:
        numbered_items = "\n".join(f"{n}: {content}" for n, content in item_lines.items())

        if keywords:
            keyword_instruction = (
                f"KEYWORDS TO INJECT: {', '.join(keywords)}\n\n"
                "For each keyword, find the most relevant existing bullet and weave it in naturally. "
                "Only skip a keyword if it is already present (case-insensitive) in the bullets."
            )
            rule3 = (
                "Inject as many of the listed keywords as possible into existing bullets "
                "where they fit naturally. Do not force a keyword into an unrelated bullet."
            )
        else:
            keyword_instruction = (
                f"JOB DESCRIPTION:\n{jd_text}\n\n"
                "Identify the most important ATS keywords and weave them into the bullet lines."
            )
            rule3 = "Only modify a line if a keyword genuinely improves it."

        system = "\n".join([
            "You are an ATS resume optimization specialist.",
            "CRITICAL: Do NOT change the original structure of the resume or CV .tex file.",
            "You only inject the selected keywords in the appropriate place in the resume or CV.",
            "The structure, formatting, sections, and all existing content must remain exactly as-is.",
            "",
            "Strict rules:",
            "1. Do NOT rewrite or restructure bullets — only insert keywords into existing lines.",
            "2. Do NOT invent experience or technologies not already implied by the bullet.",
            f"3. {rule3}",
            "4. Every output line MUST start with \\item.",
            "5. Return ONLY a JSON object: {\"<line_number>\": \"<full modified \\item line>\", ...}",
            "   Include ONLY lines you actually changed. If no changes needed, return {}.",
            "6. No explanation, no markdown fences, no full file — just the JSON object.",
        ])

        user = (
            f"{keyword_instruction}\n\n"
            f"RESUME BULLET LINES (line_number: content):\n{numbered_items}\n\n"
            "Return a JSON object of changed lines only."
        )

        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw = (resp.choices[0].message.content or "").strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw).strip()

                patches: dict = json.loads(raw)
                if not isinstance(patches, dict):
                    raise ValueError("Response was not a JSON object.")

                modified_lines = list(lines)
                applied = 0
                for line_no_str, new_content in patches.items():
                    try:
                        line_no = int(line_no_str)
                    except ValueError:
                        continue
                    if not (1 <= line_no <= len(modified_lines)):
                        log.warning(f"Patch line {line_no} out of range — skipped.")
                        continue
                    if not new_content.strip().startswith(r"\item"):
                        log.warning(f"Patch line {line_no} missing \\item — skipped.")
                        continue
                    if not _braces_balanced(new_content):
                        log.warning(f"Patch line {line_no} unbalanced braces — skipped.")
                        continue
                    modified_lines[line_no - 1] = new_content
                    applied += 1

                log.info(f"Phase 1: {applied} bullet(s) updated by LLM.")
                lines = modified_lines
                break

            except (json.JSONDecodeError, ValueError) as e:
                log.warning(f"Phase 1 attempt {attempt + 1}: JSON parse error — {e}")
            except openai.RateLimitError:
                log.warning("Phase 1: rate limit — waiting 30s")
                time.sleep(30)
            except openai.APIError as e:
                log.error(f"Phase 1: API error — {e}")
                break

    # ── Phase 2: dynamic fallback for any still-missing keywords ──────────────

    if keywords:
        current_tex = "\n".join(lines)
        _, still_missing = check_keyword_coverage(current_tex, keywords)
        if still_missing:
            log.info(f"Phase 2: {len(still_missing)} keyword(s) still missing — inserting dynamically.")
            lines = _phase2_append(lines, still_missing, client)

    return "\n".join(lines), warnings


def _braces_balanced(text: str) -> bool:
    return text.count("{") == text.count("}")
