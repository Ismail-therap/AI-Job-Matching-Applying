"""
tools/score_match.py — Multi-Dimensional Job-Resume Scorer

Scores how well a resume matches a job description across 5 dimensions (100 pts total):

  1. Semantic Similarity  (35 pts) — sentence-transformer cosine similarity
  2. Skills Coverage      (25 pts) — OpenAI extracts JD skills; matched against resume
  3. Title Alignment      (15 pts) — cosine similarity of role titles
  4. Experience Match     (15 pts) — required years vs. candidate years
  5. Location Fit         (10 pts) — rule-based remote/hybrid/on-site check

Score calibration note (from AIApply):
  all-MiniLM-L6-v2 cosine scores for resume-vs-JD comparisons fall in the 0.30–0.55 range.
  Raw cosine is mapped from the [0.20, 0.65] window to [0, 35] pts.
"""

from __future__ import annotations

import json
import logging
import os
import re

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # Lazy-loaded; decorated with @st.cache_resource in app.py


def get_model():
    """
    Load and return the sentence-transformer model.
    Call this inside a @st.cache_resource wrapper in app.py for Streamlit efficiency.
    """
    global _model
    if _model is None:
        log.info(f"Loading sentence-transformer '{MODEL_NAME}' (first run ~80MB download)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Model loaded.")
    return _model


def score(
    jd_text: str,
    resume_plain: str,
    meta: dict,
    client: openai.OpenAI,
    model=None,
) -> dict:
    """
    Compute multi-dimensional match score.

    Args:
        jd_text:      Plain text of the job description.
        resume_plain: LaTeX-stripped plain text of the resume.
        meta:         {company_name, role_title, safe_company} from extract_meta.
        client:       Initialized OpenAI client.
        model:        Loaded sentence-transformer model (from get_model()).

    Returns:
        {
            "total": int,                        # 0–100
            "dimensions": {
                "semantic_similarity": int,      # 0–35
                "skills_coverage":     int,      # 0–25
                "title_alignment":     int,      # 0–15
                "experience_match":    int,      # 0–15
                "location_fit":        int,      # 0–10
            },
            "matched_skills":   list[str],
            "missing_skills":   list[str],
            "all_skills":       list[str],
        }
    """
    if model is None:
        model = get_model()

    d1 = _score_semantic(jd_text, resume_plain, model)
    d2, matched_skills, missing_skills, all_skills = _score_skills(jd_text, resume_plain, client)
    d3 = _score_title(jd_text, resume_plain, meta, model)
    d4 = _score_experience(jd_text, resume_plain, client)
    d5 = _score_location(jd_text)

    total = d1 + d2 + d3 + d4 + d5

    return {
        "total": total,
        "dimensions": {
            "semantic_similarity": d1,
            "skills_coverage":     d2,
            "title_alignment":     d3,
            "experience_match":    d4,
            "location_fit":        d5,
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "all_skills": all_skills,
    }


# ── Dimension 1: Semantic Similarity (35 pts) ─────────────────────────────────

def _score_semantic(jd_text: str, resume_plain: str, model) -> int:
    """Cosine similarity between resume and JD embeddings, mapped to 0–35 pts."""
    if not jd_text.strip() or not resume_plain.strip():
        return 0

    embeddings = model.encode(
        [resume_plain[:5000], jd_text[:5000]],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cosine = float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0])

    # Map raw cosine range [0.20, 0.65] → [0, 35]
    lo, hi = 0.20, 0.65
    pts = int(max(0, min(35, (cosine - lo) / (hi - lo) * 35)))
    log.info(f"Semantic similarity: cosine={cosine:.3f} → {pts}/35 pts")
    return pts


# ── Dimension 2: Skills Coverage (25 pts) ─────────────────────────────────────

def _score_skills(
    jd_text: str, resume_plain: str, client: openai.OpenAI
) -> tuple[int, list[str], list[str], list[str]]:
    """OpenAI extracts required skills; match them against the resume."""
    prompt = (
        "Extract up to 15 required technical skills, tools, or qualifications from this job description. "
        "Focus on hard skills, not soft skills. "
        "Return ONLY a JSON array of short strings. No explanation, no markdown.\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:3000]}"
    )

    all_skills: list[str] = []
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw).strip()
        all_skills = json.loads(raw)
        if not isinstance(all_skills, list):
            all_skills = []
    except Exception as e:
        log.warning(f"Skills extraction failed: {e}")
        return 12, [], [], []  # neutral fallback

    resume_lower = resume_plain.lower()
    matched = [s for s in all_skills if s.lower() in resume_lower]
    missing = [s for s in all_skills if s.lower() not in resume_lower]

    if not all_skills:
        return 12, [], [], []

    pts = int(len(matched) / len(all_skills) * 25)
    log.info(f"Skills: {len(matched)}/{len(all_skills)} matched → {pts}/25 pts")
    return pts, matched, missing, all_skills


# ── Dimension 3: Title Alignment (15 pts) ─────────────────────────────────────

def _score_title(jd_text: str, resume_plain: str, meta: dict, model) -> int:
    """Cosine similarity between the JD role title and the candidate's most recent title."""
    jd_role = meta.get("role_title", "")
    if not jd_role or jd_role == "Unknown":
        return 7  # neutral

    candidate_title = ""
    date_pattern = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}"
        r"|\b\d{4}\s*[–\-]\s*(?:\d{4}|Present|Current)",
        re.IGNORECASE,
    )
    lines = resume_plain.splitlines()
    for i, line in enumerate(lines):
        if date_pattern.search(line):
            if i > 0 and lines[i - 1].strip():
                candidate_title = lines[i - 1].strip()
                break

    if not candidate_title:
        candidate_title = next((l.strip() for l in lines if len(l.strip()) > 5), jd_role)

    embeddings = model.encode(
        [jd_role, candidate_title],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cosine = float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0])
    pts = int(max(0, min(15, cosine * 15)))
    log.info(f"Title alignment: '{candidate_title}' vs '{jd_role}' → cosine={cosine:.3f} → {pts}/15 pts")
    return pts


# ── Dimension 4: Experience Match (15 pts) ────────────────────────────────────

def _score_experience(jd_text: str, resume_plain: str, client: openai.OpenAI) -> int:
    """Compare JD required years vs. candidate total years of experience."""
    required_years = _extract_required_years(jd_text)
    candidate_years = _estimate_candidate_years(resume_plain)

    if required_years is None:
        required_years = _openai_extract_years(jd_text, client)

    if required_years is None:
        return 10  # neutral — no requirement stated

    if candidate_years is None:
        return 8  # neutral — can't determine from resume

    gap = candidate_years - required_years

    if gap >= 0:
        pts = 15
    elif gap == -1:
        pts = 12
    elif gap == -2:
        pts = 8
    elif gap == -3:
        pts = 4
    else:
        pts = 2

    log.info(f"Experience: required={required_years}y candidate={candidate_years}y gap={gap} → {pts}/15 pts")
    return pts


def _extract_required_years(text: str) -> int | None:
    patterns = [
        r"(\d+)\+?\s*(?:or more\s*)?years?\s+of\s+(?:related\s+)?(?:professional\s+)?experience",
        r"minimum\s+(?:of\s+)?(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
        r"(\d+)[–-](\d+)\s+years?\s+of\s+experience",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _estimate_candidate_years(resume_plain: str) -> int | None:
    import datetime
    current_year = datetime.datetime.now().year
    total = 0
    found_any = False

    spans = re.findall(
        r"(\d{4})\s*[–\-]\s*(\d{4}|present|current)",
        resume_plain,
        re.IGNORECASE,
    )
    for start_str, end_str in spans:
        start = int(start_str)
        end = current_year if end_str.lower() in ("present", "current") else int(end_str)
        if 1980 <= start <= current_year and start <= end <= current_year + 1:
            total += end - start
            found_any = True

    return total if found_any else None


def _openai_extract_years(jd_text: str, client: openai.OpenAI) -> int | None:
    prompt = (
        "How many years of work experience does this job description require? "
        "Reply with ONLY a single integer (e.g. 3). If unspecified, reply with null.\n\n"
        f"{jd_text[:2000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        val = (resp.choices[0].message.content or "").strip().lower()
        if val in ("null", "none"):
            return None
        return int(val)
    except Exception:
        return None


# ── Dimension 5: Location Fit (10 pts) ────────────────────────────────────────

def _score_location(jd_text: str) -> int:
    lower = jd_text.lower()

    if any(kw in lower for kw in ["fully remote", "100% remote", "work from anywhere", "remote-first"]):
        return 10
    if "remote" in lower and "hybrid" not in lower:
        return 9
    if "hybrid" in lower:
        return 7
    if any(kw in lower for kw in ["on-site", "onsite", "in-office", "in office", "in person", "on site"]):
        return 3
    return 5
