"""
tools/extract_meta.py — Job Metadata Extractor

Uses OpenAI to extract company name and role title from a job description.
Falls back to regex if the API response is malformed JSON.
"""

from __future__ import annotations

import json
import os
import re

import openai

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def extract(jd_text: str, client: openai.OpenAI) -> dict:
    """
    Extract company name and role title from the job description.

    Args:
        jd_text: Plain text of the job description.
        client:  Initialized OpenAI client.

    Returns:
        {
            "company_name": str,   # e.g. "Stripe"
            "role_title":  str,    # e.g. "Senior ML Engineer"
            "safe_company": str,   # filesystem-safe version of company_name
        }
    """
    prompt = (
        "Extract the company name and job title from this job description. "
        'Return ONLY valid JSON in this exact format: {"company": "...", "role": "..."} '
        "If you cannot determine either, use \"Unknown\". No explanation, no markdown.\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:3000]}"
    )

    meta = {"company": "Unknown", "role": "Unknown"}

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw).strip()
        meta = json.loads(raw)
    except (json.JSONDecodeError, IndexError, openai.APIError):
        # Regex fallback
        company_match = re.search(
            r"\bat\s+([A-Z][A-Za-z0-9\s&,\.]+?)[\.,\n\(\)]", jd_text
        )
        role_match = re.search(
            r"(?:position|role|title|job)[:\s]+([^\n]{5,80})", jd_text, re.IGNORECASE
        )
        if company_match:
            meta["company"] = company_match.group(1).strip()
        if role_match:
            meta["role"] = role_match.group(1).strip()

    company_name = str(meta.get("company", "Unknown")).strip() or "Unknown"
    role_title = str(meta.get("role", "Unknown")).strip() or "Unknown"
    safe_company = re.sub(r"[^\w]", "_", company_name).strip("_") or "Unknown"

    return {
        "company_name": company_name,
        "role_title": role_title,
        "safe_company": safe_company,
    }
