"""
tools/visa_check.py — Visa / Sponsorship Restriction Detector

Pure Python keyword scan — no API calls.
Returns a flag and the specific phrases found so the UI can display them prominently.

Update the KEYWORDS list in the workflow (workflows/job_matching.md) to add more phrases
as you encounter them in the wild.
"""

KEYWORDS = [
    "no sponsorship",
    "will not sponsor",
    "cannot sponsor",
    "unable to sponsor",
    "not able to sponsor",
    "does not sponsor",
    "sponsorship is not available",
    "must be authorized to work",
    "must be legally authorized",
    "work authorization required",
    "authorized to work in the us",
    "authorized to work in the united states",
    "no visa",
    "no h-1b",
    "h1b",
    "us citizens only",
    "u.s. citizens only",
    "citizens and permanent residents",
    "green card holders only",
    "not eligible for sponsorship",
    "do not offer sponsorship",
    "does not offer sponsorship",
    "without sponsorship",
    "without the need for sponsorship",
]


def check(jd_text: str) -> dict:
    """
    Scan job description text for visa/sponsorship restriction phrases.

    Args:
        jd_text: Plain text of the job description.

    Returns:
        {
            "flagged": bool,
            "matched_phrases": list[str]  # the exact KEYWORD strings that matched
        }
    """
    lower = jd_text.lower()
    matched = [kw for kw in KEYWORDS if kw in lower]
    return {
        "flagged": len(matched) > 0,
        "matched_phrases": matched,
    }
