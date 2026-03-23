# Workflow: Job Matching & Application Package

## Objective

Score a candidate's fit for a specific job and, if the candidate chooses to apply, generate an ATS-optimized resume PDF and a tailored cover letter PDF saved to an organized output folder.

## Required Inputs

| Input | Source | Required |
|---|---|---|
| Job description | Paste / URL / PDF / DOCX / TXT | Yes |
| Resume | `.tex` file upload | Yes |
| Cover letter template | `.tex` file upload | No (default template used) |

## Tools Used (in order)

1. `tools/parse_jd.py` — normalize JD to plain text
2. `tools/extract_meta.py` — Claude extracts company + role
3. `tools/visa_check.py` — keyword scan for sponsorship restrictions
4. `tools/score_match.py` — 5-dimension scoring (100 pts)
5. `tools/tailor_resume.py` — Claude rewrites resume bullets (on Apply)
6. `tools/generate_cover.py` — Claude generates/adapts cover letter (on Apply)
7. `tools/compile_pdf.py` — LaTeX.Online REST API → PDF bytes (on Apply)

## Step-by-Step Flow

### Phase 1: Input Collection
- JD accepted via three modes: paste text, fetch URL, or upload file
- Resume `.tex` decoded to raw string + plain text (LaTeX stripped)
- Cover letter `.tex` decoded if provided; `None` if not

### Phase 2: Scoring (on "Analyze" click)
1. Claude extracts company name + role title → stored in session
2. Keyword scan for visa/sponsorship phrases → displayed as red error banner if found
3. Five-dimension scoring:
   - **Semantic Similarity (35 pts)**: `all-MiniLM-L6-v2` cosine similarity
     - Cosine range 0.20–0.65 mapped linearly to 0–35 pts
     - Scores in practice: 0.30–0.55 for relevant matches
   - **Skills Coverage (25 pts)**: Claude extracts up to 15 required skills from JD;
     fuzzy substring match against resume plain text
   - **Title Alignment (15 pts)**: cosine similarity between JD role and
     candidate's most-recent job title (extracted by date-line heuristic)
   - **Experience Match (15 pts)**: regex extracts required years from JD;
     candidate years summed from date spans in resume; gap table applied
   - **Location Fit (10 pts)**: rule-based keyword scan
     - Fully remote: 10, Remote (no hybrid): 9, Hybrid: 7, On-site: 3, Unknown: 5
4. Results displayed: gauge chart (0–100), dimension bar chart, skill pills

### Phase 3: Application Package (on "Apply for This Job" click)
1. **Tailor Resume**: Claude identifies work-experience sections by line number,
   rewrites `\item` bullets section by section, validates brace balance
2. **Generate Cover Letter**: one of three modes (see tools/generate_cover.py)
3. **Compile Resume PDF**: LaTeX.Online POST request; 2s delay before next compile
4. **Compile Cover Letter PDF**: LaTeX.Online POST request
5. **Save**: both PDFs + source `.tex` files saved to `outputs/{Company}_{timestamp}/`
6. Download buttons rendered in UI

## Expected Outputs

```
outputs/{Company}_{YYYYMMDD_HHMMSS}/
├── {Company}_Resume.pdf
├── {Company}_Resume.tex         (tailored source)
├── {Company}_CoverLetter.pdf
└── {Company}_CoverLetter.tex    (generated source)
```

## Score Interpretation

| Score | Label |
|---|---|
| 75–100 | Strong match — recommended to apply |
| 50–74 | Moderate match — consider applying |
| 0–49 | Weak match — significant gaps identified |

The visa warning is displayed regardless of score. The Apply button is always available; the score is advisory only.

## Visa / Sponsorship Keywords

The following phrases trigger the visa warning (case-insensitive substring match).
Update this list directly in `tools/visa_check.py → KEYWORDS` as new phrases are found:

```
no sponsorship, will not sponsor, cannot sponsor, unable to sponsor,
not able to sponsor, does not sponsor, sponsorship is not available,
must be authorized to work, must be legally authorized,
work authorization required, authorized to work in the us,
authorized to work in the united states, no visa, no h-1b, h1b,
us citizens only, u.s. citizens only, citizens and permanent residents,
green card holders only, not eligible for sponsorship,
do not offer sponsorship, does not offer sponsorship,
without sponsorship, without the need for sponsorship
```

## Known Quirks & Calibration Notes

### Sentence-Transformer Model
- Model: `all-MiniLM-L6-v2`, ~80MB, auto-downloads on first run
- Cold-start: 3–8 seconds; Streamlit `@st.cache_resource` loads it once per process
- Typical cosine score range for resume-vs-JD: 0.30–0.55
- Mapping window [0.20, 0.65] → [0, 35 pts]; do not adjust unless scores are consistently 0 or 35

### LaTeX.Online Rate Limits
- Free service; two compilations per job application = two requests
- Default 2s delay between requests (`LATEXONLINE_DELAY_SECONDS` in `.env`)
- On 429 (rate limit): the tool waits 30s and retries once
- On persistent failure: increase delay to 10+ or install MiKTeX locally

### Cover Letter Template Modes
| Condition | Mode | Behavior |
|---|---|---|
| User uploads .tex with `%%COMPANY%%` etc. | A — Token injection | Replace tokens with Claude-generated values |
| User uploads .tex without tokens | B — Body rewrite | Claude rewrites paragraphs, preserving structure |
| No template uploaded | C — Default template | `default_cover_letter.tex` used as skeleton |

### Resume Section Discovery
Claude is asked to identify work-experience sections by line number in Pass 1.
Known failure cases:
- Tabular-based layouts (e.g., `\begin{tabular}` for experience)
- Very short resumes with no `\item` bullets
- Non-standard section names

If section discovery fails, a warning is shown and the original resume is used unchanged.

### LaTeX Validation
After every Claude rewrite:
- Brace balance: `text.count("{") == text.count("}")`
- Bullet check: every line must start with `\item`
- Cover letter: must contain `\begin{document}` and `\end{document}`
Failure in any check → original content is kept + warning shown in UI.

## Error Table

| Situation | What Happens | Fix |
|---|---|---|
| URL fetch returns 403 | `ValueError` shown in UI | Paste the JD text directly instead |
| LaTeX.Online returns non-PDF | Warning + compile log shown | Fix LaTeX errors in the source .tex |
| Section discovery returns [] | Warning; original resume used | Resume structure may be non-standard |
| Anthropic API rate limit | `RateLimitError`; one retry after 30s | Wait and re-click Analyze/Apply |
| ANTHROPIC_API_KEY not set | App stops with error message | Add key to `.env` and restart |

## Self-Improvement Log

*Add notes here when you discover new issues, calibration improvements, or better approaches.*

- 2026-03-22: Initial version. LaTeX.Online as primary PDF compiler, no local LaTeX required.
