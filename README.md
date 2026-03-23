# JobMatchingApp

An AI-powered job application assistant built with Streamlit. Upload your resume, paste a job description, and get an instant match score — then generate a tailored CV and cover letter as ready-to-download PDFs.

---

## What It Does

1. **Scores your fit** across 5 dimensions (100 pts total)
2. **Detects visa/sponsorship restrictions** and warns you before you apply
3. **Identifies missing ATS keywords** from the job description
4. **Injects keywords** into your CV at a section you choose
5. **Writes a tailored cover letter** (max 1 page, 3 paragraphs)
6. **Compiles both documents to PDF** and lets you download them instantly

---

## Scoring Breakdown

| Dimension | Points | Method |
|---|---|---|
| Semantic Similarity | 35 | Sentence-transformer cosine similarity |
| Skills Coverage | 25 | GPT-4o extracts JD skills, matched against resume |
| Title Alignment | 15 | Cosine similarity of role titles |
| Experience Match | 15 | Required years vs. candidate years |
| Location Fit | 10 | Rule-based remote / hybrid / on-site check |

**Score bands:** ≥75 = Strong match · 50–74 = Moderate · <50 = Weak

---

## Requirements

- Python 3.10+
- [MiKTeX](https://miktex.org/download) installed and `pdflatex` on your system PATH (for PDF compilation)
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Ismail-therap/JobMatchingApp.git
cd JobMatchingApp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB). This is cached after the first download.

### 3. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
```

> `.env` is in `.gitignore` — it will never be committed to GitHub.

---

## Running the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## How to Use

### Step 1 — Upload your resume
Click **Browse files** and select your `.tex` resume file.

### Step 2 — Upload cover letter template *(optional)*
Upload a `.tex` cover letter template. If skipped, a built-in default template is used.

### Step 3 — Provide the job description
Use any of the three tabs:
- **Paste** — paste the full JD text, click **Use This Text**
- **URL** — paste a job posting URL, click **Fetch**
- **Upload File** — upload a `.pdf`, `.docx`, or `.txt` file

### Step 4 — Analyze
Click **Analyze Job Match**. The right panel shows:
- Match score gauge and per-dimension bar chart
- Matched skills (green) and missing skills (red)
- Visa/sponsorship warning if detected

### Step 5 — Customize keyword injection
- Choose which CV section to inject keywords into
- Deselect any keywords you don't want added

### Step 6 — Apply
Click **Apply — Generate Tailored CV + Cover Letter**. The app will:
1. Inject selected keywords into your CV
2. Write a tailored cover letter
3. Compile both to PDF
4. Show download buttons

### Step 7 — Download
Download your **CV (PDF)** and **Cover Letter (PDF)** directly from the browser.

Files are also saved locally to `outputs/YYYY-MM-DD/{Company}/`.

### Step 8 — Analyze another job
Click **Analyze New Job** — your resume stays loaded, just paste a new JD.

---

## Project Structure

```
JobMatchingApp/
├── app.py                    # Single unified Streamlit app
├── app1_analyze.py           # Standalone analyzer (2-app workflow)
├── app2_apply.py             # Standalone CV updater (2-app workflow)
├── default_cover_letter.tex  # Fallback cover letter template
├── requirements.txt
├── .env                      # Your API key (not committed)
│
├── tools/
│   ├── compile_pdf.py        # LaTeX → PDF via local pdflatex
│   ├── extract_meta.py       # Extract company name and role from JD
│   ├── generate_cover.py     # Cover letter generation (OpenAI)
│   ├── parse_jd.py           # JD ingestion (text / URL / file)
│   ├── parse_resume.py       # .tex resume parser and LaTeX stripper
│   ├── score_match.py        # 5-dimension scoring engine
│   ├── tailor_resume.py      # ATS keyword injection into .tex
│   └── visa_check.py         # Visa/sponsorship keyword scanner
│
├── workflows/
│   └── job_matching.md       # SOP for the full workflow
│
└── outputs/                  # Generated PDFs and .tex files (gitignored)
    └── YYYY-MM-DD/
        └── CompanyName/
            ├── Resume.pdf
            ├── Resume.tex
            ├── CoverLetter.pdf
            └── CoverLetter.tex
```

---

## Two-App Workflow (Alternative)

If you prefer a split workflow:

```bash
# Terminal 1 — analyze the job
streamlit run app1_analyze.py

# Terminal 2 — generate documents
streamlit run app2_apply.py
```

App 1 saves the analysis to `.tmp/last_analysis.json`. App 2 picks it up automatically.

---

## Troubleshooting

**PDF compilation failed**
- Confirm `pdflatex` is on your PATH: run `pdflatex --version` in a terminal
- MiKTeX auto-downloads missing packages on first compile — requires internet access
- The `.tex` source is always saved even if PDF compilation fails

**"OPENAI_API_KEY is not set"**
- Confirm your `.env` file is in the `JobMatchingApp/` folder (not a parent folder)
- Confirm the key starts with `sk-`

**App loads a spinner and nothing appears**
- Normal on first run — the AI scoring model (~80 MB) is downloading
- Subsequent runs are instant (model is cached)

**Missing skills list is empty after analysis**
- OpenAI couldn't extract skills from the JD — try pasting the JD text directly instead of fetching via URL

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | OpenAI GPT-4o |
| Semantic scoring | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF compilation | pdflatex (MiKTeX) |
| URL scraping | httpx + BeautifulSoup4 |
| Visualization | Plotly |

---

## License

MIT
