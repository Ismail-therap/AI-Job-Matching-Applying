"""
App 1 — Job Match Analyzer

Usage:
    streamlit run app1_analyze.py

What it does:
  1. Upload your resume (.tex)
  2. Provide a job description (paste / URL / file)
  3. Click Analyze → see your match score, skill gaps, and ATS keywords to add
  4. Analysis is saved to .tmp/last_analysis.json for App 2 to use
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import openai
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

import tools.extract_meta as extract_meta
import tools.parse_jd as parse_jd
import tools.parse_resume as parse_resume
import tools.score_match as score_match
import tools.visa_check as visa_check

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="App 1 — Job Match Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ANALYSIS_FILE = Path(".tmp/last_analysis.json")


@st.cache_resource
def get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-your-key-here":
        st.error("OPENAI_API_KEY is not set in your .env file.")
        st.stop()
    return openai.OpenAI(api_key=api_key)


@st.cache_resource(show_spinner="Loading AI scoring model (first run only)...")
def get_st_model():
    return score_match.get_model()


def _init_state():
    defaults = {
        "resume_tex":      "",
        "resume_plain":    "",
        "resume_filename": "",
        "jd_text":         "",
        "scoring_done":    False,
        "meta":            {},
        "visa_result":     {},
        "score_result":    {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_score():
    st.session_state.scoring_done = False
    st.session_state.meta = {}
    st.session_state.visa_result = {}
    st.session_state.score_result = {}


# ── Charts ────────────────────────────────────────────────────────────────────

def _score_band(total: int) -> tuple[str, str]:
    if total >= 75:
        return "Strong match — recommended to apply", "#22c55e"
    elif total >= 50:
        return "Moderate match — consider applying", "#f59e0b"
    return "Weak match — significant gaps identified", "#ef4444"


def _gauge_chart(total: int) -> go.Figure:
    label, color = _score_band(total)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=total,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50],   "color": "#fee2e2"},
                {"range": [50, 75],  "color": "#fef3c7"},
                {"range": [75, 100], "color": "#dcfce7"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def _dimension_chart(dimensions: dict) -> go.Figure:
    labels = {
        "semantic_similarity": "Semantic Similarity (35)",
        "skills_coverage":     "Skills Coverage (25)",
        "title_alignment":     "Title Alignment (15)",
        "experience_match":    "Experience Match (15)",
        "location_fit":        "Location Fit (10)",
    }
    maxes = {"semantic_similarity": 35, "skills_coverage": 25,
             "title_alignment": 15, "experience_match": 15, "location_fit": 10}
    keys = list(labels.keys())
    values = [dimensions.get(k, 0) for k in keys]
    max_values = [maxes[k] for k in keys]
    pct = [v / m * 100 for v, m in zip(values, max_values)]
    colors = ["#22c55e" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444" for p in pct]
    fig = go.Figure(go.Bar(
        x=values, y=[labels[k] for k in keys], orientation="h",
        marker_color=colors,
        text=[f"{v}/{m}" for v, m in zip(values, max_values)],
        textposition="outside",
    ))
    fig.update_layout(
        height=260, margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, 35], title="Points"),
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    client = get_openai_client()
    model  = get_st_model()

    st.title("🔍 App 1 — Job Match Analyzer")
    st.caption("Score your fit and identify ATS keywords to add. Then use **App 2** to update your CV and generate a cover letter.")

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: inputs ──────────────────────────────────────────────────────────
    with left:
        st.subheader("Resume / CV")
        if st.session_state.resume_tex:
            st.success(f"✅ {st.session_state.resume_filename}")
            with st.expander("Change resume"):
                _resume_uploader()
        else:
            _resume_uploader()

        st.divider()

        st.subheader("Job Description")
        tab_paste, tab_url, tab_file = st.tabs(["Paste", "URL", "Upload File"])

        with tab_paste:
            pasted = st.text_area("Paste JD", height=180, label_visibility="collapsed",
                                  placeholder="Paste the full job description here...")
            if st.button("Use This Text", key="btn_paste"):
                if pasted.strip():
                    st.session_state.jd_text = parse_jd.from_text(pasted)
                    _reset_score()
                    st.success(f"Loaded ({len(st.session_state.jd_text)} chars).")

        with tab_url:
            url = st.text_input("URL", placeholder="https://...", key="jd_url")
            if st.button("Fetch", key="btn_url"):
                if url.strip():
                    with st.spinner("Fetching..."):
                        try:
                            st.session_state.jd_text = parse_jd.from_url(url.strip())
                            _reset_score()
                            st.success(f"Fetched ({len(st.session_state.jd_text)} chars).")
                        except ValueError as e:
                            st.error(str(e))

        with tab_file:
            jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"],
                                       label_visibility="collapsed", key="jd_file")
            if jd_file and st.button("Use This File", key="btn_jd_file"):
                try:
                    st.session_state.jd_text = parse_jd.from_file(jd_file.read(), jd_file.name)
                    _reset_score()
                    st.success(f"Loaded ({len(st.session_state.jd_text)} chars).")
                except Exception as e:
                    st.error(str(e))

        if st.session_state.jd_text:
            with st.expander("Preview JD"):
                st.text(st.session_state.jd_text[:600] + ("..." if len(st.session_state.jd_text) > 600 else ""))

        st.divider()

        ready = bool(st.session_state.resume_tex) and bool(st.session_state.jd_text)
        if st.button("🔍 Analyze Match", type="primary", use_container_width=True,
                     disabled=not ready, key="btn_analyze"):
            _reset_score()
            with st.spinner("Analyzing..."):
                meta = extract_meta.extract(st.session_state.jd_text, client)
                st.session_state.meta = meta
                st.session_state.visa_result = visa_check.check(st.session_state.jd_text)
                st.session_state.score_result = score_match.score(
                    jd_text=st.session_state.jd_text,
                    resume_plain=st.session_state.resume_plain,
                    meta=meta,
                    client=client,
                    model=model,
                )
                st.session_state.scoring_done = True
                _save_analysis()

        if not ready:
            missing = []
            if not st.session_state.resume_tex: missing.append("resume")
            if not st.session_state.jd_text: missing.append("job description")
            st.caption(f"Upload your {' and '.join(missing)} to continue.")

    # ── RIGHT: results ─────────────────────────────────────────────────────────
    with right:
        if st.session_state.scoring_done:
            _show_results()


def _resume_uploader():
    f = st.file_uploader("Upload .tex resume", type=["tex"],
                         label_visibility="collapsed", key="resume_upload")
    if f:
        tex, plain = parse_resume.parse(f.read())
        st.session_state.resume_tex      = tex
        st.session_state.resume_plain    = plain
        st.session_state.resume_filename = f.name
        _reset_score()
        st.success(f"Loaded: {f.name}")


def _show_results():
    meta   = st.session_state.meta
    visa   = st.session_state.visa_result
    result = st.session_state.score_result

    if visa.get("flagged"):
        phrases = visa.get("matched_phrases", [])
        st.error(
            f"🚫 **VISA / SPONSORSHIP WARNING**\n\n"
            f"Likely does not offer visa sponsorship.\n\n"
            f"Detected: {', '.join(f'`{p}`' for p in phrases)}"
        )

    company = meta.get("company_name", "Unknown")
    role    = meta.get("role_title", "Unknown")
    st.subheader(f"{role}  ·  {company}")

    total = result.get("total", 0)
    col_g, col_b = st.columns(2)
    with col_g:
        st.plotly_chart(_gauge_chart(total), use_container_width=True)
    with col_b:
        st.plotly_chart(_dimension_chart(result.get("dimensions", {})), use_container_width=True)

    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])

    col_m, col_x = st.columns(2)
    with col_m:
        st.markdown("**✅ Matched Skills**")
        if matched:
            st.markdown(" ".join(
                f'<span style="background:#dcfce7;color:#166534;padding:2px 8px;'
                f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block">{s}</span>'
                for s in matched
            ), unsafe_allow_html=True)
        else:
            st.caption("None detected")

    with col_x:
        st.markdown("**❌ Missing Skills (ATS Keywords to Add)**")
        if missing:
            st.markdown(" ".join(
                f'<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;'
                f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block">{s}</span>'
                for s in missing
            ), unsafe_allow_html=True)
        else:
            st.caption("No gaps — great match!")

    st.divider()
    st.success(
        "✅ **Analysis saved.** Open **App 2** to inject these keywords into your CV and generate a cover letter.\n\n"
        "```\nstreamlit run app2_apply.py\n```"
    )


def _save_analysis():
    """Save analysis result to .tmp/last_analysis.json for App 2 to pick up."""
    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "jd_text":       st.session_state.jd_text,
        "meta":          st.session_state.meta,
        "missing_skills": st.session_state.score_result.get("missing_skills", []),
        "all_skills":    st.session_state.score_result.get("all_skills", []),
        "score_result":  st.session_state.score_result,
    }
    ANALYSIS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
