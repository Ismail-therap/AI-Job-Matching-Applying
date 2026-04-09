"""
JobMatchingApp — Single Unified App
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import openai
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

import tools.compile_pdf as compile_pdf
import tools.cost_tracker as cost_tracker
import tools.extract_meta as extract_meta
import tools.generate_cover as generate_cover
import tools.parse_jd as parse_jd
import tools.parse_resume as parse_resume
import tools.score_match as score_match
import tools.tailor_resume as tailor_resume
import tools.visa_check as visa_check

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ANALYSIS_FILE = Path(".tmp/last_analysis.json")


# ── CSS ────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
    <style>
    /* ── Base & font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #f1f5f9; }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 1280px; }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 0 0 24px 24px;
        padding: 32px 40px 28px;
        margin: -1rem -1rem 28px -1rem;
        color: white;
    }
    .hero h1 {
        font-size: 2rem; font-weight: 700; margin: 0 0 4px;
        letter-spacing: -0.5px; color: white;
    }
    .hero p { font-size: 0.95rem; opacity: 0.75; margin: 0; color: white; }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 10px;
        color: rgba(255,255,255,0.9); letter-spacing: 0.5px;
    }

    /* ── Cards ── */
    .card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase; color: #64748b; margin-bottom: 12px;
        display: flex; align-items: center; gap: 6px;
    }
    .card-title-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #4f46e5;
        display: inline-block;
    }

    /* ── Step badge ── */
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 24px; height: 24px; border-radius: 50%;
        background: #4f46e5; color: white;
        font-size: 0.75rem; font-weight: 700; margin-right: 8px;
        flex-shrink: 0;
    }
    .step-row {
        display: flex; align-items: center;
        font-size: 0.85rem; font-weight: 600; color: #1e293b; margin-bottom: 12px;
    }

    /* ── Status pills ── */
    .pill-loaded {
        display: inline-flex; align-items: center; gap: 6px;
        background: #f0fdf4; border: 1px solid #bbf7d0;
        color: #15803d; border-radius: 8px; padding: 6px 12px;
        font-size: 0.83rem; font-weight: 500; width: 100%;
    }
    .pill-empty {
        display: inline-flex; align-items: center; gap: 6px;
        background: #f8fafc; border: 1.5px dashed #cbd5e1;
        color: #94a3b8; border-radius: 8px; padding: 6px 12px;
        font-size: 0.83rem; width: 100%; margin-bottom: 8px;
    }

    /* ── Score card ── */
    .score-header {
        display: flex; align-items: flex-start; justify-content: space-between;
        margin-bottom: 16px;
    }
    .score-company { font-size: 1.1rem; font-weight: 700; color: #0f172a; }
    .score-role { font-size: 0.87rem; color: #64748b; margin-top: 2px; }
    .score-badge-strong {
        background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0;
        border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
        white-space: nowrap;
    }
    .score-badge-moderate {
        background: #fef3c7; color: #92400e; border: 1px solid #fde68a;
        border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
        white-space: nowrap;
    }
    .score-badge-weak {
        background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;
        border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
        white-space: nowrap;
    }

    /* ── Skill badges ── */
    .skill-matched {
        display: inline-block; background: #f0fdf4; color: #15803d;
        border: 1px solid #bbf7d0; border-radius: 6px;
        padding: 3px 10px; font-size: 0.79rem; font-weight: 500;
        margin: 3px 3px 3px 0;
    }
    .skill-missing {
        display: inline-block; background: #fff1f2; color: #be123c;
        border: 1px solid #fecdd3; border-radius: 6px;
        padding: 3px 10px; font-size: 0.79rem; font-weight: 500;
        margin: 3px 3px 3px 0;
    }
    .skill-section-title {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.6px;
        text-transform: uppercase; color: #64748b; margin-bottom: 8px;
    }

    /* ── Keyword callout ── */
    .kw-callout {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border: 1.5px solid #fbbf24; border-radius: 10px;
        padding: 12px 16px; margin-bottom: 14px;
        display: flex; align-items: center; gap: 10px;
    }
    .kw-callout-icon { font-size: 1.2rem; }
    .kw-callout-text {
        font-size: 0.85rem; font-weight: 600; color: #78350f;
    }

    /* ── Download cards ── */
    .dl-card {
        background: linear-gradient(135deg, #f8faff, #eef2ff);
        border: 1px solid #c7d2fe; border-radius: 12px;
        padding: 18px 20px; text-align: center;
    }
    .dl-card-icon { font-size: 2rem; margin-bottom: 6px; }
    .dl-card-title { font-size: 0.9rem; font-weight: 600; color: #3730a3; margin-bottom: 4px; }
    .dl-card-sub { font-size: 0.78rem; color: #6366f1; }

    /* ── Visa warning ── */
    .visa-warning {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1.5px solid #fb7185; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 16px;
    }
    .visa-warning-title {
        font-size: 0.92rem; font-weight: 700; color: #be123c; margin-bottom: 4px;
    }
    .visa-warning-body { font-size: 0.82rem; color: #9f1239; }

    /* ── Cost table ── */
    .cost-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 0; border-bottom: 1px solid #f1f5f9;
        font-size: 0.83rem;
    }
    .cost-label { color: #475569; }
    .cost-value { font-weight: 600; color: #1e293b; }
    .cost-total {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0 0; font-size: 0.9rem; font-weight: 700; color: #4f46e5;
    }

    /* ── Streamlit widget overrides ── */
    div[data-testid="stTabs"] button {
        font-size: 0.83rem !important; font-weight: 500 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; letter-spacing: 0.3px !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
    }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px !important; font-weight: 600 !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px !important; border: 1px solid #e2e8f0 !important;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 8px !important; font-size: 0.85rem !important;
    }
    div[data-testid="stSelectbox"] > div {
        border-radius: 8px !important;
    }
    div[data-testid="stMultiSelect"] > div {
        border-radius: 8px !important;
    }

    /* Status widget */
    div[data-testid="stStatus"] {
        border-radius: 10px !important;
    }

    /* Metric */
    div[data-testid="stMetric"] {
        background: #f8fafc; border-radius: 10px;
        padding: 12px; border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Cached resources ────────────────────────────────────────────────────────────

@st.cache_resource
def get_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-your-key-here":
        st.error("OPENAI_API_KEY is not set in your .env file.")
        st.stop()
    return openai.OpenAI(api_key=api_key)


@st.cache_resource(show_spinner="Loading AI scoring model (first run only)...")
def get_model():
    return score_match.get_model()


# ── Session state ───────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "resume_tex":      "",
        "resume_plain":    "",
        "resume_filename": "",
        "resume_sections": [],
        "cover_tex":       None,
        "cover_filename":  "",
        "jd_text":         "",
        "meta":            {},
        "visa_result":     {},
        "score_result":    {},
        "scoring_done":    False,
        "package_done":    False,
        "output_paths":    {},
        "output_warnings": [],
        "cover_tex_out":   "",
        "refine_status":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_analysis():
    st.session_state.scoring_done    = False
    st.session_state.package_done    = False
    st.session_state.meta            = {}
    st.session_state.visa_result     = {}
    st.session_state.score_result    = {}
    st.session_state.output_paths    = {}
    st.session_state.output_warnings = []
    for k in ("kw_section_idx", "kw_selected"):
        st.session_state.pop(k, None)


# ── Charts ──────────────────────────────────────────────────────────────────────

def _score_band(total):
    if total >= 75:
        return "Strong Match", "#22c55e", "score-badge-strong"
    elif total >= 50:
        return "Moderate Match", "#f59e0b", "score-badge-moderate"
    return "Weak Match", "#ef4444", "score-badge-weak"


def _gauge_chart(total):
    _, color, _ = _score_band(total)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=total,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 42, "color": color}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                     "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],   "color": "#fee2e2"},
                {"range": [50, 75],  "color": "#fef3c7"},
                {"range": [75, 100], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": total},
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _dimension_chart(dimensions):
    labels = {
        "semantic_similarity": "Semantic",
        "skills_coverage":     "Skills",
        "title_alignment":     "Job Title",
        "experience_match":    "Experience",
        "location_fit":        "Location",
    }
    maxes = {
        "semantic_similarity": 35, "skills_coverage": 25,
        "title_alignment": 15, "experience_match": 15, "location_fit": 10,
    }
    keys       = list(labels.keys())
    values     = [dimensions.get(k, 0) for k in keys]
    max_values = [maxes[k] for k in keys]
    pct        = [v / m * 100 for v, m in zip(values, max_values)]
    colors     = ["#4f46e5" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444" for p in pct]
    fig = go.Figure(go.Bar(
        x=values,
        y=[f"{labels[k]}  ({maxes[k]}pts)" for k in keys],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v}/{m}" for v, m in zip(values, max_values)],
        textposition="outside",
        textfont={"size": 11, "family": "Inter"},
    ))
    fig.update_layout(
        height=220, margin=dict(t=10, b=10, l=10, r=50),
        xaxis=dict(range=[0, 37], showgrid=True, gridcolor="#f1f5f9",
                   title=None, tickfont={"size": 10}),
        yaxis=dict(autorange="reversed", tickfont={"size": 11}),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _is_streamlit_cloud() -> bool:
    """Detect if running on Streamlit Community Cloud."""
    return os.getenv("STREAMLIT_SHARING_MODE") is not None


def _output_folder(safe_company: str) -> Path | None:
    """Return the local output folder, or None on web deployment."""
    if _is_streamlit_cloud():
        return None
    custom = st.session_state.get("local_save_path", "").strip()
    base = Path(custom) if custom else Path(__file__).parent / "outputs"
    folder = base / datetime.now().strftime("%Y-%m-%d") / safe_company
    try:
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    except Exception as e:
        st.warning(f"Could not create output folder `{folder}`: {e}\n\nFiles will be available via download buttons only.")
        return None


def _save_analysis():
    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "jd_text":        st.session_state.jd_text,
        "meta":           st.session_state.meta,
        "missing_skills": st.session_state.score_result.get("missing_skills", []),
        "score_result":   st.session_state.score_result,
    }
    ANALYSIS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Uploaders ───────────────────────────────────────────────────────────────────

def _resume_uploader():
    f = st.file_uploader("Upload .tex resume", type=["tex"],
                         label_visibility="collapsed", key="resume_upload")
    if f:
        tex = f.read().decode("utf-8", errors="replace")
        if tex == st.session_state.resume_tex:
            return
        plain = parse_resume.strip_latex(tex)
        st.session_state.resume_tex      = tex
        st.session_state.resume_plain    = plain
        st.session_state.resume_filename = f.name
        st.session_state.resume_sections = tailor_resume.extract_sections_and_subsections(tex)
        _reset_analysis()


def _cover_uploader():
    f = st.file_uploader("Upload .tex cover letter template", type=["tex"],
                         label_visibility="collapsed", key="cover_upload")
    if f:
        new_tex = f.read().decode("utf-8", errors="replace")
        if new_tex == st.session_state.cover_tex:
            return
        st.session_state.cover_tex      = new_tex
        st.session_state.cover_filename = f.name


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    _inject_css()
    _init_state()
    client = get_client()
    model  = get_model()

    # ── Sidebar: info ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if _is_streamlit_cloud():
            st.info("Running on web — files are downloaded via the buttons below. Local save is not available.")
        else:
            st.info("Set your output folder in the main panel before generating.")

    # ── Hero header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI-POWERED · GPT-4o</div>
        <h1>🎯 JobMatch AI</h1>
        <p>Score your resume fit, inject missing keywords, and generate a tailored cover letter — in one click.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: inputs ─────────────────────────────────────────────────────────
    with left:

        # Step 1 — Resume
        st.markdown("""<div class="card">
            <div class="card-title"><span class="card-title-dot"></span>Step 1 — Resume / CV</div>
        """, unsafe_allow_html=True)

        if st.session_state.resume_tex:
            st.markdown(f"""<div class="pill-loaded">
                ✅ <strong>{st.session_state.resume_filename}</strong>
            </div>""", unsafe_allow_html=True)
            st.write("")
            with st.expander("Replace resume"):
                _resume_uploader()
        else:
            st.markdown('<div class="pill-empty">📄 No resume loaded — upload a .tex file</div>',
                        unsafe_allow_html=True)
            _resume_uploader()

        st.markdown("</div>", unsafe_allow_html=True)

        # Step 2 — Cover Letter
        st.markdown("""<div class="card">
            <div class="card-title"><span class="card-title-dot"></span>Step 2 — Cover Letter Template <span style="font-weight:400;color:#94a3b8">(optional)</span></div>
        """, unsafe_allow_html=True)

        if st.session_state.cover_tex:
            st.markdown(f"""<div class="pill-loaded">
                ✅ <strong>{st.session_state.cover_filename}</strong>
            </div>""", unsafe_allow_html=True)
            st.write("")
            with st.expander("Replace template"):
                _cover_uploader()
        else:
            st.markdown('<div class="pill-empty">📝 No template — default formatting will be used</div>',
                        unsafe_allow_html=True)
            _cover_uploader()

        st.markdown("</div>", unsafe_allow_html=True)

        # Step 3 — Job Description
        st.markdown("""<div class="card">
            <div class="card-title"><span class="card-title-dot"></span>Step 3 — Job Description</div>
        """, unsafe_allow_html=True)

        tab_paste, tab_url, tab_file = st.tabs(["✏️  Paste", "🔗  URL", "📂  Upload File"])

        with tab_paste:
            pasted = st.text_area("Paste JD", height=160, label_visibility="collapsed",
                                  placeholder="Paste the full job description here...")
            if st.button("Use This Text →", key="btn_paste", use_container_width=True):
                if pasted.strip():
                    st.session_state.jd_text = parse_jd.from_text(pasted)
                    _reset_analysis()

        with tab_url:
            url = st.text_input("Job posting URL", placeholder="https://...", key="jd_url")
            if st.button("Fetch Job Description →", key="btn_url", use_container_width=True):
                if url.strip():
                    with st.spinner("Fetching..."):
                        try:
                            st.session_state.jd_text = parse_jd.from_url(url.strip())
                            _reset_analysis()
                        except ValueError as e:
                            st.error(str(e))

        with tab_file:
            jd_file = st.file_uploader("Upload JD file", type=["pdf", "docx", "txt"],
                                       label_visibility="collapsed", key="jd_file")
            if jd_file and st.button("Use This File →", key="btn_jd_file", use_container_width=True):
                try:
                    st.session_state.jd_text = parse_jd.from_file(jd_file.read(), jd_file.name)
                    _reset_analysis()
                except Exception as e:
                    st.error(str(e))

        if st.session_state.jd_text:
            st.markdown(f"""<div class="pill-loaded" style="margin-top:10px">
                ✅ Job description loaded &nbsp;·&nbsp;
                <span style="color:#6b7280">{len(st.session_state.jd_text):,} chars</span>
            </div>""", unsafe_allow_html=True)
            with st.expander("Preview"):
                st.text(st.session_state.jd_text[:600] +
                        ("..." if len(st.session_state.jd_text) > 600 else ""))

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Analyze button ───────────────────────────────────────────────────
        analyze_ready = bool(st.session_state.resume_tex) and bool(st.session_state.jd_text)

        if not analyze_ready:
            parts = []
            if not st.session_state.resume_tex: parts.append("resume")
            if not st.session_state.jd_text:    parts.append("job description")
            st.markdown(
                f'<p style="text-align:center;color:#94a3b8;font-size:0.83rem;margin:4px 0 8px">'
                f'Upload your {" and ".join(parts)} to continue</p>',
                unsafe_allow_html=True,
            )

        if st.button("🔍  Analyze Job Match", type="primary", use_container_width=True,
                     disabled=not analyze_ready, key="btn_analyze"):
            _reset_analysis()
            with st.spinner("Analyzing your match..."):
                meta = extract_meta.extract(st.session_state.jd_text, client)
                st.session_state.meta        = meta
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

    # ── RIGHT: results ────────────────────────────────────────────────────────
    with right:
        if st.session_state.scoring_done:
            _show_results(client)
        else:
            # Empty state
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:420px;text-align:center;color:#94a3b8;">
                <div style="font-size:3.5rem;margin-bottom:16px">📊</div>
                <div style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:8px">
                    Results will appear here
                </div>
                <div style="font-size:0.85rem;max-width:280px;line-height:1.5">
                    Upload your resume, add a job description, then click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Results panel ───────────────────────────────────────────────────────────────

def _show_results(client):
    meta   = st.session_state.meta
    visa   = st.session_state.visa_result
    result = st.session_state.score_result

    # Visa warning
    if visa.get("flagged"):
        phrases = visa.get("matched_phrases", [])
        st.markdown(f"""
        <div class="visa-warning">
            <div class="visa-warning-title">🚫 Visa / Sponsorship Warning</div>
            <div class="visa-warning-body">
                This posting likely does not offer visa sponsorship.<br>
                Detected: {', '.join(f'<code>{p}</code>' for p in phrases)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    company = meta.get("company_name", "Unknown")
    role    = meta.get("role_title",   "Unknown")
    total   = result.get("total", 0)
    _, _, badge_cls = _score_band(total)

    # Score header card
    st.markdown(f"""
    <div class="card" style="margin-bottom:14px">
        <div class="score-header">
            <div>
                <div class="score-company">{company}</div>
                <div class="score-role">{role}</div>
            </div>
            <div class="{badge_cls}">{_score_band(total)[0]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Company name override (inside expander to keep it tidy)
    with st.expander("✏️ Edit company name (if incorrectly detected)"):
        override = st.text_input("Company name", value=company, key="company_override",
                                 label_visibility="collapsed")
        if override.strip() and override != company:
            st.session_state.meta["company_name"] = override
            st.session_state.meta["safe_company"] = re.sub(r"[^\w]", "_", override).strip("_")

    # Charts
    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown('<div class="card" style="padding:14px">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Overall Score</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_gauge_chart(total), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card" style="padding:14px">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Score Breakdown</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_dimension_chart(result.get("dimensions", {})), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Skills
    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_m, col_x = st.columns(2)
    with col_m:
        st.markdown('<div class="skill-section-title">✅ Matched Skills</div>', unsafe_allow_html=True)
        if matched:
            st.markdown(
                "".join(f'<span class="skill-matched">{s}</span>' for s in matched),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span style="color:#94a3b8;font-size:0.83rem">None detected</span>',
                        unsafe_allow_html=True)
    with col_x:
        st.markdown('<div class="skill-section-title">❌ Missing Skills</div>', unsafe_allow_html=True)
        if missing:
            st.markdown(
                "".join(f'<span class="skill-missing">{s}</span>' for s in missing),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span style="color:#94a3b8;font-size:0.83rem">No gaps — great match!</span>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Keyword placement ────────────────────────────────────────────────────
    sections = st.session_state.resume_sections
    if sections and missing:
        st.markdown("""
        <div class="kw-callout">
            <span class="kw-callout-icon">⚠️</span>
            <span class="kw-callout-text">Required — choose the CV section to receive missing keywords before applying</span>
        </div>
        """, unsafe_allow_html=True)

        section_labels = []
        for _, name, lvl in sections:
            prefix = "  └ " if lvl == "subsection" else ""
            section_labels.append(f"{prefix}{name}")

        skill_keywords = ("skill", "technical", "competenc", "tool", "method", "software")
        default_idx = next(
            (i for i, (_, n, _) in enumerate(sections)
             if any(k in n.lower() for k in skill_keywords)),
            0,
        )

        sel_idx = st.selectbox(
            "📌 Insert keywords into section:",
            range(len(sections)),
            index=default_idx,
            format_func=lambda i: section_labels[i],
            key="kw_section_idx",
        )
        _, sel_name, _ = sections[sel_idx]

        st.markdown("**Select keywords to inject** — remove any you don't want:")
        selected_kws = st.multiselect(
            "Keywords",
            options=missing,
            default=missing,
            key="kw_selected",
            label_visibility="collapsed",
        )
        if selected_kws:
            st.markdown(f"""<div class="pill-loaded" style="margin-top:6px">
                ✅ {len(selected_kws)} keyword(s) → <strong>{sel_name}</strong>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("No keywords selected — keyword injection will be skipped.")

        st.write("")

    # ── Output folder picker ─────────────────────────────────────────────────
    if not st.session_state.package_done and not _is_streamlit_cloud():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Output Folder</div>',
                    unsafe_allow_html=True)
        st.caption("Files will be saved as: `<folder> / YYYY-MM-DD / CompanyName /`")

        col_path, col_browse = st.columns([5, 1])
        with col_path:
            typed = st.text_input(
                "Output folder path",
                value=st.session_state.get("local_save_path", ""),
                placeholder="Leave blank to use outputs/ inside the app folder",
                label_visibility="collapsed",
            )
            st.session_state["local_save_path"] = typed
        with col_browse:
            if st.button("Browse", use_container_width=True, key="btn_browse_folder"):
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.wm_attributes("-topmost", 1)
                    chosen = filedialog.askdirectory(title="Select output folder")
                    root.destroy()
                    if chosen:
                        st.session_state["local_save_path"] = chosen.replace("\\", "/")
                        st.rerun()
                except Exception as e:
                    st.warning(f"Browse failed: {e} — type the path manually.")

        current = st.session_state.get("local_save_path", "").strip()
        preview = current if current else str(Path(__file__).parent / "outputs")
        st.markdown(
            f'<p style="font-size:0.78rem;color:#64748b;margin:6px 0 0">Save location: <code>{preview}/YYYY-MM-DD/CompanyName/</code></p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Apply button ─────────────────────────────────────────────────────────
    if not st.session_state.package_done:
        st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
        if st.button("📄  Generate Tailored CV + Cover Letter",
                     key="btn_apply", type="primary", use_container_width=True):
            _run_package(client)

    if st.session_state.package_done:
        _show_package_results(client)


# ── Application package ─────────────────────────────────────────────────────────

def _run_package(client):
    cost_tracker.reset()
    meta         = st.session_state.meta
    safe_company = meta.get("safe_company") or re.sub(
        r"[^\w]", "_", meta.get("company_name", "Unknown")
    ).strip("_") or "Unknown"

    keywords      = st.session_state.score_result.get("missing_skills") or []
    jd_text       = st.session_state.jd_text
    warnings      = []
    tailored_tex  = st.session_state.resume_tex
    cover_tex_out = ""
    resume_pdf_bytes: bytes | None = None
    cover_pdf_bytes:  bytes | None = None

    with st.status("⚙️ Building your application package...", expanded=True) as pkg_status:

        # STEP 1 — Inject keywords
        sections       = st.session_state.get("resume_sections", [])
        kw_section_idx = st.session_state.get("kw_section_idx")
        kw_selected    = st.session_state.get("kw_selected", [])
        kw_input_str   = ", ".join(kw_selected) if kw_selected else ""

        if sections and kw_section_idx is not None and kw_input_str:
            section_line_idx, section_name, _ = sections[kw_section_idx]
            st.write(f"**1 / 4** — Inserting {len(kw_selected)} keyword(s) into **{section_name}**…")
            try:
                tailored_tex = tailor_resume.insert_keyword_line(
                    st.session_state.resume_tex, section_line_idx, kw_input_str
                )
                injected, still_missing = tailor_resume.check_keyword_coverage(tailored_tex, kw_selected)
                if injected:
                    st.write(f"✅ Confirmed: {', '.join(injected)}")
                if still_missing:
                    st.write(f"⚠️ Not confirmed: {', '.join(still_missing)}")
                    warnings.append(f"Keywords not confirmed: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword insert error: {e}")
                st.write("⚠️ Insert failed — original CV used.")
        else:
            st.write(f"**1 / 4** — Injecting {len(keywords)} keyword(s) into CV…")
            try:
                tailored_tex, tailor_warn = tailor_resume.tailor(
                    tex_str=st.session_state.resume_tex,
                    jd_text=jd_text,
                    client=client,
                    keywords=keywords if keywords else None,
                )
                warnings.extend(tailor_warn)
                if tailor_warn:
                    st.write(f"⚠️ {tailor_warn[0]}")
                else:
                    st.write("✅ Keywords injected into CV.")
                if keywords:
                    injected, still_missing = tailor_resume.check_keyword_coverage(tailored_tex, keywords)
                    if injected:
                        st.write(f"✅ Confirmed: {', '.join(injected)}")
                    if still_missing:
                        st.write(f"⚠️ Could not place: {', '.join(still_missing)}")
                        warnings.append(f"Keywords not placed: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword injection error: {e}")
                st.write("⚠️ Injection failed — original CV used.")

        # STEP 2 — Cover letter
        st.write("**2 / 4** — Writing tailored cover letter…")
        try:
            cover_tex_out, cover_warn = generate_cover.generate(
                cover_tex=st.session_state.cover_tex,
                jd_text=jd_text,
                meta=meta,
                resume_plain=st.session_state.resume_plain,
                client=client,
                default_template_path="default_cover_letter.tex",
                resume_tex=st.session_state.resume_tex,
            )
            warnings.extend(cover_warn)
            st.session_state.cover_tex_out = cover_tex_out
            mode = "your template" if st.session_state.cover_tex else "default template"
            st.write(f"✅ Cover letter written ({mode}).")
        except Exception as e:
            warnings.append(f"Cover letter failed: {e}")
            st.write(f"❌ Cover letter failed: {e}")
            pkg_status.update(label="❌ Cover letter generation failed.", state="error")
            return

        # STEP 3 — Compile PDFs
        st.write("**3 / 4** — Compiling PDFs…")
        resume_pdf_bytes = compile_pdf.compile_from_string(
            tailored_tex, f"{safe_company}_Resume.tex", ".tmp"
        )
        if resume_pdf_bytes:
            st.write(f"✅ Resume PDF compiled ({len(resume_pdf_bytes) // 1024} KB).")
        else:
            log_text = compile_pdf.get_compile_log(tailored_tex, f"{safe_company}_Resume.tex", ".tmp")
            warnings.append(f"Resume PDF failed.\n\npdflatex log:\n{log_text or 'No log.'}")
            st.write("❌ Resume PDF failed — see warnings.")

        cover_pdf_bytes = compile_pdf.compile_from_string(
            cover_tex_out, f"{safe_company}_CoverLetter.tex", ".tmp"
        )
        if cover_pdf_bytes:
            st.write(f"✅ Cover letter PDF compiled ({len(cover_pdf_bytes) // 1024} KB).")
        else:
            log_text = compile_pdf.get_compile_log(cover_tex_out, f"{safe_company}_CoverLetter.tex", ".tmp")
            warnings.append(f"Cover letter PDF failed.\n\npdflatex log:\n{log_text or 'No log.'}")
            st.write("❌ Cover letter PDF failed — see warnings.")

        # STEP 4 — Save files
        st.write("**4 / 4** — Saving files…")
        output_paths = {}
        folder = _output_folder(safe_company)

        # Always keep bytes for download buttons / ZIP
        if resume_pdf_bytes:
            output_paths["resume_pdf_bytes"] = resume_pdf_bytes
        if cover_pdf_bytes:
            output_paths["cover_pdf_bytes"] = cover_pdf_bytes
        output_paths["resume_tex"] = tailored_tex
        output_paths["cover_tex"]  = cover_tex_out

        pdf_count = (1 if resume_pdf_bytes else 0) + (1 if cover_pdf_bytes else 0)

        if folder is not None:
            # Local save
            (folder / "Resume.tex").write_text(tailored_tex, encoding="utf-8")
            (folder / "CoverLetter.tex").write_text(cover_tex_out, encoding="utf-8")
            if resume_pdf_bytes:
                p = folder / "Resume.pdf"
                p.write_bytes(resume_pdf_bytes)
                output_paths["resume_pdf"] = str(p)
            if cover_pdf_bytes:
                p = folder / "CoverLetter.pdf"
                p.write_bytes(cover_pdf_bytes)
                output_paths["cover_pdf"] = str(p)
            output_paths["folder"] = str(folder)
            st.write(f"✅ {pdf_count}/2 PDF(s) + .tex files saved → `{folder}`")
        else:
            st.write(f"✅ {pdf_count}/2 PDF(s) ready — use the download buttons below.")

        if pdf_count == 2:
            pkg_status.update(label="✅ Application package ready!", state="complete", expanded=False)
        elif pdf_count == 1:
            pkg_status.update(label="⚠️ Partial — 1/2 PDFs compiled.", state="complete", expanded=True)
        else:
            pkg_status.update(label="❌ PDFs failed — .tex files saved.", state="error", expanded=True)

    st.session_state.output_paths    = output_paths
    st.session_state.output_warnings = warnings
    st.session_state.cost_summary    = cost_tracker.get_summary()
    st.session_state.package_done    = True
    st.rerun()


# ── Cover letter refinement ─────────────────────────────────────────────────────

def _refine_cover(client):
    feedback = st.session_state.get("cover_feedback", "").strip()
    if not feedback:
        st.session_state.refine_status = "no_feedback"
        return

    with st.spinner("Refining cover letter…"):
        new_tex, warn = generate_cover.refine(
            cover_tex_out=st.session_state.cover_tex_out,
            feedback=feedback,
            jd_text=st.session_state.jd_text,
            meta=st.session_state.meta,
            resume_plain=st.session_state.resume_plain,
            client=client,
        )
        st.session_state.cover_tex_out = new_tex
        st.session_state.output_warnings.extend(warn)

        safe_company = st.session_state.meta.get("safe_company", "Company")
        cover_pdf_bytes = compile_pdf.compile_from_string(
            new_tex, f"{safe_company}_CoverLetter.tex", ".tmp"
        )

        if cover_pdf_bytes:
            st.session_state.output_paths["cover_pdf_bytes"] = cover_pdf_bytes
            folder = _output_folder(safe_company)
            if folder is not None:
                (folder / "CoverLetter.pdf").write_bytes(cover_pdf_bytes)
                (folder / "CoverLetter.tex").write_text(new_tex, encoding="utf-8")
            st.session_state.refine_status = "success"
        else:
            st.session_state.refine_status = "error"

    st.rerun()


# ── Download results panel ──────────────────────────────────────────────────────

def _build_zip(paths: dict, safe_company: str) -> bytes:
    """Bundle all output files into a ZIP (files at root, no date subfolder)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if "resume_pdf_bytes" in paths:
            zf.writestr("Resume.pdf", paths["resume_pdf_bytes"])
        if "cover_pdf_bytes" in paths:
            zf.writestr("CoverLetter.pdf", paths["cover_pdf_bytes"])
        if "resume_tex" in paths:
            zf.writestr("Resume.tex", paths["resume_tex"])
        if "cover_tex" in paths:
            zf.writestr("CoverLetter.tex", paths["cover_tex"])
    return buf.getvalue()


def _show_package_results(client):
    paths       = st.session_state.output_paths
    warnings    = st.session_state.output_warnings
    folder      = paths.get("folder", "")
    safe_company = st.session_state.meta.get("safe_company", "Company")

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="card-title-dot"></span>Download Your Application</div>',
                unsafe_allow_html=True)

    if _is_streamlit_cloud():
        # ── Web: single ZIP download ─────────────────────────────────────────
        zip_bytes = _build_zip(paths, safe_company)
        st.download_button(
            f"⬇️  Download {safe_company}.zip (Resume + Cover Letter)",
            data=zip_bytes,
            file_name=f"{safe_company}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
        st.caption("ZIP contains: Resume.pdf · CoverLetter.pdf · Resume.tex · CoverLetter.tex")
    else:
        # ── Local: individual download buttons + saved path ──────────────────
        if folder:
            st.markdown(f'<p style="font-size:0.8rem;color:#64748b;margin:0 0 14px">Saved to: <code>{folder}</code></p>',
                        unsafe_allow_html=True)

        col_r, col_c = st.columns(2)
        with col_r:
            st.markdown("""<div class="dl-card">
                <div class="dl-card-icon">📄</div>
                <div class="dl-card-title">Tailored Resume</div>
                <div class="dl-card-sub">Keywords injected · PDF</div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            if "resume_pdf_bytes" in paths:
                st.download_button(
                    "⬇️  Download Resume (PDF)",
                    data=paths["resume_pdf_bytes"],
                    file_name="Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.error("Resume PDF failed — .tex source saved.")

        with col_c:
            st.markdown("""<div class="dl-card">
                <div class="dl-card-icon">✉️</div>
                <div class="dl-card-title">Cover Letter</div>
                <div class="dl-card-sub">AI-tailored · PDF</div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            if "cover_pdf_bytes" in paths:
                st.download_button(
                    "⬇️  Download Cover Letter (PDF)",
                    data=paths["cover_pdf_bytes"],
                    file_name="CoverLetter.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.error("Cover letter PDF failed — .tex source saved.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Warnings
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} warning(s)"):
            for w in warnings:
                st.warning(w)

    # Cost summary
    cs = st.session_state.get("cost_summary")
    if cs and cs["total_tokens"] > 0:
        with st.expander(f"💰 API cost this run — **${cs['total_cost_usd']:.4f}**"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total tokens",  f"{cs['total_tokens']:,}")
            c2.metric("Prompt tokens", f"{cs['total_prompt']:,}")
            c3.metric("Output tokens", f"{cs['total_completion']:,}")
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            rows_html = ""
            for call in cs["calls"]:
                rows_html += f"""
                <div class="cost-row">
                    <span class="cost-label">{call.label}</span>
                    <span class="cost-value">${call.total_cost:.4f}
                        <span style="font-weight:400;color:#94a3b8;font-size:0.78rem">
                            ({call.total_tokens:,} tokens)
                        </span>
                    </span>
                </div>"""
            st.markdown(f"""
            {rows_html}
            <div class="cost-total">
                <span>Total</span>
                <span>${cs['total_cost_usd']:.4f}</span>
            </div>
            <p style="font-size:0.75rem;color:#94a3b8;margin-top:8px">
                gpt-4o: $2.50/M input · $10.00/M output
            </p>
            """, unsafe_allow_html=True)

    # Refine cover letter section
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="card-title-dot"></span>Refine Cover Letter</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:0 0 12px">Not satisfied? Describe what you\'d like changed and we\'ll regenerate it.</p>',
                unsafe_allow_html=True)

    refine_status = st.session_state.get("refine_status", "")
    if refine_status == "success":
        st.success("Cover letter updated! Use the download button above to get the new version.")
    elif refine_status == "error":
        st.error("Refinement failed — PDF compilation error. The .tex source was saved.")
    elif refine_status == "no_feedback":
        st.warning("Please describe what you'd like changed before regenerating.")

    st.text_area(
        "Your feedback",
        key="cover_feedback",
        placeholder="e.g. Make the opening stronger, mention my Python skills more, shorten the closing paragraph…",
        height=100,
        label_visibility="collapsed",
    )
    if st.button("✏️  Regenerate Cover Letter", key="btn_refine_cover", use_container_width=True):
        st.session_state.refine_status = ""
        _refine_cover(client)
    st.markdown('</div>', unsafe_allow_html=True)

    # New job button
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="text-align:center;padding:20px">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#64748b;margin-bottom:12px">Your CV and cover letter template stay loaded — just paste a new job description.</p>',
                unsafe_allow_html=True)
    if st.button("🔄  Analyze Another Job", key="btn_new_job", type="primary", use_container_width=True):
        st.session_state.jd_text         = ""
        st.session_state.scoring_done    = False
        st.session_state.package_done    = False
        st.session_state.meta            = {}
        st.session_state.visa_result     = {}
        st.session_state.score_result    = {}
        st.session_state.output_paths    = {}
        st.session_state.output_warnings = []
        st.session_state.cover_tex_out   = ""
        st.session_state.refine_status   = ""
        for k in ("kw_section_idx", "kw_selected", "cover_feedback"):
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
