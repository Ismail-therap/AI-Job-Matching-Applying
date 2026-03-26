"""
JobMatchingApp — Single Unified App
Built by merging app1_analyze.py + app2_apply.py

Usage:
    streamlit run app.py

Workflow:
  1. Upload resume (.tex) and optional cover letter template (.tex)
  2. Provide a job description (paste / URL / file)
  3. Analyze — score the match, detect visa restrictions, identify missing keywords
  4. Apply — inject keywords into CV, generate cover letter, compile PDFs, download
  5. Analyze New Job — resume stays loaded, just paste a new JD
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
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
    page_title="Job Matching App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ANALYSIS_FILE = Path(".tmp/last_analysis.json")


# ── Cached resources ───────────────────────────────────────────────────────────

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


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "resume_tex":      "",
        "resume_plain":    "",
        "resume_filename": "",
        "resume_sections": [],   # [(line_idx, name, level), ...]
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
    # Clear keyword widget state so selectbox/multiselect start fresh for new analysis.
    # Without this, Streamlit raises StreamlitAPIException when the new job's missing
    # skills differ from the previous selection stored in session state.
    for k in ("kw_section_idx", "kw_selected"):
        st.session_state.pop(k, None)


# ── Charts ─────────────────────────────────────────────────────────────────────

def _score_band(total):
    if total >= 75:
        return "Strong match — recommended to apply", "#22c55e"
    elif total >= 50:
        return "Moderate match — consider applying", "#f59e0b"
    return "Weak match — significant gaps identified", "#ef4444"


def _gauge_chart(total):
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


def _dimension_chart(dimensions):
    labels = {
        "semantic_similarity": "Semantic Similarity (35)",
        "skills_coverage":     "Skills Coverage (25)",
        "title_alignment":     "Title Alignment (15)",
        "experience_match":    "Experience Match (15)",
        "location_fit":        "Location Fit (10)",
    }
    maxes = {
        "semantic_similarity": 35, "skills_coverage": 25,
        "title_alignment": 15, "experience_match": 15, "location_fit": 10,
    }
    keys       = list(labels.keys())
    values     = [dimensions.get(k, 0) for k in keys]
    max_values = [maxes[k] for k in keys]
    pct        = [v / m * 100 for v, m in zip(values, max_values)]
    colors     = ["#22c55e" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444" for p in pct]
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _output_folder(safe_company: str) -> Path:
    folder = Path("outputs") / datetime.now().strftime("%Y-%m-%d") / safe_company
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_analysis():
    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "jd_text":        st.session_state.jd_text,
        "meta":           st.session_state.meta,
        "missing_skills": st.session_state.score_result.get("missing_skills", []),
        "score_result":   st.session_state.score_result,
    }
    ANALYSIS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Uploaders ──────────────────────────────────────────────────────────────────

def _resume_uploader():
    f = st.file_uploader("Upload .tex resume", type=["tex"],
                         label_visibility="collapsed", key="resume_upload")
    if f:
        # st.file_uploader returns the same file on every rerender (not just on new uploads).
        # Guard against spurious re-processing: only act when the content actually changed.
        tex = f.read().decode("utf-8", errors="replace")
        if tex == st.session_state.resume_tex:
            return
        plain = parse_resume.strip_latex(tex)
        st.session_state.resume_tex      = tex
        st.session_state.resume_plain    = plain
        st.session_state.resume_filename = f.name
        st.session_state.resume_sections = tailor_resume.extract_sections_and_subsections(tex)
        _reset_analysis()
        st.success(f"Loaded: {f.name}")


def _cover_uploader():
    f = st.file_uploader("Upload .tex cover letter template", type=["tex"],
                         label_visibility="collapsed", key="cover_upload")
    if f:
        new_tex = f.read().decode("utf-8", errors="replace")
        if new_tex == st.session_state.cover_tex:
            return
        st.session_state.cover_tex      = new_tex
        st.session_state.cover_filename = f.name
        st.success(f"Loaded: {f.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    client = get_client()
    model  = get_model()

    st.title("🎯 Job Matching App")
    st.caption("Score your fit, then generate a tailored resume + cover letter with one click.")

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: inputs ──────────────────────────────────────────────────────────
    with left:

        st.subheader("Resume / CV")
        if st.session_state.resume_tex:
            st.success(f"✅ Loaded: **{st.session_state.resume_filename}**")
            with st.expander("Change resume"):
                _resume_uploader()
        else:
            _resume_uploader()

        st.divider()

        st.subheader("Cover Letter Template (optional)")
        if st.session_state.cover_tex:
            st.success(f"✅ Loaded: **{st.session_state.cover_filename}**")
            with st.expander("Change template"):
                _cover_uploader()
        else:
            _cover_uploader()
            st.info("No template — default will be used.", icon="ℹ️")

        st.divider()

        st.subheader("Job Description")
        tab_paste, tab_url, tab_file = st.tabs(["Paste", "URL", "Upload File"])

        with tab_paste:
            pasted = st.text_area("Paste JD", height=180, label_visibility="collapsed",
                                  placeholder="Paste the full job description here...")
            if st.button("Use This Text", key="btn_paste"):
                if pasted.strip():
                    st.session_state.jd_text = parse_jd.from_text(pasted)
                    _reset_analysis()
                    st.success(f"Loaded ({len(st.session_state.jd_text)} chars).")

        with tab_url:
            url = st.text_input("URL", placeholder="https://...", key="jd_url")
            if st.button("Fetch", key="btn_url"):
                if url.strip():
                    with st.spinner("Fetching..."):
                        try:
                            st.session_state.jd_text = parse_jd.from_url(url.strip())
                            _reset_analysis()
                            st.success(f"Fetched ({len(st.session_state.jd_text)} chars).")
                        except ValueError as e:
                            st.error(str(e))

        with tab_file:
            jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"],
                                       label_visibility="collapsed", key="jd_file")
            if jd_file and st.button("Use This File", key="btn_jd_file"):
                try:
                    st.session_state.jd_text = parse_jd.from_file(jd_file.read(), jd_file.name)
                    _reset_analysis()
                    st.success(f"Loaded ({len(st.session_state.jd_text)} chars).")
                except Exception as e:
                    st.error(str(e))

        if st.session_state.jd_text:
            with st.expander("Preview JD"):
                st.text(st.session_state.jd_text[:600] +
                        ("..." if len(st.session_state.jd_text) > 600 else ""))

        st.divider()

        analyze_ready = bool(st.session_state.resume_tex) and bool(st.session_state.jd_text)
        if st.button("🔍 Analyze Job Match", type="primary", use_container_width=True,
                     disabled=not analyze_ready, key="btn_analyze"):
            _reset_analysis()
            with st.spinner("Analyzing..."):
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

        if not analyze_ready:
            parts = []
            if not st.session_state.resume_tex: parts.append("resume")
            if not st.session_state.jd_text:    parts.append("job description")
            st.caption(f"Upload your {' and '.join(parts)} to continue.")

    # ── RIGHT: results ─────────────────────────────────────────────────────────
    with right:
        if st.session_state.scoring_done:
            _show_results(client)


# ── Results panel ──────────────────────────────────────────────────────────────

def _show_results(client):
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

    override = st.text_input(
        "Company name for output folder (edit if wrong)",
        value=company, key="company_override",
    )
    if override.strip() and override != company:
        st.session_state.meta["company_name"] = override
        st.session_state.meta["safe_company"] = re.sub(r"[^\w]", "_", override).strip("_")

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
        st.markdown("**❌ Missing Skills (to inject into CV)**")
        if missing:
            st.markdown(" ".join(
                f'<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;'
                f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block">{s}</span>'
                for s in missing
            ), unsafe_allow_html=True)
        else:
            st.caption("No gaps — great match!")

    st.divider()

    # ── Keyword placement ──────────────────────────────────────────────────────
    sections = st.session_state.resume_sections
    if sections and missing:
        st.markdown(
            """
            <div style="background:#fffbeb;border:2px solid #f59e0b;border-radius:8px;padding:16px 20px;margin-bottom:12px">
            <span style="font-size:1.1em;font-weight:700;color:#92400e">
            ⚠️ Step required before applying — choose where to insert missing keywords
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_labels = []
        for _, name, lvl in sections:
            prefix = "  └ " if lvl == "subsection" else ""
            section_labels.append(f"{prefix}{name}")

        # Smart default: prefer a Skills / Technical section, fall back to index 0
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

        st.markdown("**Keywords to insert** — click × to remove any you don't want:")
        selected_kws = st.multiselect(
            "Keywords",
            options=missing,
            default=missing,
            key="kw_selected",
            label_visibility="collapsed",
        )
        if selected_kws:
            st.success(
                f"✅ {len(selected_kws)} keyword(s) will be inserted into **{sel_name}**."
            )
        else:
            st.warning("No keywords selected — keyword injection will be skipped.")

        st.divider()

    if not st.session_state.package_done:
        st.markdown("### Ready to apply?")
        if st.button("📄 Apply — Generate Tailored CV + Cover Letter",
                     key="btn_apply", type="primary", use_container_width=True):
            _run_package(client)

    if st.session_state.package_done:
        _show_package_results()


# ── Application package ────────────────────────────────────────────────────────

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

    with st.status("⚙️ Generating application package...", expanded=True) as pkg_status:

        # STEP 1 — Inject keywords into CV
        sections       = st.session_state.get("resume_sections", [])
        kw_section_idx = st.session_state.get("kw_section_idx")
        kw_selected    = st.session_state.get("kw_selected", [])
        kw_input_str   = ", ".join(kw_selected) if kw_selected else ""

        if sections and kw_section_idx is not None and kw_input_str:
            # ── Direct insert: user chose section + selected keywords ───────
            section_line_idx, section_name, _ = sections[kw_section_idx]
            st.write(f"**Step 1 / 4** — Inserting {len(kw_selected)} keyword(s) into **{section_name}**...")
            try:
                tailored_tex = tailor_resume.insert_keyword_line(
                    st.session_state.resume_tex, section_line_idx, kw_input_str
                )
                injected, still_missing = tailor_resume.check_keyword_coverage(tailored_tex, kw_selected)
                if injected:
                    st.write(f"✅ Confirmed in CV: {', '.join(injected)}")
                if still_missing:
                    st.write(f"⚠️ Not confirmed after insert: {', '.join(still_missing)}")
                    warnings.append(f"Keywords not confirmed: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword insert error: {e}")
                st.write("⚠️ Insert failed — original CV used.")
        else:
            # ── LLM-based injection fallback ────────────────────────────────
            st.write(f"**Step 1 / 4** — Injecting {len(keywords)} keyword(s) into CV...")
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
                        st.write(f"✅ Confirmed in CV: {', '.join(injected)}")
                    if still_missing:
                        st.write(f"⚠️ Could not place: {', '.join(still_missing)}")
                        warnings.append(f"Keywords not placed: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword injection error: {e}")
                st.write("⚠️ Injection failed — original CV used.")

        # STEP 2 — Generate cover letter
        st.write("**Step 2 / 4** — Writing tailored cover letter...")
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
            mode = "your template" if st.session_state.cover_tex else "default template"
            st.write(f"✅ Cover letter written ({mode}).")
        except Exception as e:
            warnings.append(f"Cover letter failed: {e}")
            st.write(f"❌ Cover letter failed: {e}")
            pkg_status.update(label="❌ Cover letter generation failed.", state="error")
            return

        # STEP 3 — Compile PDFs
        st.write("**Step 3 / 4** — Compiling PDFs...")
        resume_pdf_bytes = compile_pdf.compile_from_string(
            tailored_tex, f"{safe_company}_Resume.tex", ".tmp"
        )
        if resume_pdf_bytes:
            st.write(f"✅ Resume PDF compiled ({len(resume_pdf_bytes) // 1024} KB).")
        else:
            log_text = compile_pdf.get_compile_log(tailored_tex, f"{safe_company}_Resume.tex", ".tmp")
            warnings.append(f"Resume PDF failed.\n\npdflatex log:\n{log_text or 'No log.'}")
            st.write("❌ Resume PDF failed — see warnings below after completion.")

        cover_pdf_bytes = compile_pdf.compile_from_string(
            cover_tex_out, f"{safe_company}_CoverLetter.tex", ".tmp"
        )
        if cover_pdf_bytes:
            st.write(f"✅ Cover letter PDF compiled ({len(cover_pdf_bytes) // 1024} KB).")
        else:
            log_text = compile_pdf.get_compile_log(cover_tex_out, f"{safe_company}_CoverLetter.tex", ".tmp")
            warnings.append(f"Cover letter PDF failed.\n\npdflatex log:\n{log_text or 'No log.'}")
            st.write("❌ Cover letter PDF failed — see warnings below after completion.")

        # STEP 4 — Save files
        st.write("**Step 4 / 4** — Saving files...")
        output_paths = {}
        folder = _output_folder(safe_company)

        (folder / "Resume.tex").write_text(tailored_tex, encoding="utf-8")
        (folder / "CoverLetter.tex").write_text(cover_tex_out, encoding="utf-8")

        if resume_pdf_bytes:
            p = folder / "Resume.pdf"
            p.write_bytes(resume_pdf_bytes)
            output_paths["resume_pdf"]       = str(p)
            output_paths["resume_pdf_bytes"] = resume_pdf_bytes

        if cover_pdf_bytes:
            p = folder / "CoverLetter.pdf"
            p.write_bytes(cover_pdf_bytes)
            output_paths["cover_pdf"]       = str(p)
            output_paths["cover_pdf_bytes"] = cover_pdf_bytes

        output_paths["folder"] = str(folder)
        pdf_count = (1 if resume_pdf_bytes else 0) + (1 if cover_pdf_bytes else 0)
        st.write(f"✅ Saved {pdf_count}/2 PDF(s) + .tex sources to: {folder}")

        if pdf_count == 2:
            pkg_status.update(label="✅ Done! Application package ready.", state="complete", expanded=False)
        elif pdf_count == 1:
            pkg_status.update(label="⚠️ Partial — 1/2 PDFs compiled.", state="complete", expanded=True)
        else:
            pkg_status.update(label="❌ PDFs failed — .tex files saved.", state="error", expanded=True)

    st.session_state.output_paths    = output_paths
    st.session_state.output_warnings = warnings
    st.session_state.cost_summary    = cost_tracker.get_summary()
    st.session_state.package_done    = True
    st.rerun()


# ── Download results panel ─────────────────────────────────────────────────────

def _show_package_results():
    paths    = st.session_state.output_paths
    warnings = st.session_state.output_warnings
    folder   = paths.get("folder", "")

    if folder:
        st.success(f"✅ **Saved to:** `{folder}`")

    col_r, col_c = st.columns(2)
    with col_r:
        if "resume_pdf_bytes" in paths:
            st.download_button(
                "⬇️ Download CV (PDF)",
                data=paths["resume_pdf_bytes"],
                file_name="Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.error("CV PDF not generated — .tex source saved.")

    with col_c:
        if "cover_pdf_bytes" in paths:
            st.download_button(
                "⬇️ Download Cover Letter (PDF)",
                data=paths["cover_pdf_bytes"],
                file_name="CoverLetter.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.error("Cover letter PDF not generated — .tex source saved.")

    if warnings:
        with st.expander(f"⚠️ {len(warnings)} warning(s)"):
            for w in warnings:
                st.warning(w)

    # ── Cost summary ───────────────────────────────────────────────────────────
    cs = st.session_state.get("cost_summary")
    if cs and cs["total_tokens"] > 0:
        with st.expander(f"💰 API cost this run — ${cs['total_cost_usd']:.4f}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total tokens", f"{cs['total_tokens']:,}")
            c2.metric("Prompt tokens", f"{cs['total_prompt']:,}")
            c3.metric("Output tokens", f"{cs['total_completion']:,}")
            st.markdown("**Breakdown by step:**")
            rows = ""
            for call in cs["calls"]:
                rows += (
                    f"| {call.label} | {call.model} | "
                    f"{call.prompt_tokens:,} | {call.completion_tokens:,} | "
                    f"${call.total_cost:.4f} |\n"
                )
            st.markdown(
                "| Step | Model | Prompt tokens | Output tokens | Cost |\n"
                "|---|---|---|---|---|\n" + rows
            )
            st.caption("Prices: gpt-4o $2.50/M input · $10.00/M output")

    st.divider()
    st.markdown("#### Analyze another job?")
    st.caption("Your CV and cover letter template stay loaded — just paste a new JD.")
    if st.button("🔄 Analyze New Job", key="btn_new_job", type="primary", use_container_width=True):
        st.session_state.jd_text         = ""
        st.session_state.scoring_done    = False
        st.session_state.package_done    = False
        st.session_state.meta            = {}
        st.session_state.visa_result     = {}
        st.session_state.score_result    = {}
        st.session_state.output_paths    = {}
        st.session_state.output_warnings = []
        # Clear keyword selections so multiselect refreshes for the next job
        for k in ("kw_section_idx", "kw_selected"):
            st.session_state.pop(k, None)
        st.rerun()


if __name__ == "__main__":
    main()
