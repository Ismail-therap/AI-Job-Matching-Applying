"""
App 2 — CV Updater & Cover Letter Generator

Usage:
    streamlit run app2_apply.py

What it does:
  1. Loads the analysis saved by App 1 (.tmp/last_analysis.json)
  2. Upload your resume (.tex) and optional cover letter template (.tex)
  3. Shows the ATS keywords that will be injected
  4. Click Apply → injects keywords, writes cover letter, compiles both PDFs
  5. Download PDFs from the browser or find them in outputs/YYYY-MM-DD/{Company}/
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
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

import tools.compile_pdf as compile_pdf
import tools.generate_cover as generate_cover
import tools.parse_resume as parse_resume
import tools.tailor_resume as tailor_resume

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="App 2 — CV Updater & Cover Letter",
    page_icon="📄",
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


def _init_state():
    defaults = {
        "resume_tex":      "",
        "resume_plain":    "",
        "resume_filename": "",
        "cover_tex":       None,
        "cover_filename":  "",
        "analysis":        None,
        "package_done":    False,
        "output_paths":    {},
        "output_warnings": [],
        "resume_sections": [],   # list of (line_idx, name, level)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _output_folder(safe_company: str) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = Path("outputs") / date_str / safe_company
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    client = get_openai_client()

    st.title("📄 App 2 — CV Updater & Cover Letter Generator")
    st.caption("Injects ATS keywords from your App 1 analysis and generates a tailored cover letter.")

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: inputs ──────────────────────────────────────────────────────────
    with left:

        # ── Load analysis from App 1 ──────────────────────────────────────────
        st.subheader("Analysis from App 1")
        analysis = _load_analysis()

        if analysis:
            meta    = analysis.get("meta", {})
            company = meta.get("company_name", "Unknown")
            role    = meta.get("role_title", "Unknown")
            missing = analysis.get("missing_skills", [])
            st.success(f"✅ Loaded: **{role}** at **{company}**")
            if missing:
                st.markdown("**ATS keywords to inject:**")
                st.markdown(" ".join(
                    f'<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;'
                    f'border-radius:12px;font-size:0.82em;margin:2px;display:inline-block">{s}</span>'
                    for s in missing
                ), unsafe_allow_html=True)
            else:
                st.info("No missing keywords found — resume already matches well.")
        else:
            st.warning(
                "No analysis found. Run **App 1** first to generate the analysis, "
                "or the job description and keywords won't be available."
            )
            st.caption("```\nstreamlit run app1_analyze.py\n```")

        st.divider()

        # ── Resume upload ─────────────────────────────────────────────────────
        st.subheader("Resume (.tex)")
        if st.session_state.resume_tex:
            st.success(f"✅ {st.session_state.resume_filename}")
            with st.expander("Change resume"):
                _resume_uploader()
        else:
            _resume_uploader()

        st.divider()

        # ── Keyword placement selector ────────────────────────────────────────
        st.subheader("📌 Keyword Placement")
        sections = st.session_state.resume_sections
        if sections and analysis:
            missing_kws = analysis.get("missing_skills", [])

            # Build display labels with indentation for subsections
            section_labels = []
            for _, name, lvl in sections:
                prefix = "  └ " if lvl == "subsection" else ""
                section_labels.append(f"{prefix}{name}")

            sel_idx = st.selectbox(
                "Insert keywords into section:",
                range(len(sections)),
                format_func=lambda i: section_labels[i],
                key="kw_section_idx",
            )
            _, sel_name, _ = sections[sel_idx]

            st.markdown("**Keywords to insert** — click × to remove any you don't want:")
            selected_kws = st.multiselect(
                "Keywords",
                options=missing_kws,
                default=missing_kws,
                key="kw_selected",
                label_visibility="collapsed",
            )
            if selected_kws:
                st.caption(
                    f"Will insert {len(selected_kws)} keyword(s) as **Applied Research Terms** "
                    f"subsection in **{sel_name}**."
                )
            else:
                st.warning("No keywords selected — Step 1 will be skipped.")
        elif not sections and st.session_state.resume_tex:
            st.info("No \\\\section commands found in the uploaded resume.")
        else:
            st.caption("Upload resume and run App 1 analysis to enable keyword placement.")

        st.divider()

        # ── Cover letter template (optional) ──────────────────────────────────
        st.subheader("Cover Letter Template (.tex) — Optional")
        if st.session_state.cover_tex:
            st.success(f"✅ {st.session_state.cover_filename}")
            with st.expander("Change template"):
                _cover_uploader()
        else:
            _cover_uploader()
            st.info("No template uploaded — a default template will be used.", icon="ℹ️")

        st.divider()

        # ── Editable company name ─────────────────────────────────────────────
        if analysis:
            meta     = analysis.get("meta", {})
            company  = meta.get("company_name", "Unknown")
            override = st.text_input(
                "Company name for output folder (edit if wrong)",
                value=company, key="company_override",
            )
            if override.strip() and override != company:
                analysis["meta"]["company_name"] = override
                analysis["meta"]["safe_company"]  = re.sub(r"[^\w]", "_", override).strip("_")
                st.session_state.analysis = analysis

        # ── Apply button ──────────────────────────────────────────────────────
        apply_ready = bool(st.session_state.resume_tex)
        if st.button(
            "⚡ Apply — Update CV + Generate Cover Letter",
            type="primary", use_container_width=True,
            disabled=not apply_ready, key="btn_apply",
        ):
            st.session_state.package_done    = False
            st.session_state.output_paths    = {}
            st.session_state.output_warnings = []
            _run_package(client, analysis)

        if not apply_ready:
            st.caption("Upload your resume (.tex) to continue.")

    # ── RIGHT: results ─────────────────────────────────────────────────────────
    with right:
        if st.session_state.package_done:
            _show_results()


# ── Uploaders ─────────────────────────────────────────────────────────────────

def _resume_uploader():
    f = st.file_uploader("Upload .tex resume", type=["tex"],
                         label_visibility="collapsed", key="resume_upload")
    if f:
        tex, plain = parse_resume.parse(f.read())
        st.session_state.resume_tex      = tex
        st.session_state.resume_plain    = plain
        st.session_state.resume_filename = f.name
        st.session_state.package_done    = False
        st.session_state.resume_sections = tailor_resume.extract_sections_and_subsections(tex)
        st.success(f"Loaded: {f.name}")


def _cover_uploader():
    f = st.file_uploader(
        "Upload .tex cover letter template", type=["tex"],
        label_visibility="collapsed", key="cover_upload",
        help="Optional. The body will be fully rewritten for this specific job.",
    )
    if f:
        st.session_state.cover_tex      = f.read().decode("utf-8", errors="replace")
        st.session_state.cover_filename = f.name
        st.success(f"Loaded: {f.name}")


# ── Load analysis from App 1 ──────────────────────────────────────────────────

def _load_analysis() -> dict | None:
    if st.session_state.analysis:
        return st.session_state.analysis
    if ANALYSIS_FILE.exists():
        try:
            data = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
            st.session_state.analysis = data
            return data
        except Exception:
            return None
    return None


# ── Generation pipeline ───────────────────────────────────────────────────────

def _run_package(client: openai.OpenAI, analysis: dict | None):
    meta = (analysis or {}).get("meta", {})
    safe_company = meta.get("safe_company") or re.sub(
        r"[^\w]", "_", meta.get("company_name", "Unknown")
    ).strip("_") or "Unknown"

    jd_text  = (analysis or {}).get("jd_text", "")
    keywords = (analysis or {}).get("missing_skills", [])

    warnings: list[str] = []
    tailored_tex  = st.session_state.resume_tex
    cover_tex_out = ""
    resume_pdf_bytes: bytes | None = None
    cover_pdf_bytes:  bytes | None = None

    with st.status("⚙️ Generating application package...", expanded=True) as pkg_status:

        # STEP 1 — Inject ATS keywords
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
                    st.write(f"✅ Confirmed in tex: {', '.join(injected)}")
                if still_missing:
                    st.write(f"⚠️ Not confirmed after insert: {', '.join(still_missing)}")
                    warnings.append(f"Keywords not confirmed in tex: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword insert error — original used: {e}")
                st.write(f"⚠️ Insert failed — original resume used.")
        else:
            # ── LLM-based injection (original behaviour) ────────────────────
            st.write(f"**Step 1 / 4** — Injecting {len(keywords)} ATS keyword(s) into resume bullets...")
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
                    st.write("✅ Keywords injected into resume bullets.")
                if keywords:
                    injected, still_missing = tailor_resume.check_keyword_coverage(tailored_tex, keywords)
                    if injected:
                        st.write(f"✅ Keywords confirmed in CV: {', '.join(injected)}")
                    if still_missing:
                        st.write(f"⚠️ Could not place (no suitable bullet): {', '.join(still_missing)}")
                        warnings.append(f"Keywords not injected: {', '.join(still_missing)}")
            except Exception as e:
                warnings.append(f"Keyword injection error — original used: {e}")
                st.write(f"⚠️ Keyword injection failed — original resume used.")

        # STEP 2 — Generate cover letter
        st.write("**Step 2 / 4** — Writing tailored cover letter (max 1 page)...")
        try:
            cover_tex_out, cover_warn = generate_cover.generate(
                cover_tex=st.session_state.cover_tex,
                jd_text=jd_text,
                meta=meta,
                resume_plain=st.session_state.resume_plain,
                client=client,
                default_template_path="default_cover_letter.tex",
            )
            warnings.extend(cover_warn)
            mode = "your template" if st.session_state.cover_tex else "default template"
            st.write(f"✅ Cover letter written ({mode}).")
        except Exception as e:
            warnings.append(f"Cover letter failed: {e}")
            st.write(f"❌ Cover letter generation failed: {e}")
            pkg_status.update(label="❌ Cover letter generation failed.", state="error")
            return

        # STEP 3 — Compile PDFs
        st.write("**Step 3 / 4** — Compiling PDFs with local MiKTeX...")

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
        output_paths: dict = {}
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
        st.write(f"✅ Saved {pdf_count}/2 PDF(s) + .tex sources → `{folder}`")

        if pdf_count == 2:
            pkg_status.update(label="✅ Done! Application package ready.", state="complete", expanded=False)
        elif pdf_count == 1:
            pkg_status.update(label="⚠️ Partial — 1/2 PDFs generated.", state="complete", expanded=True)
        else:
            pkg_status.update(label="❌ PDFs failed — .tex files saved.", state="error", expanded=True)

    st.session_state.output_paths    = output_paths
    st.session_state.output_warnings = warnings
    st.session_state.package_done    = True
    st.rerun()


# ── Results panel ─────────────────────────────────────────────────────────────

def _show_results():
    paths    = st.session_state.output_paths
    warnings = st.session_state.output_warnings
    folder   = paths.get("folder", "")

    if folder:
        st.success(f"✅ **Saved to:** `{folder}`")

    col_r, col_c = st.columns(2)
    with col_r:
        if "resume_pdf_bytes" in paths:
            st.download_button(
                "⬇️ Download Resume (PDF)",
                data=paths["resume_pdf_bytes"],
                file_name="Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.error("Resume PDF not generated — .tex source saved.")

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

    st.divider()
    st.markdown("#### Apply for another job?")
    st.caption("Go back to App 1 to analyze a new job description.")
    st.code("streamlit run app1_analyze.py")


if __name__ == "__main__":
    main()
