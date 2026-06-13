import os
import re
import tempfile
from datetime import datetime
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from src.evaluator import (
    compute_mrr,
    compute_ndcg,
    compute_precision_recall_f1,
    compute_processing_stats,
)
from src.matcher import generate_explanation, rank_candidates
from src.parser import ResumeParser
from src.preprocessor import TextPreprocessor

st.set_page_config(
    page_title="AI Resume Screening & Skill Matching System",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Core Layout Styles */
    .stApp {
        background: linear-gradient(135deg, #051F20 0%, #0B2B26 50%, #163832 100%);
        background-attachment: fixed;
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11,43,38,0.9) 0%, rgba(5,31,32,0.9) 100%);
        border-right: 1px solid rgba(142,182,155,0.2);
        backdrop-filter: blur(15px);
    }
    [data-testid="stSidebar"] * {
        color: #DAF1DE !important;
    }

    /* Custom Section Headers */
    .section-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #8EB69B, #DAF1DE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }

    /* Section Padding helper */
    .custom-section {
        padding: 80px 0 40px 0;
    }

    /* Section 1 — Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 100px 20px;
        min-height: 80vh;
    }
    .hero-title-large {
        font-size: 4.5rem !important;
        font-weight: 800 !important;
        color: #DAF1DE !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.03em !important;
    }
    .hero-subtitle-large {
        font-size: 1.5rem;
        color: #8EB69B;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .hero-description {
        font-size: 1.2rem;
        color: #b8d4c8;
        max-width: 750px;
        margin-bottom: 2.5rem;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    .explore-btn {
        display: inline-block;
        background: linear-gradient(135deg, #235347, #8EB69B);
        color: white !important;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.85rem 2.5rem;
        border: 1px solid rgba(142,182,155,0.5);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(35,83,71,0.3);
        transition: all 0.3s ease;
        text-decoration: none !important;
        position: relative;
        z-index: 1;
    }
    .explore-btn:hover {
        background: linear-gradient(135deg, #8EB69B, #235347);
        box-shadow: 0 0 20px rgba(142,182,155,0.4);
        transform: translateY(-3px);
    }

    /* Section 2 — About Page Section */
    .about-text {
        font-size: 1.15rem;
        color: #b8d4c8;
        text-align: center;
        max-width: 900px;
        margin: 0 auto 3rem auto;
        line-height: 1.75;
    }
    .value-card {
        background: rgba(35,83,71,0.3);
        border: 1px solid rgba(218,241,222,0.1);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        padding: 2.2rem 1.8rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 240px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .value-card:hover {
        border-color: rgba(142,182,155,0.5);
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(142,182,155,0.15);
    }
    .value-card-icon {
        font-size: 2.8rem;
        margin-bottom: 0.75rem;
    }
    .value-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #DAF1DE;
        margin-bottom: 0.75rem;
    }
    .value-card-desc {
        font-size: 0.95rem;
        color: #b8d4c8;
        line-height: 1.5;
    }

    /* Section 3 — Key Features Section */
    .feature-card {
        background: rgba(35,83,71,0.25);
        border: 1px solid rgba(218,241,222,0.08);
        border-radius: 12px;
        backdrop-filter: blur(12px);
        padding: 1.6rem 1.4rem;
        transition: all 0.3s ease;
        height: 180px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.2);
    }
    .feature-card:hover {
        border-color: rgba(142,182,155,0.5);
        transform: translateY(-3px);
        box-shadow: 0 0 20px rgba(142,182,155,0.2);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #DAF1DE;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #8EB69B;
        line-height: 1.4;
    }

    /* Existing app helpers */
    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8EB69B;
        margin-bottom: 0.5rem;
    }
    .badge-shortlisted {
        display: inline-block;
        background: rgba(35,83,71,0.5);
        color: #DAF1DE;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(142,182,155,0.5);
        backdrop-filter: blur(8px);
        box-shadow: 0 0 12px rgba(35,83,71,0.2);
    }
    .badge-unsuitable {
        display: inline-block;
        background: rgba(107,44,44,0.5);
        color: #ffb3b3;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(255,150,150,0.5);
        backdrop-filter: blur(8px);
        box-shadow: 0 0 12px rgba(107,44,44,0.2);
    }
    div[data-testid="stMetric"] {
        background: rgba(35,83,71,0.3);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(218,241,222,0.1);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #b8d4c8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #DAF1DE !important;
        font-weight: 700 !important;
    }

    /* Expander Cards */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(218,241,222,0.08);
        border-radius: 16px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.15);
        overflow: hidden;
    }
    [data-testid="stExpander"] > div:first-child {
        background: rgba(35,83,71,0.15);
    }
    [data-testid="stExpanderDetails"] {
        padding: 1rem 1.5rem 1.5rem;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #235347, #8EB69B);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border: 1px solid rgba(142,182,155,0.5);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(35,83,71,0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #8EB69B, #235347);
        box-shadow: 0 0 20px rgba(142,182,155,0.4);
        transform: translateY(-2px);
    }
    div.stDownloadButton > button:first-child {
        background: linear-gradient(135deg, #235347, #8EB69B);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: 1px solid rgba(142,182,155,0.5);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(35,83,71,0.3);
        transition: all 0.3s ease;
    }
    div.stDownloadButton > button:first-child:hover {
        background: linear-gradient(135deg, #8EB69B, #235347);
        box-shadow: 0 0 20px rgba(142,182,155,0.4);
        transform: translateY(-2px);
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(142,182,155,0.1);
        box-shadow: 0 4px 32px rgba(0,0,0,0.15);
    }
    [data-testid="stDataFrame"] th {
        background: rgba(35,83,71,0.3);
        backdrop-filter: blur(8px);
        color: #DAF1DE !important;
    }
    /* Results Section Headers */
    .results-header {
        background: linear-gradient(90deg, #8EB69B, #DAF1DE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _status_badge(status: str) -> str:
    if status == "Shortlisted":
        return "🟢 Shortlisted"
    return "🔴 Unsuitable"


def _extract_standardized_terms(text: str, preprocessor: TextPreprocessor) -> Set[str]:
    """Extract normalized technical terms using the spaCy synonym dictionary."""
    if not text or not text.strip():
        return set()

    cleaned = preprocessor.clean_text(text)
    normalized = preprocessor.preprocess(text)
    terms: Set[str] = set()

    for abbrev, canonical in preprocessor.synonym_dict.items():
        abbrev_pattern = r"\b" + re.escape(abbrev) + r"\b"
        if re.search(abbrev_pattern, cleaned) or canonical in normalized:
            terms.add(canonical)

    return terms


def _compute_skill_gaps(
    job_text: str, resume_text: str, preprocessor: TextPreprocessor
) -> List[str]:
    """Return JD technical terms absent from the candidate resume."""
    jd_terms = _extract_standardized_terms(job_text, preprocessor)
    resume_terms = _extract_standardized_terms(resume_text, preprocessor)
    return sorted(jd_terms - resume_terms)


def _format_skill_label(term: str) -> str:
    return term.title() if len(term.split()) <= 4 else term


def main():
    # Section 1 - Hero
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title-large">TalentMatch AI</h1>
            <p class="hero-subtitle-large">INTELLIGENT RESUME SCREENING POWERED BY NLP</p>
            <p class="hero-description">Upload resumes, enter a job description, and let AI rank the best candidates instantly.</p>
            <a href="#about" class="explore-btn">Explore &rarr;</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2 - About
    st.markdown(
        """
        <div id="about" class="custom-section" style="scroll-margin-top: 80px;">
            <h2 class="section-title">What is TalentMatch AI?</h2>
            <p class="about-text">
                TalentMatch AI is an enterprise-grade resume screening and candidate ranking system powered by advanced natural language processing.
                By leveraging state-of-the-art sentence embeddings and semantic alignment, it goes beyond keyword matching to find the best talent.
                Our system evaluates professional skills, calculates semantic compatibility, and provides detailed explainability for hiring decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown(
            """
            <div class="value-card">
                <div class="value-card-icon">⚡</div>
                <div class="value-card-title">Fast Screening</div>
                <div class="value-card-desc">Processes multiple resumes in seconds using optimized batch embedding generation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="value-card">
                <div class="value-card-icon">🎯</div>
                <div class="value-card-title">Accurate Matching</div>
                <div class="value-card-desc">Measures semantic overlap using SentenceTransformers and spaCy NLP tokenization.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="value-card">
                <div class="value-card-icon">📊</div>
                <div class="value-card-title">Data-Driven Insights</div>
                <div class="value-card-desc">Generates PDF/CSV reports, interactive skill gap analysis, and ranking leaderboard.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 3 - Features
    st.markdown(
        """
        <div id="features" class="custom-section" style="scroll-margin-top: 80px; margin-top: 60px;">
            <h2 class="section-title">Key Features</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feat_col1, feat_col2, feat_col3 = st.columns(3, gap="medium")
    with feat_col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Semantic Skill Matching</div>
                <div class="feature-desc">Goes beyond simple keywords to match candidate resumes by semantic skill similarity.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Candidate Ranking</div>
                <div class="feature-desc">Ranks all candidates automatically using normalized cosine similarity scores.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Multi-format Resume Parsing</div>
                <div class="feature-desc">Supports parsing and analysis from PDF, DOCX, and plain TXT files.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📥</div>
                <div class="feature-title">Export Reports</div>
                <div class="feature-desc">Exports rank leaderboards and detailed resumes comparison reports to CSV and PDF.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI Explainability</div>
                <div class="feature-desc">Explains matching scores in plain, transparent language to justify hiring choices.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🕳️</div>
                <div class="feature-title">Skill Gap Analysis</div>
                <div class="feature-desc">Uncovers key standardized technical terms missing from the candidate's profile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 4 - Start Screening
    st.markdown(
        """
        <div id="tool" class="custom-section" style="scroll-margin-top: 80px; margin-top: 60px;">
            <h2 class="section-title">🚀 Start Screening</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ System Configuration")
        threshold_percent = st.slider(
            "Minimum Match Score Threshold (%)",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            help="Candidates with similarity score above this threshold will be shortlisted",
        )
        threshold = threshold_percent / 100.0
        st.markdown(f"**Current Threshold:** {threshold_percent}%")
        st.divider()
        st.caption("Powered by `all-MiniLM-L6-v2` · Cosine Similarity · spaCy NLP")

    col_jd, col_upload = st.columns(2, gap="large")

    with col_jd:
        st.markdown(
            '<p class="section-label">Job Description</p>', unsafe_allow_html=True
        )

        job_role = st.text_input(
            "🎯 Job Role / Title",
            value=st.session_state.get("job_role", ""),
            placeholder="e.g., Data Analyst, Software Engineer, DevOps Engineer...",
            help="Enter the job role or title for this position",
        )
        st.session_state["job_role"] = job_role

        job_description = st.text_area(
            "Job Description",
            value=st.session_state.get("job_description", ""),
            height=300,
            placeholder="Paste or type your job description here...",
            help="Enter the complete job description for candidate screening",
            label_visibility="collapsed",
        )
        st.session_state["job_description"] = job_description

    with col_upload:
        st.markdown(
            '<p class="section-label">Resume Upload</p>', unsafe_allow_html=True
        )
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Select one or more resume files to evaluate against the job description",
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) ready for screening")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            font-size: 1.2rem;
            font-weight: 700;
            padding: 0.75rem 2rem;
            border: 2px solid #34d399;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
            box-shadow: 0 12px 32px rgba(16, 185, 129, 0.6);
            transform: translateY(-2px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_button = st.button(
            "⚡ Run AI Screening Analytics",
            type="primary",
            use_container_width=True,
            help="Process all uploaded resumes against the job description",
        )

    if run_button:
        if not job_description.strip():
            st.error("❌ Please provide a job description before running the analysis.")
            return

        if not uploaded_files:
            st.error(
                "❌ Please upload at least one resume file before running the analysis."
            )
            return

        # Progress bar for processing stages
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            temp_dir = tempfile.mkdtemp()
            resume_paths = []

            status_text.text("📂 Loading resumes...")
            progress_bar.progress(10)

            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                resume_paths.append(temp_file_path)

            status_text.text("🔍 Preprocessing text...")
            progress_bar.progress(30)

            parser = ResumeParser()
            preprocessor = TextPreprocessor()
            resume_texts = {
                path: parser.parse_resume(path) or "" for path in resume_paths
            }

            status_text.text("🧠 Generating embeddings...")
            progress_bar.progress(60)

            results = rank_candidates(
                resume_paths=resume_paths,
                job_description_text=job_description,
                threshold=threshold,
            )

            status_text.text("📊 Computing similarity scores...")
            progress_bar.progress(85)

            for file_path in resume_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)

            status_text.text("✅ Screening complete!")
            progress_bar.progress(100)

            # Clear progress bar after short delay
            import time

            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            st.success("✅ Analysis completed successfully!")
            st.session_state["results"] = results
            st.session_state["screening_done"] = True
            st.session_state["job_role"] = job_role
            st.session_state["job_description"] = job_description
            st.session_state["threshold"] = threshold
            st.session_state["threshold_percent"] = threshold_percent

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")

    # Now display results from session state if available
    if st.session_state.get('screening_done') and st.session_state.get('results'):
        results = st.session_state["results"]
        job_role = st.session_state["job_role"]
        job_description = st.session_state["job_description"]
        threshold = st.session_state["threshold"]
        threshold_percent = st.session_state["threshold_percent"]

        st.divider()

        total_resumes = len(results)
        shortlisted_count = sum(1 for r in results if r["status"] == "Shortlisted")
        unsuitable_count = total_resumes - shortlisted_count
        avg_match_score = (
            np.mean([r["similarity_score"] * 100 for r in results])
            if results
            else 0
        )
        processing_times = [r["processing_time"] for r in results]
        avg_inference_s = (
            float(np.mean(processing_times)) if processing_times else 0.0
        )
        inference_speed_label = (
            f"{avg_inference_s:.2f}s / resume"
            if avg_inference_s >= 1.0
            else f"{1 / avg_inference_s:.1f} resumes/s"
            if avg_inference_s > 0
            else "—"
        )

        st.markdown('<h3 class="results-header">📊 Screening Analytics</h3>', unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Resumes", total_resumes)
        with metric_col2:
            st.metric("Suitable", shortlisted_count)
        with metric_col3:
            st.metric("Unsuitable", unsuitable_count)
        with metric_col4:
            st.metric("Avg Match Score", f"{avg_match_score:.1f}%")

        st.divider()

        st.markdown('<h3 class="results-header">🏆 Candidate Ranking Leaderboard</h3>', unsafe_allow_html=True)

        df_data = []
        for result in results:
            score_pct = result["similarity_score"] * 100
            df_data.append(
                {
                    "Candidate": os.path.basename(result["resume_path"]),
                    "Match Score (%)": round(score_pct, 2),
                    "Status": _status_badge(result["status"]),
                    "Processing Time (s)": round(result["processing_time"], 2),
                }
            )

        df = pd.DataFrame(df_data)
        df = df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        # Reorder columns to have Rank first
        columns_order = ["Rank", "Candidate", "Match Score (%)", "Status", "Processing Time (s)"]
        df = df[columns_order]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Candidate": st.column_config.TextColumn(
                    "Candidate", width="large"
                ),
                "Match Score (%)": st.column_config.ProgressColumn(
                    "Match Score (%)",
                    help="Cosine similarity with job description",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Processing Time (s)": st.column_config.NumberColumn(
                    "Processing Time (s)", width="small", format="%.2f"
                ),
            },
        )

        shortlist_df = df[df["Status"].str.contains("Shortlisted")].copy()
        if not shortlist_df.empty:
            st.download_button(
                label="⬇️ Download Shortlist Report (CSV)",
                data=shortlist_df.to_csv(index=False),
                file_name="shortlisted_candidates_report.csv",
                mime="text/csv",
                use_container_width=True,
                help="Export ranked shortlisted candidates only",
            )
        else:
            st.caption(
                "No shortlisted candidates matched the current threshold — "
                "adjust the slider to generate a shortlist report."
            )

        st.divider()

        st.markdown('<h3 class="results-header">🔍 Candidate Deep-Dive Analysis</h3>', unsafe_allow_html=True)

        for idx, result in enumerate(results, 1):
            candidate_name = os.path.basename(result["resume_path"])
            score_pct = result["similarity_score"] * 100
            badge = _status_badge(result["status"])

            # Color-coded label based on score and status
            if result["status"] == "Unsuitable":
                score_label = "🔴 Low"
            elif score_pct >= 75:
                score_label = "🟢 Excellent"
            elif score_pct >= 50:
                score_label = "🟡 Moderate"
            else:
                score_label = "🔴 Low"

            with st.expander(
                f"#{idx} · {candidate_name}  ·  {score_pct:.1f}% match ({score_label})  ·  {badge}",
                expanded=(idx == 1),
            ):
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                with detail_col1:
                    st.metric("Match Score", f"{score_pct:.2f}%")
                with detail_col2:
                    st.metric("Status", result["status"])
                with detail_col3:
                    st.metric(
                        "Processing Time", f"{result['processing_time']:.2f}s"
                    )

                st.markdown("---")

                # Matched Skills Section
                matched_skills = result.get("matched_skills", [])
                st.markdown('<h4 class="results-header">✅ Matched Skills</h4>', unsafe_allow_html=True)
                if matched_skills:
                    skill_badges = [
                        f'<span class="badge-shortlisted">{skill}</span>'
                        for skill in matched_skills
                    ]
                    st.markdown(" ".join(skill_badges), unsafe_allow_html=True)
                else:
                    st.caption(
                        "No matched skills detected from the predefined skill list."
                    )

                # Missing Skills Section
                missing_skills = result.get("missing_skills", [])
                st.markdown('<h4 class="results-header">❌ Missing Skills</h4>', unsafe_allow_html=True)
                if missing_skills:
                    skill_badges = [
                        f'<span class="badge-unsuitable">{skill}</span>'
                        for skill in missing_skills
                    ]
                    st.markdown(" ".join(skill_badges), unsafe_allow_html=True)
                else:
                    st.caption(
                        "All required skills from the job description are present in this resume."
                    )

                st.markdown("---")

                # Explainability Module
                st.markdown('<h4 class="results-header">🧠 Why this score?</h4>', unsafe_allow_html=True)
                explanation = generate_explanation(
                    candidate_name,
                    result["similarity_score"],
                    result.get("matched_skills", []),
                    result.get("missing_skills", []),
                    threshold,
                    result.get("matched_skills_ratio", 1.0),
                    result.get("total_jd_skills", 0),
                )
                st.info(explanation)

                st.markdown("---")
                st.markdown('<h4 class="results-header">📋 Model Decision Interpretability</h4>', unsafe_allow_html=True)

                if result["status"] == "Shortlisted":
                    st.success(
                        f"**{candidate_name}** exceeded the {threshold_percent}% "
                        f"similarity threshold with a score of **{score_pct:.2f}%**. "
                        f"Semantic embedding alignment indicates strong skill-profile fit."
                    )
                else:
                    st.warning(
                        f"**{candidate_name}** scored **{score_pct:.2f}%**, "
                        f"below the {threshold_percent}% threshold. "
                        f"Profile semantic overlap with the job description is insufficient."
                    )

                st.info(
                    f"Processed in **{result['processing_time']:.2f}s** via "
                    f"SentenceTransformer (`all-MiniLM-L6-v2`) cosine similarity pipeline."
                )

                if result.get("error"):
                    st.error(f"Processing note: {result['error']}")

        # Candidate Comparison Overview (only when 2+ resumes)
        if len(results) >= 2:
            st.divider()
            st.markdown('<h3 class="results-header">📊 Candidate Comparison Overview</h3>', unsafe_allow_html=True)

            comparison_data = []
            for result in results:
                score_pct = result["similarity_score"] * 100
                matched_count = len(result.get("matched_skills", []))
                missing_count = len(result.get("missing_skills", []))

                comparison_data.append(
                    {
                        "Candidate": os.path.basename(result["resume_path"]),
                        "Match Score (%)": score_pct,
                        "Matched Skills Count": matched_count,
                        "Missing Skills Count": missing_count,
                        "Status": result["status"],
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
            comparison_df["Rank"] = range(1, len(comparison_df) + 1)
            # Reorder columns
            columns_order = ["Rank", "Candidate", "Match Score (%)", "Matched Skills Count", "Missing Skills Count", "Status"]
            comparison_df = comparison_df[columns_order]

            # Apply color coding to Status column
            def color_status(val):
                color = "#22c55e" if val == "Shortlisted" else "#ef4444"
                return f"color: {color}; font-weight: bold"

            # Apply color coding to Match Score column
            def color_score(val):
                color = "#22c55e" if val >= threshold_percent else "#ef4444"
                return f"color: {color}; font-weight: bold"

            styled_df = comparison_df.style.applymap(
                color_status, subset=["Status"]
            )
            styled_df = styled_df.applymap(color_score, subset=["Match Score (%)"])

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Candidate": st.column_config.TextColumn(
                        "Candidate", width="large"
                    ),
                    "Match Score (%)": st.column_config.NumberColumn(
                        "Match Score (%)", format="%.2f%%", width="medium"
                    ),
                    "Matched Skills Count": st.column_config.NumberColumn(
                        "Matched Skills Count", width="medium"
                    ),
                    "Missing Skills Count": st.column_config.NumberColumn(
                        "Missing Skills Count", width="medium"
                    ),
                    "Status": st.column_config.TextColumn("Status", width="medium"),
                },
            )

        # Evaluation Metrics Section
        st.divider()
        st.markdown('<h3 class="results-header">📈 Evaluation Metrics</h3>', unsafe_allow_html=True)

        precision, recall, f1 = compute_precision_recall_f1(results, threshold)
        mrr = compute_mrr(results)
        ndcg = compute_ndcg(results)
        avg_processing_time = compute_processing_stats(results)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Precision", f"{precision:.2f}")
            st.metric("F1-Score", f"{f1:.2f}")
            st.metric("NDCG", f"{ndcg:.2f}")
        with metric_col2:
            st.metric("Recall", f"{recall:.2f}")
            st.metric("MRR", f"{mrr:.2f}")
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s / resume")

        st.caption("Metrics are computed based on skill matching and semantic similarity thresholds.")

        st.divider()
        st.markdown('<h3 class="results-header">📥 Export Results</h3>', unsafe_allow_html=True)

        # CSV Export with Matched Skills and Missing Skills
        export_data = []
        for r in results:
            export_data.append(
                {
                    "Candidate": os.path.basename(r["resume_path"]),
                    "Match Score (%)": f"{r['similarity_score'] * 100:.2f}",
                    "Matched Skills": ", ".join(r.get("matched_skills", [])),
                    "Missing Skills": ", ".join(r.get("missing_skills", [])),
                    "Status": r["status"],
                    "Processing Time (s)": f"{r['processing_time']:.2f}",
                }
            )
        export_df = pd.DataFrame(export_data)
        # Convert Match Score to float for sorting
        export_df["Match Score (%)"] = export_df["Match Score (%)"].astype(float)
        export_df = export_df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
        export_df["Rank"] = range(1, len(export_df) + 1)
        # Reorder columns
        columns_order = ["Rank", "Candidate", "Match Score (%)", "Matched Skills", "Missing Skills", "Status", "Processing Time (s)"]
        export_df = export_df[columns_order]
        csv_data = export_df.to_csv(index=False)

        # PDF Export
        def generate_pdf_report(results, job_role, job_description):
            def clean(t):
                if not isinstance(t, str):
                    return t
                t = t.replace("—", "-")
                t = t.replace("“", '"').replace("”", '"')
                t = t.replace("‘", "'").replace("’", "'")
                return t

            pdf = FPDF()
            pdf.add_page()

            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(
                0,
                10,
                clean("TalentMatch AI — Screening Report"),
                ln=True,
                align="C",
            )
            pdf.ln(5)

            # Date and Time
            pdf.set_font("Arial", "", 10)
            pdf.cell(
                0,
                8,
                clean(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                ln=True,
                align="C",
            )
            pdf.ln(10)

            # Job Role
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, clean(f"Job Role: {job_role}"), ln=True)
            pdf.ln(5)

            # Job Description snippet
            pdf.set_font("Arial", "", 10)
            jd_snippet = (
                job_description[:300] + "..."
                if len(job_description) > 300
                else job_description
            )
            pdf.multi_cell(0, 6, clean(f"Job Description: {jd_snippet}"))
            pdf.ln(10)

            # Sort results by similarity score descending for PDF
            sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)

            # Candidate Table
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, clean("Candidate Results"), ln=True)
            pdf.ln(5)

            # Table header
            pdf.set_font("Arial", "B", 10)
            pdf.cell(20, 8, clean("Rank"), border=1)
            pdf.cell(60, 8, clean("Name"), border=1)
            pdf.cell(30, 8, clean("Score"), border=1)
            pdf.cell(30, 8, clean("Status"), border=1)
            pdf.ln()

            # Table rows
            pdf.set_font("Arial", "", 9)
            for idx, result in enumerate(sorted_results, 1):
                pdf.cell(20, 8, clean(str(idx)), border=1)
                pdf.cell(
                    60,
                    8,
                    clean(os.path.basename(result["resume_path"])[:25]),
                    border=1,
                )
                pdf.cell(
                    30,
                    8,
                    clean(f"{result['similarity_score'] * 100:.1f}%"),
                    border=1,
                )
                pdf.cell(30, 8, clean(result["status"]), border=1)
                pdf.ln()

            pdf.ln(10)

            # Skills per candidate
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, clean("Skills Analysis"), ln=True)
            pdf.ln(5)

            for idx, result in enumerate(sorted_results, 1):
                pdf.set_font("Arial", "B", 10)
                pdf.cell(
                    0,
                    8,
                    clean(f"#{idx} - {os.path.basename(result['resume_path'])}"),
                    ln=True,
                )
                pdf.ln(3)

                pdf.set_font("Arial", "", 9)
                matched = result.get("matched_skills", [])
                missing = result.get("missing_skills", [])

                if matched:
                    pdf.cell(0, 6, clean(f"Matched: {', '.join(matched)}"), ln=True)
                else:
                    pdf.cell(0, 6, clean("Matched: None"), ln=True)

                if missing:
                    pdf.cell(0, 6, clean(f"Missing: {', '.join(missing)}"), ln=True)
                else:
                    pdf.cell(0, 6, clean("Missing: None"), ln=True)

                pdf.ln(5)

            return bytes(pdf.output())

        pdf_data = generate_pdf_report(results, job_role, job_description)

        # Side by side download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_data,
                file_name="screening_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="⬇️ Download as PDF",
                data=pdf_data,
                file_name="screening_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
