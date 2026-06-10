import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from typing import List

from src.parser import ResumeParser
from src.preprocessor import TextPreprocessor
from src.embedder import EmbeddingEngine
from src.matcher import rank_candidates


# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 AI Resume Screening & Skill Matching System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ System Configuration")
    
    # Threshold slider
    threshold_percent = st.sidebar.slider(
        "Minimum Match Score Threshold (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Candidates with similarity score above this threshold will be shortlisted"
    )
    
    threshold = threshold_percent / 100.0
    
    st.sidebar.markdown(f"**Current Threshold:** {threshold_percent}%")
    st.sidebar.markdown("---")
    
    # Main layout - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Job Description")
        st.markdown("Paste the job description below:")
        
        job_description = st.text_area(
            "Job Description",
            height=300,
            placeholder="Paste the complete job description here including required skills, qualifications, and responsibilities...",
            help="Enter the full job description text for candidate matching"
        )
    
    with col2:
        st.markdown("### 📁 Resume Upload")
        st.markdown("Upload candidate resumes (PDF, DOCX, or TXT):")
        
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Select one or more resume files to evaluate against the job description"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} file(s) selected for processing")
    
    # Execution button
    st.markdown("---")
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        run_button = st.button(
            "⚡ Run AI Screening Analytics",
            type="primary",
            use_container_width=True,
            help="Process all uploaded resumes against the job description"
        )
    
    # Process when button is clicked
    if run_button:
        if not job_description.strip():
            st.error("❌ Please provide a job description before running the analysis.")
            return
        
        if not uploaded_files:
            st.error("❌ Please upload at least one resume file before running the analysis.")
            return
        
        # Show processing status
        with st.spinner("🔄 Processing resumes and calculating similarity scores..."):
            try:
                # Save uploaded files to temporary directory
                temp_dir = tempfile.mkdtemp()
                resume_paths = []
                
                for uploaded_file in uploaded_files:
                    # Create temporary file path
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    # Write uploaded file to temporary location
                    with open(temp_file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    resume_paths.append(temp_file_path)
                
                # Save job description to temporary file
                jd_temp_path = os.path.join(temp_dir, "job_description.txt")
                with open(jd_temp_path, 'w', encoding='utf-8') as f:
                    f.write(job_description)
                
                # Run the ranking analysis
                results = rank_candidates(
                    resume_paths=resume_paths,
                    job_description_path=jd_temp_path,
                    threshold=threshold
                )
                
                # Clean up temporary files
                for file_path in resume_paths + [jd_temp_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)
                
                # Display results
                st.success("✅ Analysis completed successfully!")
                st.markdown("---")
                
                # Calculate KPI metrics
                total_resumes = len(results)
                shortlisted_count = sum(1 for r in results if r['status'] == 'Shortlisted')
                avg_score = np.mean([r['similarity_score'] for r in results]) * 100 if results else 0
                
                # Display KPI cards
                st.markdown("### 📊 Key Performance Indicators")
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                
                with kpi_col1:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{total_resumes}</div>
                            <div class="kpi-label">Total Resumes Evaluated</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col2:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{shortlisted_count}</div>
                            <div class="kpi-label">Shortlisted Candidates</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col3:
                    st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-value">{avg_score:.1f}%</div>
                            <div class="kpi-label">Average Match Score</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display ranked results table
                st.markdown("### 🏆 Candidate Ranking Leaderboard")
                
                # Prepare data for DataFrame
                df_data = []
                for idx, result in enumerate(results, 1):
                    df_data.append({
                        'Rank': idx,
                        'Candidate': os.path.basename(result['resume_path']),
                        'Match Score (%)': f"{result['similarity_score'] * 100:.2f}",
                        'Status': result['status'],
                        'Processing Time (s)': f"{result['processing_time']:.2f}"
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display DataFrame with styling
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn('Rank', width='small'),
                        'Candidate': st.column_config.TextColumn('Candidate', width='large'),
                        'Match Score (%)': st.column_config.ProgressColumn(
                            'Match Score (%)',
                            help='Similarity score with job description',
                            format='%.2f',
                            min_value=0,
                            max_value=100
                        ),
                        'Status': st.column_config.TextColumn('Status', width='medium'),
                        'Processing Time (s)': st.column_config.NumberColumn('Processing Time (s)', width='small', format='%.2f')
                    }
                )
                
                st.markdown("---")
                
                # Display detailed analysis for each candidate
                st.markdown("### 🔍 Detailed Candidate Analysis")
                
                for idx, result in enumerate(results, 1):
                    candidate_name = os.path.basename(result['resume_path'])
                    status_icon = "✅" if result['status'] == 'Shortlisted' else "❌"
                    
                    with st.expander(f"{status_icon} #{idx} - {candidate_name} (Score: {result['similarity_score']*100:.2f}%)"):
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.metric("Similarity Score", f"{result['similarity_score']*100:.2f}%")
                            st.metric("Status", result['status'])
                        
                        with col_detail2:
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                            st.metric("Threshold", f"{threshold_percent}%")
                        
                        st.markdown("---")
                        st.markdown("#### 📋 Model Decision Interpretability")
                        
                        # Display processing information
                        st.info(f"""
                        **Processing Details:**
                        - File processed in {result['processing_time']:.2f} seconds
                        - Compared against job description using cosine similarity
                        - Status determined by threshold: {threshold_percent}%
                        """)
                        
                        # Note about key technical words
                        st.warning("""
                        **Note:** Key technical word overlap analysis can be enhanced by implementing 
                        additional NLP features such as keyword extraction, named entity recognition, 
                        or skill-specific vocabulary matching in future iterations.
                        """)
                
                # Download results option
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='resume_screening_results.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.error("Please check your input files and try again.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🤖 AI Resume Screening System | Powered by NLP & Machine Learning</p>
    <p>Using Sentence Transformers (all-MiniLM-L6-v2) & Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
