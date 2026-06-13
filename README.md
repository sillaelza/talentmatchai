# TalentMatch AI

**Intelligent Resume Screening & Skill Matching System powered by NLP**

TalentMatch AI is an automated resume screening tool that uses sentence embeddings and cosine similarity to match candidate resumes against job descriptions, rank candidates by relevance, and provide explainable scoring along with skill-gap analysis.

---

## Project Description

Recruitment teams often spend significant time manually screening resumes against job requirements. TalentMatch AI automates this process by:

- Parsing resumes in PDF, DOCX, and TXT formats
- Generating semantic embeddings for both resumes and job descriptions using a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`)
- Computing cosine similarity between resume and job description embeddings
- Extracting and comparing skills to identify matched and missing skills
- Ranking candidates and classifying them as "Shortlisted" or "Unsuitable" based on a configurable threshold
- Providing human-readable explanations for each match decision
- Reporting evaluation metrics (Precision, Recall, F1-Score, MRR, NDCG)
- Exporting results as CSV and PDF reports

---

## Approach / Solution Overview

1. **Resume & JD Ingestion** — Uploaded resumes (PDF/DOCX/TXT) and the job description text are parsed and cleaned in `src/parser.py` and `src/preprocessor.py`.
2. **Embedding Generation** — Both resume text and job description text are converted into dense vector embeddings using `SentenceTransformer (all-MiniLM-L6-v2)` in `src/embedder.py`.
3. **Similarity & Skill Matching** — `src/matcher.py` computes cosine similarity between embeddings, extracts skills from a predefined skill list, and identifies matched vs. missing skills.
4. **Scoring & Classification** — A similarity score is computed and adjusted based on the proportion of matched skills. Candidates are classified as Shortlisted or Unsuitable based on a user-configurable threshold.
5. **Explainability** — `generate_explanation()` produces a plain-language reason for each candidate's score and status.
6. **Evaluation** — `src/evaluator.py` computes Precision, Recall, F1-Score, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG) for the ranking quality.
7. **Interface** — A Streamlit app (`app.py`) provides an interactive dashboard for uploading resumes, entering job descriptions, viewing rankings, and exporting reports.

---

## Features

- Multi-format resume parsing (PDF, DOCX, TXT)
- Semantic similarity matching using SBERT embeddings
- Skill extraction and skill-gap analysis (matched/missing skills)
- Configurable match score threshold
- Candidate ranking leaderboard
- Candidate comparison overview table
- Explainability section ("Why this score?")
- Evaluation metrics dashboard (Precision, Recall, F1, MRR, NDCG, processing time)
- Export results as CSV and PDF
- Interactive, modern Streamlit UI

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI Framework | Streamlit |
| NLP / Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`), spaCy |
| Similarity | scikit-learn (cosine similarity) |
| Data Handling | pandas, numpy |
| PDF Reports | fpdf2 |
| Resume Parsing | PyPDF / python-docx (PDF, DOCX, TXT support) |

---

## Project Structure

```
resume-ai/
├── app.py                  # Streamlit application (UI & workflow)
├── src/
│   ├── parser.py           # Resume parsing (PDF, DOCX, TXT)
│   ├── preprocessor.py      # Text cleaning and normalization
│   ├── embedder.py          # Sentence embedding generation
│   ├── matcher.py            # Similarity scoring, skill matching, ranking
│   └── evaluator.py          # Evaluation metrics (Precision, Recall, F1, MRR, NDCG)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/talentmatch-ai.git
cd talentmatch-ai
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Usage Guide

1. Open the application in your browser.
2. On the landing page, click **Explore** to go to the screening tool.
3. Enter the **Job Role/Title** and **Job Description** in the provided fields.
4. Upload one or more resumes (PDF, DOCX, or TXT) using the file uploader.
5. (Optional) Adjust the **Minimum Match Score Threshold** in the sidebar to control how strict the shortlisting should be.
6. Click **Run AI Screening Analytics**.
7. Review the results:
   - **Screening Analytics** — summary metrics
   - **Candidate Ranking Leaderboard** — ranked list of candidates
   - **Candidate Deep-Dive Analysis** — detailed score, matched/missing skills, and explanation per candidate
   - **Candidate Comparison Overview** — side-by-side comparison table
   - **Evaluation Metrics** — Precision, Recall, F1, MRR, NDCG, average processing time
8. Download the results as a **CSV** or **PDF** report using the Export Results section.

---

## Evaluation Metrics

The system computes the following metrics based on skill matching and semantic similarity thresholds:

- **Precision** — proportion of shortlisted candidates that are genuinely relevant
- **Recall** — proportion of relevant candidates that were correctly shortlisted
- **F1-Score** — harmonic mean of Precision and Recall
- **MRR (Mean Reciprocal Rank)** — measures how high the first relevant candidate is ranked
- **NDCG (Normalized Discounted Cumulative Gain)** — measures the overall quality of the ranking order
- **Average Processing Time** — average time taken to process each resume

---

## Notes

- All models used are publicly available, open-source pre-trained models.
- Uploaded resumes are processed in-memory/temporarily and are not stored beyond the screening session.
- The system is designed to be modular, allowing each component (parsing, preprocessing, embedding, matching, evaluation) to be modified or extended independently.
