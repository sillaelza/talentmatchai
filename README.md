# 🚀 TalentMatch AI: Advanced Resume Screening & Skill Matching Engine

TalentMatch AI is a production-grade, enterprise-ready Automated Resume Screening and Talent Acquisition analytics platform. Built using automated Natural Language Processing (NLP) pipelines, the system parses unstructured candidate profiles, normalizes technical domain jargon, and leverages dense vector semantic models to accurately score and rank applicants against precise job requirements.

---

## 🛠️ Core Engineering Architecture

The platform is engineered using a decoupled, modular design to ensure high scalability, rapid inference execution, and clean maintainability:

* **Robust Text Extraction Layer (`parser.py`):** Automatically detects, processes, and extracts unstructured text streams from `.pdf`, `.docx`, and `.txt` files with automated fallback encoding layers.
* **NLP Text Preprocessing Engine (`preprocessor.py`):** Utilizes `spaCy` linguistic tokenization pipelines to clean raw inputs, eliminate linguistic noise/stopwords, and run a custom domain dictionary mapping layer to handle synonyms and technical abbreviations (e.g., standardizing tokens like `ML`, `AI`, `ReactJS`, or `dApp`).
* **Vector Transformation Matrix (`embedder.py`):** Converts clean text representations into high-dimensional dense vector embeddings using a pre-trained `SentenceTransformer` architecture (`all-MiniLM-L6-v2`) to extract deep semantic contextual intent.
* **Cosine Similarity Matching Engine (`matcher.py`):** Computes the exact mathematical similarity between the Job Description vector and applicant profile matrices. It tracks processing latency per file and implements a strict threshold cutoff logic to categorize candidates immediately.

---

## 📁 System Directory Structure

```text
clayays/
│
├── data/                            
│   ├── resumes/                     # Source directory for candidate profiles
│   └── job_descriptions/            # Target core requirement files
│
├── src/                             
│   ├── __init__.py                  # Package initialization
│   ├── parser.py                    # Multi-format document parser
│   ├── preprocessor.py              # Advanced text cleaner & token mapper
│   ├── embedder.py                  # Sentence Transformer vector matrix
│   └── matcher.py                   # Scoring engine & ranking algorithms
│
├── app.py                           # Streamlit Interactive Dashboard UI
├── generate_evaluation.py           # IR Performance validation script
└── requirements.txt                 # Frozen dependency configuration
