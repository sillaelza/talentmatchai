import time
import os
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.parser import ResumeParser
from src.preprocessor import TextPreprocessor
from src.embedder import EmbeddingEngine
import re


# Predefined skill list for matching
SKILL_LIST = [
    'python', 'javascript', 'html', 'css', 'flask', 'django', 'mongodb',
    'tensorflow', 'scikit-learn', 'sklearn', 'numpy', 'pandas', 'matplotlib',
    'opencv', 'nlp', 'natural language processing', 'machine learning',
    'deep learning', 'sql', 'git', 'docker', 'rest api', 'restful api',
    'node.js', 'nodejs', 'react', 'reactjs', 'regression', 'classification',
    'neural network', 'data preprocessing', 'feature engineering',
    'cosine similarity', 'tf-idf', 'tfidf', 'spacy', 'bert', 'transformers'
]


def extract_skills_from_text(text: str, skill_list: List[str]) -> List[str]:
    """
    Extract skills from text that match the predefined skill list.

    Args:
        text: Input text to search for skills
        skill_list: List of skills to search for

    Returns:
        List of skills found in the text
    """
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in skill_list:
        # Check if skill appears in text (case-insensitive)
        if skill.lower() in text_lower:
            found_skills.append(skill)

    return found_skills


def analyze_skill_gap(job_description_text: str, resume_text: str, skill_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyze skill gap between job description and resume.

    Args:
        job_description_text: Job description text
        resume_text: Resume text
        skill_list: List of skills to check

    Returns:
        Tuple of (matched_skills, missing_skills, jd_skills)
    """
    jd_skills = extract_skills_from_text(job_description_text, skill_list)
    resume_skills = extract_skills_from_text(resume_text, skill_list)

    # Skills present in both JD and resume
    matched_skills = list(set(jd_skills) & set(resume_skills))

    # Skills in JD but not in resume
    missing_skills = list(set(jd_skills) - set(resume_skills))

    # Sort for consistent display
    matched_skills.sort()
    missing_skills.sort()
    jd_skills.sort()

    return matched_skills, missing_skills, jd_skills


def rank_candidates(
    resume_paths: List[str],
    job_description_text: str,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Rank candidates based on similarity to job description using cosine similarity.

    Args:
        resume_paths: List of paths to resume files (PDF, DOCX, TXT)
        job_description_text: Job description text string
        threshold: Similarity threshold for shortlisting (default: 0.5)

    Returns:
        List of dictionaries containing candidate information sorted by similarity score
        (highest to lowest). Each dictionary contains:
        - resume_path: Path to the resume file
        - similarity_score: Cosine similarity score (0-1)
        - status: "Shortlisted" or "Unsuitable"
        - processing_time: Time taken to process the file in seconds
    """
    # Initialize components
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    embedder = EmbeddingEngine()

    results = []

    # Process job description
    print("Processing job description from text input")
    jd_start_time = time.time()

    # Use job description text directly
    jd_text = job_description_text
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text is empty")

    # Preprocess job description
    jd_cleaned = preprocessor.preprocess(jd_text)

    # Generate job description embedding
    jd_embedding = embedder.generate_embedding(jd_cleaned)

    jd_processing_time = time.time() - jd_start_time
    print(f"Job description processed in {jd_processing_time:.2f} seconds")

    # Process each resume
    print(f"\nProcessing {len(resume_paths)} resume(s)...")

    for resume_path in resume_paths:
        print(f"\nProcessing: {os.path.basename(resume_path)}")
        resume_start_time = time.time()

        try:
            # Parse resume
            resume_text = parser.parse_resume(resume_path)
            if not resume_text:
                print(f"Failed to parse resume: {resume_path}")
                results.append({
                    'resume_path': resume_path,
                    'similarity_score': 0.0,
                    'status': 'Unsuitable',
                    'processing_time': time.time() - resume_start_time,
                    'error': 'Failed to parse resume'
                })
                continue

            # Preprocess resume
            resume_cleaned = preprocessor.preprocess(resume_text)

            # Generate resume embedding
            resume_embedding = embedder.generate_embedding(resume_cleaned)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(
                resume_embedding.reshape(1, -1),
                jd_embedding.reshape(1, -1)
            )
            raw_cosine_score = float(similarity_matrix[0][0])

            # Analyze skill gap
            matched_skills, missing_skills, jd_skills = analyze_skill_gap(
                job_description_text, resume_text, SKILL_LIST
            )
            
            total_jd_skills = len(jd_skills)
            # Calculate matched skills ratio
            if total_jd_skills == 0:
                # If JD has no skills from our list, avoid division by zero (use 1.0 as default ratio)
                matched_skills_ratio = 1.0
            else:
                matched_skills_ratio = len(matched_skills) / total_jd_skills

            # Normalize raw cosine score from (-1 to 1) to (0 to 1)
            normalized_cosine_score = (raw_cosine_score + 1) / 2

            # Blend normalized cosine similarity and skill ratio with 50-50 weight
            similarity_score = (0.5 * matched_skills_ratio + 0.5 * normalized_cosine_score)

            # Determine status based on threshold and skill ratio
            # Shortlisted if score is above threshold AND matched skills ratio is at least 0.25
            if total_jd_skills == 0:
                # If JD has no skills, just use threshold
                status = "Shortlisted" if similarity_score >= threshold else "Unsuitable"
            else:
                status = "Shortlisted" if (similarity_score >= threshold and matched_skills_ratio >= 0.25) else "Unsuitable"

            processing_time = time.time() - resume_start_time

            # Store result
            result = {
                'resume_path': resume_path,
                'similarity_score': similarity_score,
                'status': status,
                'processing_time': processing_time,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'total_jd_skills': total_jd_skills,
                'matched_skills_ratio': matched_skills_ratio
            }

            results.append(result)

            print(f"Similarity Score: {similarity_score:.4f} ({similarity_score*100:.2f}%)")
            print(f"Status: {status}")
            print(f"Processing Time: {processing_time:.2f} seconds")

        except Exception as e:
            print(f"Error processing {resume_path}: {e}")
            results.append({
                'resume_path': resume_path,
                'similarity_score': 0.0,
                'status': 'Unsuitable',
                'processing_time': time.time() - resume_start_time,
                'error': str(e),
                'matched_skills': [],
                'missing_skills': [],
                'total_jd_skills': 0,
                'matched_skills_ratio': 0.0
            })

    # Sort results by similarity score (highest to lowest)
    sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    # Print summary
    print("\n" + "="*60)
    print("RANKING SUMMARY")
    print("="*60)

    for idx, result in enumerate(sorted_results, 1):
        print(f"\n{idx}. {os.path.basename(result['resume_path'])}")
        print(f"   Score: {result['similarity_score']:.4f} ({result['similarity_score']*100:.2f}%)")
        print(f"   Status: {result['status']}")
        print(f"   Time: {result['processing_time']:.2f}s")

    shortlisted_count = sum(1 for r in sorted_results if r['status'] == 'Shortlisted')
    print(f"\nTotal Candidates: {len(sorted_results)}")
    print(f"Shortlisted: {shortlisted_count}")
    print(f"Unsuitable: {len(sorted_results) - shortlisted_count}")

    return sorted_results


def generate_explanation(filename: str, score: float, matched_skills: List[str], missing_skills: List[str], threshold: float, matched_skills_ratio: float = 1.0, total_jd_skills: int = 0) -> str:
    """
    Generate a human-readable explanation for the matching decision.

    Args:
        filename: Candidate filename
        score: Similarity score (0-1 range)
        matched_skills: List of matched skills
        missing_skills: List of missing skills
        threshold: Threshold score (0-1 range)
        matched_skills_ratio: Ratio of matched skills to total JD skills (0-1)
        total_jd_skills: Total number of skills in JD from our list

    Returns:
        Human-readable explanation string
    """
    score_pct = score * 100
    threshold_pct = threshold * 100
    ratio_pct = matched_skills_ratio * 100

    # Check if skill ratio is sufficient (>=25%)
    skill_ratio_sufficient = matched_skills_ratio >= 0.25 or total_jd_skills == 0

    if score >= threshold and skill_ratio_sufficient:
        # Suitable candidate
        top_skills = matched_skills[:3]
        skills_str = ", ".join(top_skills)
        if len(matched_skills) > 3:
            skills_str += f", and {len(matched_skills) - 3} more"
        ratio_info = f" They have {len(matched_skills)}/{total_jd_skills} skills ({ratio_pct:.0f}% match) from the job description." if total_jd_skills > 0 else ""
        return f"This candidate is a strong match.{ratio_info} Their score is a 50-50 blend of semantic similarity and skill match ratio. They possess {len(matched_skills)} required skills including {skills_str}."
    else:
        # Unsuitable candidate
        if not skill_ratio_sufficient:
            return f"This candidate only has {len(matched_skills)}/{total_jd_skills} skills ({ratio_pct:.0f}% match) from the job description, which is below the required 25%."
        elif missing_skills:
            top_missing = missing_skills[:3]
            missing_str = ", ".join(top_missing)
            if len(missing_skills) > 3:
                missing_str += f", and {len(missing_skills) - 3} more"
            return f"This candidate falls short of the {threshold_pct:.0f}% threshold. While they have {len(matched_skills)}/{total_jd_skills} matching skills, they are missing key skills such as {missing_str} which are critical for this role."
        else:
            return f"This candidate falls short of the {threshold_pct:.0f}% threshold with a score of {score_pct:.1f}%."


def calculate_similarity_percentage(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity percentage between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity percentage (0-100)
    """
    similarity_matrix = cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )
    similarity_score = float(similarity_matrix[0][0])
    return similarity_score * 100


if __name__ == "__main__":
    # Example usage
    # Note: Replace with actual file paths for testing

    # Example resume paths
    resume_paths = [
        "path/to/resume1.pdf",
        "path/to/resume2.docx",
        "path/to/resume3.txt"
    ]

    # Example job description text
    job_description_text = "Job description text here"

    # Rank candidates with 50% threshold
    # results = rank_candidates(resume_paths, job_description_text, threshold=0.5)

    print("Matcher module loaded successfully.")
    print("Use rank_candidates() function to process resumes against a job description.")
