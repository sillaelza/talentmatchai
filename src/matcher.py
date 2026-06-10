import time
import os
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.parser import ResumeParser
from src.preprocessor import TextPreprocessor
from src.embedder import EmbeddingEngine


def rank_candidates(
    resume_paths: List[str],
    job_description_path: str,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Rank candidates based on similarity to job description using cosine similarity.
    
    Args:
        resume_paths: List of paths to resume files (PDF, DOCX, TXT)
        job_description_path: Path to the job description file
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
    print(f"Processing job description: {job_description_path}")
    jd_start_time = time.time()
    
    # Parse job description
    jd_text = parser.parse_resume(job_description_path)
    if not jd_text:
        raise ValueError(f"Failed to parse job description: {job_description_path}")
    
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
            similarity_score = float(similarity_matrix[0][0])
            
            # Determine status based on threshold
            status = "Shortlisted" if similarity_score >= threshold else "Unsuitable"
            
            processing_time = time.time() - resume_start_time
            
            # Store result
            result = {
                'resume_path': resume_path,
                'similarity_score': similarity_score,
                'status': status,
                'processing_time': processing_time
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
                'error': str(e)
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
    
    # Example job description path
    job_description_path = "path/to/job_description.txt"
    
    # Rank candidates with 50% threshold
    # results = rank_candidates(resume_paths, job_description_path, threshold=0.5)
    
    print("Matcher module loaded successfully.")
    print("Use rank_candidates() function to process resumes against a job description.")
