from typing import Dict, List, Tuple

import numpy as np


def compute_precision_recall_f1(
    results: List[Dict], threshold: float
) -> Tuple[float, float, float]:
    """
    Treats candidates above threshold with matched skills as True Positives.
    Returns precision, recall, f1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for result in results:
        is_shortlisted = result["status"] == "Shortlisted"
        has_matched_skills = len(result.get("matched_skills", [])) > 0

        if is_shortlisted and has_matched_skills:
            tp += 1
        elif is_shortlisted and not has_matched_skills:
            fp += 1
        elif not is_shortlisted and has_matched_skills:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_mrr(results: List[Dict]) -> float:
    """
    Mean Reciprocal Rank - ranks the first relevant (shortlisted) candidate.
    Returns MRR score.
    """
    for idx, result in enumerate(results):
        if result["status"] == "Shortlisted":
            return 1.0 / (idx + 1)  # Ranks are 1-based
    return 0.0  # No relevant candidate found


def compute_ndcg(results: List[Dict]) -> float:
    """
    Normalized Discounted Cumulative Gain - evaluates ranking quality.
    Returns NDCG score.
    """
    if not results:
        return 0.0

    # Assign relevance scores: Shortlisted = 1, Unsuitable = 0
    relevance_scores = [1 if r["status"] == "Shortlisted" else 0 for r in results]

    # Compute DCG
    dcg = 0.0
    for i, score in enumerate(relevance_scores):
        dcg += score / np.log2(i + 2)  # i+2 because log2(1) is 0, so rank 1 is 2

    # Compute Ideal DCG (sort relevance scores in descending order)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(ideal_relevance_scores):
        idcg += score / np.log2(i + 2)

    # Compute NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def compute_processing_stats(results: List[Dict]) -> float:
    """
    Returns average processing time per resume.
    """
    processing_times = [r["processing_time"] for r in results if "processing_time" in r]
    return float(np.mean(processing_times)) if processing_times else 0.0
