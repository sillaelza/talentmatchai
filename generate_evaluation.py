import numpy as np
from typing import List
from sklearn.metrics import precision_score, recall_score, ndcg_score


def calculate_mean_reciprocal_rank(relevant_items: List[int], ranked_list: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR measures the rank of the first relevant item in a ranked list.
    It is commonly used in information retrieval and recommendation systems.
    
    Args:
        relevant_items: List of indices/IDs of relevant items
        ranked_list: List of indices/IDs in ranked order (highest to lowest)
        
    Returns:
        MRR score (0 to 1), where 1 means the first item is relevant
    """
    if not relevant_items or not ranked_list:
        return 0.0
    
    # Find the rank of the first relevant item
    for rank, item in enumerate(ranked_list, start=1):
        if item in relevant_items:
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg(relevance_scores: List[int], predicted_scores: List[float], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG measures the quality of ranking by considering the position of relevant items.
    It accounts for the fact that relevant items appearing higher in the ranking are more valuable.
    
    Args:
        relevance_scores: List of binary relevance scores (1 = relevant, 0 = irrelevant)
        predicted_scores: List of predicted similarity scores (higher = more relevant)
        k: Optional cutoff position for calculation (default: None = use all items)
        
    Returns:
        NDCG score (0 to 1), where 1 means perfect ranking
    """
    if not relevance_scores or not predicted_scores:
        return 0.0
    
    # Ensure lists have same length
    if len(relevance_scores) != len(predicted_scores):
        raise ValueError("relevance_scores and predicted_scores must have the same length")
    
    # Reshape for sklearn's ndcg_score function
    relevance_array = np.array([relevance_scores])
    predicted_array = np.array([predicted_scores])
    
    # Calculate NDCG
    try:
        if k is None:
            k = len(relevance_scores)
        
        ndcg = ndcg_score(relevance_array, predicted_array, k=k)
        return float(ndcg)
    except Exception as e:
        print(f"Error calculating NDCG: {e}")
        return 0.0


def calculate_precision_recall(y_true: List[int], y_pred: List[int]) -> tuple:
    """
    Calculate Precision and Recall metrics.
    
    Args:
        y_true: List of true labels (1 = relevant, 0 = irrelevant)
        y_pred: List of predicted labels (1 = relevant, 0 = irrelevant)
        
    Returns:
        Tuple of (precision, recall) scores
    """
    if not y_true or not y_pred:
        return 0.0, 0.0
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return float(precision), float(recall)


def run_validation_evaluation():
    """
    Run a quick validation evaluation using mock ground truth data.
    
    This function demonstrates the calculation of key evaluation metrics
    for the resume screening system using sample data.
    """
    print("="*70)
    print("RESUME SCREENING SYSTEM - VALIDATION METRICS")
    print("="*70)
    print()
    
    # Mock ground truth data (1 = relevant/correctly ranked, 0 = irrelevant/incorrect)
    # This represents the ideal ranking where 1s should appear before 0s
    ground_truth = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    
    # Mock predicted similarity scores (higher = more relevant)
    # In a real scenario, these would come from the cosine similarity calculations
    predicted_scores = [0.95, 0.88, 0.45, 0.82, 0.35, 0.78, 0.28, 0.42, 0.71, 0.31]
    
    # Convert predicted scores to binary predictions based on threshold (e.g., 0.5)
    threshold = 0.5
    predicted_labels = [1 if score >= threshold else 0 for score in predicted_scores]
    
    print("Ground Truth (Relevance):", ground_truth)
    print("Predicted Scores:       ", [f"{s:.2f}" for s in predicted_scores])
    print("Predicted Labels (>=0.5):", predicted_labels)
    print(f"Threshold: {threshold}")
    print()
    
    # Calculate Precision and Recall
    precision, recall = calculate_precision_recall(ground_truth, predicted_labels)
    
    print("-"*70)
    print("CLASSIFICATION METRICS")
    print("-"*70)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print()
    
    # Calculate MRR
    # Get indices of relevant items (where ground_truth = 1)
    relevant_indices = [i for i, rel in enumerate(ground_truth) if rel == 1]
    
    # Create ranked list based on predicted scores (sorted by score, descending)
    ranked_indices = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
    
    mrr = calculate_mean_reciprocal_rank(relevant_indices, ranked_indices)
    
    print("-"*70)
    print("RANKING METRICS")
    print("-"*70)
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  - First relevant item at position: {ranked_indices.index(relevant_indices[0]) + 1 if relevant_indices else 'N/A'}")
    print()
    
    # Calculate NDCG at different cutoffs
    print("-"*70)
    print("NORMALIZED DISCOUNTED CUMULATIVE GAIN (NDCG)")
    print("-"*70)
    
    for k in [3, 5, 10]:
        ndcg = calculate_ndcg(ground_truth, predicted_scores, k=k)
        print(f"NDCG@{k}:  {ndcg:.4f}")
    
    print()
    
    # Calculate overall NDCG (all items)
    ndcg_all = calculate_ndcg(ground_truth, predicted_scores)
    print(f"NDCG@All: {ndcg_all:.4f}")
    print()
    
    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total Items Evaluated: {len(ground_truth)}")
    print(f"Relevant Items: {sum(ground_truth)}")
    print(f"Irrelevant Items: {len(ground_truth) - sum(ground_truth)}")
    print()
    print("Key Metrics:")
    print(f"  - Precision: {precision*100:.2f}%")
    print(f"  - Recall: {recall*100:.2f}%")
    print(f"  - MRR: {mrr:.4f}")
    print(f"  - NDCG@5: {calculate_ndcg(ground_truth, predicted_scores, k=5):.4f}")
    print(f"  - NDCG@10: {calculate_ndcg(ground_truth, predicted_scores, k=10):.4f}")
    print("="*70)
    print()
    
    return {
        'precision': precision,
        'recall': recall,
        'mrr': mrr,
        'ndcg_5': calculate_ndcg(ground_truth, predicted_scores, k=5),
        'ndcg_10': calculate_ndcg(ground_truth, predicted_scores, k=10),
        'ndcg_all': ndcg_all
    }


if __name__ == "__main__":
    # Run the validation evaluation
    metrics = run_validation_evaluation()
    
    print("\n[SUCCESS] Validation evaluation completed successfully!")
    print("\nThese metrics can be used to assess the ranking quality of the")
    print("resume screening system. In production, you would:")
    print("  1. Collect human-annotated ground truth data")
    print("  2. Run the system on a test dataset")
    print("  3. Compare system rankings against ground truth")
    print("  4. Use these metrics to tune the threshold and model parameters")
