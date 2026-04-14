def rank_resumes(resumes, scores):
    """
    Rank resumes from highest to lowest similarity score.

    Args:
        resumes (list[str]): List of resume texts
        scores (list[float]): Similarity scores

    Returns:
        list[dict]: Ranked resumes with scores
    """
    if len(resumes) != len(scores):
        raise ValueError("Resumes and scores must have the same length.")

    ranked_results = [
        {"resume": resume, "score": score}
        for resume, score in zip(resumes, scores)
    ]

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results