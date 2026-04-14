from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(resumes, job_description):
    """
    Compute similarity scores between a job description and a list of resumes.

    Args:
        resumes (list[str]): List of resume texts
        job_description (str): Job description text

    Returns:
        list[float]: Similarity score for each resume
    """
    if not resumes:
        return []

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    cleaned_resumes = [resume if isinstance(resume, str) else "" for resume in resumes]
    documents = [job_description] + cleaned_resumes

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarity_scores[0].tolist()