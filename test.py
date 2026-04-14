import pandas as pd

from src.preprocessing import preprocess_text
from src.similarity import compute_similarity
from src.ranking import rank_resumes


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    df = pd.read_csv("data/resumes.csv")

    print("Original Columns:", df.columns.tolist())

    # Keep only required columns
    df = df[["Resume_str", "Category"]].dropna()

    if df.empty:
        raise ValueError("Dataset is empty after removing null values.")

    # -----------------------------
    # 2. Preprocess resumes
    # -----------------------------
    print("Preprocessing resumes...")

    df["cleaned_resume"] = df["Resume_str"].apply(preprocess_text)

    # Use cleaned text for ML
    cleaned_resumes = df["cleaned_resume"].tolist()

    # Keep original resumes for display
    original_resumes = df["Resume_str"].tolist()

    # -----------------------------
    # 3. Job Description (Input Query)
    # -----------------------------
    job_description = """
    We are looking for a Data Scientist with strong experience in:
    Python, Machine Learning, Pandas, NumPy, SQL, Statistics, and Data Analysis.
    """

    # -----------------------------
    # 4. Compute similarity scores
    # -----------------------------
    print("Calculating similarity scores...")

    scores = compute_similarity(cleaned_resumes, job_description)

    # -----------------------------
    # 5. Rank resumes
    # -----------------------------
    print("Ranking resumes...")

    ranked_results = rank_resumes(
        resumes=original_resumes,
        scores=scores
    )

    # Optional safety sort (extra check)
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # -----------------------------
    # 6. Display Top Results
    # -----------------------------
    print("\n===== TOP MATCHED CANDIDATES =====\n")

    for i, item in enumerate(ranked_results[:5], start=1):
        print(f"Rank {i}")
        print(f"Score: {round(item['score'] * 100, 2)}%")
        print(f"Resume Preview: {item['resume'][:250]}")
        print("-" * 60)

    # -----------------------------
    # 7. Save output to CSV
    # -----------------------------
    output_df = pd.DataFrame(ranked_results)
    output_df.to_csv("ranked_results.csv", index=False)

    print("\nResults saved to ranked_results.csv")

    # -----------------------------
    # 8. Optional: Average score (for viva/report)
    # -----------------------------
    avg_score = sum(scores) / len(scores)
    print("\nAverage Match Score:", round(avg_score * 100, 2), "%")


if __name__ == "__main__":
    main()
