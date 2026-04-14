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

    # -----------------------------
    # 2. Preprocess resumes
    # -----------------------------
    print("Preprocessing resumes...")

    df["cleaned_resume"] = df["Resume_str"].apply(preprocess_text)

    resumes = df["cleaned_resume"].tolist()

    # -----------------------------
    # 3. Job Description (input query)
    # -----------------------------
    job_description = """
    We are looking for a Data Scientist with strong experience in:
    Python, Machine Learning, Pandas, NumPy, SQL, and Statistics.
    """

    # -----------------------------
    # 4. Compute similarity scores
    # -----------------------------
    print("Calculating similarity scores...")

    scores = compute_similarity(resumes, job_description)

    # -----------------------------
    # 5. Rank resumes
    # -----------------------------
    print("Ranking resumes...")

    ranked_results = rank_resumes(
        resumes=df["Resume_str"].tolist(),  # original resumes (for display)
        scores=scores
    )

    # -----------------------------
    # 6. Display Top Results
    # -----------------------------
    print("\n===== TOP MATCHED CANDIDATES =====\n")

    for i, item in enumerate(ranked_results[:10], start=1):
        print(f"Rank {i}")
        print(f"Score: {round(item['score'] * 100, 2)}%")
        print(f"Resume Preview: {item['resume'][:250]}")
        print("-" * 60)

    # -----------------------------
    # 7. Save output to CSV (important for project)
    # -----------------------------
    output_df = pd.DataFrame(ranked_results)
    output_df.to_csv("ranked_results.csv", index=False)

    print("\nResults saved to ranked_results.csv")


if __name__ == "__main__":
    main()
