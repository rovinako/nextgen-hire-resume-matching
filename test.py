import pandas as pd
from src.similarity import compute_similarity


def main():
    df = pd.read_csv("data/resumes.csv")
    print("Columns:", df.columns.tolist())

    resume_column = "Resume_str"

    sample_df = df[[resume_column, "Category"]].dropna().sample(20, random_state=42)
    resumes = sample_df[resume_column].tolist()
    categories = sample_df["Category"].tolist()

    job_description = """
    We are looking for a Data Scientist with experience in Python,
    machine learning, statistics, pandas, and scikit-learn.
    """

    scores = compute_similarity(resumes, job_description)

    ranked = list(zip(resumes, categories, scores))
    ranked.sort(key=lambda x: x[2], reverse=True)

    for i, (resume, category, score) in enumerate(ranked, start=1):
        print(f"Rank {i}: Score = {score:.3f} | Category = {category}")
        print(resume[:200])
        print("-" * 50)


if __name__ == "__main__":
    main()