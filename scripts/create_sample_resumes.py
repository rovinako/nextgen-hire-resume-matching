import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/resumes.csv")

# Make output folder
os.makedirs("sample_resumes", exist_ok=True)

# Pick 10 random resumes
sample_df = df.sample(10, random_state=42)

for i, row in sample_df.iterrows():
    text = row["Resume_str"]
    category = row["Category"]

    filename = f"resume_{i}_{category}.txt"
    filepath = os.path.join("sample_resumes", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

print("Sample resumes created in /sample_resumes/")