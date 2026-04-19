import pandas as pd
import os

df = pd.read_csv("data/resumes.csv")

hr_df = df[df["Category"].str.contains("HR", case=False, na=False)]

output_folder = "sample_resumes"
os.makedirs(output_folder, exist_ok=True)

sample_df = hr_df.sample(n=min(10, len(hr_df)), random_state=42)

for i, row in sample_df.iterrows():
    text = row["Resume_str"]
    filename = f"hr_resume_{i}.txt"

    with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
        f.write(text)

print(f"{len(sample_df)} HR resumes saved to '{output_folder}/'")