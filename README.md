# nextgen-hire-resume-matching

NextGen Hire is an AI-based Resume–Job Matching System. The goal of the project is to help recruiters compare resumes with a job description and identify the most suitable candidates using Natural Language Processing (NLP).

The system analyzes the text of resumes and job descriptions, applies preprocessing techniques, converts the text into numerical representations using TF-IDF, and then calculates similarity scores with cosine similarity. Based on these scores, resumes are ranked from best match to lowest match.

## Project Objective

The purpose of this project is to automate part of the hiring process by helping recruiters screen resumes more efficiently. Instead of manually reviewing many applications, the system provides a ranked list of candidates based on how closely their resumes match a job description.

## Features

- Enter a job description through the UI
- Upload resumes in PDF, DOCX, DOC, or TXT format
- Preprocess resume and job description text
- Compute similarity scores using TF-IDF and cosine similarity
- Rank resumes from highest to lowest match
- Display top matched candidates in a simple UI
- Highlight matched skills from uploaded resumes

## Technologies Used

- Python
- CustomTkinter
- Pandas
- Scikit-learn
- NLTK
- PyPDF
- python-docx

## Project Structure

```text
nextgen-hire-resume-matching/
├── app/
│   ├── app.py
│   └── resume_reader.py
├── data/
│   └── resumes.csv
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── similarity.py
│   └── ranking.py
├── test.py
├── README.md
└── requirements.txt
