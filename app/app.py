import os
import re
from tkinter import filedialog, messagebox
import customtkinter as ctk

from src.similarity import compute_similarity
from src.ranking import rank_resumes
from app.resume_reader import read_resume


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


BG = "#1a1a1a"
CARD = "#2a2a2a"
BORDER = "#3a3a3a"
GREEN = "#1D9E75"
WHITE = "#ffffff"
MUTED = "#9a9a9a"
RED = "#E24B4A"


def extract_skills(text: str, skill_list: list[str]) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


class NextGenHireApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NextGen Hire")
        self.geometry("1000x760")
        self.configure(fg_color=BG)

        self.resume_files = []
        self.resume_texts = {}
        self.skill_list = [
            "Python", "SQL", "Machine Learning", "Data Analysis", "Pandas",
            "NumPy", "Scikit-learn", "Statistics", "Deep Learning", "NLP",
            "TensorFlow", "PyTorch", "Excel", "Java", "C++", "AWS"
        ]

        self.build_ui()

    def build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="NextGen Hire",
            text_color=WHITE,
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(side="left", padx=25, pady=10)

        ctk.CTkLabel(
            header,
            text="AI Resume Matching System",
            text_color=MUTED,
            font=ctk.CTkFont(size=12)
        ).pack(side="right", padx=25)

        # Main container
        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Top section: job description
        jd_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        jd_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            jd_card,
            text="Job Description",
            text_color=WHITE,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            jd_card,
            text="Paste the job description below.",
            text_color=MUTED,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(0, 8))

        self.jd_box = ctk.CTkTextbox(
            jd_card,
            height=160,
            fg_color=BG,
            border_width=1,
            border_color=BORDER,
            text_color=WHITE
        )
        self.jd_box.pack(fill="x", padx=20, pady=(0, 18))

        # Middle section: upload + buttons
        upload_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        upload_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            upload_card,
            text="Upload Resumes",
            text_color=WHITE,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            upload_card,
            text="Upload PDF, DOCX, DOC, or TXT resume files.",
            text_color=MUTED,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        btn_row = ctk.CTkFrame(upload_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="Upload Files",
            fg_color=GREEN,
            hover_color="#157a5c",
            command=self.upload_files,
            width=140
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="Clear Files",
            fg_color=RED,
            hover_color="#b63a39",
            command=self.clear_files,
            width=140
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="Run Matching",
            fg_color="#2d7dd2",
            hover_color="#1f5ea0",
            command=self.run_analysis,
            width=160
        ).pack(side="right")

        self.file_count_label = ctk.CTkLabel(
            upload_card,
            text="No files uploaded",
            text_color=MUTED,
            font=ctk.CTkFont(size=12)
        )
        self.file_count_label.pack(anchor="w", padx=20, pady=(0, 8))

        self.file_list_box = ctk.CTkTextbox(
            upload_card,
            height=110,
            fg_color=BG,
            border_width=1,
            border_color=BORDER,
            text_color=WHITE
        )
        self.file_list_box.pack(fill="x", padx=20, pady=(0, 18))
        self.file_list_box.insert("end", "Uploaded files will appear here.\n")
        self.file_list_box.configure(state="disabled")

        # Bottom section: results
        results_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
        results_card.pack(fill="both", expand=True)

        ctk.CTkLabel(
            results_card,
            text="Results",
            text_color=WHITE,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            results_card,
            text="Top ranked resumes will appear below.",
            text_color=MUTED,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        self.results_box = ctk.CTkTextbox(
            results_card,
            fg_color=BG,
            border_width=1,
            border_color=BORDER,
            text_color=WHITE
        )
        self.results_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def upload_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Resume Files",
            filetypes=[
                ("Resume files", "*.pdf *.docx *.doc *.txt"),
                ("All files", "*.*")
            ]
        )

        if not paths:
            return

        self.resume_files = list(paths)
        self.resume_texts = {}

        self.file_list_box.configure(state="normal")
        self.file_list_box.delete("1.0", "end")

        loaded_count = 0
        for path in self.resume_files:
            try:
                text = read_resume(path)
                if text and text.strip():
                    self.resume_texts[os.path.basename(path)] = text
                    self.file_list_box.insert("end", f"{os.path.basename(path)}\n")
                    loaded_count += 1
            except Exception as e:
                self.file_list_box.insert("end", f"Error reading {os.path.basename(path)}: {e}\n")

        self.file_list_box.configure(state="disabled")
        self.file_count_label.configure(text=f"{loaded_count} file(s) loaded")

    def clear_files(self):
        self.resume_files = []
        self.resume_texts = {}
        self.file_count_label.configure(text="No files uploaded")

        self.file_list_box.configure(state="normal")
        self.file_list_box.delete("1.0", "end")
        self.file_list_box.insert("end", "Uploaded files will appear here.\n")
        self.file_list_box.configure(state="disabled")

    def run_analysis(self):
        job_description = self.jd_box.get("1.0", "end").strip()

        if not job_description:
            messagebox.showwarning("Missing Job Description", "Please enter a job description.")
            return

        if not self.resume_texts:
            messagebox.showwarning("No Resumes", "Please upload at least one resume.")
            return

        names = list(self.resume_texts.keys())
        texts = list(self.resume_texts.values())

        try:
            scores = compute_similarity(texts, job_description)
            ranked = rank_resumes(names, scores)

            text_lookup = dict(zip(names, texts))

            self.results_box.delete("1.0", "end")
            self.results_box.insert("end", "===== TOP MATCHED CANDIDATES =====\n\n")

            for i, item in enumerate(ranked[:5], start=1):
                name = item["resume"]
                score = item["score"]
                resume_text = text_lookup.get(name, "")
                found_skills = extract_skills(resume_text, self.skill_list)

                self.results_box.insert("end", f"Rank {i}\n")
                self.results_box.insert("end", f"Candidate: {name}\n")
                self.results_box.insert("end", f"Score: {score:.2%}\n")

                if found_skills:
                    self.results_box.insert("end", f"Matched Skills: {', '.join(found_skills[:8])}\n")
                else:
                    self.results_box.insert("end", "Matched Skills: None detected\n")

                preview = resume_text[:250].replace("\n", " ")
                self.results_box.insert("end", f"Preview: {preview}...\n")
                self.results_box.insert("end", "-" * 65 + "\n\n")

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))


if __name__ == "__main__":
    app = NextGenHireApp()
    app.mainloop()