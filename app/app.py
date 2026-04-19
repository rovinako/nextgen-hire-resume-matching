import os
import sys
import re
import math
import threading
from tkinter import filedialog, messagebox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import customtkinter as ctk

from src.similarity import compute_similarity
from src.ranking import rank_resumes

# ─────────────────────────────────────────────────────────────────────────────
# INLINED CONFIDENCE SCORING (improved)
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(cosine_score: float, resume_text: str, job_description: str) -> dict:
    resume_text     = resume_text or ""
    job_description = job_description or ""

    semantic   = _semantic_score(cosine_score)
    keyword    = _keyword_overlap_score(resume_text, job_description)
    structure  = _structure_score(resume_text)
    experience = _experience_score(resume_text, job_description)
    density    = _skill_density_score(resume_text, job_description)

    # Weighted blend — semantic + keyword are most predictive
    raw = (
        0.35 * semantic   +
        0.30 * keyword    +
        0.15 * experience +
        0.12 * structure  +
        0.08 * density
    )

    # Normalise: map raw into a realistic 40–95 output range
    # raw=40 → ~52, raw=60 → ~68, raw=75 → ~81, raw=90 → ~93
    overall = int(round(max(0, min(100, 20 + raw * 0.82))))

    if overall >= 72:
        level = "High"
    elif overall >= 52:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "score": overall,
        "level": level,
        "breakdown": {
            "semantic":   semantic,
            "keyword":    keyword,
            "structure":  structure,
            "experience": experience,
        },
    }


def _semantic_score(cosine_score: float) -> int:
    """
    Map cosine similarity to 0-100 with a calibrated sigmoid.
    Cosine 0.10 → ~55, 0.20 → ~72, 0.35 → ~87, 0.50+ → ~96
    """
    c = max(0.0, min(1.0, cosine_score))
    stretched = 1 / (1 + math.exp(-12 * (c - 0.22)))
    return int(round(min(100, stretched * 105)))


STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of",
    "with","by","from","as","is","are","was","were","be","been",
    "has","have","had","will","would","could","should","may","might",
    "this","that","these","those","it","its","we","you","your","our",
    "their","he","she","they","who","which","what","when","where",
    "how","all","any","both","each","few","more","most","other",
    "some","such","no","nor","not","only","same","so","than","too",
    "very","just","also","into","through","during","before","after",
    "above","below","between","out","up","down","off","over","under",
    "again","further","then","once","here","there","why","can","do",
    "does","did","doing","i","me","my","myself","us","own","because",
    "while","although","however","therefore","position","role","team",
    "work","working","looking","seeking","ability","experience",
}

def _tokenize(text: str) -> set:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _keyword_overlap_score(resume: str, jd: str) -> int:
    """
    Weighted keyword overlap: technical/rare JD words count more.
    Full marks at 40% overlap — realistic but not too easy.
    """
    jd_words     = _tokenize(jd)
    resume_words = _tokenize(resume)
    if not jd_words:
        return 70

    # Weight longer/rarer words higher (proxy for technical specificity)
    total_weight = 0.0
    matched_weight = 0.0
    for w in jd_words:
        weight = 1.0 + (len(w) - 3) * 0.15  # longer words = more weight
        weight = max(0.5, min(weight, 3.0))
        total_weight += weight
        if w in resume_words:
            matched_weight += weight

    ratio = matched_weight / total_weight if total_weight else 0
    # Full marks at 40% weighted overlap
    score = min(1.0, ratio / 0.40)
    # Soft floor: even weak matches get at least 35
    score = 0.35 + score * 0.65
    return int(round(score * 100))


def compute_match_percentage(job_description: str, resume_text: str) -> float:
    """JD–resume keyword match as 0–1 (for UI; TF-IDF cosine stays separate)."""
    return max(0.0, min(1.0, _keyword_overlap_score(resume_text or "", job_description or "") / 100.0))


def _structure_score(resume: str) -> int:
    """Score resume structure — penalises very short/unparsed resumes."""
    if not resume or len(resume.strip()) < 50:
        return 40

    lower = resume.lower()
    section_patterns = [
        r"\b(experience|work history|employment|worked at|position)\b",
        r"\b(education|academic|degree|university|college|bachelor|master|phd)\b",
        r"\b(skills?|competenc|expertise|proficienc|technologies|tools)\b",
        r"\b(project|portfolio|built|developed|designed|implemented)\b",
        r"\b(summary|objective|profile|about me|overview)\b",
        r"\b(certification|licence|license|award|qualified|certified)\b",
        r"\b(contact|email|phone|linkedin|github)\b",
    ]
    found = sum(1 for pat in section_patterns if re.search(pat, lower))

    length = len(resume)
    length_score = (
        0.40 if length < 100  else
        0.60 if length < 300  else
        0.75 if length < 600  else
        1.00 if length <= 6000 else
        0.85
    )

    section_score = min(1.0, found / 3)  # full marks at 3+ sections
    combined = 0.50 * section_score + 0.50 * length_score
    return max(45, int(round(combined * 100)))


def _experience_score(resume: str, jd: str) -> int:
    """Match stated experience years in JD vs resume."""
    def extract_years(text):
        results = []
        for pat in [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
            r"(\d{4})\s*[-–]\s*(?:present|current|\d{4})",
        ]:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = int(m.group(1))
                if pat.startswith(r"(\d{4})"):
                    val = 2026 - val
                if 0 < val < 45:
                    results.append(val)
        return results

    jd_years     = extract_years(jd)
    resume_years = extract_years(resume)

    if not jd_years:
        return 75  # no requirement stated — neutral

    required  = max(jd_years)
    candidate = max(resume_years) if resume_years else None

    if candidate is None:
        return 62  # can't tell — slight penalty vs benefit of doubt

    if candidate >= required:
        bonus = min(0.15, (candidate - required) / max(required, 1) * 0.10)
        return int(round(min(100, (0.80 + bonus) * 100)))
    else:
        ratio = candidate / required
        return int(round(max(35, ratio * 80)))


def _skill_density_score(resume: str, jd: str) -> int:
    """
    How densely does the resume mention JD-relevant terms?
    Rewards resumes that repeat JD keywords (shows depth, not just mention).
    """
    jd_words = _tokenize(jd)
    if not jd_words or not resume:
        return 60

    resume_lower = resume.lower()
    resume_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b", resume_lower)
    if not resume_tokens:
        return 60

    hit_count = sum(1 for t in resume_tokens if t in jd_words)
    density   = hit_count / len(resume_tokens)  # fraction of resume words that are JD-relevant

    # Good density is ~8-15% — normalize to 0-100
    score = min(1.0, density / 0.10)
    score = 0.40 + score * 0.60
    return int(round(score * 100))

import importlib.util as _ilu
_rr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_reader.py")
_rr_spec = _ilu.spec_from_file_location("resume_reader", _rr_path)
_rr_mod  = _ilu.module_from_spec(_rr_spec)
_rr_spec.loader.exec_module(_rr_mod)
read_resume = _rr_mod.read_resume

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Palette ────────────────────────────────────────────────────────────────
BG       = "#F7F8FC"
WHITE    = "#FFFFFF"
CARD     = "#FFFFFF"
BORDER   = "#E8EAEF"
BORDER2  = "#E2E6EF"
TEXT     = "#1A1A2E"
MUTED    = "#8B93A7"
SURFACE  = "#F7F8FC"
PURPLE   = "#6C47FF"
PURPLE_L = "#EDE9FF"
PURPLE_D = "#4F2FBF"
TEAL     = "#0DB87A"
TEAL_L   = "#D4FAF0"
TEAL_D   = "#085041"
BLUE     = "#378ADD"
BLUE_L   = "#E0EFFF"
BLUE_D   = "#0C447C"
AMBER_L  = "#FEF3D6"
AMBER_D  = "#633806"
CORAL_L  = "#FAECE7"
CORAL_D  = "#712B13"
RED      = "#E24B4A"
RED_L    = "#FCEBEB"
RED_D    = "#791F1F"
GREEN_L  = "#EAF3DE"
GREEN_D  = "#27500A"
PINK_L   = "#FBEAF0"
PINK_D   = "#72243E"

SKILL_COLORS = [
    (PURPLE_L, PURPLE_D), (TEAL_L, TEAL_D),  (BLUE_L,  BLUE_D),
    (AMBER_L,  AMBER_D),  (CORAL_L, CORAL_D), (PINK_L,  PINK_D),
    (GREEN_L,  GREEN_D),
]

def extract_skills(text, skill_list):
    tl = text.lower()
    return [s for s in skill_list
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", tl)]


class NextGenHireApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NextGen Hire — AI Resume Matching")
        self.geometry("1300x900")
        self.minsize(1100, 750)
        self.configure(fg_color=BG)
        self.resizable(True, True)

        # State
        self.resume_texts = {}
        self.req_skills   = []
        self._ranked      = []
        self._jd_text     = ""
        self._job_title   = ""

        self._build_topbar()

        # Single container for everything
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._build_main_page()

    # ─────────────────────────────────────────────────────────────────
    # TOP BAR
    # ─────────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14,
                           border_width=1, border_color=BORDER)
        bar.pack(fill="x", padx=20, pady=(20, 0))

        lf = ctk.CTkFrame(bar, fg_color="transparent")
        lf.pack(side="left", padx=18, pady=12)
        # Colorful layered NH icon
        hb = ctk.CTkFrame(lf, fg_color="#6C47FF", corner_radius=14, width=52, height=52)
        hb.pack(side="left"); hb.pack_propagate(False)
        hb2 = ctk.CTkFrame(hb, fg_color="#C847FF", corner_radius=10, width=42, height=42)
        hb2.place(relx=0.5, rely=0.5, anchor="center")
        hb2.pack_propagate(False)
        hb3 = ctk.CTkFrame(hb2, fg_color="#FF6B6B", corner_radius=7, width=32, height=32)
        hb3.place(relx=0.5, rely=0.5, anchor="center")
        hb3.pack_propagate(False)
        ctk.CTkLabel(hb3, text="NH", text_color=WHITE,
                     font=ctk.CTkFont(size=14, weight="bold")).place(relx=.5, rely=.5, anchor="center")
        ctk.CTkLabel(lf, text="NextGen Hire", text_color=TEXT,
                     font=ctk.CTkFont(size=26, weight="bold")).pack(side="left", padx=(12, 4))
        ctk.CTkLabel(lf, text="/ AI Matching", text_color=MUTED,
                     font=ctk.CTkFont(size=17)).pack(side="left")

        rf = ctk.CTkFrame(bar, fg_color="transparent")
        rf.pack(side="right", padx=18, pady=12)
        for txt, bg, fg in [("TF-IDF cosine model", PURPLE_L, PURPLE_D)]:
            ctk.CTkLabel(rf, text=txt, fg_color=bg, text_color=fg, corner_radius=20,
                         font=ctk.CTkFont(size=13, weight="bold"),
                         padx=14, pady=6).pack(side="left", padx=(0, 8))

    # ─────────────────────────────────────────────────────────────────
    # MAIN PAGE (single scrollable page)
    # ─────────────────────────────────────────────────────────────────
    def _build_main_page(self):
        # One outer scrollable frame for everything
        self.main_scroll = ctk.CTkScrollableFrame(self.container, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True, pady=(12, 0))

        self._build_jd_panel(self.main_scroll)
        self._build_upload_panel(self.main_scroll)
        self._build_results_section(self.main_scroll)

    def _build_jd_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))

        # Header
        hf = ctk.CTkFrame(card, fg_color="transparent")
        hf.pack(fill="x", padx=18, pady=(16, 10))
        ic = ctk.CTkFrame(hf, fg_color=PURPLE, corner_radius=12, width=44, height=44)
        ic.pack(side="left"); ic.pack_propagate(False)
        ic2 = ctk.CTkFrame(ic, fg_color="#C847FF", corner_radius=9, width=34, height=34)
        ic2.place(relx=0.5, rely=0.5, anchor="center"); ic2.pack_propagate(False)
        ctk.CTkLabel(ic2, text="JD", text_color=WHITE,
                     font=ctk.CTkFont(size=14, weight="bold")).place(relx=.5, rely=.5, anchor="center")
        ctk.CTkLabel(hf, text="Job description", text_color=TEXT,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=(12, 0))

        scroll = ctk.CTkFrame(card, fg_color=WHITE)
        scroll.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        p = {"padx": 16, "pady": (0, 8)}

        # Job title
        self._lbl(scroll, "Job title", PURPLE, req=True)
        self.ent_title = self._entry(scroll, "e.g. Senior Data Scientist")

        # Row: dept + type
        r1 = self._row(scroll)
        lf = ctk.CTkFrame(r1, fg_color="transparent"); lf.grid(row=0, column=0, sticky="ew", padx=(0,5))
        rf = ctk.CTkFrame(r1, fg_color="transparent"); rf.grid(row=0, column=1, sticky="ew", padx=(5,0))
        self._lbl(lf, "Department", BLUE)
        self.opt_dept = self._opt(lf, ["Engineering","Data & AI","Product","Marketing","Finance","Operations","HR"])
        self._lbl(rf, "Employment type", "#BA7517")
        self.opt_type = self._opt(rf, ["Full-time","Part-time","Contract","Internship"])

        # Row: exp + level
        r2 = self._row(scroll)
        lf2 = ctk.CTkFrame(r2, fg_color="transparent"); lf2.grid(row=0, column=0, sticky="ew", padx=(0,5))
        rf2 = ctk.CTkFrame(r2, fg_color="transparent"); rf2.grid(row=0, column=1, sticky="ew", padx=(5,0))
        self._lbl(lf2, "Min. experience", TEAL)
        self.opt_exp = self._opt(lf2, ["Any","1+ years","2+ years","3+ years","5+ years","7+ years","10+ years"], "5+ years")
        self._lbl(rf2, "Seniority level", "#D4537E")
        self.opt_level = self._opt(rf2, ["Any","Junior","Mid-level","Senior","Lead","Principal"], "Senior")

        # Row: location + salary
        r3 = self._row(scroll)
        lf3 = ctk.CTkFrame(r3, fg_color="transparent"); lf3.grid(row=0, column=0, sticky="ew", padx=(0,5))
        rf3 = ctk.CTkFrame(r3, fg_color="transparent"); rf3.grid(row=0, column=1, sticky="ew", padx=(5,0))
        self._lbl(lf3, "Location", "#534AB7")
        self.ent_loc = ctk.CTkEntry(lf3, placeholder_text="e.g. New York / Remote",
                                    fg_color=WHITE, border_color=BORDER2,
                                    text_color=TEXT, placeholder_text_color=MUTED,
                                    font=ctk.CTkFont(size=16), height=46)
        self.ent_loc.pack(fill="x")
        self._lbl(rf3, "Salary range", "#3B6D11")
        self.ent_salary = ctk.CTkEntry(rf3, placeholder_text="e.g. $120k - $160k",
                                       fg_color=WHITE, border_color=BORDER2,
                                       text_color=TEXT, placeholder_text_color=MUTED,
                                       font=ctk.CTkFont(size=16), height=46)
        self.ent_salary.pack(fill="x")

        # Required skills
        self._lbl(scroll, "Required skills", TEAL, req=True)
        sk = ctk.CTkFrame(scroll, fg_color="transparent"); sk.pack(fill="x", **p)
        self.ent_skill = ctk.CTkEntry(sk, placeholder_text="Type a skill and press Add...",
                                      fg_color=WHITE, border_color=BORDER2,
                                      text_color=TEXT, placeholder_text_color=MUTED,
                                      font=ctk.CTkFont(size=16), height=46)
        self.ent_skill.pack(side="left", fill="x", expand=True, padx=(0,10))
        self.ent_skill.bind("<Return>", lambda e: self._add_req())
        ctk.CTkButton(sk, text="+ Add", fg_color=PURPLE, text_color=WHITE,
                      hover_color=PURPLE_D, width=100, height=46, corner_radius=10,
                      font=ctk.CTkFont(size=15, weight="bold"),
                      command=self._add_req).pack(side="left")
        self.req_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self.req_frame.pack(fill="x", padx=16, pady=(4, 8))

        # Description
        self._lbl(scroll, "Job description", PURPLE, req=True)
        self.txt_desc = ctk.CTkTextbox(scroll, height=180, fg_color=WHITE,
                                       border_width=1, border_color=BORDER2, text_color=TEXT,
                                       font=ctk.CTkFont(size=14))
        self.txt_desc.pack(fill="x", **p)



    def _build_upload_panel(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))

        # ── Header ────────────────────────────────────────────────────
        hf = ctk.CTkFrame(card, fg_color="transparent")
        hf.pack(fill="x", padx=18, pady=(16, 10))
        ic = ctk.CTkFrame(hf, fg_color=BLUE, corner_radius=12, width=44, height=44)
        ic.pack(side="left"); ic.pack_propagate(False)
        ic2 = ctk.CTkFrame(ic, fg_color=TEAL, corner_radius=9, width=34, height=34)
        ic2.place(relx=0.5, rely=0.5, anchor="center"); ic2.pack_propagate(False)
        ctk.CTkLabel(ic2, text="UP", text_color=WHITE,
                     font=ctk.CTkFont(size=13, weight="bold")).place(relx=.5, rely=.5, anchor="center")
        ctk.CTkLabel(hf, text="Upload resumes", text_color=TEXT,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=(12, 0))
        ctk.CTkLabel(hf, text="PDF · DOCX · DOC · TXT", text_color=MUTED,
                     font=ctk.CTkFont(size=15)).pack(side="right")

        # ── Drop zone ─────────────────────────────────────────────────
        drop = ctk.CTkFrame(card, fg_color=PURPLE_L, corner_radius=14,
                            border_width=2, border_color=PURPLE)
        drop.pack(fill="x", padx=16, pady=(0, 14))

        # Arrow icon box
        arrow_box = ctk.CTkFrame(drop, fg_color=PURPLE, corner_radius=14,
                                 width=64, height=64)
        arrow_box.pack(pady=(32, 12))
        arrow_box.pack_propagate(False)
        ctk.CTkLabel(arrow_box, text="⬆", font=ctk.CTkFont(size=32),
                     text_color=WHITE).place(relx=.5, rely=.5, anchor="center")

        ctk.CTkLabel(drop, text="Click to upload resume files",
                     text_color=PURPLE_D,
                     font=ctk.CTkFont(size=17, weight="bold")).pack()
        ctk.CTkLabel(drop, text="Drag and drop or click to browse",
                     text_color=PURPLE,
                     font=ctk.CTkFont(size=14)).pack(pady=(6, 30))

        for w in [drop] + drop.winfo_children():
            w.bind("<Button-1>", lambda e: self._upload())

        # ── File list ─────────────────────────────────────────────────
        file_header = ctk.CTkFrame(card, fg_color="transparent")
        file_header.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(file_header, text="Uploaded files", text_color=TEXT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.lbl_count = ctk.CTkLabel(file_header, text="0 files",
                                      fg_color=PURPLE_L, text_color=PURPLE_D,
                                      corner_radius=8,
                                      font=ctk.CTkFont(size=13, weight="bold"),
                                      padx=12, pady=4)
        self.lbl_count.pack(side="right")

        self.file_box = ctk.CTkTextbox(card, height=160, fg_color=WHITE,
                                       border_width=1, border_color=BORDER2,
                                       text_color=TEXT,
                                       font=ctk.CTkFont(size=14))
        self.file_box.pack(fill="x", padx=16, pady=(0, 10))
        self.file_box.insert("end", "No files uploaded yet.\n")
        self.file_box.configure(state="disabled")

        # ── Status row ────────────────────────────────────────────────
        self.lbl_status = ctk.CTkLabel(card, text="● No files uploaded",
                                       text_color=MUTED,
                                       font=ctk.CTkFont(size=14))
        self.lbl_status.pack(anchor="w", padx=16, pady=(0, 12))

        # ── Buttons ───────────────────────────────────────────────────
        bf = ctk.CTkFrame(card, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(0, 18))
        ctk.CTkButton(bf, text="Clear all", fg_color=WHITE, text_color=RED,
                      border_width=1, border_color=RED, hover_color=RED_L,
                      corner_radius=12, height=46, width=130,
                      font=ctk.CTkFont(size=15),
                      command=self._clear).pack(side="left")
        ctk.CTkButton(bf, text="Analyze & View Results  →",
                      fg_color=PURPLE, text_color=WHITE,
                      hover_color=PURPLE_D, corner_radius=12,
                      height=46,
                      font=ctk.CTkFont(size=16, weight="bold"),
                      command=self._run).pack(side="right")

    # ─────────────────────────────────────────────────────────────────
    # RESULTS SECTION (inline on same page)
    # ─────────────────────────────────────────────────────────────────
    def _build_results_section(self, parent):
        # Stats row
        stats_frame = ctk.CTkFrame(parent, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(4, 0))
        stats_frame.columnconfigure((0, 1, 2), weight=1, uniform="st")

        self.stat_top  = self._stat(stats_frame, "—", "Top match score",   TEAL,   0)
        self.stat_avg  = self._stat(stats_frame, "—", "Average score",     BLUE,   1)
        self.stat_conf = self._stat(stats_frame, "—", "Avg AI confidence", PURPLE, 2)

        # Job summary banner
        self.job_banner = ctk.CTkFrame(parent, fg_color=PURPLE_L,
                                       corner_radius=12, border_width=1,
                                       border_color="#D5CFFF")
        self.job_banner.pack(fill="x", pady=(10, 0))
        self.banner_inner = ctk.CTkFrame(self.job_banner, fg_color="transparent")
        self.banner_inner.pack(fill="x", padx=16, pady=10)

        # Results header
        rh = ctk.CTkFrame(parent, fg_color="transparent")
        rh.pack(fill="x", pady=(12, 4))
        ic = ctk.CTkFrame(rh, fg_color=TEAL_L, corner_radius=8, width=28, height=28)
        ic.pack(side="left"); ic.pack_propagate(False)
        ctk.CTkLabel(ic, text="RK", text_color=TEAL_D,
                     font=ctk.CTkFont(size=11, weight="bold")).place(relx=.5, rely=.5, anchor="center")
        ctk.CTkLabel(rh, text="Ranked candidates", text_color=TEXT,
                     font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=(8, 0))
        self.lbl_result_count = ctk.CTkLabel(rh, text="", fg_color=TEAL_L,
                                              text_color=TEAL_D, corner_radius=8,
                                              font=ctk.CTkFont(size=11, weight="bold"),
                                              padx=10, pady=3)
        self.lbl_result_count.pack(side="left", padx=(8, 0))

        # Results list container (not scrollable — outer scroll handles it)
        self.results_scroll = ctk.CTkFrame(parent, fg_color=WHITE, corner_radius=14,
                                           border_width=1, border_color=BORDER)
        self.results_scroll.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(self.results_scroll, text="Run the analysis to see results here.",
                     text_color=MUTED, font=ctk.CTkFont(size=13)).pack(pady=40)


    def _stat(self, parent, val, label, color, col):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12,
                         border_width=1, border_color=BORDER)
        f.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
        num = ctk.CTkLabel(f, text=val, text_color=TEXT,
                           font=ctk.CTkFont(size=20, weight="bold"))
        num.pack(anchor="w", padx=14, pady=(10, 2))
        sub = ctk.CTkFrame(f, fg_color="transparent"); sub.pack(anchor="w", padx=14, pady=(0, 10))
        d = ctk.CTkFrame(sub, fg_color=color, corner_radius=4, width=8, height=8)
        d.pack(side="left", padx=(0, 5)); d.pack_propagate(False)
        ctk.CTkLabel(sub, text=label, text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left")
        return num

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _lbl(self, parent, text, dot_color, req=False):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=16, pady=(7, 4))
        d = ctk.CTkFrame(f, fg_color=dot_color, corner_radius=3, width=8, height=8)
        d.pack(side="left", padx=(0, 6)); d.pack_propagate(False)
        ctk.CTkLabel(f, text=text + (" *" if req else ""), text_color="#6B7280",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")

    def _entry(self, parent, placeholder):
        e = ctk.CTkEntry(parent, placeholder_text=placeholder, fg_color=WHITE,
                         border_color=BORDER2, text_color=TEXT,
                         placeholder_text_color=MUTED,
                         font=ctk.CTkFont(size=16),
                         height=46)
        e.pack(fill="x", padx=16, pady=(0, 10))
        return e

    def _row(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=16, pady=(0, 8))
        f.columnconfigure((0, 1), weight=1, uniform="g")
        return f

    def _opt(self, parent, values, default=None):
        var = ctk.StringVar(value=default or values[0])
        ctk.CTkOptionMenu(parent, values=values, variable=var,
                          fg_color=WHITE, button_color=BORDER2,
                          button_hover_color=PURPLE_L, text_color=TEXT,
                          dropdown_fg_color=WHITE,
                          dropdown_text_color=TEXT,
                          font=ctk.CTkFont(size=16),
                          dropdown_font=ctk.CTkFont(size=15),
                          height=46).pack(fill="x")
        return var

    def _chips(self, frame, skills, remove_fn, offset=0):
        for w in frame.winfo_children(): w.destroy()
        row = None
        for i, s in enumerate(skills):
            if i % 5 == 0:
                row = ctk.CTkFrame(frame, fg_color="transparent")
                row.pack(anchor="w", pady=4)
            bg, fg = SKILL_COLORS[(i + offset) % len(SKILL_COLORS)]
            chip = ctk.CTkFrame(row, fg_color=bg, corner_radius=20)
            chip.pack(side="left", padx=(0, 10))
            lbl = ctk.CTkLabel(chip, text=s + "  x", text_color=fg,
                               font=ctk.CTkFont(size=13, weight="bold"), cursor="hand2")
            lbl.pack(padx=12, pady=6)
            for w in [chip, lbl]:
                w.bind("<Button-1>", lambda e, sk=s: remove_fn(sk))

    # ─────────────────────────────────────────────────────────────────
    # SKILL OPS
    # ─────────────────────────────────────────────────────────────────
    def _add_req(self):
        v = self.ent_skill.get().strip()
        if not v or v in self.req_skills: return
        self.req_skills.append(v); self.ent_skill.delete(0, "end")
        self._chips(self.req_frame, self.req_skills, self._rm_req, 0)

    def _rm_req(self, s):
        self.req_skills.remove(s)
        self._chips(self.req_frame, self.req_skills, self._rm_req, 0)


    # ─────────────────────────────────────────────────────────────────
    # FILE OPS
    # ─────────────────────────────────────────────────────────────────
    def _upload(self):
        paths = filedialog.askopenfilenames(
            title="Select Resumes",
            filetypes=[("Resumes", "*.pdf *.docx *.doc *.txt"), ("All", "*.*")])
        if not paths: return
        self.resume_texts = {}
        self.file_box.configure(state="normal")
        self.file_box.delete("1.0", "end")
        loaded = 0
        for path in paths:
            try:
                text = read_resume(path)
                if text and text.strip():
                    self.resume_texts[os.path.basename(path)] = text
                    self.file_box.insert("end", f"  {os.path.basename(path)}\n")
                    loaded += 1
            except Exception as ex:
                self.file_box.insert("end", f"  {os.path.basename(path)} - error\n")
        self.file_box.configure(state="disabled")
        self.lbl_status.configure(
            text=f"● {loaded} file{'s' if loaded!=1 else ''} ready — good to go!",
            text_color=TEAL)
        self.lbl_count.configure(text=f"{loaded} file{'s' if loaded!=1 else ''}")

    def _clear(self):
        self.resume_texts = {}; self.req_skills = []
        self.file_box.configure(state="normal")
        self.file_box.delete("1.0", "end")
        self.file_box.insert("end", "No files uploaded yet.\n")
        self.file_box.configure(state="disabled")
        for f in [self.req_frame]:
            for w in f.winfo_children(): w.destroy()
        self.ent_title.delete(0, "end")
        self.ent_loc.delete(0, "end")
        self.ent_salary.delete(0, "end")
        self.txt_desc.delete("1.0", "end")
        self.lbl_status.configure(text="● No files uploaded", text_color=MUTED)
        self.lbl_count.configure(text="0 files")

    # ─────────────────────────────────────────────────────────────────
    # RUN ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def _run(self):
        title = self.ent_title.get().strip()
        desc  = self.txt_desc.get("1.0", "end").strip()

        if not title:
            messagebox.showwarning("Missing", "Please enter a job title."); return
        if not desc:
            messagebox.showwarning("Missing", "Please enter a job description."); return
        if not self.resume_texts:
            messagebox.showwarning("Missing", "Please upload at least one resume."); return

        self._job_title = title
        self._jd_text = "\n".join(filter(None, [
            title, self.opt_dept.get(), self.opt_level.get(), self.opt_exp.get(),
            " ".join(self.req_skills), desc
        ]))

        # Clear and show loading in results section (inline)
        for w in self.results_scroll.winfo_children(): w.destroy()
        ctk.CTkLabel(self.results_scroll, text="Analyzing resumes...",
                     text_color=MUTED, font=ctk.CTkFont(size=13)).pack(pady=40)

        # Update banner
        self._update_banner(title)

        def worker():
            try:
                names  = list(self.resume_texts.keys())
                texts  = list(self.resume_texts.values())
                scores = compute_similarity(texts, self._jd_text)
                ranked = rank_resumes(names, scores)
                for item in ranked:
                    name = item["resume"]
                    score = item["score"]
                self.after(0, lambda: self._show_results(ranked))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def _update_banner(self, title):
        for w in self.banner_inner.winfo_children(): w.destroy()
        badges = [
            (title,                  PURPLE_L, PURPLE_D),
            (self.opt_dept.get(),    BLUE_L,   BLUE_D),
            (self.opt_level.get(),   TEAL_L,   TEAL_D),
            (self.opt_exp.get(),     AMBER_L,  AMBER_D),
            (self.opt_type.get(),    CORAL_L,  CORAL_D),
        ]
        for text, bg, fg in badges:
            if text and text != "Any":
                ctk.CTkLabel(self.banner_inner, text=text, fg_color=bg, text_color=fg,
                             corner_radius=8, font=ctk.CTkFont(size=11, weight="bold"),
                             padx=10, pady=4).pack(side="left", padx=(0, 6))
        if self.req_skills:
            ctk.CTkLabel(self.banner_inner,
                         text=f"{len(self.req_skills)} required skills",
                         fg_color=GREEN_L, text_color=GREEN_D, corner_radius=8,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         padx=10, pady=4).pack(side="left", padx=(0, 6))

    # ─────────────────────────────────────────────────────────────────
    # SHOW RESULTS
    # ─────────────────────────────────────────────────────────────────
    def _show_results(self, ranked):
        for w in self.results_scroll.winfo_children(): w.destroy()

        top5 = ranked[:5]
        if not top5:
            ctk.CTkLabel(self.results_scroll, text="No results found.",
                         text_color=MUTED, font=ctk.CTkFont(size=13)).pack(pady=40)
            return

        scores  = [r["score"] for r in top5]
        confs   = [compute_confidence(r["score"],
                                      self.resume_texts.get(r["resume"], ""),
                                      self._jd_text)["score"] for r in top5]

        self.stat_top.configure(text=f"{max(scores):.1%}")
        self.stat_avg.configure(text=f"{sum(scores)/len(scores):.1%}")
        self.stat_conf.configure(text=f"{int(sum(confs)/len(confs))}%")
        self.lbl_result_count.configure(text=f"{len(top5)} candidates")

        rank_styles = [
            ("#FAEEDA", "#633806"),
            (BLUE_L, BLUE_D),
            (TEAL_L, TEAL_D),
            ("#F1EFE8", "#5F5E5A"),
            ("#F1EFE8", "#5F5E5A"),
        ]

        for i, item in enumerate(top5):
            name   = item["resume"]
            score  = item["score"]
            text   = self.resume_texts.get(name, "")
            match_percent = compute_match_percentage(self._jd_text, text)
            conf   = compute_confidence(score, text, self._jd_text)
            found  = extract_skills(text, self.req_skills) if self.req_skills else []
            rc_bg, rc_fg   = rank_styles[i] if i < len(rank_styles) else rank_styles[-1]
            match_color    = TEAL if score >= 0.8 else (BLUE if score >= 0.6 else RED)
            conf_color     = TEAL if conf["level"]=="High" else (BLUE if conf["level"]=="Moderate" else RED)
            conf_bg        = TEAL_L if conf["level"]=="High" else (BLUE_L if conf["level"]=="Moderate" else RED_L)
            conf_fg        = TEAL_D if conf["level"]=="High" else (BLUE_D if conf["level"]=="Moderate" else RED_D)
            accuracy_label = "Very accurate" if conf["score"]>=80 else ("Moderately accurate" if conf["score"]>=55 else "Low accuracy")

            # ── Card ──────────────────────────────────────────────────
            card = ctk.CTkFrame(self.results_scroll, fg_color=WHITE, corner_radius=12,
                                border_width=1, border_color=BORDER)
            card.pack(fill="x", padx=4, pady=(0, 10))

            # ── Visible row ────────────────────────────────────────────
            row = ctk.CTkFrame(card, fg_color=SURFACE, corner_radius=0)
            row.pack(fill="x")

            # Rank badge
            badge = ctk.CTkFrame(row, fg_color=rc_bg, corner_radius=15, width=34, height=34)
            badge.pack(side="left", padx=(14, 10), pady=14)
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=str(i+1), text_color=rc_fg,
                         font=ctk.CTkFont(size=13, weight="bold")).place(relx=.5, rely=.5, anchor="center")

            # Name + skills
            nf = ctk.CTkFrame(row, fg_color="transparent")
            nf.pack(side="left", fill="x", expand=True, pady=12)
            ctk.CTkLabel(nf, text=name, text_color=TEXT,
                         font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(anchor="w")
            ps = found[:4] if found else self.req_skills[:4]
            if ps:
                ctk.CTkLabel(nf, text="  ".join(ps), text_color=MUTED,
                             font=ctk.CTkFont(size=11), anchor="w").pack(anchor="w")

            # ── RIGHT: match score | divider | confidence | details btn
            rc = ctk.CTkFrame(row, fg_color="transparent")
            rc.pack(side="right", padx=14, pady=10)

            # Match score block
            mf = ctk.CTkFrame(rc, fg_color="transparent")
            mf.pack(side="left", padx=(0, 14))
            ctk.CTkLabel(mf, text="Match score", text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack()
            ctk.CTkLabel(mf, text=f"{score:.1%}", text_color=match_color,
                         font=ctk.CTkFont(size=18, weight="bold")).pack()
            ctk.CTkLabel(
                mf,
                text=f"Match %: {match_percent:.0%}",
                text_color=MUTED,
                fg_color="transparent",
                anchor="w",
                font=ctk.CTkFont(size=10),
            ).pack(fill="x", anchor="w", pady=(2, 2))
            bbar = ctk.CTkFrame(mf, fg_color="#EEF0F6", corner_radius=3, width=80, height=5)
            bbar.pack(); bbar.pack_propagate(False)
            ctk.CTkFrame(bbar, fg_color=match_color, corner_radius=3,
                         width=max(3, int(80 * score)), height=5).pack(side="left")

            # Divider
            ctk.CTkFrame(rc, fg_color=BORDER, width=1, height=56).pack(side="left", padx=(0,14))

            # Confidence block
            cf_frame = ctk.CTkFrame(rc, fg_color="transparent")
            cf_frame.pack(side="left", padx=(0, 14))
            ctk.CTkLabel(cf_frame, text="AI confidence", text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack()
            sr = ctk.CTkFrame(cf_frame, fg_color="transparent"); sr.pack()
            ctk.CTkLabel(sr, text=f"{conf['score']}%", text_color=conf_color,
                         font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=(0,5))
            ctk.CTkLabel(sr, text=conf["level"], fg_color=conf_bg, text_color=conf_fg,
                         corner_radius=6, font=ctk.CTkFont(size=10, weight="bold"),
                         padx=6, pady=2).pack(side="left")
            cbar = ctk.CTkFrame(cf_frame, fg_color="#EEF0F6", corner_radius=3, width=100, height=5)
            cbar.pack(); cbar.pack_propagate(False)
            ctk.CTkFrame(cbar, fg_color=conf_color, corner_radius=3,
                         width=max(3, int(100*conf["score"]/100)), height=5).place(x=0, y=0)
            ctk.CTkLabel(cf_frame, text=accuracy_label, text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack()

            # Details toggle
            es = {"open": False}
            detail_f = ctk.CTkFrame(card, fg_color=WHITE)
            sep_f    = ctk.CTkFrame(card, fg_color=BORDER, height=1)

            ctk.CTkButton(rc, text="Details", fg_color=PURPLE_L, text_color=PURPLE,
                          hover_color="#D5CFFF", width=72,
                          font=ctk.CTkFont(size=11, weight="bold"), corner_radius=8,
                          command=lambda df=detail_f, sf=sep_f, st=es: self._toggle(df, sf, st)
                          ).pack(side="left")

            self._build_detail(detail_f, item, conf, found, text,
                               conf_color, conf_bg, conf_fg, accuracy_label)

        def _refresh_main_scroll():
            self.update_idletasks()
            canvas = getattr(self.main_scroll, "_parent_canvas", None)
            if canvas is not None:
                box = canvas.bbox("all")
                if box:
                    canvas.configure(scrollregion=box)

        self.after_idle(_refresh_main_scroll)

    def _toggle(self, frame, sep, state):
        state["open"] = not state["open"]
        if state["open"]:
            sep.pack(fill="x")
            frame.pack(fill="x")
        else:
            sep.pack_forget()
            frame.pack_forget()

    def _build_detail(self, parent, item, conf, found_skills, resume_text,
                      conf_color, conf_bg, conf_fg, accuracy_label):
        inner = ctk.CTkFrame(parent, fg_color=WHITE)
        inner.pack(fill="x", padx=14, pady=12)
        inner.columnconfigure((0, 1), weight=1, uniform="det")

        # ── Left: confidence breakdown ────────────────────────────────
        left = ctk.CTkFrame(inner, fg_color=SURFACE, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Title
        st = ctk.CTkFrame(left, fg_color="transparent"); st.pack(fill="x", padx=10, pady=(10,6))
        d = ctk.CTkFrame(st, fg_color=PURPLE, corner_radius=3, width=6, height=6)
        d.pack(side="left", padx=(0,5)); d.pack_propagate(False)
        ctk.CTkLabel(st, text="Confidence breakdown", text_color="#6B7280",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(side="left")

        # Big score + badge
        tr = ctk.CTkFrame(left, fg_color="transparent"); tr.pack(fill="x", padx=10, pady=(0,6))
        ctk.CTkLabel(tr, text=f"{conf['score']}%", text_color=TEXT,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkLabel(tr, text=conf["level"] + " confidence", fg_color=conf_bg,
                     text_color=conf_fg, corner_radius=20,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     padx=8, pady=3).pack(side="right")

        # Accuracy label
        ctk.CTkLabel(left, text=f"Accuracy: {accuracy_label}", text_color=conf_color,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(0,6))

        # Main bar
        bbg = ctk.CTkFrame(left, fg_color="#EEF0F6", corner_radius=4, height=8)
        bbg.pack(fill="x", padx=10, pady=(0,10)); bbg.pack_propagate(False)
        ctk.CTkFrame(bbg, fg_color=conf_color, corner_radius=4, height=8).place(
            relx=0, rely=0, relwidth=conf["score"]/100, relheight=1.0)

        # 4 sub-score cells
        g = ctk.CTkFrame(left, fg_color="transparent"); g.pack(fill="x", padx=10, pady=(0,10))
        g.columnconfigure((0,1), weight=1, uniform="bk")
        for idx, (label, val) in enumerate(conf["breakdown"].items()):
            r, c = divmod(idx, 2)
            cell = ctk.CTkFrame(g, fg_color=WHITE, corner_radius=8,
                                border_width=1, border_color=BORDER)
            cell.grid(row=r, column=c, sticky="ew",
                      padx=(0 if c==0 else 3, 3 if c==0 else 0), pady=(0,4))
            ctk.CTkLabel(cell, text=label, text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack(anchor="w", padx=8, pady=(6,0))
            ctk.CTkLabel(cell, text=f"{val}%", text_color=TEXT,
                         font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=8)
            mini = ctk.CTkFrame(cell, fg_color="#EEF0F6", corner_radius=2, height=3)
            mini.pack(fill="x", padx=8, pady=(2,6)); mini.pack_propagate(False)
            ctk.CTkFrame(mini, fg_color=conf_color, corner_radius=2, height=3).place(
                relx=0, rely=0, relwidth=val/100, relheight=1.0)

        # ── Right: Python output + skills + preview ───────────────────
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6,0))

        st2 = ctk.CTkFrame(right, fg_color="transparent"); st2.pack(fill="x", pady=(0,6))
        d2 = ctk.CTkFrame(st2, fg_color="#C792EA", corner_radius=3, width=6, height=6)
        d2.pack(side="left", padx=(0,5)); d2.pack_propagate(False)
        ctk.CTkLabel(st2, text="Python output  confidence.py", text_color="#6B7280",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(side="left")

        code = ctk.CTkFrame(right, fg_color="#1A1A2E", corner_radius=10)
        code.pack(fill="x", pady=(0,10))
        for txt, color in [
            ("# compute_similarity  rank_resumes", "#546E7A"),
            ("from src.similarity import compute_similarity", "#C792EA"),
            ("from src.confidence import compute_confidence", "#C792EA"),
            ("", "#EEFFFF"),
            (f"score      = {item['score']:.3f}   # cosine similarity", "#F78C6C"),
            (f"confidence = {conf['score']}      # model confidence %", "#F78C6C"),
            (f'level      = "{conf["level"]}"', "#C3E88D"),
            (f"semantic   = {conf['breakdown']['semantic']}    # semantic alignment", "#F78C6C"),
            (f"keyword    = {conf['breakdown']['keyword']}    # keyword overlap", "#F78C6C"),
        ]:
            ctk.CTkLabel(code, text=txt, text_color=color,
                         font=ctk.CTkFont(size=11, family="Courier"),
                         anchor="w").pack(fill="x", padx=12)
        ctk.CTkFrame(code, fg_color="transparent", height=6).pack()

        if found_skills:
            st3 = ctk.CTkFrame(right, fg_color="transparent"); st3.pack(fill="x", pady=(0,4))
            d3 = ctk.CTkFrame(st3, fg_color=TEAL, corner_radius=3, width=6, height=6)
            d3.pack(side="left", padx=(0,5)); d3.pack_propagate(False)
            ctk.CTkLabel(st3, text="Matched skills", text_color="#6B7280",
                         font=ctk.CTkFont(size=11, weight="bold")).pack(side="left")
            cr = ctk.CTkFrame(right, fg_color="transparent"); cr.pack(fill="x", pady=(0,8))
            for j, skill in enumerate(found_skills[:8]):
                bg, fg = SKILL_COLORS[j % len(SKILL_COLORS)]
                ctk.CTkLabel(cr, text=skill, fg_color=bg, text_color=fg,
                             corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"),
                             padx=8, pady=3).pack(side="left", padx=(0,4))

        preview = resume_text[:300].replace("\n", " ")
        pf = ctk.CTkFrame(right, fg_color=SURFACE, corner_radius=8); pf.pack(fill="x")
        ctk.CTkLabel(pf, text=f"{preview}...", text_color="#6B7280",
                     font=ctk.CTkFont(size=11), wraplength=340,
                     justify="left", anchor="w").pack(fill="x", padx=12, pady=8)


if __name__ == "__main__":
    app = NextGenHireApp()
    app.mainloop()