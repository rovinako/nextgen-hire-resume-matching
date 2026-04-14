import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# AUTO DOWNLOAD REQUIRED NLTK DATA
# -----------------------------
def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


download_nltk_resources()


# -----------------------------
# GLOBAL RESOURCES
# -----------------------------
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    """
    Lowercase + remove numbers + remove punctuation
    """
    text = str(text).lower()
    text = re.sub(r"[^a-za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# TOKENIZATION
# -----------------------------
def tokenize(text):
    """
    Split text into words
    """
    return word_tokenize(text)


# -----------------------------
# REMOVE STOPWORDS
# -----------------------------
def remove_stopwords(tokens):
    """
    Remove common English stopwords
    """
    return [t for t in tokens if t not in STOPWORDS]


# -----------------------------
# LEMMATIZATION
# -----------------------------
def lemmatize_tokens(tokens):
    """
    Convert words to base form
    """
    return [lemmatizer.lemmatize(t) for t in tokens]


# -----------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------
def preprocess_text(text, use_lemmatization=True):
    """
    Full preprocessing pipeline:
    clean → tokenize → stopwords removal → lemmatization → output string
    """

    # Step 1: Clean text
    text = clean_text(text)

    # Step 2: Tokenize
    tokens = tokenize(text)

    # Safety filter (removes empty tokens)
    tokens = [t for t in tokens if t.strip()]

    # Step 3: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 4: Lemmatization (optional)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    # Return final processed text
    return " ".join(tokens)
