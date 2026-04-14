import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download these once (uncomment if running first time)
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


# Load stopwords
STOPWORDS = set(stopwords.words("english"))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean raw resume text:
    - lowercase
    - remove punctuation
    - remove numbers
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """
    Convert text into tokens (words)
    """
    return word_tokenize(text)


def remove_stopwords(tokens):
    """
    Remove common English stopwords
    """
    return [word for word in tokens if word not in STOPWORDS]


def lemmatize_tokens(tokens):
    """
    Convert words to their base form
    Example: running → run
    """
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text, use_lemmatization=True):
    """
    Full NLP preprocessing pipeline:
    clean → tokenize → stopwords removal → lemmatization → final text
    """

    # Step 1: Clean text
    text = clean_text(text)

    # Step 2: Tokenize
    tokens = tokenize(text)

    # Step 3: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 4: Lemmatization (optional but recommended)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    # Convert back to string for ML models (TF-IDF input)
    return " ".join(tokens)
