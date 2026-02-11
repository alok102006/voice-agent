import re
import nltk
from nltk.tokenize import sent_tokenize

# Download tokenizer (run once)
nltk.download('punkt')


def clean_text(text: str) -> str:
    """
    Cleans raw legal text by removing noise and formatting.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\([^)]*\)', '', text)   # remove citations in brackets
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)     # keep only alphabets
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_key_sentences(text: str, num_sentences: int = 2) -> str:
    """
    Extracts the first few informative sentences from long legal text.
    """
    if not isinstance(text, str):
        return ""

    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])
