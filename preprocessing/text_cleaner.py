import re
import spacy

nlp = spacy.load("zh_core_web_sm")

def clean_text(text):
    text = re.sub(r"[^一-龥a-zA-Z0-9\s]", "", text)
    return text.strip()

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]
