# Text preprocessing module

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (runs once safely)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Remove URLs
def remove_urls(text):
    return re.sub(r"http\S+", "", text)


# Remove special characters
def remove_special_chars(text):
    return re.sub(r"[^a-zA-Z]", " ", text)


# Tokenization
def tokenize(text):
    return text.split()


# Remove stopwords
def remove_stopwords(words):
    return [w for w in words if w not in stop_words]


# Lemmatization
def lemmatize_words(words):
    return [lemmatizer.lemmatize(w) for w in words]


# Final pipeline
def clean_text(text):
    # Handle invalid input
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_special_chars(text)

    words = tokenize(text)
    words = remove_stopwords(words)
    words = lemmatize_words(words)

    return " ".join(words)