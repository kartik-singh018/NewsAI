import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Remove stopwords and short words
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # 6. Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # 7. Join words back together
    return " ".join(tokens)