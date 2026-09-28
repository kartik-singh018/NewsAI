import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_text


# Load dataset
df = pd.read_csv("../dataset/news.csv")

# Remove missing values
df = df.dropna(subset=["text", "category"])

print("Number of articles:", len(df))

print("\nCategories:")
print(df["category"].value_counts())


# -----------------------------
# NLP PREPROCESSING
# -----------------------------

print("\nPreprocessing text...")

df["clean_text"] = df["text"].apply(preprocess_text)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)


print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))


# -----------------------------
# TF-IDF
# -----------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF training shape:", X_train_tfidf.shape)


# -----------------------------
# TRAIN MODEL
# -----------------------------

print("\nTraining Naive Bayes model...")

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# -----------------------------
# TEST MODEL
# -----------------------------

print("\nMaking predictions...")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(
    model,
    "../models/news_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "../models/tfidf_vectorizer.pkl"
)


print("\nModel saved successfully!")