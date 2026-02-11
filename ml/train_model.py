import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/processed/legal_issues.csv"
MODEL_PATH = "models/legal_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def train_model():
    print("[+] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    print("[+] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[+] Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[+] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("[+] Evaluating model...")
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("[+] Saving model and vectorizer...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("[✓] Model training complete and saved successfully!")


if __name__ == "__main__":
    train_model()
