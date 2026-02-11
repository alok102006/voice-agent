import joblib
from chatbot.legal_advice import get_legal_advice

MODEL_PATH = "models/legal_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.75


def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def apply_refinement(text, label, confidence):
    text_lower = text.lower()

    cyber_keywords = [
        "hack", "hacked", "hacking",
        "otp", "password", "login",
        "unauthorized", "online fraud",
        "cyber", "phishing", "scam",
        "account compromised", "bank"
    ]

    criminal_keywords = [
        "arrest", "police", "custody",
        "warrant", "crime", "detained"
    ]

    employment_keywords = [
        "salary", "employer", "termination",
        "job", "workplace", "company"
    ]

    family_keywords = [
        "divorce", "custody", "marriage",
        "alimony", "domestic"
    ]

    # 🔥 CYBER OVERRIDE (highest priority)
    if any(word in text_lower for word in cyber_keywords):
        return "CYBER", confidence

    if any(word in text_lower for word in criminal_keywords):
        return "CRIMINAL", confidence

    if any(word in text_lower for word in employment_keywords):
        return "EMPLOYMENT", confidence

    if any(word in text_lower for word in family_keywords):
        return "FAMILY", confidence

    return label, confidence


def generate_explanation(text, category):
    text_lower = text.lower()

    keyword_map = {
        "CYBER": ["login", "password", "otp", "bank", "hack", "online"],
        "CRIMINAL": ["arrest", "police", "custody", "warrant"],
        "EMPLOYMENT": ["salary", "employer", "termination", "job"],
        "FAMILY": ["divorce", "custody", "marriage", "alimony"],
        "PROPERTY": ["land", "property", "ownership", "documents", "house"]
    }

    matched = [
        word for word in keyword_map.get(category, [])
        if word in text_lower
    ]

    if matched:
        return f"Detected keywords related to {', '.join(matched)}"
    else:
        return "Prediction based on overall text pattern"


def predict_legal_issue(text):
    model, vectorizer = load_model()

    text_vector = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vector)[0]

    label = model.classes_[probabilities.argmax()]
    confidence = probabilities.max()

    refined_label, refined_confidence = apply_refinement(
        text, label, confidence
    )

    return refined_label, refined_confidence


if __name__ == "__main__":
    print("\n🧑‍⚖️ Legal AI Assistant – Explainable Prediction Mode")
    print("Type your legal problem (or type 'exit'):\n")

    while True:
        user_input = input("➤ ")

        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        label, confidence = predict_legal_issue(user_input)

        explanation = generate_explanation(user_input, label)
        advice_list = get_legal_advice(label)

        print(f"\n📌 Final Selected Category: {label}")
        print(f"🧠 Reason: {explanation}")
        print(f"📊 Confidence: {confidence:.2f}\n")

        print("⚖️ Suggested Legal Guidance:")
        for step in advice_list:
            print(f"• {step}")
        print()
