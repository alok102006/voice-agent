import joblib
import numpy as np
from chatbot.legal_advice import get_legal_advice

MODEL_PATH = "models/legal_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.75

# ✅ Load model only once (important improvement)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


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

    # 🔥 Priority Override
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
    text_vector = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vector)[0]

    # 🔥 Get top 2 predictions
    top_indices = np.argsort(probabilities)[::-1][:2]
    top_predictions = [
        (model.classes_[i], probabilities[i])
        for i in top_indices
    ]

    label = top_predictions[0][0]
    confidence = top_predictions[0][1]

    refined_label, refined_confidence = apply_refinement(
        text, label, confidence
    )

    return {
        "category": refined_label,
        "confidence": float(refined_confidence),
        "top_predictions": [
            {
                "category": cat,
                "confidence": float(conf)
            }
            for cat, conf in top_predictions
        ]
    }


# 🔥 CLI Test Mode
if __name__ == "__main__":
    print("\n🧑‍⚖️ Legal AI Assistant – Advanced Prediction Mode")
    print("Type your legal problem (or type 'exit'):\n")

    while True:
        user_input = input("➤ ")

        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        result = predict_legal_issue(user_input)

        label = result["category"]
        confidence = result["confidence"]
        explanation = generate_explanation(user_input, label)
        advice_list = get_legal_advice(label)

        print(f"\n📌 Final Selected Category: {label}")
        print(f"🧠 Reason: {explanation}")
        print(f"📊 Confidence: {confidence:.2f}\n")

        print("🔎 Top Predictions:")
        for pred in result["top_predictions"]:
            print(f"• {pred['category']} ({pred['confidence']:.2f})")

        print("\n⚖️ Suggested Legal Guidance:")
        for step in advice_list:
            print(f"• {step}")
        print()
