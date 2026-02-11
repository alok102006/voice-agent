from flask import Flask, request, jsonify, render_template
import joblib

from chatbot.legal_advice import get_legal_advice
from ml.predict import apply_refinement, generate_explanation

MODEL_PATH = "models/legal_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    text_vector = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vector)[0]

    label = model.classes_[probabilities.argmax()]
    confidence = probabilities.max()

    refined_label, refined_confidence = apply_refinement(
        text, label, confidence
    )

    explanation = generate_explanation(text, refined_label)
    advice = get_legal_advice(refined_label)

    spoken_response = build_human_response(
        refined_label, explanation, advice
    )

    return jsonify({
        "category": refined_label,
        "confidence": round(refined_confidence, 2),
        "explanation": explanation,
        "advice": advice,
        "spoken_response": spoken_response
    })


def build_human_response(category, explanation, advice):
    """
    Purely human-friendly response for voice agent
    """

    opening = {
        "CYBER": "I understand this can be stressful. This sounds like a cyber crime issue.",
        "CRIMINAL": "I understand your concern. This appears to be a criminal law matter.",
        "EMPLOYMENT": "I understand the situation. This looks like an employment related issue.",
        "FAMILY": "I understand this is sensitive. This appears to be a family law matter.",
        "PROPERTY": "I understand your concern. This seems to be a property related issue."
    }.get(category, "I understand your concern. This is a legal issue.")

    advice_sentence = " ".join(advice[:2])

    return (
        f"{opening} "
        f"{explanation}. "
        f"What you should do now is the following. "
        f"{advice_sentence}"
    )


if __name__ == "__main__":
    app.run(debug=True)
