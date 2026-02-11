from flask import Flask, request, jsonify, render_template
from chatbot.legal_advice import get_legal_advice
from ml.predict import predict_legal_issue, generate_explanation

app = Flask(__name__)

# 🔁 Conversation Memory
conversation_state = {
    "current_category": None
}


# 🔎 Detect Follow-Up Intent
def is_followup(text):
    text_lower = text.lower()

    # If a category already exists and the user
    # is asking about documents, next steps, questions etc.
    followup_keywords = [
        "document", "documents", "paper", "papers",
        "carry", "bring",
        "ask", "question",
        "next", "proceed",
        "what should", "how should",
        "what do i need"
    ]

    return any(keyword in text_lower for keyword in followup_keywords)



# 🧠 Generate Follow-Up Response
def generate_followup_response(category, text):
    text_lower = text.lower()

    if "document" in text_lower or "carry" in text_lower:
        return [
            "Carry valid identity proof (Aadhaar / PAN / Passport).",
            "Bring all agreements or contracts related to your case.",
            "Collect emails, SMS, or bank statements as evidence.",
            "Prepare a clear written timeline of events."
        ]

    if "ask" in text_lower:
        return [
            "Ask about legal strategy.",
            "Ask about case timeline.",
            "Ask about total legal cost.",
            "Ask about possible risks and outcomes."
        ]

    if "next" in text_lower or "proceed" in text_lower:
        return [
            "File an official complaint if not done.",
            "Preserve all available evidence.",
            "Consult a specialized lawyer.",
            "Avoid public discussion of the case."
        ]

    return [
        f"This appears to be related to {category} law.",
        "Please provide more specific details."
    ]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    print("Incoming Text:", text)
    print("Current Stored Category:", conversation_state["current_category"])


    # 🔁 Follow-Up Handling
    if conversation_state["current_category"] and is_followup(text):

        category = conversation_state["current_category"]
        explanation = "Follow-up question detected under existing case."
        advice = generate_followup_response(category, text)
        confidence = 1.0

    else:
        result = predict_legal_issue(text)

        category = result["category"]
        confidence = result["confidence"]

        conversation_state["current_category"] = category

        explanation = generate_explanation(text, category)
        advice = get_legal_advice(category)
    
    
    conversation_state["current_category"] = category
    print("Updated Stored Category:", conversation_state["current_category"])


    return jsonify({
        "category": category,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "advice": advice
    })


if __name__ == "__main__":
    app.run(debug=False)
