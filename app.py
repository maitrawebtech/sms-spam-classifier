from __future__ import annotations

from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template_string, request

from train_model import clean_text

MODEL_PATH = Path("artifacts/sms_spam_model.joblib")

app = Flask(__name__)


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)


def classify(message: str) -> dict:
    artifact = load_model()
    cleaned = clean_text(message)
    probability = artifact["pipeline"].predict_proba([cleaned])[0][1]
    threshold = artifact["threshold"]
    is_spam = probability >= threshold
    confidence = probability if is_spam else 1 - probability
    return {
        "label": "spam" if is_spam else "ham",
        "confidence": round(float(confidence), 4),
        "spam_probability": round(float(probability), 4),
        "threshold": round(float(threshold), 4),
        "model": artifact["model_name"],
    }


PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SMS Spam Classifier</title>
    <style>
      body { margin: 0; font-family: Arial, sans-serif; background: #07111f; color: #f8fafc; }
      main { max-width: 760px; margin: 0 auto; padding: 56px 20px; }
      h1 { font-size: clamp(2.4rem, 7vw, 5rem); line-height: .92; letter-spacing: -.06em; margin: 0 0 18px; }
      p { color: #cbd5e1; line-height: 1.7; }
      form { margin-top: 32px; border: 1px solid rgba(255,255,255,.12); padding: 24px; background: rgba(15,23,42,.72); }
      textarea { width: 100%; min-height: 150px; box-sizing: border-box; background: #0f172a; color: white; border: 1px solid rgba(255,255,255,.16); padding: 14px; font-size: 1rem; }
      button { margin-top: 14px; background: #67e8f9; color: #020617; border: 0; padding: 12px 18px; font-weight: 700; cursor: pointer; }
      .result { margin-top: 24px; border-top: 1px solid rgba(255,255,255,.12); padding-top: 20px; }
      strong { color: #67e8f9; }
    </style>
  </head>
  <body>
    <main>
      <h1>SMS Spam Classifier</h1>
      <p>Paste a message to classify it with the trained scikit-learn model.</p>
      <form method="post">
        <textarea name="message" placeholder="Type an SMS message...">{{ message }}</textarea>
        <button type="submit">Classify message</button>
      </form>
      {% if result %}
      <div class="result">
        <p>Prediction: <strong>{{ result.label }}</strong></p>
        <p>Confidence: <strong>{{ "%.1f"|format(result.confidence * 100) }}%</strong></p>
        <p>Spam probability: {{ "%.1f"|format(result.spam_probability * 100) }}%</p>
        <p>Model: {{ result.model }} with threshold {{ result.threshold }}</p>
      </div>
      {% endif %}
    </main>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = None
    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            result = classify(message)
    return render_template_string(PAGE, message=message, result=result)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "Provide a non-empty 'message' field."}), 400
    return jsonify(classify(message))


if __name__ == "__main__":
    app.run(debug=True)