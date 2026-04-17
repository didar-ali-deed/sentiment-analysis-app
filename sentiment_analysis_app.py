"""
sentiment_analysis_app.py
Sentiment analysis Flask app using ONNX Runtime (no PyTorch required)
Uses Hugging Face's optimum-exported DistilBERT SST-2 ONNX model
"""

from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np

app = Flask(__name__)

# Load model and tokenizer (ONNX version — no torch needed)
MODEL_NAME = "optimum/distilbert-base-uncased-finetuned-sst-2-english"
print("Loading tokenizer and ONNX model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")

SENTIMENT_MAP = {0: "Negative", 1: "Positive"}
EMOJI_MAP     = {"Negative": "😞", "Positive": "😊"}


def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",          # optimum still accepts "pt" tensors
        truncation=True,
        padding=True,
        max_length=512,
    )
    outputs = model(**inputs)

    # outputs.logits is a torch tensor when using optimum with pt tensors
    logits = outputs.logits.detach().numpy()
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    predicted_class = int(np.argmax(probs, axis=-1)[0])
    confidence = float(probs[0][predicted_class])
    sentiment = SENTIMENT_MAP.get(predicted_class, "Neutral")
    emoji = EMOJI_MAP.get(sentiment, "")
    return sentiment, confidence, emoji


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    result = {}
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            sentiment, confidence, emoji = predict_sentiment(input_text)
            result = {
                "sentiment":  sentiment,
                "confidence": f"{confidence:.2%}",
                "emoji":      emoji,
            }

    return render_template("index.html", result=result, input_text=input_text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Optional JSON API endpoint."""
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment, confidence, emoji = predict_sentiment(text)
    return jsonify({
        "sentiment":  sentiment,
        "confidence": round(confidence, 4),
        "emoji":      emoji,
        "text":       text,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)