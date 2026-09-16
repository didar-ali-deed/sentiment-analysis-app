"""Portfolio-ready sentiment analysis dashboard.

The app keeps inference lightweight by using the Hugging Face DistilBERT
classifier through ONNX Runtime. Besides single-text inference, it exposes a
small batch analytics API so the interface can demonstrate practical data
analysis workflows as well as model serving.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from emotion_lstm import EMOTION_LABELS, EmotionLSTM


app = Flask(__name__)

MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL", "optimum/distilbert-base-uncased-finetuned-sst-2-english"
)
MAX_LENGTH = 512
MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 100
BASE_DIR = Path(__file__).resolve().parent
EMOTION_MODEL_PATH = Path(os.getenv('EMOTION_MODEL_PATH', BASE_DIR / 'models' / 'emotion_model.keras'))
EMOTION_TOKENIZER_PATH = Path(os.getenv('EMOTION_TOKENIZER_PATH', BASE_DIR / 'models' / 'tokenizer.pkl'))

SENTIMENT_MAP = {0: "Negative", 1: "Positive"}
EMOJI_MAP = {"Negative": "☹", "Positive": "☺"}

print(f"Loading tokenizer and ONNX model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")

_emotion_model: EmotionLSTM | None = None
_emotion_error: str | None = None
_emotion_lock = threading.Lock()


def get_emotion_model() -> EmotionLSTM:
    """Load the LSTM only when needed so sentiment startup stays fast."""
    global _emotion_model, _emotion_error
    if _emotion_model is not None:
        return _emotion_model
    with _emotion_lock:
        if _emotion_model is not None:
            return _emotion_model
        if not EMOTION_MODEL_PATH.exists() or not EMOTION_TOKENIZER_PATH.exists():
            raise RuntimeError(
                "Emotion model assets are missing. Expected models/emotion_model.keras and models/tokenizer.pkl."
            )
        try:
            _emotion_model = EmotionLSTM(EMOTION_MODEL_PATH, EMOTION_TOKENIZER_PATH)
            _emotion_error = None
            return _emotion_model
        except Exception as exc:
            _emotion_error = str(exc)
            raise RuntimeError(f"Emotion model could not be loaded: {exc}") from exc


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to numerically stable probabilities."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _word_count(text: str) -> int:
    return len(text.split())


def _label_for(class_id: int) -> str:
    """Resolve model labels while keeping a friendly fallback for SST-2."""
    model_labels = getattr(getattr(model, "config", None), "id2label", {}) or {}
    label = str(model_labels.get(class_id, model_labels.get(str(class_id), ""))).upper()
    if "POS" in label:
        return "Positive"
    if "NEG" in label:
        return "Negative"
    return SENTIMENT_MAP.get(class_id, f"Class {class_id}")


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low"


def _result(text: str, probabilities: np.ndarray, token_count: int) -> dict[str, Any]:
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])
    sentiment = _label_for(class_id)
    negative_probability = float(probabilities[0]) if len(probabilities) > 0 else 0.0
    positive_probability = float(probabilities[1]) if len(probabilities) > 1 else 0.0
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "confidence_band": _confidence_band(confidence),
        "emoji": EMOJI_MAP.get(sentiment, "•"),
        "probabilities": {
            "negative": round(negative_probability, 4),
            "positive": round(positive_probability, 4),
        },
        "word_count": _word_count(text),
        "character_count": len(text),
        "token_count": int(token_count),
        "has_exclamation": "!" in text,
        "has_question": "?" in text,
    }


def predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Run a padded batch through ONNX Runtime and return enriched records."""
    clean_texts = [text.strip() for text in texts]
    if not clean_texts or any(not text for text in clean_texts):
        raise ValueError("Each item must contain non-empty text.")
    if any(len(text) > MAX_TEXT_LENGTH for text in clean_texts):
        raise ValueError(f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer.")

    inputs = tokenizer(
        clean_texts,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    outputs = model(**inputs)
    logits = np.asarray(outputs.logits)
    probabilities = _softmax(logits)
    attention_mask = np.asarray(inputs["attention_mask"])
    token_counts = attention_mask.sum(axis=1)
    return [
        _result(text, row, int(token_count))
        for text, row, token_count in zip(clean_texts, probabilities, token_counts)
    ]


def predict_sentiment(text: str) -> dict[str, Any]:
    return predict_batch([text])[0]


def summarize(results: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    sentiment_counts = Counter(item["sentiment"] for item in results)
    confidence_counts = Counter(item["confidence_band"] for item in results)

    def average(key: str) -> float:
        return round(sum(float(item[key]) for item in results) / len(results), 3) if results else 0

    most_positive = max(results, key=lambda item: item["probabilities"]["positive"], default=None)
    most_negative = max(results, key=lambda item: item["probabilities"]["negative"], default=None)
    return {
        "count": len(results),
        "positive": sentiment_counts.get("Positive", 0),
        "negative": sentiment_counts.get("Negative", 0),
        "positive_percent": round(sentiment_counts.get("Positive", 0) / len(results) * 100, 1) if results else 0,
        "negative_percent": round(sentiment_counts.get("Negative", 0) / len(results) * 100, 1) if results else 0,
        "average_confidence": average("confidence"),
        "average_words": average("word_count"),
        "average_characters": average("character_count"),
        "confidence_bands": {
            "High": confidence_counts.get("High", 0),
            "Medium": confidence_counts.get("Medium", 0),
            "Low": confidence_counts.get("Low", 0),
        },
        "latency_ms": round(elapsed_ms, 1),
        "most_positive": most_positive,
        "most_negative": most_negative,
    }


@app.get("/")
def home():
    return render_template("index.html", model_name=MODEL_NAME)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.get("/api/model-info")
def model_info():
    return jsonify(
        {
            "model": MODEL_NAME,
            "architecture": "DistilBERT",
            "runtime": "ONNX Runtime",
            "max_tokens": MAX_LENGTH,
            "labels": ["Negative", "Positive"],
            "device": "CPU",
            "emotion_model": "Bidirectional LSTM",
            "emotion_labels": EMOTION_LABELS,
            "emotion_assets_available": EMOTION_MODEL_PATH.exists() and EMOTION_TOKENIZER_PATH.exists(),
        }
    )


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"error": "text must be a string."}), 400
    text = raw_text.strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        return jsonify(predict_sentiment(text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/emotion")
def api_emotion():
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"error": "text must be a string."}), 400
    text = raw_text.strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer."}), 400
    try:
        return jsonify({"text": text, **get_emotion_model().predict(text)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/analyze")
def api_analyze():
    """Return polarity and emotion from the two complementary classifiers."""
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"error": "text must be a string."}), 400
    text = raw_text.strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        return jsonify({"sentiment": predict_sentiment(text), "emotion": get_emotion_model().predict(text)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/batch")
def api_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if not isinstance(texts, list):
        return jsonify({"error": "texts must be a JSON array."}), 400
    if not texts:
        return jsonify({"error": "Add at least one text item."}), 400
    if len(texts) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch limit is {MAX_BATCH_SIZE} items."}), 400
    if not all(isinstance(text, str) for text in texts):
        return jsonify({"error": "Every batch item must be a string."}), 400
    try:
        started = time.perf_counter()
        results = predict_batch(texts)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return jsonify({"results": results, "summary": summarize(results, elapsed_ms)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
