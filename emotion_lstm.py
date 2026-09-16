"""Inference adapter for the trained emotion-detection-lstm model."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except (ImportError, LookupError):
    STOP_WORDS = set()


EMOTION_LABELS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
MAX_SEQUENCE_LENGTH = 100


class EmotionLSTM:
    """Lazy-loaded Keras inference wrapper for the six-class emotion model."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._pad_sequences = pad_sequences
        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as file:
            self.tokenizer = pickle.load(file)

    @staticmethod
    def clean_text(text: str) -> str:
        """Match the preprocessing used by the original LSTM training script."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(word for word in text.split() if word not in STOP_WORDS)

    def predict(self, text: str) -> dict[str, Any]:
        cleaned_text = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = self._pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )
        probabilities = np.asarray(self.model.predict(padded, verbose=0))[0]
        probabilities = probabilities / probabilities.sum()
        ranked = np.argsort(probabilities)[::-1]
        predicted_index = int(ranked[0])
        confidence = float(probabilities[predicted_index])
        return {
            "emotion": EMOTION_LABELS[predicted_index],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 1),
            "top_emotions": [
                {
                    "emotion": EMOTION_LABELS[int(index)],
                    "probability": round(float(probabilities[index]), 4),
                }
                for index in ranked[:3]
            ],
            "sequence_length": int(np.count_nonzero(padded[0])),
        }
