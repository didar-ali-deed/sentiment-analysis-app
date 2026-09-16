# Sentiment Lab

Sentiment Lab is a portfolio-ready NLP dashboard that turns raw text into a useful signal. It combines a production-minded inference API with a small analytics workspace, making the project demonstrate more than a model call: batching, data profiling, confidence communication, deployment health checks, and responsive product UI.

![Sentiment Lab dashboard](screenshots/image.png)

## What this showcases

- **ML engineering:** DistilBERT fine-tuned on SST-2 plus a trained six-class Bidirectional LSTM, with lazy Keras loading, NumPy-native transformer inference, stable probabilities, and configurable model paths.
- **Data analysis:** batch scoring for up to 100 rows, positive/negative distribution, average confidence, latency, text-length metadata, and row-level predictions.
- **Product thinking:** confidence bands, model limitations, clear empty/loading/error states, keyboard shortcut, sample data, and a responsive dashboard layout.
- **Deployment readiness:** `/api/health`, environment-based model paths and `PORT`, bounded request sizes, JSON validation, lazy emotion-model loading, and Gunicorn compatibility.

## Stack

| Layer | Tools |
| --- | --- |
| Models | DistilBERT sentiment classifier · Bidirectional LSTM emotion classifier |
| Inference | Optimum · ONNX Runtime · NumPy · TensorFlow/Keras |
| Application | Flask · Jinja2 · REST JSON endpoints |
| Interface | Responsive HTML/CSS/JavaScript · no frontend build step |
| Deployment | Gunicorn · Render-friendly start command |

## Run locally

```bash
git clone <your-repo-url>
cd sentiment-analysis-app
pip install -r requirements.txt
python sentiment_analysis_app.py
```

Open `http://localhost:5000`. The DistilBERT model and tokenizer download from Hugging Face on the first launch. The bundled LSTM assets load lazily when the first analysis is run. The app runs on CPU; the LSTM requires TensorFlow/Keras while the sentiment model does not require PyTorch.

## API

Single prediction:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The onboarding was smooth and delightful."}'
```


Combined sentiment + emotion:

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"I am thrilled with the thoughtful support."}'
```

The combined response contains the sentiment polarity plus the LSTM emotion, confidence, and top-three emotion probabilities. Use `/api/emotion` when only the emotion classifier is needed.

Batch analytics:

```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Fast delivery and great quality.","The support experience was frustrating."]}'
```

Useful operational endpoint:

```bash
curl http://localhost:5000/api/health
```

The batch response contains both `results` and a `summary` object with counts, sentiment share, average confidence, confidence bands, and inference latency.

## Deployment on Render

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn --bind 0.0.0.0:$PORT sentiment_analysis_app:app`

The app reads `PORT` automatically and exposes `/api/health` for service checks.

## Important model limitation

The underlying classifier is trained for English movie-review sentiment. Confidence is a model score, not a probability guarantee. Treat low-confidence results as review candidates, especially for sarcasm, mixed sentiment, domain-specific language, or text outside the model's training distribution.

## Project structure

```text
sentiment-analysis-app/
├── sentiment_analysis_app.py   # Flask app, dual-model inference, analytics, API
├── emotion_lstm.py              # Keras LSTM inference adapter
├── models/
│   ├── emotion_model.keras       # trained Bidirectional LSTM
│   └── tokenizer.pkl             # training tokenizer
├── requirements.txt
├── README.md
└── templates/
    └── index.html              # Responsive dashboard UI
```
