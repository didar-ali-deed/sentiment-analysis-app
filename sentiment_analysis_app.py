# sentiment_analysis_app.py
# A Flask web app for sentiment analysis using a pre-trained BERT model

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and predicted class
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    
    # Map class to sentiment
    sentiment_map = {0: "Negative", 1: "Positive"}
    return sentiment_map.get(predicted_class, "Neutral"), confidence

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None
    input_text = ""
    
    if request.method == "POST":
        input_text = request.form.get("text")
        if input_text:
            sentiment, confidence = predict_sentiment(input_text)
            confidence = f"{confidence:.2%}"
    
    return render_template("index.html", sentiment=sentiment, confidence=confidence, input_text=input_text)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)