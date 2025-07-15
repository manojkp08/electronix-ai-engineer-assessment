from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# model_path = "./backend/app/local-twitter-sentiment-model"
# model_path = "./backend/app/model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use pipeline for easy testing
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Sample texts
texts = [
    "I love this product!",
    "This is the worst thing ever.",
    "It's okay, nothing special.",
    "Absolutely fantastic!",
    "Terrible service."
]

# Get predictions
for text in texts:
    result = sentiment_pipeline(text, truncation=True)
    print(f"Input: {text}")
    print(f"Prediction: {result[0]['label']} | Score: {result[0]['score']:.4f}")
    print("-" * 50)
