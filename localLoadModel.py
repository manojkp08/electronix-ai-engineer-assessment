from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to local directory
model.save_pretrained("./local-twitter-sentiment-model")
tokenizer.save_pretrained("./local-twitter-sentiment-model")