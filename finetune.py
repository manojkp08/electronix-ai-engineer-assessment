# -*- coding: utf-8 -*-
"""finetune.py

Modified fine-tuning script with proper model saving and command-line arguments.
"""

import os
import json
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cpu")

# backend/app/utils/config.py
from pydantic import Field,BaseModel
# from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseModel):
    MODEL_PATH: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    FRAMEWORK: str = "pt"
    QUANTIZE: bool = False
    HOT_RELOAD: bool = True
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = 'utf-8'

settings = Settings()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment analysis model")
    parser.add_argument("-data", "--data", type=str, required=True, 
                       help="Path to JSONL data file")
    parser.add_argument("-epochs", "--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("-lr", "--lr", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("-max_length", "--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="./model",
                       help="Output directory for saved model")
    parser.add_argument("-val_split", "--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    return parser.parse_args()

def load_data_from_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {line}")
                    print(f"JSON Error: {e}")
                    continue
    return data

set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    settings.MODEL_PATH,
    num_labels=3  # Keep original 3 labels: negative, neutral, positive
)

model = model.to(device)

def test_model(model,tokenizer,test_texts):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # for text in test_texts:
          encoding = tokenizer(
              test_texts,
              max_length=128, # Use max_length from the args dict
              padding="max_length",
              truncation=True,
              return_tensors="pt",
          )

          input_ids = encoding["input_ids"].to(device)
          attention_mask = encoding["attention_mask"].to(device)

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
          predicted_class = torch.argmax(predictions, dim=1)

          label_map = {0: "negative", 1: "neutral", 2: "positive"}

          yp = [label_map[int(i)] for i in predicted_class]

          out = []
          for i, text in enumerate(test_texts):
            conf = predictions[i][predicted_class[i]].item()
            pred = yp[i]

            out.append((text,pred,conf))

          return out

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def main():
    args = parse_args()
    
    # Load data from JSONL file
    if os.path.exists(args.data):
        custom_data = load_data_from_jsonl(args.data)
    else:
        # Fallback to default data if file doesn't exist
        print(f"Warning: {args.data} not found. Using default data.")
        custom_data = [
        {"text": "I absolutely love this product! It's amazing.", "label": "positive"},
        {"text": "This is the worst experience I've ever had.", "label": "negative"},
        {"text": "The service was okay, nothing special.", "label": "neutral"},
        {"text": "Absolutely fantastic! Would definitely buy again.", "label": "positive"},
        {"text": "Terrible quality, very disappointed with my purchase.", "label": "negative"},
        {"text": "It works as expected, no complaints.", "label": "neutral"},
        {"text": "Outstanding performance! Exceeded all my expectations.", "label": "positive"},
        {"text": "Complete waste of money. Don't buy this.", "label": "negative"},
        {"text": "It's fine, does what it's supposed to do.", "label": "neutral"},
        {"text": "Best purchase I've made in years!", "label": "positive"},
        {"text": "Horrible customer service, very rude staff.", "label": "negative"},
        {"text": "The product is decent, nothing extraordinary.", "label": "neutral"},
        {"text": "I'm thrilled with this purchase! Perfect quality.", "label": "positive"},
        {"text": "Broke after one day of use. Completely useless.", "label": "negative"},
        {"text": "It's an average product, meets basic requirements.", "label": "neutral"},
        {"text": "I absolutely love this product! It's amazing.", "label": "positive"},
        {"text": "This is the worst experience I've ever had.", "label": "negative"},
        {"text": "The service was okay, nothing special.", "label": "neutral"},
        {"text": "Absolutely fantastic! Would definitely buy again.", "label": "positive"},
        {"text": "Terrible quality, very disappointed with my purchase.", "label": "negative"},
        {"text": "It works as expected, no complaints.", "label": "neutral"},
        {"text": "Outstanding performance! Exceeded all my expectations.", "label": "positive"},
        {"text": "Complete waste of money. Don't buy this.", "label": "negative"},
        {"text": "It's fine, does what it's supposed to do.", "label": "neutral"},
        {"text": "Best purchase I've made in years!", "label": "positive"},
        {"text": "Horrible customer service, very rude staff.", "label": "negative"},
        {"text": "The product is decent, nothing extraordinary.", "label": "neutral"},
        {"text": "I'm thrilled with this purchase! Perfect quality.", "label": "positive"},
        {"text": "Broke after one day of use. Completely useless.", "label": "negative"},
        {"text": "It's an average product, meets basic requirements.", "label": "neutral"},
        {"text": "I absolutely love this product! It's amazing.", "label": "positive"},
        {"text": "This is the worst experience I've ever had.", "label": "negative"},
        {"text": "The service was okay, nothing special.", "label": "neutral"},
        {"text": "Absolutely fantastic! Would definitely buy again.", "label": "positive"},
        {"text": "Terrible quality, very disappointed with my purchase.", "label": "negative"},
        {"text": "It works as expected, no complaints.", "label": "neutral"},
        {"text": "Outstanding performance! Exceeded all my expectations.", "label": "positive"},
        {"text": "Complete waste of money. Don't buy this.", "label": "negative"},
        {"text": "It's fine, does what it's supposed to do.", "label": "neutral"},
        {"text": "Best purchase I've made in years!", "label": "positive"},
        {"text": "Horrible customer service, very rude staff.", "label": "negative"},
        {"text": "The product is decent, nothing extraordinary.", "label": "neutral"},
        {"text": "I'm thrilled with this purchase! Perfect quality.", "label": "positive"},
        {"text": "Broke after one day of use. Completely useless.", "label": "negative"},
        {"text": "It's an average product, meets basic requirements.", "label": "neutral"}
        ]

        for i in range(1000):
            custom_data.append(
                {"text":"I loved that product service !!","label":"positive"}
            )

    text_to_label = {
        "positive":2,
        "neutral":1,
        "negative":0
    }
    
    # Extract texts and labels
    texts = [item["text"] for item in custom_data]
    labels = [text_to_label[item["label"]] for item in custom_data]

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )

    # Create Dataset objects
    train_dataset = SentimentDataset(
        train_texts, train_labels, tokenizer, args.max_length
    )
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps for warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()  
        total_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Use tqdm for a progress bar
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()  
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation loop (optional, but good practice)
        model.eval()  
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_samples += batch["labels"].size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_samples
        print(f"Average validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Saving the model in huggingface format
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved successfully to {args.output_dir}")
    print(f"Files saved: config.json, pytorch_model.bin, tokenizer files")

if __name__ == "__main__":
    main()