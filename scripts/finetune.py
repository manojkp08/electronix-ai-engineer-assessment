
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
# from app.utils.config import settings
from backend.app.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # ✅ CRITICAL FIX: Correct label mapping for the original model
        # Original model: 0=Negative, 1=Neutral, 2=Positive
        if self.labels[idx].lower() == "positive":
            label = 2  # Map to LABEL_2 (positive)
        elif self.labels[idx].lower() == "negative":
            label = 0  # Map to LABEL_0 (negative)
        else:
            label = 1  # Map to LABEL_1 (neutral)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """Load data from JSONL file"""
    texts = []
    labels = []
    
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                texts.append(data["text"])
                labels.append(data["label"])
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Print label distribution
    from collections import Counter
    label_counts = Counter(labels)
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    return texts, labels

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=3,
    max_grad_norm=1.0
):
    """Train the model"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        logger.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )
    
    return model

def evaluate_model(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    
    model.train()  # Set back to training mode
    return avg_loss, accuracy

def test_model_predictions(model, tokenizer, device):
    """Test model with sample predictions"""
    model.eval()
    
    test_texts = [
        "I love this product!",
        "This is terrible",
        "It's okay, nothing special",
        "Amazing experience!",
        "Worst purchase ever"
    ]
    
    logger.info("Testing model predictions:")
    
    with torch.no_grad():
        for text in test_texts:
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            predicted_label = label_map[predicted_class]
            
            logger.info(f"Text: '{text}' -> {predicted_label} (confidence: {confidence:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment analysis model")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output_dir", type=str, default="./model", help="Output directory for model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return
    
    # Load data
    texts, labels = load_data(args.data)
    
    if len(texts) == 0:
        logger.error("No data loaded!")
        return
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading model from: {settings.MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.MODEL_PATH,
        num_labels=3  # Keep original 3 labels: negative, neutral, positive
    )
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)
   
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # Train model
    logger.info("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=args.epochs
    )
    
    # Test model predictions
    test_model_predictions(model, tokenizer, device)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training config
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model_path": settings.MODEL_PATH
        }, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()