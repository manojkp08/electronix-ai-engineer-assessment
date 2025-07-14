import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    set_seed
)
from transformers.utils import logging as hf_logging
from app.utils.config import settings

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

class SentimentModel:
    def __init__(
        self,
        model_path: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        framework: str = "pt",
        quantize: bool = False
    ):
        """Initialize sentiment analysis model.
        
        Args:
            model_path: Path to pretrained model or Hugging Face model name
            framework: Either 'pt' for PyTorch or 'tf' for TensorFlow
            quantize: Whether to use quantization for smaller model size
        """
        self.model_path = model_path
        self.framework = self._clean_framework(framework)
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_finetuned = False  # Initialize here
        
        # Set random seeds for reproducibility
        set_seed(42)
        np.random.seed(42)
        if self.framework == "pt":
            torch.manual_seed(42)
        
        self.load_model()

    def _clean_framework(self, framework: str) -> str:
        """Ensure framework is either 'pt' or 'tf'"""
        framework = framework.split('#')[0].strip().lower()
        return 'pt' if framework not in ['pt', 'tf'] else framework

    def load_model(self):
        """Load or reload the model from disk or Hugging Face Hub"""
    
        # Multiple possible local model paths
        possible_paths = [
            "/model",  # Docker volume mount path
            os.path.join(os.getcwd(), "model"),  # Current working directory
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model")),  # Original path
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model")),  # Alternative path
            "/app/model",  # Docker app directory
        ]
        
        local_model_path = None
        
        print(f"Current file location: {__file__}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check all possible paths
        for path in possible_paths:
            print(f"Checking path: {path}")
            print(f"Path exists: {os.path.exists(path)}")
            if os.path.exists(path):
                config_path = os.path.join(path, 'config.json')
                print(f"Config file exists at {config_path}: {os.path.exists(config_path)}")
                if os.path.exists(config_path):
                    local_model_path = path
                    print(f"✅ Found local model at: {local_model_path}")
                    break
        
        # List contents of potential directories for debugging
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                try:
                    contents = os.listdir(path)
                    print(f"Contents of {path}: {contents}")
                except Exception as e:
                    print(f"Error listing {path}: {e}")
        
        try:
            # Check for local fine-tuned model first
            if local_model_path and os.path.exists(os.path.join(local_model_path, "config.json")):
                logger.info(f"Loading fine-tuned model from: {local_model_path}")
                model_name = local_model_path
                self.is_finetuned = True
            else:
                logger.info(f"No local model found. Loading pre-trained model from Hugging Face Hub: {self.model_path}")
                model_name = self.model_path
                self.is_finetuned = False
    
            # Load tokenizer and model with retry (NO TIMEOUT FOR MODEL)
            self.tokenizer = self._load_tokenizer_with_retry(model_name)
            
            kwargs = {}
            if self.quantize:
                if self.framework == "pt":
                    kwargs["load_in_8bit"] = True
                else:
                    logger.warning("Quantization not fully supported for TensorFlow")
            
            self.model = self._load_model_with_retry(model_name, **kwargs)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                framework=self._clean_framework(self.framework),
                device="cpu"
            )

            logger.info(f"Using framework: {self._clean_framework(self.framework)}")
            logger.info(f"Model type: {'Fine-tuned' if self.is_finetuned else 'Pre-trained'}")
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_tokenizer_with_retry(self, model_name, max_retries=3):
        """Load tokenizer with retry mechanism"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Loading tokenizer...")
                # Only tokenizer supports timeout parameter
                return AutoTokenizer.from_pretrained(
                    model_name,
                    timeout=300,  # 5 minutes timeout - WORKS FOR TOKENIZER
                    use_fast=True
                )
            except Exception as e:
                logger.warning(f"Tokenizer load attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)  # Wait 10 seconds before retry
                else:
                    raise e

    def _load_model_with_retry(self, model_name, max_retries=3, **kwargs):
        """Load model with retry mechanism"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Loading model...")
                # ✅ FIXED: Remove timeout parameter - it's not supported by model
                return AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    from_tf=self.framework == "tf",
                    # timeout=300,  # ❌ REMOVED - This causes the error
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Model load attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)  # Wait 10 seconds before retry
                else:
                    raise e

    async def predict(self, text: str) -> Dict[str, str]:
        """Predict sentiment for given text"""
        if not self.pipeline:
            raise ValueError("Model not loaded")
        
        try:
            result = self.pipeline(text)
            if not result:
                return {"label": "neutral", "score": 0.5}
            
            prediction = result[0]
            
            # ✅ CRITICAL FIX: Handle both pre-trained and fine-tuned model outputs
            if self.is_finetuned:
                # Fine-tuned model outputs: LABEL_0, LABEL_1, LABEL_2
                label_map = {
                    "LABEL_0": "negative",  # 0 -> negative
                    "LABEL_1": "neutral",   # 1 -> neutral
                    "LABEL_2": "positive"   # 2 -> positive
                }
                raw_label = prediction["label"]
                label = label_map.get(raw_label, "neutral")
                
                logger.debug(f"Fine-tuned model output: {raw_label} -> {label}")
                
            else:
                # Pre-trained model outputs: negative, neutral, positive
                label = prediction["label"].lower()
                if label not in ["positive", "negative", "neutral"]:
                    # Fallback mapping
                    label = "positive" if label in ["pos", "positive"] else "negative"
                
                logger.debug(f"Pre-trained model output: {label}")
            
            return {
                "label": label,
                "score": float(prediction["score"])
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"label": "error", "score": 0.0}

    def test_predictions(self):
        """Test model with sample predictions for debugging"""
        test_texts = [
            "I love this product!",
            "This is terrible",
            "It's okay, nothing special",
            "Amazing experience!",
            "Worst purchase ever"
        ]
        
        logger.info("Testing model predictions:")
        
        for text in test_texts:
            try:
                result = self.pipeline(text)
                prediction = result[0]
                
                if self.is_finetuned:
                    label_map = {
                        "LABEL_0": "negative",
                        "LABEL_1": "neutral", 
                        "LABEL_2": "positive"
                    }
                    raw_label = prediction["label"]
                    label = label_map.get(raw_label, "neutral")
                else:
                    label = prediction["label"].lower()
                
                logger.info(f"'{text}' -> {label} (score: {prediction['score']:.4f})")
                
            except Exception as e:
                logger.error(f"Error testing '{text}': {str(e)}")

    def reload_model(self):
        """Reload model (useful after fine-tuning)"""
        logger.info("Reloading model...")
        self.load_model()
        logger.info("Model reloaded successfully")