import os
import logging
from typing import Dict, Optional
import numpy as np
from pathlib import Path
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
        local_model_path = Path("/app/model"),
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
        self.local_model_path = local_model_path
        self.framework = self._clean_framework(framework)
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
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

        try:
            
            # logger.info(f"🌐 Loading pre-trained model from Hugging Face Hub: {self.model_path}")
            # model_name = self.model_path
            # self.is_finetuned = False

            logger.info(f'Loading the finetuned model from the local path: {self.local_model_path}')
            model_name = self.local_model_path
            self.is_finetuned = True


            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_name))

            kwargs = {}
            if self.quantize:
                if self.framework == "pt":
                    kwargs["load_in_8bit"] = True
                else:
                    logger.warning("Quantization not fully supported for TensorFlow")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                from_tf=self.framework == "tf",
                **kwargs
            )
            
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


    async def predict(self, text: str) -> Dict[str, str]:
        if not self.pipeline:
            raise ValueError("Model not loaded")

        try:
            result = self.pipeline(text)
            if not result:
                return {"label": "neutral", "score": 0.5}

            prediction = result[0]

            label = prediction["label"].lower()
            if label not in ["positive", "negative", "neutral"]:
                label = "neutral"  # fallback for safety

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