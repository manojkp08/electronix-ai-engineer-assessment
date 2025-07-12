# backend/app/utils/logger.py
import logging
import sys
from app.utils.config import settings

def setup_logging():

    """Safe logging configuration with fallback"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper().strip(), logging.INFO)

    """Configure logging for the application"""
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )