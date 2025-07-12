# backend/app/utils/config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
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