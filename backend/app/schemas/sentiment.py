# backend/app/schemas/sentiment.py
import strawberry
from pydantic import BaseModel

@strawberry.input
class SentimentInput:
    text: str

@strawberry.type
class SentimentResult:
    label: str
    score: float

# Pydantic model for REST API compatibility
class SentimentResponse(BaseModel):
    label: str
    score: float