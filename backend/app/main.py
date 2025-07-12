# backend/app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
from app.services.model_service import SentimentModel
from app.schemas.sentiment import SentimentInput, SentimentResult
from app.services.watcher import start_watcher
from app.utils.logger import setup_logging
from app.utils.config import settings

# Initialize logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for binary sentiment analysis using Hugging Face Transformers",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model service
model_service = SentimentModel(
    model_path=settings.MODEL_PATH,
    framework=settings.FRAMEWORK,
    quantize=settings.QUANTIZE
)

# Start the model watcher if hot reload is enabled
if settings.HOT_RELOAD:
    start_watcher(model_service, settings.MODEL_PATH)

# GraphQL Schema
@strawberry.type
class Query:
    @strawberry.field
    async def health(self) -> str:
        return "OK"

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def predict_sentiment(self, input: SentimentInput) -> SentimentResult:
        """Predict sentiment for given text"""
        prediction = await model_service.predict(input.text)
        return SentimentResult(
            label=prediction["label"],
            score=prediction["score"]
        )

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")

# REST endpoint for compatibility
@app.post("/predict", response_model=SentimentResult)
async def predict(text: str):
    """Predict sentiment (REST endpoint)"""
    prediction = await model_service.predict(text)
    return prediction

@app.get("/health")
async def health_check():
    return {"status": "healthy"}