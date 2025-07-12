<<<<<<< HEAD
# sentiment-analysis-electronix-assessment
=======
# Sentiment Analysis System

An end-to-end microservice for binary sentiment analysis with a React frontend, Python backend, and model fine-tuning capabilities.

## Features

- **Backend**: FastAPI with GraphQL and REST endpoints
- **Frontend**: React with TypeScript and Apollo Client
- **Model**: Hugging Face Transformers with support for PyTorch and TensorFlow
- **Fine-tuning**: CLI script for model training
- **Hot Reload**: Automatic model reloading without service restart
- **Containerization**: Docker Compose for easy deployment
- **CI/CD**: GitHub Actions workflow for testing and building

## Setup

1. Clone the repository
2. Install Docker and Docker Compose
3. Run `docker-compose up --build`
4. Access the frontend at `http://localhost:3000`

## Usage

### Backend API

- GraphQL endpoint: `POST /graphql`
- REST endpoint: `POST /predict`
- Health check: `GET /health`

### Fine-tuning

To fine-tune the model:

```bash
python scripts/finetune.py --data example_data.jsonl --epochs 3 --lr 3e-5
>>>>>>> 9f03b06 (initial commit)
