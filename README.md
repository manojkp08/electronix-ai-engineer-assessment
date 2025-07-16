# Sentiment Analysis System

## Overview
A complete sentiment analysis solution consisting of:
* Python backend with machine learning model
* React frontend interface
* Model training capabilities
* Containerized deployment

## Technical Components

### Backend Service
* Framework: FastAPI with GraphQL and REST endpoints
* Model: Uses Hugging Face's transformer models (default: Twitter-RoBERTa-base)
* Features:
   * Sentiment prediction API
   * Automatic model reloading
   * Health monitoring

### Frontend Application
* Built with React and TypeScript
* Features:
   * Text input for sentiment analysis
   * Results display
   * Error handling
 
### Screenshots
<img width="1749" height="1220" alt="image" src="https://github.com/user-attachments/assets/8e72d589-55be-42c3-8084-12963f20ce64" />
<img width="1871" height="1305" alt="image" src="https://github.com/user-attachments/assets/758ca3eb-f1e0-436b-8503-adbfe7809558" />
<img width="1775" height="1252" alt="image" src="https://github.com/user-attachments/assets/036377d7-eed1-4b2d-8fdb-d809d2e4e007" />

## Hugginface Model used
<img width="2817" height="1453" alt="image" src="https://github.com/user-attachments/assets/22f4b8c3-1b8d-4a23-812c-d1974498202d" />


### Model Training
* Fine-tuning script included
* Supports custom datasets in JSONL format
* Training features:
   * Learning rate scheduling
   * Gradient clipping
   * Multi Class Cross Entropy

## Getting Started

### Prerequisites
* Docker
* Docker Compose

### Installation
1. Clone this repository
2. Run the system: `docker-compose up --build`
3. Access the application at `http://localhost:3000`

## Usage

### API Endpoints
* **GraphQL**: `http://localhost:8000/graphql`
* **REST**: POST `http://localhost:8000/predict`
* **Health Check**: GET `http://localhost:8000/health`

### Training the Model
To train with custom data:
```bash
python scripts/finetune.py --data your_data.jsonl --epochs 3 --lr 3e-5
```

## Development

### Project Structure
* `backend/`: Python service and model code
* `frontend/`: React application
* `scripts/`: Training and utility scripts

### Configuration
Environment variables can be set in `.env` files for both backend and frontend.
