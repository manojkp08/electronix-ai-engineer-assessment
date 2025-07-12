import { gql } from '@apollo/client';

// 1. Sentiment Prediction Query
export const PREDICT_SENTIMENT = gql`
  mutation PredictSentiment($input: SentimentInput!) {
    predictSentiment(input: $input) {
      label
      score
    }
  }
`;

// 2. Health Check Query
export const HEALTH_CHECK = gql`
  query HealthCheck {
    health
  }
`;

// 3. Type Definitions for TypeScript
export type SentimentResponse = {
  predictSentiment: {
    label: 'positive' | 'negative';
    score: number;
  };
};

export type SentimentInput = {
  text: string;
};