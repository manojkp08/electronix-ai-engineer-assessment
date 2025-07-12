// frontend/src/graphql/mutations.ts
import { gql } from '@apollo/client';

export const PREDICT_SENTIMENT = gql`
  mutation PredictSentiment($input: SentimentInput!) {
    predictSentiment(input: $input) {
      label
      score
    }
  }
`;