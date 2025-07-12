import { useMutation } from '@apollo/client';
import { PREDICT_SENTIMENT, SentimentResponse } from '../graphql/queries';
import { useState } from 'react';

type SentimentResult = {
  label: 'positive' | 'negative';
  score: number;
};

type UseSentimentReturn = {
  analyze: (text: string) => Promise<void>;
  result: SentimentResult | null;
  loading: boolean;
  error: string | null;
  reset: () => void;
};

export const useSentiment = (): UseSentimentReturn => {
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [predict] = useMutation<SentimentResponse>(PREDICT_SENTIMENT, {
    onError: (err) => {
      setError(err.message);
      setLoading(false);
    },
  });

  const analyze = async (text: string) => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const { data } = await predict({
        variables: { input: { text } },
      });

      if (data?.predictSentiment) {
        setResult({
          label: data.predictSentiment.label,
          score: data.predictSentiment.score,
        });
      }
    } catch (err) {
      setError('Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return { analyze, result, loading, error, reset };
};