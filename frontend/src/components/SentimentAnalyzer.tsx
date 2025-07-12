// frontend/src/components/SentimentAnalyzer.tsx
import React, { useState, useEffect } from 'react';
import { useMutation } from '@apollo/client';
import { PREDICT_SENTIMENT } from '../graphql/mutations';
import ThemeToggle from './ThemeToggle';
import styled from 'styled-components';

interface SentimentResult {
  label: string;
  score: number;
}

const Container = styled.div`
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
  border-radius: 8px;
  background: ${({ theme }) => theme.background};
  color: ${({ theme }) => theme.text};
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 150px;
  padding: 1rem;
  margin-bottom: 1rem;
  border: 1px solid ${({ theme }) => theme.border};
  border-radius: 4px;
  background: ${({ theme }) => theme.inputBackground};
  color: ${({ theme }) => theme.text};
  font-size: 1rem;
  resize: vertical;
`;

const Button = styled.button`
  padding: 0.75rem 1.5rem;
  background: ${({ theme }) => theme.primary};
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;

  &:hover {
    background: ${({ theme }) => theme.primaryHover};
  }

  &:disabled {
    background: ${({ theme }) => theme.disabled};
    cursor: not-allowed;
  }
`;

const ResultContainer = styled.div`
  margin-top: 2rem;
  padding: 1.5rem;
  border-radius: 4px;
  background: ${({ theme }) => theme.resultBackground};
`;

const ScoreBar = styled.div<{ score: number }>`
  height: 10px;
  background: linear-gradient(
    to right,
    ${({ theme }) => theme.negative} 0%,
    ${({ theme }) => theme.negative} 50%,
    ${({ theme }) => theme.positive} 50%,
    ${({ theme }) => theme.positive} 100%
  );
  margin-top: 1rem;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: -5px;
    left: ${({ score }) => (score * 100)}%;
    width: 2px;
    height: 20px;
    background: ${({ theme }) => theme.text};
  }
`;

const SentimentAnalyzer: React.FC = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [typingTimeout, setTypingTimeout] = useState<NodeJS.Timeout | null>(null);

  const [predictSentiment] = useMutation(PREDICT_SENTIMENT);

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const { data } = await predictSentiment({
        variables: { input: { text } },
      });

      setResult(data.predictSentiment);
    } catch (err) {
      setError('Failed to analyze sentiment. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (typingTimeout) {
      clearTimeout(typingTimeout);
    }

    if (text.trim()) {
      const timeout = setTimeout(() => {
        handlePredict();
      }, 1000);

      setTypingTimeout(timeout);
    }

    return () => {
      if (typingTimeout) {
        clearTimeout(typingTimeout);
      }
    };
  }, [text]);

  return (
    <Container>
      <ThemeToggle />
      <h1>Sentiment Analysis</h1>
      <p>Enter text to analyze its sentiment:</p>
      
      <TextArea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type your text here..."
      />
      
      <Button onClick={handlePredict} disabled={isLoading || !text.trim()}>
        {isLoading ? 'Analyzing...' : 'Analyze Sentiment'}
      </Button>
      
      {error && <p style={{ color: 'red' }}>{error}</p>}
      
      {result && (
        <ResultContainer>
          <h2>Result</h2>
          <p>
            Sentiment: <strong>{result.label}</strong>
          </p>
          <p>
            Confidence: <strong>{(result.score * 100).toFixed(2)}%</strong>
          </p>
          <ScoreBar score={result.score} />
        </ResultContainer>
      )}
    </Container>
  );
};

export default SentimentAnalyzer;