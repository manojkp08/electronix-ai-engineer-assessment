// frontend/src/App.tsx
import React, { useState } from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { ApolloClient, InMemoryCache, ApolloProvider } from '@apollo/client';
import SentimentAnalyzer from './components/SentimentAnalyzer';
import { lightTheme, darkTheme } from './theme';

const client = new ApolloClient({
  uri: process.env.REACT_APP_API_URL || 'http://localhost:8000/graphql',
  cache: new InMemoryCache(),
});

export const ThemeContext = React.createContext({
  theme: 'light',
  toggleTheme: () => {},
});

const AppContainer = styled.div`
  min-height: 100vh;
  background: ${({ theme }) => theme.background};
  color: ${({ theme }) => theme.text};
  transition: all 0.3s ease;
`;

const App: React.FC = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <ApolloProvider client={client}>
      <ThemeContext.Provider value={{ theme, toggleTheme }}>
        <ThemeProvider theme={theme === 'light' ? lightTheme : darkTheme}>
          <AppContainer>
            <SentimentAnalyzer />
          </AppContainer>
        </ThemeProvider>
      </ThemeContext.Provider>
    </ApolloProvider>
  );
};

export default App;
