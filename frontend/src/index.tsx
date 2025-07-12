import React from 'react';
import ReactDOM from 'react-dom/client';
import { ApolloClient, InMemoryCache, ApolloProvider } from '@apollo/client';
import App from './App';
// import reportWebVitals from './reportWebVitals';
import { ThemeContextProvider } from './context/theme.context';

// 1. Apollo Client Setup
const client = new ApolloClient({
  uri: process.env.REACT_APP_API_URL || 'http://localhost:8000/graphql',
  cache: new InMemoryCache(),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
    },
  },
});

// 2. Root Rendering
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <ApolloProvider client={client}>
      <ThemeContextProvider>
        <App />
      </ThemeContextProvider>
    </ApolloProvider>
  </React.StrictMode>
);

// 3. Performance Monitoring (Optional)
// reportWebVitals();