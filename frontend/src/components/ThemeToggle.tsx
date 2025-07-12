// frontend/src/components/ThemeToggle.tsx
import React, { useContext } from 'react';
import styled from 'styled-components';
import { ThemeContext } from '../App';

const ToggleButton = styled.button`
  position: absolute;
  top: 1rem;
  right: 1rem;
  padding: 0.5rem;
  background: ${({ theme }) => theme.background};
  color: ${({ theme }) => theme.text};
  border: 1px solid ${({ theme }) => theme.border};
  border-radius: 4px;
  cursor: pointer;
`;

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <ToggleButton onClick={toggleTheme}>
      {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
    </ToggleButton>
  );
};

export default ThemeToggle;