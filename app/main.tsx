import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './globals.css';
import { FightLab } from './fight-lab';

const container = document.getElementById('root');

if (!container) {
  throw new Error('Missing #root container in index.html');
}

createRoot(container).render(
  <StrictMode>
    <FightLab />
  </StrictMode>,
);
