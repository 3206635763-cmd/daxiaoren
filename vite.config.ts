import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/postcss';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The overseas and mainland deployments share this repo, so absolute URLs in
// `index.html` are injected at build time from the SITE_URL build variable.
// When it is unset the placeholders collapse to root-relative paths.
const siteUrl = (process.env.SITE_URL ?? '').replace(/\/+$/, '');

export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  resolve: {
    alias: { '@': fileURLToPath(new URL('.', import.meta.url)) },
  },
  plugins: [
    react(),
    {
      name: 'inject-site-url',
      transformIndexHtml(html: string) {
        return html.replaceAll('__SITE_URL__', siteUrl);
      },
    },
  ],
});
