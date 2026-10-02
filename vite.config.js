import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    target: 'es2022',
    rollupOptions: {
      input: {
        main: 'index.html',
        diagnostics: 'diagnostics.html',
      },
    },
  },
  worker: {
    format: 'es',
  },
  server: {
    port: 5173,
    watch: {
      ignored: ['**/_spike/**', '**/.cache/**'],
    },
  },
  preview: {
    port: 4173,
  },
});
