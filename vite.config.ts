import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist/client',
    rollupOptions: {
      input: {
        main: 'index.html',
        worker: 'worker.html',
        p2p: 'p2p.html',
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8787',
        ws: true,
      },
      '/api': {
        target: 'http://localhost:8787',
      },
    },
  },
  assetsInclude: ['**/*.wgsl'],
});
