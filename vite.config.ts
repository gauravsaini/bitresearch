import { defineConfig } from 'vite';
import { fileURLToPath } from 'url';
export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist/client',
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        worker: fileURLToPath(new URL('./worker.html', import.meta.url)),
        p2p: fileURLToPath(new URL('./p2p.html', import.meta.url)),
        validate: fileURLToPath(new URL('./validate.html', import.meta.url)),
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
