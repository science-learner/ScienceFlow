import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Note: all backend communication (HTTP + SSE) is bridged through the Wails Go
// layer (see ../bindings/scienceflow and api/client.ts), so there is no Vite
// /api proxy. The dev server below only serves the React frontend when running
// inside `wails3 dev` or in a plain browser (plain browser has no runtime).
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 45000,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});