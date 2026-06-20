import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'esnext',   // three.js WebGPU build uses top-level await
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext', // same fix for dev mode
    },
  },
})
