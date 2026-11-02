import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 5174 keeps this app independent of the incident dashboard (5173).
export default defineConfig({
  plugins: [react()],
  server: { port: 5174, strictPort: true },
  preview: { port: 5174, strictPort: true },
})
