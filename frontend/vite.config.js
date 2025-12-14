import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,          // 0.0.0.0
    port: 4200,

    // 🔴 VITE 5 İÇİN DOĞRU FORMAT
    allowedHosts: [
      '.trycloudflare.com'
    ],

    hmr: {
      clientPort: 443
    },

    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false
      }
    }
  }
});
