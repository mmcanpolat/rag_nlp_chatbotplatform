import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,          // 0.0.0.0
    port: 4200,
    strictPort: false,

    // 🔴 Cloudflare için KRİTİK
    allowedHosts: 'all',

    hmr: {
      clientPort: 443
    },

    // (API varsa önerilir)
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false
      }
    }
  }
});
