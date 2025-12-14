import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,          // 0.0.0.0
    port: 4200,
    strictPort: false,

    // 🔴 KRİTİK SATIRLAR
    allowedHosts: 'all',

    hmr: {
      clientPort: 443    // Cloudflare üzerinden geldiği için
    }
  }
});

