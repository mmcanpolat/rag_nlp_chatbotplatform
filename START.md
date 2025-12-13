# Başlatma Talimatları

## Backend'i Başlatma

Backend sunucusu çalışmıyorsa şu komutu çalıştırın:

```bash
cd bil482-project
npm start
```

veya

```bash
cd bil482-project/backend
node server.js
```

Backend başarıyla başladığında şu mesajı göreceksiniz:
```
==================================================
  RAG Platform API
==================================================
  Port: 3000
  SuperAdmin: admin@ragplatform.com
==================================================
```

## Frontend'i Başlatma

Ayrı bir terminal'de:

```bash
cd bil482-project/frontend
ng serve
```

veya

```bash
cd bil482-project
npm run frontend
```

## Her İkisini Birlikte Başlatma

İki ayrı terminal açın:

**Terminal 1 (Backend):**
```bash
cd bil482-project
npm start
```

**Terminal 2 (Frontend):**
```bash
cd bil482-project
npm run frontend
```

## Giriş Bilgileri

**SuperAdmin:**
- Kullanıcı Adı: `admin@ragplatform.com`
- Şifre: `Admin123!@#`

## Sorun Giderme

### 404 Hatası
- Backend'in çalıştığından emin olun (port 3000)
- `http://localhost:3000/api/health` adresine tarayıcıdan erişmeyi deneyin

### CORS Hatası
- Backend'de CORS ayarlarını kontrol edin
- Frontend'in `http://localhost:4200` adresinde çalıştığından emin olun

### Port Zaten Kullanımda
- Port 3000 veya 4200 kullanımdaysa, farklı portlar kullanın
- Backend için: `PORT=3001 node backend/server.js`
- Frontend için: `ng serve --port 4201`

