# 🚀 Projeyi Başlatma Rehberi

## 📍 Yöntem 1: Local (Bilgisayarında)

### Adım 1: Bağımlılıkları Kur

```bash
cd bil482-project
cd python_services
pip install -r requirements.txt
```

### Adım 2: API Key Ayarla

İki `.env` dosyası oluştur:

**`backend_fastapi/.env`** dosyası:
```bash
cd ../backend_fastapi
echo "PORT=3000" > .env
echo "OPENAI_API_KEY=sk-proj-BURAYA-KEY-INIZI-GIRIN" >> .env
echo "REQUEST_TIMEOUT=600000" >> .env
```

**`python_services/.env`** dosyası:
```bash
cd ../python_services
echo "OPENAI_API_KEY=sk-proj-BURAYA-KEY-INIZI-GIRIN" > .env
```

> ⚠️ `BURAYA-KEY-INIZI-GIRIN` yerine gerçek OpenAI API key'inizi yazın!

### Adım 3: Backend'i Başlat (Terminal 1)

```bash
cd ../backend_fastapi
python run.py
```

Başarılı olursa şunu göreceksin:
```
INFO:     Uvicorn running on http://0.0.0.0:3000
```

### Adım 4: Frontend'i Başlat (Terminal 2 - YENİ TERMİNAL)

```bash
cd bil482-project/frontend_streamlit
streamlit run app.py --server.port 8501
```

veya

```bash
python run.py
```

Başarılı olursa şunu göreceksin:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Adım 5: Tarayıcıda Aç

- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:3000/api/health

### 🔑 Giriş Bilgileri

- **Email:** `admin@ragplatform.com`
- **Şifre:** `Admin123!@#`

---

## 📍 Yöntem 2: Google Colab

### Adım 1: Colab'ı Aç

1. Google Colab'a git: https://colab.research.google.com/
2. Yeni notebook oluştur
3. GitHub'dan projeyi yükle veya `notebooks/colab_setup_python.ipynb` dosyasını aç

### Adım 2: Setup Notebook'unu Çalıştır

`notebooks/colab_setup_python.ipynb` dosyasındaki tüm hücreleri sırayla çalıştır:

1. **Hücre 1:** Projeyi GitHub'dan yükle
2. **Hücre 2:** Bağımlılıkları kur
3. **Hücre 3:** API key'i ayarla (Colab Secrets kullan veya gizli input)
4. **Hücre 4:** Servisleri başlat
5. **Hücre 5:** Public URL al

### Adım 3: Public URL'i Kullan

Colab port forwarding veya localtunnel ile public URL alıp tarayıcıda aç.

---

## 🛠️ Sorun Giderme

### Backend başlamıyor?

```bash
# Port 3000 kullanılıyor mu kontrol et
lsof -i :3000

# Kullanılıyorsa öldür
kill -9 <PID>
```

### Frontend başlamıyor?

```bash
# Port 8501 kullanılıyor mu kontrol et
lsof -i :8501

# Kullanılıyorsa öldür
kill -9 <PID>
```

### API Key hatası?

- `.env` dosyalarının doğru yerde olduğundan emin ol
- API key'in doğru olduğundan emin ol
- `.env` dosyalarında boşluk veya tırnak işareti olmamalı

### Import hatası?

```bash
# Python path'i kontrol et
cd python_services
python -c "import sys; print(sys.path)"
```

---

## 📝 Hızlı Komutlar

### Her şeyi tek seferde başlat (Local)

```bash
# Terminal 1
cd bil482-project/backend_fastapi && python run.py &

# Terminal 2
cd bil482-project/frontend_streamlit && streamlit run app.py
```

### Servisleri durdur

```bash
# Backend'i durdur
pkill -f "uvicorn"

# Frontend'i durdur
pkill -f "streamlit"
```

---

## ✅ Başarı Kontrolü

1. Backend çalışıyor mu?
   ```bash
   curl http://localhost:3000/api/health
   ```
   `{"status":"ok"}` dönmeli

2. Frontend çalışıyor mu?
   - Tarayıcıda http://localhost:8501 açılmalı

3. Giriş yapabiliyor musun?
   - Email: `admin@ragplatform.com`
   - Şifre: `Admin123!@#`

---

## 🎯 Sonraki Adımlar

1. ✅ Giriş yap
2. ✅ Şirket oluştur (SuperAdmin)
3. ✅ Agent oluştur
4. ✅ Veri yükle (PDF, CSV, TXT, vb.)
5. ✅ Chat yap
6. ✅ Benchmark çalıştır

