# Python Versiyonu - Başlatma Talimatları

## 🚀 Hızlı Başlangıç

### 1. Bağımlılıkları Kur

```bash
cd python_services
pip install -r requirements.txt
```

### 2. API Key Ayarla

```bash
# backend_fastapi/.env
OPENAI_API_KEY=sk-proj-BURAYA-KEY-INIZI-GIRIN
PORT=3000

# python_services/.env
OPENAI_API_KEY=sk-proj-BURAYA-KEY-INIZI-GIRIN
```

### 3. Backend'i Başlat

```bash
cd backend_fastapi
python run.py
```

veya

```bash
uvicorn main:app --host 0.0.0.0 --port 3000
```

### 4. Frontend'i Başlat (Yeni Terminal)

```bash
cd frontend_streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

veya

```bash
python run.py
```

### 5. Tarayıcıda Aç

- Frontend: http://localhost:8501
- Backend API: http://localhost:3000

## 🔑 Giriş Bilgileri

- Email: `admin@ragplatform.com`
- Şifre: `Admin123!@#`

## 📝 Notlar

- Backend ve Frontend ayrı process'ler olarak çalışır
- Backend FastAPI (port 3000)
- Frontend Streamlit (port 8501)
- Tüm Python servisleri (RAG, evaluator, ingestor) aynen çalışır

