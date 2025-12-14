# RAG SaaS Platform - Python Versiyonu

**100% Python - FastAPI + Streamlit**

## 🎯 Özellikler

- ✅ **Backend:** FastAPI (Node.js yerine)
- ✅ **Frontend:** Streamlit (Angular yerine)
- ✅ **RAG Servisleri:** Aynen çalışıyor (Python)
- ✅ **Colab Uyumlu:** Tek dil, kolay kurulum

## 📁 Yeni Proje Yapısı

```
bil482-project/
├── backend_fastapi/          # FastAPI Backend
│   ├── main.py              # API endpoints
│   └── run.py               # Başlatma scripti
├── frontend_streamlit/       # Streamlit Frontend
│   ├── app.py               # Streamlit UI
│   └── run.py               # Başlatma scripti
├── python_services/          # RAG Servisleri (aynı)
│   ├── scripts/
│   │   ├── rag_engine.py
│   │   ├── ingestor.py
│   │   └── evaluator.py
│   └── requirements.txt
└── notebooks/
    └── colab_setup_python.ipynb  # Python-only Colab kurulumu
```

## 🚀 Kurulum

### Local

```bash
# 1. Bağımlılıkları kur
cd python_services
pip install -r requirements.txt

# 2. API Key ayarla
echo "OPENAI_API_KEY=sk-proj-BURAYA-KEY" > backend_fastapi/.env
echo "OPENAI_API_KEY=sk-proj-BURAYA-KEY" > python_services/.env

# 3. Backend başlat (Terminal 1)
cd backend_fastapi
python run.py

# 4. Frontend başlat (Terminal 2)
cd frontend_streamlit
streamlit run app.py
```

### Colab

1. `notebooks/colab_setup_python.ipynb` dosyasını aç
2. Tüm hücreleri sırayla çalıştır
3. Public URL al (Colab port forwarding veya localtunnel)

## 📊 API Endpoints

Tüm endpoint'ler aynı (Express.js'deki gibi):
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/admin/companies`
- `POST /api/admin/companies`
- `GET /api/agents`
- `POST /api/agents`
- `POST /api/chat`
- `POST /api/upload`
- `POST /api/benchmark`

## 🔑 Giriş Bilgileri

- Email: `admin@ragplatform.com`
- Şifre: `Admin123!@#`

## 💡 Avantajlar

- ✅ Tek dil (Python)
- ✅ Daha kolay kurulum
- ✅ Colab'da daha hızlı
- ✅ Node.js/Angular bağımlılığı yok
- ✅ Mevcut Python servisleri aynen çalışıyor

