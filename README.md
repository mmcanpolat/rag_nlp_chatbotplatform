# RAG SaaS Platform - Python Versiyonu

**100% Python - FastAPI + Streamlit**

Akıllı RAG (Retrieval-Augmented Generation) tabanlı SaaS chatbot platformu. Şirketler kendi chatbot'larını oluşturup veri setlerini yükleyebilir, akademik metriklerle değerlendirme yapabilir.

## 🎯 Özellikler

- ✅ **Backend:** FastAPI (Python)
- ✅ **Frontend:** Streamlit (Python)
- ✅ **RAG Engine:** LangChain + FAISS
- ✅ **Modeller:** GPT-4o-mini, BERT Turkish Cased, BERT Turkish Sentiment
- ✅ **Değerlendirme:** Cosine Similarity, ROUGE-L, BLEU, F1 Score, Accuracy
- ✅ **Colab Uyumlu:** Tek dil, kolay kurulum

## 📁 Proje Yapısı

```
bil482-project/
├── backend_fastapi/          # FastAPI Backend
│   ├── main.py              # API endpoints
│   └── run.py               # Başlatma scripti
├── frontend_streamlit/       # Streamlit Frontend
│   ├── app.py               # Streamlit UI
│   └── run.py               # Başlatma scripti
├── python_services/          # RAG Servisleri
│   ├── scripts/
│   │   ├── rag_engine.py    # RAG motoru
│   │   ├── ingestor.py      # Döküman işleme
│   │   └── evaluator.py     # Model değerlendirme
│   └── requirements.txt     # Python bağımlılıkları
├── notebooks/
│   └── colab_setup_python.ipynb  # Colab kurulum
└── archived_js/             # Eski JavaScript dosyaları (arşiv)
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

## 🔑 Giriş Bilgileri

- **Email:** `admin@ragplatform.com`
- **Şifre:** `Admin123!@#`

## 📊 API Endpoints

- `POST /api/auth/login` - Giriş
- `POST /api/auth/logout` - Çıkış
- `GET /api/admin/companies` - Şirket listesi (SuperAdmin)
- `POST /api/admin/companies` - Şirket oluştur (SuperAdmin)
- `GET /api/agents` - Agent listesi
- `POST /api/agents` - Agent oluştur
- `POST /api/chat` - Chat sorgusu
- `POST /api/upload` - Dosya yükleme
- `POST /api/benchmark` - Benchmark çalıştır

## 💡 Teknik Detaylar

- **Embedding Modelleri:** `paraphrase-multilingual-MiniLM-L12-v2`, `text-embedding-3-large`
- **Vector DB:** FAISS
- **Chunk Size:** 750 karakter, 100 overlap
- **Top-K Retrieval:** 3 chunk
- **Değerlendirme Metrikleri:** Cosine Similarity, ROUGE-L, BLEU, F1, Accuracy

## 📝 Notlar

- Tüm veriler memory'de tutuluyor (production için veritabanı eklenebilir)
- Session token'lar memory'de saklanıyor
- FAISS index'leri `python_services/data/faiss_index/` altında
- Grafikler `frontend_streamlit/assets/plots/` altına kaydediliyor

## 🔄 Eski Versiyon

JavaScript versiyonu (Node.js + Angular) `archived_js/` klasöründe arşivlenmiş durumda.
