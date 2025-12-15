# Colab'te Projeyi Başlatma - Adım Adım

## 1. Projeyi Colab'e Yükle

### Yöntem A: Git Clone (Önerilen)
```python
!git clone https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git
%cd rag_nlp_chatbotplatform
```

### Yöntem B: ZIP Upload
1. Colab'te **Files** sekmesine tıkla
2. Projeyi ZIP olarak yükle
3. Aç:
```python
!unzip bil482-project.zip
%cd bil482-project
```

## 2. Bağımlılıkları Kur

```python
!pip install -r python_services/requirements.txt
```

## 3. API Key Ayarla

### Colab Secrets (Önerilen)
1. Sol menüden **🔑 Secrets** sekmesine tıkla
2. **+ Add Secret** → `OPENAI_API_KEY` → API key'ini gir
3. Kod:
```python
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

### Veya Direkt
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-...'
```

## 4. .env Dosyalarını Oluştur

```python
from pathlib import Path
import os

Path("backend_fastapi").mkdir(exist_ok=True)
Path("python_services").mkdir(exist_ok=True)

# API Key'i al
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# .env dosyalarını oluştur
with open("backend_fastapi/.env", "w") as f:
    f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nREQUEST_TIMEOUT=600000")

with open("python_services/.env", "w") as f:
    f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")

os.environ['API_BASE_URL'] = "http://localhost:3000"
os.environ['GRADIO_SHARE'] = "true"
```

## 5. Backend Başlat

```python
import subprocess
import sys
import time

# Backend'i arka planda başlat
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"],
    cwd="backend_fastapi"
)

time.sleep(3)
print("✅ Backend başlatıldı: http://localhost:3000")
```

## 6. Frontend (Gradio) Başlat

```python
# Gradio'yu başlat
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    env={**os.environ, "GRADIO_SHARE": "true"}
)

print("✅ Gradio başlatıldı")
print("🌐 Public URL terminal çıktısında görünecek")
print("   'Running on public URL: https://xxxxx.gradio.live' satırını ara")
```

## Tek Hücrede Hepsi (Kopyala-Yapıştır)

```python
# ============================================
# 1. Projeyi Yükle
# ============================================
!git clone https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git
%cd rag_nlp_chatbotplatform

# ============================================
# 2. Bağımlılıkları Kur
# ============================================
!pip install -r python_services/requirements.txt

# ============================================
# 3. API Key Ayarla
# ============================================
from google.colab import userdata
import os
from pathlib import Path

OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# ============================================
# 4. .env Dosyalarını Oluştur
# ============================================
Path("backend_fastapi").mkdir(exist_ok=True)
Path("python_services").mkdir(exist_ok=True)
Path("frontend_gradio/assets/plots").mkdir(parents=True, exist_ok=True)

with open("backend_fastapi/.env", "w") as f:
    f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nREQUEST_TIMEOUT=600000")

with open("python_services/.env", "w") as f:
    f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")

os.environ['API_BASE_URL'] = "http://localhost:3000"
os.environ['GRADIO_SHARE'] = "true"

# ============================================
# 5. Backend Başlat
# ============================================
import subprocess
import sys
import time

backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"],
    cwd="backend_fastapi"
)
time.sleep(3)
print("✅ Backend: http://localhost:3000")

# ============================================
# 6. Frontend (Gradio) Başlat
# ============================================
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    env={**os.environ, "GRADIO_SHARE": "true"}
)

print("✅ Gradio başlatıldı")
print("🔑 Giriş: admin@ragplatform.com / Admin123!@#")
print("\n💡 Public URL terminal çıktısında görünecek")
print("   'Running on public URL:' satırını ara")
```

## Kullanım

1. Colab'te yeni notebook oluştur
2. Yukarıdaki kodu tek hücreye yapıştır
3. Çalıştır (Shift+Enter)
4. Terminal çıktısında Gradio public URL'yi bul
5. URL'yi tarayıcıda aç

## Notlar

- **Backend**: `http://localhost:3000` (Colab içinde)
- **Frontend**: `http://localhost:7860` (Colab içinde)
- **Public URL**: Gradio otomatik oluşturur (`https://xxxxx.gradio.live`)
- **Giriş**: `admin@ragplatform.com` / `Admin123!@#`

