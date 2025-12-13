# 🚀 RAG SaaS Platform - Kurulum Rehberi

## 📋 Ön Gereksinimler

- **Node.js** 18+ ([nodejs.org](https://nodejs.org))
- **Python** 3.10+ ([python.org](https://python.org))
- **Angular CLI** 17+ (otomatik kurulacak)
- **OpenAI API Key** ([platform.openai.com](https://platform.openai.com))

---

## 🖥️ YEREL KURULUM (Local)

### 1. Projeyi İndir

```bash
git clone https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git
cd rag_nlp_chatbotplatform
```

### 2. Python Bağımlılıklarını Kur

```bash
cd python_services
pip install -r requirements.txt
cd ..
```

### 3. Backend Bağımlılıklarını Kur

```bash
npm install
```

### 4. Frontend Bağımlılıklarını Kur

```bash
cd frontend
npm install
cd ..
```

### 5. Angular CLI Kur (Global)

```bash
npm install -g @angular/cli@17
```

### 6. API Key'i Ayarla

**Backend için:**
```bash
# backend/.env dosyası oluştur
cat > backend/.env << EOF
PORT=3000
NODE_ENV=development
OPENAI_API_KEY=sk-proj-BURAYA-KENDI-KEY-INIZI-GIRIN
PYTHON_EXECUTABLE=python3
PYTHON_SERVICES_PATH=../python_services/scripts
REQUEST_TIMEOUT=600000
EOF
```

**Python Services için:**
```bash
# python_services/.env dosyası oluştur
cat > python_services/.env << EOF
OPENAI_API_KEY=sk-proj-BURAYA-KENDI-KEY-INIZI-GIRIN
EOF
```

### 7. Servisleri Başlat

**Terminal 1 - Backend:**
```bash
npm start
```

**Terminal 2 - Frontend:**
```bash
cd frontend
ng serve
```

### 8. Tarayıcıda Aç

- Frontend: http://localhost:4200
- Backend: http://localhost:3000

### 9. Giriş Yap

**SuperAdmin:**
- Email: `admin@ragplatform.com`
- Şifre: `Admin123!@#`

---

## ☁️ GOOGLE COLAB KURULUM

### 1. GitHub'dan Clone Et

Colab'da yeni bir notebook açın ve ilk hücreyi çalıştırın:

```python
!git clone https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git
%cd rag_nlp_chatbotplatform
```

### 2. Bağımlılıkları Kur

```python
# Node.js kur
!curl -fsSL https://deb.nodesource.com/setup_18.x | bash - > /dev/null 2>&1
!apt-get install -y nodejs > /dev/null 2>&1

# Python paketleri
!pip install -q -r python_services/requirements.txt

# Backend paketleri
!npm install --silent

# Angular CLI + Frontend paketleri
!npm install -g @angular/cli@17 --silent
!cd frontend && npm install --silent

print("✅ Tüm bağımlılıklar kuruldu!")
```

### 3. API Key'i Ayarla

**Yöntem A: Colab Secrets (Önerilen)**
1. Sol menüden 🔑 (Key) ikonuna tıklayın
2. "Add new secret" → Name: `OPENAI_API_KEY`, Value: API key'iniz
3. Aşağıdaki kodu çalıştırın:

```python
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

# .env dosyalarını oluştur
with open('backend/.env', 'w') as f:
    f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nPYTHON_EXECUTABLE=python3\nREQUEST_TIMEOUT=600000")

with open('python_services/.env', 'w') as f:
    f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")

print("✅ API key ayarlandı")
```

**Yöntem B: Gizli Input (Alternatif)**
```python
from getpass import getpass
OPENAI_API_KEY = getpass("API Key (görünmez): ")

with open('backend/.env', 'w') as f:
    f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nPYTHON_EXECUTABLE=python3\nREQUEST_TIMEOUT=600000")

with open('python_services/.env', 'w') as f:
    f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")

print("✅ API key ayarlandı")
```

### 4. Servisleri Başlat

```python
import subprocess
import time

# Backend başlat
backend = subprocess.Popen(['node', 'backend/server.js'], 
                          stdout=open('/tmp/backend.log', 'w'),
                          stderr=subprocess.STDOUT)
print("🔄 Backend başlatıldı...")
time.sleep(3)

# Frontend başlat
frontend = subprocess.Popen(['ng', 'serve', '--host', '0.0.0.0', '--port', '4200', '--disable-host-check'],
                           cwd='frontend',
                           stdout=open('/tmp/frontend.log', 'w'),
                           stderr=subprocess.STDOUT)
print("🔄 Frontend başlatıldı (2-3 dakika sürebilir)...")
time.sleep(20)

print("\n✅ Servisler hazır!")
print("📍 Backend: http://localhost:3000")
print("📍 Frontend: http://localhost:4200")
```

### 5. Public URL Al

**Yöntem A: Colab Port Forwarding (Önerilen)**
- Sağ üstteki 🔗 ikonuna tıklayın
- Port 4200'i seçin
- Otomatik public URL alırsınız

**Yöntem B: ngrok (Opsiyonel)**
```python
!pip install -q pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(4200)
print(f"🌐 Public URL: {public_url}")
```

---

## ✅ Kurulum Kontrolü

### Backend Kontrolü
```bash
curl http://localhost:3000/api/health
```

### Frontend Kontrolü
Tarayıcıda http://localhost:4200 açılmalı

### Logları Görüntüleme

**Local:**
- Backend: Terminal 1'de görünür
- Frontend: Terminal 2'de görünür

**Colab:**
```python
!tail -20 /tmp/backend.log
!tail -20 /tmp/frontend.log
```

---

## 🔧 Sorun Giderme

### Port Zaten Kullanımda
```bash
# Backend için farklı port
PORT=3001 npm start

# Frontend için farklı port
cd frontend && ng serve --port 4201
```

### Python Modül Bulunamadı
```bash
pip install -r python_services/requirements.txt
```

### Node Modül Bulunamadı
```bash
npm install
cd frontend && npm install
```

### CORS Hatası
- Backend'in çalıştığından emin olun
- Frontend proxy ayarlarını kontrol edin (`frontend/proxy.conf.json`)

### API Key Hatası
- `.env` dosyalarının doğru yerde olduğundan emin olun
- API key'in geçerli olduğundan emin olun

---

## 📝 İlk Kullanım

1. **Giriş Yap:** `admin@ragplatform.com` / `Admin123!@#`
2. **Şirket Oluştur:** SuperAdmin olarak şirket oluşturun
3. **Agent Oluştur:** Şirket hesabıyla agent (chatbot) oluşturun
4. **Veri Yükle:** PDF, DOCX, TXT, CSV veya Web URL yükleyin
5. **Chat Test:** Agent ile sohbet edin
6. **Analytics:** Metrikleri görüntüleyin

---

## 🎯 Hızlı Başlangıç (Tek Komut)

**Local için:**
```bash
# Tüm bağımlılıkları kur
cd python_services && pip install -r requirements.txt && cd .. && npm install && cd frontend && npm install && cd ..

# API key'i ayarla (manuel olarak .env dosyalarını oluşturun)

# Servisleri başlat (2 ayrı terminal)
npm start  # Terminal 1
cd frontend && ng serve  # Terminal 2
```

**Colab için:**
- `notebooks/colab_setup.ipynb` dosyasını açın
- Tüm hücreleri sırayla çalıştırın

---

**🎉 Başarılar! Platformunuz hazır!**

