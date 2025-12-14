# ============================================
# RAG SaaS Platform - Colab Tek Hücre Başlatma
# ============================================
# Bu dosyayı Colab'te yeni bir hücreye yapıştır ve çalıştır
# Tek komutla tüm kurulum ve başlatma yapılır

import os
import subprocess
import sys
import time
from pathlib import Path
from getpass import getpass

print("=" * 60)
print("RAG SaaS Platform - Colab Kurulum (Gradio)")
print("=" * 60)

# 1. Projeyi yükle
print("\n[1/5] Proje yükleniyor...")
if not Path("rag_nlp_chatbotplatform").exists():
    subprocess.run(["git", "clone", "https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git"], check=True)
os.chdir("rag_nlp_chatbotplatform")
print("✅ Proje yüklendi")

# 2. Bağımlılıkları kur
print("\n[2/5] Bağımlılıklar kuruluyor (5-10 dakika)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "python_services/requirements.txt"], check=True)
print("✅ Bağımlılıklar kuruldu")

# 3. API Key al
print("\n[3/5] API Key gerekli...")
try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise KeyError
    print("✅ API key Colab Secrets'tan alındı")
except:
    OPENAI_API_KEY = getpass("OpenAI API Key girin (görünmez): ")
    if not OPENAI_API_KEY:
        raise ValueError("API Key gerekli!")

# 4. .env dosyalarını oluştur
print("\n[4/5] Yapılandırma dosyaları oluşturuluyor...")
Path("backend_fastapi").mkdir(exist_ok=True)
Path("python_services").mkdir(exist_ok=True)
Path("frontend_gradio/assets/plots").mkdir(parents=True, exist_ok=True)

with open("backend_fastapi/.env", "w") as f:
    f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nREQUEST_TIMEOUT=600000")

with open("python_services/.env", "w") as f:
    f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['API_BASE_URL'] = "http://localhost:3000"
os.environ['GRADIO_SHARE'] = "true"  # Colab'te her zaman share=True
print("✅ Yapılandırma tamamlandı")

# 5. Servisleri başlat
print("\n[5/5] Servisler başlatılıyor...")

# Eski process'leri durdur
subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "gradio"], stderr=subprocess.DEVNULL)
time.sleep(2)

# Backend başlat
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"],
    cwd="backend_fastapi",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env={**os.environ}
)
time.sleep(5)

# Frontend başlat (Gradio) - log dosyasına yazıp URL'yi oku
gradio_log_file = "/tmp/gradio_output.log"
gradio_url = None

# Frontend'i başlat - stdout'u log dosyasına yaz
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    stdout=open(gradio_log_file, "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "API_BASE_URL": "http://localhost:3000"}
)

# Gradio'nun başlamasını bekle
print("⏳ Gradio başlatılıyor (public URL oluşturuluyor, 20 saniye bekleniyor)...")
time.sleep(20)

# Log dosyasından URL'yi oku
try:
    if os.path.exists(gradio_log_file):
        with open(gradio_log_file, "r") as f:
            log_content = f.read()
            # URL'yi bul
            for line in log_content.split("\n"):
                if "Running on public URL:" in line:
                    gradio_url = line.split("Running on public URL:")[-1].strip()
                    break
                elif "https://" in line and "gradio.live" in line:
                    # Direkt URL satırı
                    parts = line.split()
                    for part in parts:
                        if "https://" in part and "gradio.live" in part:
                            gradio_url = part.strip()
                            break
                    if gradio_url:
                        break
except Exception as e:
    print(f"[!] Log okuma hatası: {e}")

print("✅ Servisler başlatıldı!")
print("\n" + "=" * 60)
print("📍 Backend: http://localhost:3000")
print("📍 Frontend: http://localhost:7860")

# Colab port forwarding - alternatif yöntem
try:
    from google.colab import output
    # Colab'in port forwarding'ini kullan
    print("\n🔗 Colab Port Forwarding:")
    print("   Sağ üstteki 🔗 ikonuna tıklayıp port 7860'i seç")
    print("   Veya aşağıdaki komutu çalıştır:")
    print("   !pip install pyngrok && python -m pyngrok http 7860")
except:
    pass

if gradio_url:
    print(f"\n🌐 Gradio Public URL: {gradio_url}")
    print(f"   👆 Bu URL'yi kopyalayıp tarayıcıda aç!")
else:
    print("\n🔗 Public URL oluşturuluyor...")
    print("   ⚠️  Birkaç saniye sonra log dosyasını kontrol et:")
    print(f"   📄 Log: {gradio_log_file}")
    print("   Veya Colab'te sağ üstteki 🔗 ikonuna tıklayıp port 7860'i seç")
    print("\n   💡 Alternatif: Aşağıdaki komutu çalıştır:")
    print("   !cat /tmp/gradio_output.log | grep 'public URL'")

print("\n🔑 Giriş: admin@ragplatform.com / Admin123!@#")
print("=" * 60)

