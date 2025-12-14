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

# Frontend başlat (Gradio) - basit ve hızlı
print("⏳ Gradio başlatılıyor...")

# Gradio'yu arka planda başlat
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env={**os.environ, "API_BASE_URL": "http://localhost:3000", "GRADIO_SHARE": "true"}
)

# URL'yi yakalamak için kısa bir süre bekle ve stdout'u oku
gradio_url = None
print("📡 Gradio public URL bekleniyor...\n")

# 10 saniye boyunca stdout'u oku (Gradio hızlı başlar)
for i in range(20):  # 20 x 0.5 = 10 saniye
    time.sleep(0.5)
    try:
        # Basit readline - blocking olabilir ama kısa süre
        import select
        ready, _, _ = select.select([frontend.stdout], [], [], 0.1)
        if ready:
            line = frontend.stdout.readline()
            if line:
                line = line.strip()
                # Sadece önemli satırları göster
                if "Running on" in line or "public URL" in line or "https://" in line:
                    print(line)
                
                # URL'yi bul
                if "Running on public URL:" in line:
                    gradio_url = line.split("Running on public URL:")[-1].strip()
                    break
                elif "https://" in line and ("gradio.live" in line or "gradio.app" in line):
                    for word in line.split():
                        if "https://" in word and ("gradio.live" in word or "gradio.app" in word):
                            gradio_url = word.strip().rstrip(".,;")
                            break
                    if gradio_url:
                        break
    except:
        continue
    
    # URL bulunduysa çık
    if gradio_url:
        break

print("\n" + "=" * 60)
print("✅ Servisler başlatıldı!")
print("=" * 60)
print("📍 Backend: http://localhost:3000")
print("📍 Frontend: http://localhost:7860")

if gradio_url:
    print(f"\n🌐 GRADIO PUBLIC URL:")
    print(f"   {gradio_url}")
    print(f"\n   👆 Bu URL'yi kopyalayıp tarayıcıda aç!")
else:
    print("\n⏳ Gradio public URL oluşturuluyor...")
    print("   Yukarıdaki çıktıda 'Running on public URL:' satırını ara")
    print("   Veya birkaç saniye bekle ve tekrar kontrol et")

print("\n🔑 Giriş: admin@ragplatform.com / Admin123!@#")
print("=" * 60)
print("\n💡 Not: Gradio arka planda çalışıyor, public URL yukarıda görünecek")

