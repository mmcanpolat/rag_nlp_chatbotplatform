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

# 3. API Key al - Colab Secrets'tan okuma (zorunlu)
print("\n[3/5] API Key Colab Secrets'tan okunuyor...")
OPENAI_API_KEY = None

try:
    from google.colab import userdata
    # Colab Secrets'tan oku
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        print("✅ API key Colab Secrets'tan alındı")
    else:
        # Secrets'ta yoksa hata ver
        raise ValueError("OPENAI_API_KEY bulunamadı")
except ImportError:
    # Colab dışında çalışıyorsa (local test için)
    print("⚠️ Colab ortamı değil, manuel API key isteniyor...")
    OPENAI_API_KEY = getpass("OpenAI API Key girin (görünmez): ")
    if not OPENAI_API_KEY:
        raise ValueError("❌ API Key gerekli!")
except ValueError as e:
    # Secrets'ta key yoksa
    print(f"\n❌ HATA: {str(e)}")
    print("\n📋 Colab Secrets'a API Key ekleme adımları:")
    print("   1. Sol menüden 🔑 Secrets sekmesine tıkla")
    print("   2. + Add Secret butonuna tıkla")
    print("   3. Name: OPENAI_API_KEY (tam olarak bu isim)")
    print("   4. Value: OpenAI API key'ini yapıştır")
    print("   5. Save butonuna tıkla")
    print("   6. Bu hücreyi tekrar çalıştır\n")
    raise ValueError("Colab Secrets'ta 'OPENAI_API_KEY' bulunamadı!")
except Exception as e:
    # Diğer hatalar
    print(f"\n❌ Colab Secrets hatası: {str(e)}")
    print("\n📋 Colab Secrets'a API Key ekleme adımları:")
    print("   1. Sol menüden 🔑 Secrets sekmesine tıkla")
    print("   2. + Add Secret butonuna tıkla")
    print("   3. Name: OPENAI_API_KEY (tam olarak bu isim)")
    print("   4. Value: OpenAI API key'ini yapıştır")
    print("   5. Save butonuna tıkla\n")
    raise

# Son kontrol - API key boş olamaz
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError("❌ API Key boş! Colab Secrets'ta 'OPENAI_API_KEY' eklediğinden emin ol.")

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

# Frontend başlat (Gradio) - public URL için share=True zorunlu
print("⏳ Gradio başlatılıyor (public URL oluşturuluyor)...")
print("   Bu işlem 10-20 saniye sürebilir...")

# Gradio'yu arka planda başlat - stdout'u yakala
gradio_log = "/tmp/gradio.log"
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    stdout=open(gradio_log, "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "API_BASE_URL": "http://localhost:3000", "GRADIO_SHARE": "true"},
    text=True,
    bufsize=1
)

# Gradio başlaması için bekleme - public URL oluşması zaman alabilir
print("📡 Gradio çıktısı dinleniyor (public URL aranıyor)...")
gradio_url = None

# 40 saniye boyunca log dosyasını kontrol et (daha uzun bekleme)
for i in range(40):
    time.sleep(1)
    try:
        if os.path.exists(gradio_log):
            with open(gradio_log, "r") as f:
                content = f.read()
                # Public URL'i ara - farklı formatlar
                if "Running on public URL:" in content:
                    for line in content.split("\n"):
                        if "Running on public URL:" in line:
                            parts = line.split("Running on public URL:")
                            if len(parts) > 1:
                                gradio_url = parts[-1].strip()
                                break
                # Alternatif format - direkt URL satırı
                if not gradio_url and "https://" in content:
                    for line in content.split("\n"):
                        if "https://" in line and ("gradio.live" in line or "gradio.app" in line or "hf.space" in line):
                            # Satırdaki URL'i bul
                            words = line.split()
                            for word in words:
                                if "https://" in word:
                                    # URL'i temizle
                                    url = word.strip().rstrip(".,;")
                                    if "gradio.live" in url or "gradio.app" in url or "hf.space" in url:
                                        gradio_url = url
                                        break
                            if gradio_url:
                                break
                
                if gradio_url:
                    print(f"\n✅ GRADIO PUBLIC URL BULUNDU: {gradio_url}\n")
                    break
    except Exception as e:
        # Hata olursa devam et
        continue

# Eğer hala bulunamadıysa, log dosyasının tamamını göster
if not gradio_url:
    print("\n⚠️ Public URL otomatik bulunamadı. Log dosyası kontrol ediliyor...")
    try:
        if os.path.exists(gradio_log):
            with open(gradio_log, "r") as f:
                content = f.read()
                print("\n📋 Gradio log dosyası içeriği:")
                print("=" * 60)
                print(content[-2000:])  # Son 2000 karakter
                print("=" * 60)
                print("\n💡 Yukarıdaki çıktıda 'Running on public URL:' veya 'https://' içeren satırı ara")
    except Exception as e:
        print(f"Log dosyası okunamadı: {e}")

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

