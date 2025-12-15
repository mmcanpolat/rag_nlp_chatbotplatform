# ============================================
# RAG SaaS Platform - Colab Hızlı Başlatma
# ============================================
# Projeyi Colab'e yükledikten sonra bu kodu çalıştır

import os
import subprocess
import sys
import time
from pathlib import Path

print("=" * 60)
print("RAG SaaS Platform - Hızlı Başlatma")
print("=" * 60)

# Mevcut dizini kontrol et
current_dir = Path.cwd()
print(f"📁 Mevcut dizin: {current_dir}")

# Proje dizinini bul
if "rag_nlp_chatbotplatform" in str(current_dir):
    project_dir = current_dir
    if project_dir.name != "rag_nlp_chatbotplatform":
        project_dir = project_dir / "rag_nlp_chatbotplatform"
else:
    # Colab'te genelde /content dizininde olur
    project_dir = Path("/content/rag_nlp_chatbotplatform")
    if not project_dir.exists():
        project_dir = Path.cwd() / "rag_nlp_chatbotplatform"

print(f"📁 Proje dizini: {project_dir}")

if not project_dir.exists():
    print("❌ Proje dizini bulunamadı!")
    print("💡 Colab'te Files sekmesinden projeyi yükleyin veya:")
    print("   !git clone https://github.com/mmcanpolat/rag_nlp_chatbotplatform.git")
    sys.exit(1)

os.chdir(project_dir)
print(f"✅ Proje dizinine geçildi: {project_dir}")

# API Key kontrolü
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        from google.colab import userdata
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    except:
        pass

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY bulunamadı!")
    print("💡 Colab Secrets'tan ekleyin veya environment variable olarak ayarlayın")
    # Devam et, belki .env dosyasında var

# .env dosyalarını oluştur
Path("backend_fastapi").mkdir(exist_ok=True)
Path("python_services").mkdir(exist_ok=True)
Path("frontend_gradio/assets/plots").mkdir(parents=True, exist_ok=True)

if OPENAI_API_KEY:
    with open("backend_fastapi/.env", "w") as f:
        f.write(f"PORT=3000\nOPENAI_API_KEY={OPENAI_API_KEY}\nREQUEST_TIMEOUT=600000")
    with open("python_services/.env", "w") as f:
        f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}")
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print("✅ API Key ayarlandı")

os.environ['API_BASE_URL'] = "http://localhost:3000"
os.environ['GRADIO_SHARE'] = "true"

# Eski process'leri durdur
print("\n🔄 Eski process'ler durduruluyor...")
subprocess.run(["pkill", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "gradio"], stderr=subprocess.DEVNULL)
time.sleep(2)

# Backend başlat
print("🚀 Backend başlatılıyor...")
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"],
    cwd="backend_fastapi",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env={**os.environ}
)
time.sleep(3)
print("✅ Backend başlatıldı (port 3000)")

# Frontend (Gradio) başlat
print("🚀 Gradio başlatılıyor...")
print("⏳ Public URL oluşturuluyor (birkaç saniye)...\n")

# Gradio'yu başlat - log dosyasına yaz
gradio_log = "/tmp/gradio.log"
frontend = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd="frontend_gradio",
    stdout=open(gradio_log, "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "API_BASE_URL": "http://localhost:3000", "GRADIO_SHARE": "true"}
)

# URL'yi bekle ve göster
gradio_url = None
for i in range(15):  # 15 saniye bekle
    time.sleep(1)
    try:
        if os.path.exists(gradio_log):
            with open(gradio_log, "r") as f:
                content = f.read()
                if "Running on public URL:" in content:
                    for line in content.split("\n"):
                        if "Running on public URL:" in line:
                            gradio_url = line.split("Running on public URL:")[-1].strip()
                            break
                elif "https://" in content and ("gradio.live" in content or "gradio.app" in content):
                    for line in content.split("\n"):
                        if "https://" in line and ("gradio.live" in line or "gradio.app" in line):
                            for word in line.split():
                                if "https://" in word and ("gradio.live" in word or "gradio.app" in word):
                                    gradio_url = word.strip().rstrip(".,;")
                                    break
                            if gradio_url:
                                break
        if gradio_url:
            break
    except:
        continue

print("\n" + "=" * 60)
print("✅ SERVİSLER BAŞLATILDI!")
print("=" * 60)
print("📍 Backend: http://localhost:3000")
print("📍 Frontend: http://localhost:7860")

if gradio_url:
    print(f"\n🌐 GRADIO PUBLIC URL:")
    print(f"   {gradio_url}")
    print(f"\n   👆 Bu URL'yi kopyalayıp tarayıcıda aç!")
else:
    print("\n⏳ Public URL oluşturuluyor...")
    print("   Birkaç saniye sonra log dosyasını kontrol edin:")
    print(f"   !cat {gradio_log} | grep 'public URL'")

print("\n🔑 Giriş: admin@ragplatform.com / Admin123!@#")
print("=" * 60)

