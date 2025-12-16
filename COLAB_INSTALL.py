#!/usr/bin/env python3
# ============================================
# RAG SaaS Platform - Paket Kurulumu
# ============================================
# Bu dosyayı SADECE BİR KERE çalıştırın
# Tüm bağımlılıkları kurar

import subprocess
import sys

print("=" * 60)
print("RAG SaaS Platform - Paket Kurulumu")
print("=" * 60)
print("\n[1/2] Bağımlılıklar kontrol ediliyor...")

required_packages = [
    "fastapi", "uvicorn[standard]", "gradio>=4.0.0", "langchain", "langchain-community",
    "langchain-huggingface", "langchain-text-splitters", "langchain-core", "transformers", 
    "torch", "sentence-transformers", "faiss-cpu", "pypdf", "docx2txt", "beautifulsoup4", 
    "requests", "python-dotenv"
]

missing = []
for pkg in required_packages:
    try:
        # Paket adını import adına çevir
        pkg_clean = pkg.split("[")[0].split(">=")[0]
        # Özel durumlar
        if pkg_clean == "langchain-text-splitters":
            pkg_import = "langchain_text_splitters"
        elif pkg_clean == "langchain-core":
            pkg_import = "langchain_core"
        else:
            pkg_import = pkg_clean.replace("-", "_")
        __import__(pkg_import)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"   {len(missing)} paket eksik, kuruluyor (5-10 dakika)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing, check=False)
    print("✅ Bağımlılıklar kuruldu")
else:
    print("✅ Tüm bağımlılıklar mevcut")

print("\n[2/2] Kurulum tamamlandı!")
print("\n" + "=" * 60)
print("✅ Paketler hazır!")
print("Şimdi projeyi başlatmak için COLAB_START.py dosyasını çalıştırın")
print("=" * 60)

