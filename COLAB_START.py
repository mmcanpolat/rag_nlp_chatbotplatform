#!/usr/bin/env python3
# ============================================
# RAG SaaS Platform - Proje Başlatma
# ============================================
# Bu dosyayı projeyi başlatmak için çalıştırın
# Önce COLAB_INSTALL.py'yi bir kere çalıştırmış olmanız gerekir

import urllib.request
import sys

print("=" * 60)
print("RAG SaaS Platform - Proje Başlatma")
print("=" * 60)
print("\n[*] Proje yükleniyor...")

# GitHub'dan direkt çalıştır
try:
    exec(urllib.request.urlopen('https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_SINGLE_FILE.py').read().decode())
except Exception as e:
    print(f"❌ Hata: {e}")
    print("\nAlternatif: GitHub'dan manuel olarak COLAB_SINGLE_FILE.py dosyasını indirip çalıştırın")
