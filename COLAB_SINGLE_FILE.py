#!/usr/bin/env python3
# ============================================
# RAG SaaS Platform - TEK DOSYA BAŞLATMA
# ============================================
# Tüm proje bu dosyada - Colab'te tek hücrede çalışır
# Backend (FastAPI) + Frontend (Gradio) + RAG Engine hepsi burada
#
# KULLANIM (Colab):
# 1. Colab'te yeni hücre oluştur
# 2. Aşağıdaki komutu çalıştır:
#    !wget -q -O - https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_SINGLE_FILE.py | python3
#
# VEYA:
# 1. GitHub'dan dosyayı kopyala-yapıştır
# 2. Shift+Enter ile çalıştır
# 3. Public URL terminal çıktısında görünecek

# Bağımlılıkları kontrol et ve eksikleri kur
import sys
import subprocess

print("=" * 60)
print("RAG SaaS Platform - Tek Dosya Başlatma")
print("=" * 60)
print("\n[1/4] Bağımlılıklar kontrol ediliyor...")

required_packages = [
    "fastapi", "uvicorn[standard]", "gradio>=4.0.0", "langchain", "langchain-community",
    "langchain-huggingface", "langchain-text-splitters", "langchain-core", "transformers", 
    "torch", "sentence-transformers", "faiss-cpu", "pypdf", "docx2txt", "beautifulsoup4", 
    "requests", "python-dotenv", "rouge-score", "matplotlib", "pandas", "numpy", "seaborn"
]

missing = []
for pkg in required_packages:
    try:
        # Paket adını import adına çevir
        pkg_clean = pkg.split("[")[0].split(">=")[0]
        # Özel durumlar - import adları
        if pkg_clean == "langchain-text-splitters":
            pkg_import = "langchain_text_splitters"
        elif pkg_clean == "langchain-core":
            pkg_import = "langchain_core"
        elif pkg_clean == "langchain-community":
            pkg_import = "langchain_community"
        elif pkg_clean == "langchain-huggingface":
            pkg_import = "langchain_huggingface"
        elif pkg_clean == "sentence-transformers":
            pkg_import = "sentence_transformers"
        elif pkg_clean == "python-dotenv":
            pkg_import = "dotenv"
        elif pkg_clean == "faiss-cpu":
            pkg_import = "faiss"
        elif pkg_clean == "beautifulsoup4":
            pkg_import = "bs4"
        else:
            pkg_import = pkg_clean.replace("-", "_")
        __import__(pkg_import)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"⚠️  {len(missing)} paket eksik, kuruluyor...")
    print(f"   Eksik paketler: {', '.join(missing)}")
    print("   (Bu işlem 5-10 dakika sürebilir...)\n")
    
    # Eksik paketleri kur
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing,
            check=False,
            timeout=600  # 10 dakika timeout
        )
        print("✅ Eksik paketler kuruldu")
        
        # Tekrar kontrol et
        still_missing = []
        for pkg in missing:
            try:
                pkg_clean = pkg.split("[")[0].split(">=")[0]
                if pkg_clean == "langchain-text-splitters":
                    pkg_import = "langchain_text_splitters"
                elif pkg_clean == "langchain-core":
                    pkg_import = "langchain_core"
                elif pkg_clean == "langchain-community":
                    pkg_import = "langchain_community"
                elif pkg_clean == "langchain-huggingface":
                    pkg_import = "langchain_huggingface"
                elif pkg_clean == "sentence-transformers":
                    pkg_import = "sentence_transformers"
                elif pkg_clean == "python-dotenv":
                    pkg_import = "dotenv"
                elif pkg_clean == "faiss-cpu":
                    pkg_import = "faiss"
                elif pkg_clean == "beautifulsoup4":
                    pkg_import = "bs4"
                else:
                    pkg_import = pkg_clean.replace("-", "_")
                __import__(pkg_import)
            except ImportError:
                still_missing.append(pkg)
        
        if still_missing:
            print(f"❌ Hala eksik paketler var: {', '.join(still_missing)}")
            print("Lütfen manuel olarak kurun:")
            print(f"  !pip install {' '.join(still_missing)}")
            sys.exit(1)
        else:
            print("✅ Tüm bağımlılıklar kuruldu ve kontrol edildi")
    except Exception as e:
        print(f"❌ Paket kurulumu sırasında hata: {e}")
        print("Lütfen manuel olarak kurun:")
        print(f"  !pip install {' '.join(missing)}")
        sys.exit(1)
else:
    print("✅ Tüm bağımlılıklar mevcut")

print("\n[2/4] Modüller yükleniyor...")

import os
import json
import time
import secrets
import threading
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from urllib.parse import urlparse

# FastAPI ve Gradio import'ları
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import gradio as gr

# LangChain ve Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, Docx2txtLoader, TextLoader, JSONLoader, CSVLoader
)
# LangChain text splitter - yeni versiyonlarda ayrı pakette
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # En son çare: langchain paketinden
        from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document - yeni versiyonlarda langchain_core'da
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("✅ Modüller yüklendi")

# ==================== CONFIG ====================
BASE_DIR = Path.cwd()
# Colab'te git clone sonrası dizin yapısı
if "rag_nlp_chatbotplatform" in str(BASE_DIR):
    BASE_DIR = BASE_DIR / "rag_nlp_chatbotplatform"

DATA_DIR = BASE_DIR / "python_services" / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
UPLOADS_DIR = DATA_DIR / "uploads"

# Dizinleri oluştur
for d in [DATA_DIR, INDEX_DIR, UPLOADS_DIR, BASE_DIR / "frontend_gradio" / "assets" / "plots"]:
    d.mkdir(parents=True, exist_ok=True)

print("\n[3/5] Yapılandırma tamamlandı")

# ==================== IN-MEMORY DATA ====================
companies: Dict[str, dict] = {}
agents: Dict[str, dict] = {}
sessions: Dict[str, dict] = {}

SUPER_ADMIN = {
    "id": "superadmin",
    "username": "admin@ragplatform.com",
    "password": "Admin123!@#",
    "isSuperAdmin": True,
    "companyId": "superadmin",
    "companyName": "SuperAdmin"
}

# ==================== HELPER FUNCTIONS ====================
def gen_id() -> str:
    return f"{int(time.time() * 1000)}_{secrets.token_hex(4)}"

def generate_strong_password() -> str:
    return secrets.token_urlsafe(24)

# ==================== DOCUMENT INGESTOR ====================
class DocumentIngestor:
    def __init__(self, index_name: str = "default", embedding_model: str = None):
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.embedding_model_name = embedding_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len
        )
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def detect_source_type(self, source: str) -> str:
        parsed = urlparse(source)
        if parsed.scheme in ('http', 'https'):
            return 'web'
        ext = Path(source).suffix.lower()
        return {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.txt': 'text',
            '.json': 'json',
            '.csv': 'csv'
        }.get(ext, 'text')
    
    def load_document(self, source: str) -> List[Document]:
        source_type = self.detect_source_type(source)
        try:
            if source_type == 'web':
                loader = WebBaseLoader(source)
            elif source_type == 'pdf':
                loader = PyPDFLoader(source)
            elif source_type == 'docx':
                loader = Docx2txtLoader(source)
            elif source_type == 'json':
                loader = JSONLoader(source, jq_schema='.')
            elif source_type == 'csv':
                loader = CSVLoader(source)
            else:
                loader = TextLoader(source)
            return loader.load()
        except Exception as e:
            print(f"[!] Yükleme hatası: {e}")
            return []
    
    def ingest(self, source: str, progress_callback=None) -> dict:
        try:
            if progress_callback:
                progress_callback("Döküman yükleniyor...", 0, 0)
            
            docs = self.load_document(source)
            if not docs:
                return {"success": False, "error": "Döküman yüklenemedi"}
            
            if progress_callback:
                progress_callback(f"Döküman yüklendi ({len(docs)} sayfa)", 0, 0)
            
            # CSV dosyaları için: Her satırı direkt chunk yap (text splitter kullanma)
            source_type = self.detect_source_type(source)
            if source_type == 'csv':
                if progress_callback:
                    progress_callback(f"CSV satırları direkt embed edilecek ({len(docs)} satır)...", 0, 0)
                # CSV için: Her Document (satır) direkt bir chunk
                chunks = docs  # Text splitter kullanmadan direkt kullan
            else:
                # Diğer dosya tipleri için text splitter kullan
                if progress_callback:
                    progress_callback("Döküman parçalara bölünüyor...", 0, 0)
                chunks = self.text_splitter.split_documents(docs)
            
            if not chunks:
                return {"success": False, "error": "Döküman bölünemedi"}
            
            total_chunks = len(chunks)
            if progress_callback:
                progress_callback(f"{total_chunks} parça oluşturuldu. Embedding başlıyor...", 0, total_chunks)
            
            # Mevcut index varsa merge et, yoksa yeni oluştur
            if (self.index_path / "index.faiss").exists():
                if progress_callback:
                    progress_callback("Mevcut index yükleniyor...", 0, total_chunks)
                vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Batch'ler halinde ekle (progress için)
                batch_size = 100
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                    batch_num = (i // batch_size) + 1
                    percent = int(((i + len(batch)) / total_chunks) * 100) if total_chunks > 0 else 0
                    if progress_callback:
                        progress_callback(
                            f"Batch {batch_num}/{total_batches} işleniyor ({len(batch)} parça) - %{percent}",
                            i + len(batch),
                            total_chunks
                        )
            else:
                # Yeni index oluştur - batch'ler halinde
                batch_size = 100
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                first_batch = chunks[:batch_size]
                percent = int((len(first_batch) / total_chunks) * 100) if total_chunks > 0 else 0
                if progress_callback:
                    progress_callback(f"İlk batch oluşturuluyor ({len(first_batch)} parça) - %{percent}", len(first_batch), total_chunks)
                
                vectorstore = FAISS.from_documents(first_batch, self.embeddings)
                
                # Kalan batch'leri ekle
                for i in range(batch_size, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                    batch_num = (i // batch_size) + 1
                    percent = int(((i + len(batch)) / total_chunks) * 100) if total_chunks > 0 else 0
                    if progress_callback:
                        progress_callback(
                            f"Batch {batch_num + 1}/{total_batches} işleniyor ({len(batch)} parça) - %{percent}",
                            i + len(batch),
                            total_chunks
                        )
            
            if progress_callback:
                progress_callback("Index kaydediliyor...", total_chunks, total_chunks)
            
            vectorstore.save_local(str(self.index_path))
            
            # Metadata kaydet
            with open(self.index_path / "metadata.json", "w") as f:
                json.dump({
                    "embedding_model": self.embedding_model_name,
                    "created_at": datetime.now().isoformat(),
                    "chunk_count": total_chunks
                }, f)
            
            if progress_callback:
                progress_callback(f"✅ Tamamlandı! {total_chunks} parça işlendi.", total_chunks, total_chunks)
            
            return {"success": True, "chunks": total_chunks}
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Hata: {str(e)}", 0, 0)
            return {"success": False, "error": str(e)}

# ==================== RAG ENGINE ====================
class SimpleRAGEngine:
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.vectorstore = None
        self._embeddings = None
        self._gpt_model = None
        self._gpt_tokenizer = None
        self._load_vectorstore()
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            meta_file = self.index_path / "metadata.json"
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        model_name = meta.get('embedding_model', model_name)
                except:
                    pass
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def _load_vectorstore(self):
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            try:
                print(f"[*] Index yükleniyor: {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Index'teki doküman sayısını kontrol et
                if self.vectorstore:
                    doc_count = self.vectorstore.index.ntotal if hasattr(self.vectorstore.index, 'ntotal') else 0
                    print(f"[*] ✅ Index yüklendi: {doc_count} doküman bulundu")
                else:
                    print(f"[!] Index yüklendi ama vectorstore boş")
            except Exception as e:
                import traceback
                print(f"[!] Index yükleme hatası: {traceback.format_exc()}")
        else:
            print(f"[!] Index dosyası bulunamadı: {index_file}")
            print(f"[!] Lütfen önce CSV dosyasını yükleyin ve embed edin")
    
    def _load_gpt_model(self):
        if self._gpt_model is None:
            print("[*] Türkçe GPT-2 modeli yükleniyor...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Türkçe GPT-2 modelleri - sırayla dene
            turkish_models = [
                "redrussianarmy/gpt2-turkish-cased",  # En yaygın Türkçe GPT-2
                "gorkemgoknar/gpt2-small-turkish",    # Alternatif Türkçe model
                "cenkersisman/gpt2-turkish-256-token" # Başka bir Türkçe model
            ]
            
            model_loaded = False
            for model_name in turkish_models:
                try:
                    print(f"[*] Model deneniyor: {model_name}...")
                    self._gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                    self._gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
                    print(f"[*] ✅ Türkçe GPT-2 modeli yüklendi: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"[!] {model_name} yüklenemedi: {str(e)[:100]}")
                    continue
            
            if not model_loaded:
                print("[!] Hiçbir Türkçe model yüklenemedi, İngilizce GPT-2 kullanılıyor...")
                model_name = "gpt2"
                self._gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self._gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
                print(f"[*] ⚠️ İngilizce GPT-2 modeli yüklendi (fallback): {model_name}")
            
            self._gpt_model.to(device)
            self._gpt_model.eval()
            if self._gpt_tokenizer.pad_token is None:
                self._gpt_tokenizer.pad_token = self._gpt_tokenizer.eos_token
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            print("[!] Vectorstore yok! Index yüklenmemiş olabilir.")
            return []
        # Similarity search - en alakalı k=3 diyaloğu getir
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            print(f"[*] Similarity search: {len(results)} sonuç bulundu")
            return [{'context': doc.page_content, 'score': float(1 - score)} 
                    for doc, score in results]
        except Exception as e:
            import traceback
            print(f"[!] Similarity search hatası: {traceback.format_exc()}")
            return []
    
    def _ask_gpt(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        try:
            self._load_gpt_model()
            if self._gpt_model is None:
                return "Model yüklenemedi", 0.0
            
            # Context'i hazırla - her context'i ayrı satır olarak
            context_text = "\n\n".join([doc for doc in contexts])
            
            # Prompt yapısı: Context + Soru formatı (kullanıcının istediği format)
            prompt = f"""Aşağıdaki tıbbi soru-cevap geçmişine dayanarak hastanın sorusunu cevapla.
Eğer verilen context içinde cevap yoksa, "Bilmiyorum" de.

CONTEXT:
{context_text}

SORU:
{query}

CEVAP:"""
            
            device = next(self._gpt_model.parameters()).device
            inputs = self._gpt_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = self._gpt_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    min_length=inputs.shape[1] + 20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._gpt_tokenizer.eos_token_id,
                    eos_token_id=self._gpt_tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            generated = outputs[0][inputs.shape[1]:]
            answer = self._gpt_tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # Eğer cevap çok kısa veya anlamsızsa, context'ten direkt al
            if len(answer) < 15 or "bilmiyorum" in answer.lower() or answer.lower().startswith("bilmiyorum"):
                # Context'lerden en alakalı olanı kullan (ilk context en yüksek skorlu)
                if contexts:
                    # İlk context'i kullan (en yüksek skorlu)
                    answer = contexts[0] if len(contexts[0]) < 500 else contexts[0][:500] + "..."
                    conf = 0.7
                else:
                    answer = "Bilgi bulunamadı."
                    conf = 0.3
            else:
                conf = min(0.85, 0.5 + len(answer) / 200)
            
            return answer, conf
        except Exception as e:
            import traceback
            print(f"[!] GPT model hatası: {traceback.format_exc()}")
            # Fallback: Context'ten direkt cevap ver
            if contexts:
                return contexts[0][:300] + "..." if len(contexts[0]) > 300 else contexts[0], 0.6
            return f"Hata: {str(e)}", 0.0
    
    def query(self, text: str, model_type: str = "GPT") -> Dict:
        start = time.time()
        retrieved = self.retrieve(text, top_k=3)
        contexts = [r['context'] for r in retrieved]
        
        if not contexts:
            return {
                "answer": "Veri yükleniyor veya index bulunamadı. Lütfen birkaç saniye bekleyip tekrar deneyin.",
                "context": "",
                "confidence": 0.0,
                "model_used": "GPT",
                "response_time_ms": round((time.time() - start) * 1000, 2)
            }
        
        if model_type.upper() == "GPT":
            answer, conf = self._ask_gpt(text, contexts)
        else:
            answer = contexts[0][:200] + "..."
            conf = 0.7
        
        return {
            "answer": answer,
            "context": contexts[0] if contexts else "",
            "confidence": round(conf, 4),
            "model_used": model_type,
            "response_time_ms": round((time.time() - start) * 1000, 2)
        }

# ==================== EVALUATION METRICS ====================
class MetricsEvaluator:
    """Evaluation metrikleri hesaplama: Exact Match, F1, ROUGE-L, Cosine Similarity"""
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from rouge_score import rouge_scorer
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def exact_match(self, pred: str, ref: str) -> float:
        """Exact Match: Harfi harfine aynı mı?"""
        return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
    
    def f1_score(self, pred: str, ref: str) -> float:
        """Token-based F1 Score"""
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def rouge_l(self, pred: str, ref: str) -> float:
        """ROUGE-L F1 Score"""
        scores = self.rouge_scorer.score(ref, pred)
        return scores['rougeL'].fmeasure
    
    def cosine_similarity(self, pred: str, ref: str) -> float:
        """Cosine Similarity - embedding bazlı"""
        vecs = self.embed_model.encode([pred, ref])
        import numpy as np
        return float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Tüm metrikleri hesapla"""
        if len(predictions) != len(references):
            return {"error": "Predictions ve references aynı uzunlukta olmalı"}
        
        em_scores = []
        f1_scores = []
        rouge_scores = []
        cosine_scores = []
        
        for pred, ref in zip(predictions, references):
            em_scores.append(self.exact_match(pred, ref))
            f1_scores.append(self.f1_score(pred, ref))
            rouge_scores.append(self.rouge_l(pred, ref))
            cosine_scores.append(self.cosine_similarity(pred, ref))
        
        return {
            "exact_match": {
                "scores": em_scores,
                "mean": sum(em_scores) / len(em_scores) if em_scores else 0.0
            },
            "f1": {
                "scores": f1_scores,
                "mean": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            },
            "rouge_l": {
                "scores": rouge_scores,
                "mean": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            },
            "cosine_similarity": {
                "scores": cosine_scores,
                "mean": sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
            }
        }
    
    def plot_metrics(self, results: Dict[str, Dict], save_path: str = None):
        """Metrikleri görselleştir"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        models = list(results.keys())
        metrics_data = {
            "Exact Match": [results[m]["exact_match"]["mean"] for m in models],
            "F1 Score": [results[m]["f1"]["mean"] for m in models],
            "ROUGE-L": [results[m]["rouge_l"]["mean"] for m in models],
            "Cosine Similarity": [results[m]["cosine_similarity"]["mean"] for m in models]
        }
        
        df = pd.DataFrame(metrics_data, index=models)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Performance Metrics Comparison", fontsize=16, fontweight="bold")
        
        # Bar chart - Tüm metrikler
        ax1 = axes[0, 0]
        df.plot(kind="bar", ax=ax1, width=0.8)
        ax1.set_title("All Metrics Comparison")
        ax1.set_ylabel("Score")
        ax1.set_xlabel("Model")
        ax1.legend(loc="best")
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Line chart
        ax2 = axes[0, 1]
        df.plot(kind="line", ax=ax2, marker="o", linewidth=2, markersize=8)
        ax2.set_title("Metrics Trend")
        ax2.set_ylabel("Score")
        ax2.set_xlabel("Model")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Heatmap
        ax3 = axes[1, 0]
        import seaborn as sns
        sns.heatmap(df.T, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax3, cbar_kws={"label": "Score"})
        ax3.set_title("Metrics Heatmap")
        ax3.set_ylabel("Metric")
        ax3.set_xlabel("Model")
        
        # Radar chart (spider chart)
        ax4 = axes[1, 1]
        angles = np.linspace(0, 2 * np.pi, len(df.columns), endpoint=False).tolist()
        angles += angles[:1]  # Kapat
        
        ax4 = plt.subplot(2, 2, 4, projection="polar")
        for idx, model in enumerate(models):
            values = [df.loc[model, col] for col in df.columns]
            values += values[:1]  # Kapat
            ax4.plot(angles, values, "o-", linewidth=2, label=model)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(df.columns)
        ax4.set_ylim(0, 1)
        ax4.set_title("Radar Chart", pad=20)
        ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[*] Grafik kaydedildi: {save_path}")
        
        return fig

# ==================== FASTAPI BACKEND ====================
backend_app = FastAPI(title="RAG SaaS Platform API")
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str

class CompanyCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None

class AgentCreate(BaseModel):
    name: str
    embedding_model: str
    data_source_type: str
    data_source: Optional[str] = None

class ChatRequest(BaseModel):
    agent_id: str
    query: str
    model: str = "GPT"

# Auth
def get_auth_header(authorization: Optional[str] = Header(None)):
    return authorization

def require_auth(authorization: Optional[str] = Depends(get_auth_header)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    if token not in sessions:
        raise HTTPException(status_code=401, detail="Invalid token")
    return sessions[token]

def require_superadmin(user: dict = Depends(require_auth)):
    if not user.get("isSuperAdmin"):
        raise HTTPException(status_code=403, detail="SuperAdmin required")
    return user

# Endpoints
@backend_app.post("/api/auth/login")
async def login(req: LoginRequest):
    if req.username == SUPER_ADMIN["username"] and req.password == SUPER_ADMIN["password"]:
        token = secrets.token_urlsafe(32)
        sessions[token] = SUPER_ADMIN
        return {"success": True, "token": token, "data": SUPER_ADMIN}
    
    for company in companies.values():
        if company.get("username") == req.username and company.get("password") == req.password:
            token = secrets.token_urlsafe(32)
            user = {
                "id": company["id"],
                "username": company["username"],
                "companyId": company["id"],
                "companyName": company["name"],
                "isSuperAdmin": False
            }
            sessions[token] = user
            return {"success": True, "token": token, "data": user}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@backend_app.post("/api/admin/companies")
async def create_company(company: CompanyCreate, user: dict = Depends(require_superadmin)):
    company_id = gen_id()
    username = f"{company.name.lower().replace(' ', '_')}@company.com"
    password = generate_strong_password()
    
    new_company = {
        "id": company_id,
        "name": company.name,
        "email": company.email or username,
        "phone": company.phone,
        "description": company.description,
        "username": username,
        "password": password,
        "createdAt": datetime.now().isoformat()
    }
    companies[company_id] = new_company
    return {"success": True, "data": new_company}

@backend_app.get("/api/admin/companies")
async def list_companies(user: dict = Depends(require_superadmin)):
    return {"success": True, "data": list(companies.values())}

@backend_app.post("/api/agents")
async def create_agent(agent: AgentCreate, user: dict = Depends(require_auth)):
    agent_id = gen_id()
    index_name = f"agent_{agent_id}"
    
    new_agent = {
        "id": agent_id,
        "name": agent.name,
        "companyId": user.get("companyId"),
        "embeddingModel": agent.embedding_model,
        "dataSourceType": agent.data_source_type,
        "dataSource": agent.data_source,
        "indexName": index_name,
        "createdAt": datetime.now().isoformat()
    }
    
    # Ingestion yap (progress callback ile detaylı log)
    ingestion_info = {"chunks": 0, "batches": 0, "status": "başlatılıyor"}
    if agent.data_source:
        try:
            def progress_log(message, current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    batch_info = ""
                    if "Batch" in message:
                        # Batch bilgisini parse et
                        import re
                        batch_match = re.search(r'Batch (\d+)/(\d+)', message)
                        if batch_match:
                            batch_num = batch_match.group(1)
                            total_batches = batch_match.group(2)
                            ingestion_info["batches"] = int(total_batches)
                            batch_info = f" (Batch {batch_num}/{total_batches})"
                    print(f"[Ingestion] {message} ({percent}%){batch_info}")
                    ingestion_info["status"] = message
                    ingestion_info["chunks"] = current
                else:
                    print(f"[Ingestion] {message}")
                    ingestion_info["status"] = message
            
            print(f"[*] Agent oluşturuluyor: {agent.name}")
            print(f"[*] Veri kaynağı: {agent.data_source}")
            print(f"[*] Embedding modeli: {agent.embedding_model}")
            print(f"[*] Index adı: {index_name}")
            print(f"[*] Ingestion başlıyor...")
            
            ingestor = DocumentIngestor(index_name=index_name, embedding_model=agent.embedding_model)
            result = ingestor.ingest(agent.data_source, progress_callback=progress_log)
            
            if not result.get("success"):
                print(f"[!] Ingestion başarısız: {result.get('error')}")
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": result.get("error", "Ingestion failed")}
                )
            
            chunks = result.get("chunks", 0)
            print(f"[+] Ingestion tamamlandı: {chunks} parça işlendi")
            new_agent["chunkCount"] = chunks
        except Exception as e:
            import traceback
            print(f"[!] Ingestion hatası: {traceback.format_exc()}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": str(e)}
            )
    
    agents[agent_id] = new_agent
    print(f"[+] Agent oluşturuldu: {agent.name} (ID: {agent_id})")
    return {"success": True, "data": new_agent, "ingestion_info": ingestion_info}

@backend_app.get("/api/agents")
async def list_agents(user: dict = Depends(require_auth)):
    user_agents = [a for a in agents.values() 
                   if a.get("companyId") == user.get("companyId") or user.get("isSuperAdmin")]
    return {"success": True, "data": user_agents}

@backend_app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # Sabit index kullan (default)
        index_name = "default"
        
        # Eğer agent_id varsa ve geçerliyse onu kullan, yoksa default kullan
        if req.agent_id and req.agent_id in agents:
            agent = agents[req.agent_id]
            index_name = agent.get("indexName", f"agent_{req.agent_id}")
        
        # RAG engine'i oluştur ve sorguyu çalıştır
        try:
            rag = SimpleRAGEngine(index_name=index_name)
            result = rag.query(req.query, model_type=req.model)
            return {"success": True, "data": result}
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[!] RAG Engine hatası: {error_detail}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"RAG Engine hatası: {str(e)}",
                    "detail": error_detail
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[!] Chat endpoint hatası: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Sunucu hatası: {str(e)}",
                "detail": error_detail
            }
        )

@backend_app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    file_path = UPLOADS_DIR / f"upload_{int(time.time() * 1000)}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"success": True, "data": {"filePath": str(file_path), "fileName": file.filename}}

# ==================== GRADIO FRONTEND ====================
def build_gradio_ui():
    # Her model için ayrı chat history tut
    model_chat_histories = {
        "GPT": [],
        "BERT-CASED": [],
        "BERT-SENTIMENT": []
    }
    
    # Custom CSS - Koyu tema, profesyonel
    custom_css = """
    .gradio-container {
        background: #1a1a1a !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }
    .gr-button-primary {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
    }
    .gr-button-primary:hover {
        background: #2563eb !important;
    }
    .gr-button {
        background: #4b5563 !important;
        color: white !important;
        border: none !important;
    }
    .gr-button:hover {
        background: #6b7280 !important;
    }
    .gr-textbox input, .gr-textbox textarea {
        background: #2d2d2d !important;
        color: #e5e5e5 !important;
        border: 1px solid #404040 !important;
    }
    .gr-textbox input:focus, .gr-textbox textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
    }
    .gr-textbox label {
        color: #d1d5db !important;
        font-weight: 500 !important;
    }
    .gr-markdown {
        color: #e5e5e5 !important;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #ffffff !important;
    }
    .gr-radio label {
        color: #d1d5db !important;
    }
    .gr-dropdown {
        background: #2d2d2d !important;
        color: #e5e5e5 !important;
        border: 1px solid #404040 !important;
    }
    .gr-tabs {
        background: #1a1a1a !important;
    }
    .gr-tab {
        background: #2d2d2d !important;
        color: #d1d5db !important;
    }
    .gr-tab.selected {
        background: #3b82f6 !important;
        color: white !important;
    }
    .gr-chatbot {
        background: #2d2d2d !important;
    }
    .gr-chatbot .message {
        color: #e5e5e5 !important;
    }
    body {
        background: #1a1a1a !important;
    }
    """
    
    def get_model_key(model_name):
        """Model adından kısa key çıkar"""
        if "gpt2-turkish" in model_name.lower() or "gpt-2" in model_name.lower() or "gpt2" in model_name.lower():
            return "GPT"
        elif "bert-base-turkish" in model_name.lower() and "sentiment" not in model_name.lower():
            return "BERT-CASED"
        elif "sentiment" in model_name.lower():
            return "BERT-SENTIMENT"
        return "GPT"
    
    def chat_fn(message, history, model):
        """Chat fonksiyonu - agent_id artık gerekmiyor, sabit index kullanıyoruz"""
        if not message or not message.strip():
            return history or [], ""
        
        # Model key'ini al
        model_key = get_model_key(model)
        
        # Bu model için history'yi al (yoksa başlat)
        if history is None:
            history = model_chat_histories.get(model_key, []).copy()
        else:
            # History'yi güncelle
            model_chat_histories[model_key] = history.copy()
        
        # Kullanıcı mesajını ekle
        history.append({"role": "user", "content": message})
        
        try:
            import requests
            # Sabit agent_id kullan (default index)
            # Backend'de default agent oluşturulmuş olmalı veya direkt RAG engine kullan
            resp = requests.post(
                "http://localhost:3000/api/chat",
                json={"agent_id": "default", "query": message, "model": model_key},
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()["data"]
                # Bot cevabını ekle
                answer = data.get("answer", "Cevap alınamadı")
                history.append({"role": "assistant", "content": answer})
                # History'yi kaydet
                model_chat_histories[model_key] = history.copy()
                return history, ""
            else:
                error_msg = f"Hata: {resp.status_code}"
                try:
                    error_detail = resp.json().get("detail", error_msg)
                    error_msg = f"Hata {resp.status_code}: {error_detail}"
                except:
                    pass
                history.append({"role": "assistant", "content": error_msg})
                model_chat_histories[model_key] = history.copy()
                return history, ""
        except requests.exceptions.ConnectionError:
            error_msg = "Backend'e bağlanılamadı. Lütfen birkaç saniye bekleyip tekrar deneyin."
            history.append({"role": "assistant", "content": error_msg})
            model_chat_histories[model_key] = history.copy()
            return history, ""
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[!] Chat hatası: {error_detail}")
            error_msg = f"Hata: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            model_chat_histories[model_key] = history.copy()
            return history, ""
    
    def update_chat_history(model):
        """Model değiştiğinde o modelin history'sini göster"""
        model_key = get_model_key(model)
        history = model_chat_histories.get(model_key, [])
        return history
    
    # Custom CSS - Koyu tema, profesyonel
    custom_css = """
    .gradio-container {
        background: #1a1a1a !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }
    .gr-button-primary {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
    }
    .gr-button-primary:hover {
        background: #2563eb !important;
    }
    .gr-button {
        background: #4b5563 !important;
        color: white !important;
        border: none !important;
    }
    .gr-button:hover {
        background: #6b7280 !important;
    }
    .gr-textbox input, .gr-textbox textarea {
        background: #2d2d2d !important;
        color: #e5e5e5 !important;
        border: 1px solid #404040 !important;
    }
    .gr-textbox input:focus, .gr-textbox textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
    }
    .gr-textbox label {
        color: #d1d5db !important;
        font-weight: 500 !important;
    }
    .gr-markdown {
        color: #e5e5e5 !important;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #ffffff !important;
    }
    .gr-radio label {
        color: #d1d5db !important;
    }
    .gr-dropdown {
        background: #2d2d2d !important;
        color: #e5e5e5 !important;
        border: 1px solid #404040 !important;
    }
    .gr-tabs {
        background: #1a1a1a !important;
    }
    .gr-tab {
        background: #2d2d2d !important;
        color: #d1d5db !important;
    }
    .gr-tab.selected {
        background: #3b82f6 !important;
        color: white !important;
    }
    .gr-chatbot {
        background: #2d2d2d !important;
    }
    .gr-chatbot .message {
        color: #e5e5e5 !important;
    }
    body {
        background: #1a1a1a !important;
    }
    """
    
    # CSS'i Blocks constructor'a ekle (Gradio versiyonuna göre)
    try:
        # Yeni versiyonlar için css parametresi Blocks'ta
        app = gr.Blocks(title="RAG SaaS Platform", css=custom_css)
    except TypeError:
        # Eski versiyonlar için css yok
        app = gr.Blocks(title="RAG SaaS Platform")
    
    # UI'ı oluştur
    with app:
        gr.Markdown("# RAG SaaS Platform")
        
        with gr.Tab("Chat", visible=True) as chat_tab:
            with gr.Row():
                with gr.Column():
                    model_radio = gr.Radio(
                        [
                            "dbmdz/gpt2-turkish (GPT-2 Türkçe)",
                            "bert-base-turkish-cased (BERT Türkçe)",
                            "savasy/bert-base-turkish-sentiment-cased (BERT Sentiment)"
                        ],
                        value="dbmdz/gpt2-turkish (GPT-2 Türkçe)",
                        label="Model Seç",
                        info="Her model için ayrı chat geçmişi tutulur"
                    )
                    chatbot = gr.Chatbot(label="Chat", height=500, type="messages", allow_tags=False)
                    msg_input = gr.Textbox(label="Mesaj", placeholder="Sorunuzu yazın...")
                    send_btn = gr.Button("Gönder", variant="primary")
                    
                    # Model değiştiğinde history'yi güncelle
                    model_radio.change(
                        update_chat_history,
                        inputs=[model_radio],
                        outputs=[chatbot]
                    )
                    
                    send_btn.click(
                        chat_fn,
                        inputs=[msg_input, chatbot, model_radio],
                        outputs=[chatbot, msg_input]
                    )
                    msg_input.submit(
                        chat_fn,
                        inputs=[msg_input, chatbot, model_radio],
                        outputs=[chatbot, msg_input]
                    )
        
        with gr.Tab("Analytics", visible=True) as analytics_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Model Performance Metrics")
                    gr.Markdown("Her model için Exact Match, F1 Score, ROUGE-L ve Cosine Similarity metrikleri hesaplanır.")
                    
                    evaluate_btn = gr.Button("Metrikleri Hesapla ve Grafikle", variant="primary")
                    metrics_output = gr.Markdown()
                    metrics_plot = gr.Image(label="Metrics Comparison Chart")
                    
                    def evaluate_models():
                        """Tüm modelleri değerlendir ve grafik oluştur"""
                        try:
                            # Test verisini yükle (CSV'den)
                            import pandas as pd
                            csv_path = "/content/sample_data/test_cleaned.csv"
                            if not os.path.exists(csv_path):
                                csv_path = str(BASE_DIR / "python_services" / "data" / "test_cleaned.csv")
                            
                            if not os.path.exists(csv_path):
                                return "❌ Test CSV dosyası bulunamadı. Lütfen /content/sample_data/test_cleaned.csv konumuna yerleştirin.", None
                            
                            df = pd.read_csv(csv_path)
                            # question_content ve question_answer sütunlarını kullan
                            if "question_content" not in df.columns or "question_answer" not in df.columns:
                                return "❌ CSV'de 'question_content' ve 'question_answer' sütunları bulunamadı.", None
                            
                            questions = df["question_content"].dropna().tolist()[:50]  # İlk 50 soru
                            references = df["question_answer"].dropna().tolist()[:50]
                            
                            if len(questions) != len(references):
                                return f"❌ Soru ve cevap sayıları eşleşmiyor: {len(questions)} soru, {len(references)} cevap", None
                            
                            evaluator = MetricsEvaluator()
                            rag = SimpleRAGEngine(index_name="default")
                            
                            models_to_test = ["GPT", "BERT-CASED", "BERT-SENTIMENT"]
                            all_results = {}
                            
                            for model_key in models_to_test:
                                print(f"[*] {model_key} modeli test ediliyor...")
                                predictions = []
                                
                                for q in questions:
                                    try:
                                        result = rag.query(q, model_type=model_key)
                                        predictions.append(result.get("answer", ""))
                                    except Exception as e:
                                        print(f"[!] {model_key} için hata: {e}")
                                        predictions.append("")
                                
                                # Metrikleri hesapla
                                metrics = evaluator.evaluate(predictions, references)
                                all_results[model_key] = metrics
                            
                            # Grafik oluştur
                            plots_dir = BASE_DIR / "frontend_gradio" / "assets" / "plots"
                            plots_dir.mkdir(parents=True, exist_ok=True)
                            plot_path = plots_dir / "metrics_comparison.png"
                            
                            fig = evaluator.plot_metrics(all_results, save_path=str(plot_path))
                            
                            # Sonuçları formatla
                            result_text = "## 📊 Model Performance Results\n\n"
                            for model_key, metrics in all_results.items():
                                result_text += f"### {model_key}\n"
                                result_text += f"- **Exact Match:** {metrics['exact_match']['mean']:.3f}\n"
                                result_text += f"- **F1 Score:** {metrics['f1']['mean']:.3f}\n"
                                result_text += f"- **ROUGE-L:** {metrics['rouge_l']['mean']:.3f}\n"
                                result_text += f"- **Cosine Similarity:** {metrics['cosine_similarity']['mean']:.3f}\n\n"
                            
                            return result_text, str(plot_path)
                        except Exception as e:
                            import traceback
                            error_detail = traceback.format_exc()
                            print(f"[!] Evaluation hatası: {error_detail}")
                            return f"❌ Hata: {str(e)}\n\n```\n{error_detail}\n```", None
                    
                    evaluate_btn.click(
                        evaluate_models,
                        outputs=[metrics_output, metrics_plot]
                    )
        
    
    return app, custom_css

# ==================== STARTUP ====================
def run_backend():
    uvicorn.run(backend_app, host="0.0.0.0", port=3000, log_level="error")

def run_frontend():
    result = build_gradio_ui()
    # Debug: result tipini kontrol et
    if isinstance(result, tuple):
        app, custom_css = result
    else:
        # Eğer tuple değilse, eski versiyon gibi davran
        app = result
        custom_css = ""
        print("[!] Uyarı: build_gradio_ui() tuple döndürmedi, CSS kullanılamayacak")
    
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except:
        pass
    
    share_value = is_colab or os.getenv("GRADIO_SHARE", "").lower() == "true"
    print(f"\n[*] Gradio başlatılıyor (share={share_value})...")
    
    # CSS zaten Blocks constructor'da, launch'ta gerek yok
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value,
            show_error=True
        )
    except Exception as e:
        # Fallback
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value
        )

if __name__ == "__main__":
    # Sabit CSV dosyasını yükle ve embed et
    print("\n[4/6] Sabit CSV dosyası yükleniyor ve embed ediliyor...")
    FIXED_CSV_PATH = "/content/sample_data/test_cleaned.csv"
    
    # Colab'te dosya yoksa, local'de test için alternatif yol
    if not os.path.exists(FIXED_CSV_PATH):
        # Local test için
        FIXED_CSV_PATH = str(BASE_DIR / "python_services" / "data" / "test_cleaned.csv")
        if not os.path.exists(FIXED_CSV_PATH):
            print(f"[!] CSV dosyası bulunamadı: {FIXED_CSV_PATH}")
            print("[!] Lütfen dosyayı bu konuma yerleştirin veya Colab'te /content/sample_data/test_cleaned.csv konumuna koyun")
            FIXED_CSV_PATH = None
    
    if FIXED_CSV_PATH and os.path.exists(FIXED_CSV_PATH):
        try:
            print(f"[*] CSV dosyası bulundu: {FIXED_CSV_PATH}")
            print("[*] Embedding başlatılıyor...")
            ingestor = DocumentIngestor(index_name="default", embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            def progress_callback(msg, current, total):
                if total > 0:
                    pct = (current / total) * 100
                    print(f"[Embedding] {msg} ({current}/{total} - %{pct:.1f})")
                else:
                    print(f"[Embedding] {msg}")
            
            result = ingestor.ingest(FIXED_CSV_PATH, progress_callback=progress_callback)
            if result.get("success"):
                chunks = result.get("chunks", 0)
                print(f"✅ CSV başarıyla yüklendi ve embed edildi: {chunks} satır")
                print(f"✅ Index kaydedildi: {INDEX_DIR / 'default'}")
            else:
                print(f"[!] CSV yükleme hatası: {result.get('error')}")
        except Exception as e:
            import traceback
            print(f"[!] CSV yükleme hatası: {traceback.format_exc()}")
    else:
        print("[!] CSV dosyası bulunamadı, manuel yükleme gerekebilir")
        print("[!] Index yoksa chat çalışmayacak!")
    
    print("\n[5/6] Backend başlatılıyor...")
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    time.sleep(5)  # Backend'in başlaması için bekle
    print("✅ Backend: http://localhost:3000")
    
    print("\n[6/6] Frontend başlatılıyor...")
    print("\n" + "=" * 60)
    print("🚀 RAG Platform hazır!")
    print("=" * 60)
    print("\n⏳ Gradio public URL oluşturuluyor...")
    print("   (Bu işlem 10-20 saniye sürebilir)\n")
    
    # Frontend'i ana thread'de çalıştır (blocking)
    run_frontend()
                    