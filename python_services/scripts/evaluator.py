#!/usr/bin/env python3
# Model değerlendirme scripti - her modeli test verisinde çalıştırıp metrikleri hesaplıyorum
# Cosine similarity, ROUGE-L, BLEU, F1, Accuracy hesaplıyorum ve grafik çiziyorum

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from rouge_score import rouge_scorer

from scripts.rag_engine import RAGEngine

# Dizinler - grafikleri frontend'e kaydediyorum
DATA_DIR = Path(__file__).parent.parent / "data"
# Streamlit frontend için plots klasörü
PLOTS_DIR = Path(__file__).parent.parent.parent / "frontend_streamlit" / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Default embedding - index'ten alınabilir ama şimdilik bu
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.65  # Bu değerin üstü "doğru" sayılıyor - cosine similarity için


class ModelEvaluator:
    # Modelleri test edip karşılaştırma raporu çıkarıyorum
    # Grafikleri plots klasörüne kaydediyorum
    
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.rag = None  # Lazy load
        self.embed_model = None  # Lazy load
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.results = {}
    
    def _load(self):
        # Modelleri lazy load yapıyorum - gerektiğinde yükleniyor
        if self.embed_model is None:
            print("[*] Embedding modeli yükleniyor...")
            # Index metadata'sından model adını alabilirim ama şimdilik default kullanıyorum
            self.embed_model = SentenceTransformer(DEFAULT_EMBEDDING)
        
        if self.rag is None:
            print("[*] RAG engine başlatılıyor...")
            self.rag = RAGEngine(self.index_name)
    
    def cosine_sim(self, text1: str, text2: str) -> float:
        # İki metin arası cosine similarity hesaplıyorum
        # Embedding'leri alıp cosine hesaplıyorum
        vecs = self.embed_model.encode([text1, text2])
        return float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
    
    def rouge_score(self, pred: str, ref: str) -> float:
        # ROUGE-L F1 skoru hesaplıyorum - metin benzerliği için
        scores = self.scorer.score(ref, pred)
        return scores['rougeL'].fmeasure
    
    def f1_score(self, pred: str, ref: str) -> float:
        # Basit token-based F1 hesaplıyorum - kelime bazında karşılaştırma
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        common = pred_tokens & ref_tokens  # Ortak kelimeler
        if not common:
            return 0.0
        
        # Precision ve recall hesaplıyorum
        prec = len(common) / len(pred_tokens) if pred_tokens else 0
        rec = len(common) / len(ref_tokens) if ref_tokens else 0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    def evaluate_one(self, question: str, expected: str, model: str) -> Dict:
        # Tek bir soru için tüm metrikleri hesaplıyorum
        res = self.rag.query(question, model)
        pred = res['answer']
        
        # Tüm metrikleri hesaplıyorum
        cos = self.cosine_sim(pred, expected)
        rouge = self.rouge_score(pred, expected)
        f1 = self.f1_score(pred, expected)
        correct = 1 if cos > THRESHOLD else 0  # Threshold üstüyse doğru sayıyorum
        
        return {
            'question': question,
            'expected': expected,
            'prediction': pred,
            'cosine': cos,
            'rouge': rouge,
            'f1': f1,
            'correct': correct,
            'confidence': res['confidence'],
            'time_ms': res['response_time_ms']
        }
    
    def run(self, test_data: List[Dict] = None) -> Dict:
        # tüm modelleri test et
        # test_data yoksa index'teki verileri kullan
        self._load()
        
        # test verisi
        if test_data is None:
            test_data = self.rag.vectorstore.docstore._dict.values() if self.rag.vectorstore else []
            # metadata'dan soru/cevap çek
            test_data = [
                {'question': d.metadata.get('question', ''), 'answer': d.page_content}
                for d in test_data if d.metadata.get('question')
            ]
        
        # sadece soru/cevap olanları al
        valid = [d for d in test_data if d.get('question') and d.get('answer')]
        
        if not valid:
            print("[!] Test verisi yok")
            return {}
        
        print(f"[*] {len(valid)} soru test edilecek")
        
        models = ['GPT', 'BERT-CASED', 'BERT-SENTIMENT']
        
        for model in models:
            print(f"\n--- {model} ---")
            
            model_results = []
            for qa in tqdm(valid, desc=model):
                r = self.evaluate_one(qa['question'], qa['answer'], model)
                model_results.append(r)
            
            df = pd.DataFrame(model_results)
            
            self.results[model] = {
                'model': model,
                'avg_cosine': df['cosine'].mean(),
                'avg_rouge': df['rouge'].mean(),
                'avg_f1': df['f1'].mean(),
                'accuracy': df['correct'].mean(),
                'avg_confidence': df['confidence'].mean(),
                'avg_time_ms': df['time_ms'].mean(),
                'predictions': df['correct'].tolist(),
                'details': model_results
            }
            
            print(f"  cosine: {df['cosine'].mean():.3f}")
            print(f"  rouge: {df['rouge'].mean():.3f}")
            print(f"  f1: {df['f1'].mean():.3f}")
            print(f"  accuracy: {df['correct'].mean():.1%}")
        
        return self.results
    
    def plot(self):
        # grafikleri oluştur ve kaydet
        if not self.results:
            print("[!] Önce run() çalıştır")
            return
        
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        colors = ['#374151', '#6B7280', '#9CA3AF']
        
        self._plot_confusion(colors)
        self._plot_metrics(colors)
        self._plot_time(colors)
        self._plot_radar(colors)
        
        print(f"[+] Grafikler: {PLOTS_DIR}")
    
    def _plot_confusion(self, colors):
        # confusion matrix grafikleri
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        for i, (model, data) in enumerate(self.results.items()):
            preds = data['predictions']
            truths = [1] * len(preds)
            cm = confusion_matrix(truths, preds, labels=[0, 1])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
                       xticklabels=['Yanlış', 'Doğru'],
                       yticklabels=['Yanlış', 'Doğru'],
                       ax=axes[i], cbar=False)
            axes[i].set_title(f'{model}')
            axes[i].set_xlabel('Tahmin')
            axes[i].set_ylabel('Gerçek')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_metrics(self, colors):
        # metrik karşılaştırma bar chart
        models = list(self.results.keys())
        metrics = ['Cosine', 'ROUGE', 'F1', 'Accuracy']
        
        data = {
            'Cosine': [self.results[m]['avg_cosine'] for m in models],
            'ROUGE': [self.results[m]['avg_rouge'] for m in models],
            'F1': [self.results[m]['avg_f1'] for m in models],
            'Accuracy': [self.results[m]['accuracy'] for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(11, 5))
        grays = ['#1F2937', '#4B5563', '#6B7280', '#9CA3AF']
        
        for i, (metric, vals) in enumerate(data.items()):
            bars = ax.bar(x + i * width, vals, width, label=metric, color=grays[i])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{v:.2f}', ha='center', fontsize=8)
        
        ax.set_ylabel('Skor')
        ax.set_title('Model Karşılaştırması')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_time(self, colors):
        # yanıt süresi grafiği
        models = list(self.results.keys())
        times = [self.results[m]['avg_time_ms'] for m in models]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, times, color=colors)
        
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{t:.0f}ms', ha='center', fontweight='bold')
        
        ax.set_ylabel('ms')
        ax.set_title('Ortalama Yanıt Süresi')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'response_time.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_radar(self, colors):
        # radar chart - çok boyutlu karşılaştırma
        cats = ['Cosine', 'ROUGE', 'F1', 'Accuracy', 'Hız']
        
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        
        max_time = max(self.results[m]['avg_time_ms'] for m in self.results)
        
        for i, (model, data) in enumerate(self.results.items()):
            vals = [
                data['avg_cosine'],
                data['avg_rouge'],
                data['avg_f1'],
                data['accuracy'],
                1 - (data['avg_time_ms'] / max_time) if max_time > 0 else 0.5
            ]
            vals += vals[:1]
            
            ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, vals, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.set_title('Genel Karşılaştırma')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'radar_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def save_json(self):
        # sonuçları json olarak kaydet
        if not self.results:
            return {}
        
        out = {m: {
            'avg_cosine': d['avg_cosine'],
            'avg_rouge': d['avg_rouge'],
            'avg_f1': d['avg_f1'],
            'accuracy': d['accuracy'],
            'avg_time_ms': d['avg_time_ms']
        } for m, d in self.results.items()}
        
        path = PLOTS_DIR / 'evaluation_results.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        
        return out


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", default="default")
    args = parser.parse_args()
    
    print("="*50)
    print("Model Değerlendirme")
    print("="*50)
    
    ev = ModelEvaluator(args.index)
    results = ev.run()
    
    if results:
        ev.plot()
        ev.save_json()
        
        # özet json çıktı
        summary = {
            'status': 'success',
            'plots': ['confusion_matrix.png', 'metrics_comparison.png', 'response_time.png', 'radar_chart.png'],
            'results': {m: {'accuracy': d['accuracy'], 'f1': d['avg_f1']} for m, d in results.items()}
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
