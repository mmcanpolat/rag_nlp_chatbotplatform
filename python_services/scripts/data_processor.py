#!/usr/bin/env python3
"""
RAG SaaS Platform - Veri İşleme Modülü
======================================
Kullanıcı tarafından yüklenen veri setlerini işler ve 
FAISS vektör indeksine dönüştürür.

Desteklenen formatlar:
    - JSON: [{"question": "...", "answer": "...", "context": "..."}]
    - CSV: question, answer, context sütunları
    - TXT: Her satır bir bağlam metni

Fonksiyonlar:
    - load_dataset(): Veri setini dosyadan yükler
    - process_dataset(): Veri setini RAG için hazırlar
    - build_faiss_index(): FAISS vektör indeksi oluşturur
    - save_index(): İndeksi diske kaydeder
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import csv

# Üst dizini yola ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# =============================================
# SABİT DEĞERLER
# =============================================

DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "faiss_index"

# Çok dilli embedding modeli - Türkçe desteği
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DataProcessor:
    """
    Veri işleme ve FAISS indeksleme sınıfı.
    
    Kullanıcı tarafından yüklenen veri setlerini:
    1. Yükler ve parse eder
    2. RAG için uygun formata dönüştürür
    3. FAISS vektör indeksi oluşturur
    
    Attributes:
        embedding_model: Metin embedding modeli
        index: FAISS vektör indeksi
        documents: İndekslenmiş dökümanlar
    """
    
    def __init__(self):
        """Veri işleyiciyi başlatır."""
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index = None
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def _load_embedding_model(self):
        """Embedding modelini yükler (lazy loading)."""
        if self.embedding_model is None:
            print("[YÜKLENIYOR] Embedding modeli yükleniyor...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("[TAMAMLANDI] Embedding modeli hazır")
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """
        Veri setini dosyadan yükler.
        
        Desteklenen formatlar:
            - .json: JSON array formatı
            - .csv: CSV dosyası (question, answer, context sütunları)
            - .txt: Her satır bir döküman
        
        Args:
            file_path: Veri seti dosya yolu
            
        Returns:
            List[Dict]: Yüklenen dökümanlar
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return self._load_json(file_path)
        elif extension == '.csv':
            return self._load_csv(file_path)
        elif extension == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {extension}")
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """JSON dosyasını yükler."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Liste değilse listeye çevir
        if isinstance(data, dict):
            data = [data]
        
        # Alanları normalize et
        normalized = []
        for item in data:
            doc = {
                'id': item.get('id', len(normalized) + 1),
                'question': item.get('question', item.get('soru', '')),
                'answer': item.get('answer', item.get('cevap', item.get('ground_truth', ''))),
                'context': item.get('context', item.get('baglam', item.get('text', '')))
            }
            # Eğer context yoksa answer'ı context olarak kullan
            if not doc['context'] and doc['answer']:
                doc['context'] = doc['answer']
            normalized.append(doc)
        
        print(f"[YÜKLENDI] {len(normalized)} döküman JSON'dan yüklendi")
        return normalized
    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """CSV dosyasını yükler."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                doc = {
                    'id': row.get('id', i + 1),
                    'question': row.get('question', row.get('soru', '')),
                    'answer': row.get('answer', row.get('cevap', '')),
                    'context': row.get('context', row.get('baglam', row.get('text', '')))
                }
                if not doc['context'] and doc['answer']:
                    doc['context'] = doc['answer']
                documents.append(doc)
        
        print(f"[YÜKLENDI] {len(documents)} döküman CSV'den yüklendi")
        return documents
    
    def _load_txt(self, file_path: Path) -> List[Dict]:
        """TXT dosyasını yükler (her satır bir döküman)."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    documents.append({
                        'id': i + 1,
                        'question': '',
                        'answer': '',
                        'context': line
                    })
        
        print(f"[YÜKLENDI] {len(documents)} döküman TXT'den yüklendi")
        return documents
    
    def process_dataset(self, documents: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Veri setini RAG için işler.
        
        Args:
            documents: Ham döküman listesi
            
        Returns:
            Tuple[List[str], List[Dict]]: (bağlam metinleri, meta veriler)
        """
        contexts = []
        metadata = []
        
        for doc in documents:
            context = doc.get('context', '')
            if context:
                contexts.append(context)
                metadata.append({
                    'id': doc.get('id'),
                    'question': doc.get('question', ''),
                    'answer': doc.get('answer', ''),
                    'context': context
                })
        
        print(f"[İŞLENDİ] {len(contexts)} bağlam metni hazırlandı")
        return contexts, metadata
    
    def build_faiss_index(self, contexts: List[str], metadata: List[Dict]) -> None:
        """
        FAISS vektör indeksi oluşturur.
        
        Args:
            contexts: Bağlam metinleri
            metadata: Döküman meta verileri
        """
        # Embedding modelini yükle
        self._load_embedding_model()
        
        print("[HESAPLANIYOR] Embedding vektörleri oluşturuluyor...")
        
        # Embeddinglieri oluştur
        embeddings = self.embedding_model.encode(
            contexts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = np.array(embeddings).astype('float32')
        
        # Kosinüs benzerliği için normalize et
        faiss.normalize_L2(embeddings)
        
        # FAISS indeksi oluştur
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = Kosinüs
        self.index.add(embeddings)
        
        # Verileri sakla
        self.documents = metadata
        self.embeddings = embeddings
        
        print(f"[TAMAMLANDI] FAISS indeksi oluşturuldu ({len(contexts)} vektör, {dimension} boyut)")
    
    def save_index(self, index_name: str = "default") -> str:
        """
        FAISS indeksini diske kaydeder.
        
        Args:
            index_name: İndeks adı
            
        Returns:
            str: Kayıt dizini yolu
        """
        if self.index is None:
            raise ValueError("Kaydedilecek indeks yok. Önce build_faiss_index() çalıştırın.")
        
        # Dizini oluştur
        save_dir = INDEX_DIR / index_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS indeksini kaydet
        index_path = save_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Dökümanları kaydet
        docs_path = save_dir / "documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'count': len(self.documents)
            }, f, ensure_ascii=False, indent=2)
        
        print(f"[KAYDEDİLDİ] İndeks kaydedildi: {save_dir}")
        return str(save_dir)
    
    def load_index(self, index_name: str = "default") -> bool:
        """
        FAISS indeksini diskten yükler.
        
        Args:
            index_name: İndeks adı
            
        Returns:
            bool: Başarılı mı
        """
        save_dir = INDEX_DIR / index_name
        index_path = save_dir / "index.faiss"
        docs_path = save_dir / "documents.json"
        
        if not index_path.exists():
            print(f"[UYARI] İndeks bulunamadı: {index_path}")
            return False
        
        # FAISS indeksini yükle
        self.index = faiss.read_index(str(index_path))
        
        # Dökümanları yükle
        with open(docs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data.get('documents', [])
        
        print(f"[YÜKLENDI] İndeks yüklendi: {len(self.documents)} döküman")
        return True
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Sorguya en benzer dökümanları bulur.
        
        Args:
            query: Arama sorgusu
            top_k: Döndürülecek sonuç sayısı
            
        Returns:
            List[Dict]: En benzer dökümanlar (skor ile birlikte)
        """
        if self.index is None:
            raise ValueError("İndeks yüklenmemiş. Önce load_index() veya build_faiss_index() çalıştırın.")
        
        # Embedding modelini yükle
        self._load_embedding_model()
        
        # Sorgu vektörünü oluştur
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # FAISS araması
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Sonuçları oluştur
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results


def process_uploaded_file(file_path: str, index_name: str = "default") -> Dict:
    """
    Yüklenen dosyayı işler ve FAISS indeksi oluşturur.
    
    Bu fonksiyon API tarafından çağrılır.
    
    Args:
        file_path: Yüklenen dosya yolu
        index_name: Oluşturulacak indeks adı
        
    Returns:
        Dict: İşlem sonucu
    """
    processor = DataProcessor()
    
    try:
        # Veri setini yükle
        documents = processor.load_dataset(file_path)
        
        # İşle
        contexts, metadata = processor.process_dataset(documents)
        
        if len(contexts) == 0:
            return {
                'success': False,
                'error': 'Veri setinde geçerli döküman bulunamadı'
            }
        
        # FAISS indeksi oluştur
        processor.build_faiss_index(contexts, metadata)
        
        # Kaydet
        save_path = processor.save_index(index_name)
        
        return {
            'success': True,
            'message': f'{len(documents)} döküman başarıyla işlendi',
            'document_count': len(documents),
            'index_path': save_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test için örnek kullanım
    import argparse
    
    parser = argparse.ArgumentParser(description="Veri seti işleyici")
    parser.add_argument("--file", "-f", type=str, help="İşlenecek dosya yolu")
    parser.add_argument("--index", "-i", type=str, default="default", help="İndeks adı")
    
    args = parser.parse_args()
    
    if args.file:
        result = process_uploaded_file(args.file, args.index)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Kullanım: python data_processor.py --file <dosya_yolu> --index <indeks_adı>")

