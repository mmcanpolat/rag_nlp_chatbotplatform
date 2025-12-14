#!/usr/bin/env python3
# Döküman işleme ve FAISS'e yükleme - PDF, Word, JSON, web sayfası vs. alıp vektör DB'ye atıyorum
# LangChain kullanıyorum çünkü manuel parsing çok uğraştırıyor

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# LangChain yükleyicileri - her dosya tipi için ayrı loader var
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Dizin ayarları - FAISS index'leri ve upload edilen dosyalar burada
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
UPLOADS_DIR = DATA_DIR / "uploads"

# Embedding modelleri - şimdilik sadece HuggingFace destekliyorum
# OpenAI embedding'i de eklenebilir ama ayrı işlem gerekiyor
EMBEDDING_MODELS = {
    'paraphrase-multilingual-MiniLM-L12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'text-embedding-3-large': 'text-embedding-3-large'  # OpenAI için ayrı işlem gerekir
}
DEFAULT_EMBEDDING = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Chunk boyutları - deneme yanılmayla buldum
# 750 karakter hem context window'a sığıyor hem anlam bütünlüğü korunuyor
# 100 overlap ile chunk'lar arası bağlantı sağlanıyor
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100


class DocumentIngestor:
    # Dökümanları alıp FAISS'e yükleyen sınıf
    # LangChain kullanıyorum çünkü manuel parsing çok uğraştırıyor
    
    def __init__(self, index_name: str = "default", embedding_model: str = None):
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.embedding_model_name = embedding_model or DEFAULT_EMBEDDING
        
        # Dizinleri oluşturuyorum - yoksa hata verir
        self.index_path.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
        self._embeddings = None  # Lazy load yapıyorum - gerektiğinde yükleniyor
        
        # Text splitter - paragraf ve cümle sınırlarına dikkat ediyor
        # Önce paragraf, sonra cümle, sonra kelime bazında bölüyor
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len
        )
        
        print(f"[*] Ingestor hazır - index: {self.index_name}")
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        # Embedding modelini lazy load yapıyorum - ilk kullanımda yükleniyor
        # Başlangıçta yüklemek gereksiz yavaşlatıyor
        if self._embeddings is None:
            print(f"[*] Embedding modeli yükleniyor: {self.embedding_model_name}")
            model_name = EMBEDDING_MODELS.get(self.embedding_model_name, self.embedding_model_name)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def detect_source_type(self, source: str) -> str:
        # Dosya tipini uzantıdan veya URL'den anlıyorum
        # URL ise web scraping yapacağım, dosya ise uzantıya bakacağım
        
        # URL mi kontrol ediyorum
        parsed = urlparse(source)
        if parsed.scheme in ('http', 'https'):
            return 'web'
        
        # Uzantıya bakıyorum
        ext = Path(source).suffix.lower()
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'txt',
            '.md': 'txt',
            '.json': 'json',
            '.csv': 'csv'
        }
        return type_map.get(ext, 'txt')
    
    def load_document(self, source: str) -> List[Document]:
        # Kaynağı LangChain loader ile yüklüyorum
        # Tip otomatik algılanıyor, her tip için uygun loader kullanılıyor
        source_type = self.detect_source_type(source)
        print(f"[*] Yükleniyor: {source_type}")
        
        try:
            if source_type == 'pdf':
                loader = PyPDFLoader(source)
                docs = loader.load()
                
            elif source_type == 'web':
                loader = WebBaseLoader(source)
                docs = loader.load()
                
            elif source_type == 'docx':
                loader = Docx2txtLoader(source)
                docs = loader.load()
                
            elif source_type == 'json':
                docs = self._load_json(source)
                
            elif source_type == 'csv':
                # CSV dosyasını UTF-8 encoding ile yüklüyorum
                # Büyük CSV dosyaları için biraz zaman alabilir
                print("[*] CSV dosyası yükleniyor (bu biraz zaman alabilir)...")
                loader = CSVLoader(
                    file_path=source,
                    encoding='utf-8',
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                        'skipinitialspace': True
                    }
                )
                docs = loader.load()
                print(f"[+] CSV: {len(docs)} satır yüklendi")
                
            else:
                loader = TextLoader(source, encoding='utf-8')
                docs = loader.load()
            
            print(f"[+] {len(docs)} döküman yüklendi")
            return docs
            
        except Exception as e:
            print(f"[!] Yükleme hatası: {e}")
            raise
    
    def _load_json(self, file_path: str) -> List[Document]:
        # JSON yükleyici - QA formatını destekliyor
        # question/answer/context veya text alanlarını arıyor
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # önce context, sonra text, sonra answer'a bak
                    content = item.get('context') or item.get('text') or item.get('answer') or item.get('content')
                    
                    if not content:
                        # hiçbiri yoksa hepsini birleştir
                        content = ' '.join(str(v) for v in item.values() if v)
                    
                    if content:
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'index': i,
                                'question': item.get('question', '')
                            }
                        ))
                elif isinstance(item, str) and item.strip():
                    docs.append(Document(page_content=item, metadata={'source': file_path, 'index': i}))
        
        elif isinstance(data, dict):
            content = data.get('content') or data.get('text') or str(data)
            docs.append(Document(page_content=content, metadata={'source': file_path}))
        
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        # dökümanları chunk'lara ayırıyor
        print(f"[*] {len(documents)} döküman parçalanıyor...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"[+] {len(chunks)} chunk oluştu")
        return chunks
    
    def load_existing_index(self) -> Optional[FAISS]:
        # varsa mevcut FAISS indexi yükle
        index_file = self.index_path / "index.faiss"
        
        if index_file.exists():
            try:
                existing = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[+] Mevcut index yüklendi")
                return existing
            except Exception as e:
                print(f"[!] Index yüklenemedi: {e}")
        return None
    
    def create_or_update_index(self, chunks: List[Document]) -> FAISS:
        # yeni index oluştur veya mevcutla birleştir
        # büyük dosyalar için batch processing
        print(f"[*] {len(chunks)} chunk için vektörler oluşturuluyor...")
        
        # büyük dosyalar için batch işleme (1000 chunk'ta bir progress göster)
        batch_size = 1000
        if len(chunks) > batch_size:
            print(f"[*] Büyük dosya tespit edildi, batch işleme yapılıyor...")
            # ilk batch ile index oluştur
            first_batch = chunks[:batch_size]
            new_index = FAISS.from_documents(first_batch, self.embeddings)
            print(f"[*] İlk {batch_size} chunk işlendi...")
            
            # kalan chunk'ları batch batch ekle
            for i in range(batch_size, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_index = FAISS.from_documents(batch, self.embeddings)
                new_index.merge_from(batch_index)
                print(f"[*] {min(i+batch_size, len(chunks))}/{len(chunks)} chunk işlendi...")
        else:
            new_index = FAISS.from_documents(chunks, self.embeddings)
        
        print(f"[+] {len(chunks)} chunk vektörize edildi")
        
        # mevcut varsa birleştir
        existing = self.load_existing_index()
        if existing:
            print("[*] Mevcut indexle birleştiriliyor...")
            existing.merge_from(new_index)
            return existing
        
        return new_index
    
    def save_index(self, index: FAISS) -> str:
        # FAISS indexi diske kaydet
        index.save_local(str(self.index_path))
        
        # meta bilgi de kaydet
        meta = {
            'index_name': self.index_name,
            'embedding_model': self.embedding_model_name,
            'chunk_size': CHUNK_SIZE
        }
        with open(self.index_path / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"[+] Index kaydedildi: {self.index_path}")
        return str(self.index_path)
    
    def ingest(self, source: str) -> dict:
        # ana pipeline: yükle -> parçala -> vektörize -> kaydet
        print("\n" + "="*50)
        print(f"Kaynak: {source}")
        print("="*50)
        
        try:
            # yükle
            docs = self.load_document(source)
            if not docs:
                return {'success': False, 'error': 'Döküman boş veya okunamadı'}
            
            # parçala
            chunks = self.split_documents(docs)
            if not chunks:
                return {'success': False, 'error': 'Chunk oluşturulamadı'}
            
            # vektörize ve kaydet
            index = self.create_or_update_index(chunks)
            save_path = self.save_index(index)
            
            result = {
                'success': True,
                'message': 'Tamamlandı',
                'documents_loaded': len(docs),
                'chunks_created': len(chunks),
                'index_path': save_path
            }
            
            print(f"\n[✓] {len(docs)} döküman -> {len(chunks)} chunk")
            return result
            
        except Exception as e:
            print(f"[!] Hata: {e}")
            return {'success': False, 'error': str(e)}


# API için kısa yollar
def ingest_from_file(file_path: str, index_name: str = "default", embedding_model: str = None) -> dict:
    return DocumentIngestor(index_name, embedding_model).ingest(file_path)

def ingest_from_url(url: str, index_name: str = "default", embedding_model: str = None) -> dict:
    return DocumentIngestor(index_name, embedding_model).ingest(url)


def main():
    parser = argparse.ArgumentParser(description="Döküman yükleyici")
    
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", "-f", help="Dosya yolu")
    source.add_argument("--url", "-u", help="Web sayfası URL'i")
    
    parser.add_argument("--index", "-i", default="default", help="Index adı")
    parser.add_argument("--embedding", "-e", default=None, help="Embedding model adı")
    
    args = parser.parse_args()
    
    src = args.file or args.url
    
    if args.file and not Path(args.file).exists():
        print(f"[!] Dosya yok: {args.file}")
        sys.exit(1)
    
    ingestor = DocumentIngestor(args.index, args.embedding)
    result = ingestor.ingest(src)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
