#!/usr/bin/env python3
# RAG soru-cevap motoru
# FAISS'ten ilgili dökümanları çekip modele veriyor

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# dizin ayarları
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# default embedding - index metadata'dan da alınabilir
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGEngine:
    # soru gelince FAISS'ten context çekip modele soruyor
    # GPT için OpenAI API, BERT'ler için basit kelime eşleştirmesi
    
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.vectorstore = None
        self._embeddings = None
        self.openai_client = None
        
        self._setup()
    
    @property
    def embeddings(self):
        # embedding modeli - lazy load
        # GPU varsa otomatik kullan, yoksa CPU
        if self._embeddings is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # önce index metadata'sından model adını al
            embedding_model = self._get_embedding_model_from_index()
            print(f"[*] Embedding modeli yükleniyor: {embedding_model} ({device})")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def _get_embedding_model_from_index(self) -> str:
        # index metadata'sından embedding model adını al
        meta_file = self.index_path / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    return meta.get('embedding_model', DEFAULT_EMBEDDING)
            except:
                pass
        return DEFAULT_EMBEDDING
    
    def _setup(self):
        # başlangıç yüklemeleri
        print("[*] RAG engine başlatılıyor...")
        
        # FAISS yükle
        self._load_vectorstore()
        
        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            print("[+] OpenAI bağlantısı OK")
        else:
            print("[!] OPENAI_API_KEY yok, GPT çalışmaz")
    
    def _load_vectorstore(self) -> bool:
        # FAISS indexi yükle
        index_file = self.index_path / "index.faiss"
        
        if not index_file.exists():
            print(f"[!] Index bulunamadı: {self.index_path}")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("[+] FAISS yüklendi")
            return True
        except Exception as e:
            print(f"[!] FAISS hatası: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        # sorguya en benzer chunk'ları getir
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        return [{
            'context': doc.page_content,
            'score': float(1 - score),  # mesafe -> benzerlik
            'metadata': doc.metadata
        } for doc, score in results]
    
    def _ask_gpt(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # OpenAI GPT'ye sor
        if not self.openai_client:
            return "API key yok", 0.0
        
        # context'leri birleştir
        ctx_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        
        prompt = f"""Aşağıdaki bilgilere göre soruyu yanıtla.
Bilgi yoksa "Bu konuda bilgi yok" de.

BİLGİLER:
{ctx_text}

SORU: {query}
CEVAP:"""
        
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Kısa ve net cevaplar ver."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                timeout=60
            )
            answer = resp.choices[0].message.content.strip()
            conf = 0.85 if len(answer) > 20 else 0.6
            return answer, conf
        except Exception as e:
            return f"GPT hatası: {e}", 0.0
    
    def _ask_bert_cased(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # BERT için basit yaklaşım
        # context'i cümlelere ayır, sorguyla en çok kelime eşleşeni seç
        # gerçek QA modeli değil ama idare eder
        text = " ".join(contexts)
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        if not sentences:
            return "Context boş", 0.0
        
        query_words = set(query.lower().split())
        
        # her cümleyi skorla
        best = ("", 0)
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > best[1]:
                best = (sent, overlap)
        
        if best[1] > 0:
            conf = min(0.9, best[1] / max(len(query_words), 1))
            return best[0] + ".", conf
        
        return sentences[0] + ".", 0.3
    
    def _ask_bert_sentiment(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # bert-sentiment için de aynı mantık
        # ileride sentiment modeli eklenebilir
        return self._ask_bert_cased(query, contexts)
    
    def query(self, text: str, model_type: str = "GPT") -> Dict:
        # ana sorgu fonksiyonu
        start = time.time()
        
        # model normalize
        model = model_type.upper().replace("-", "_")
        if model not in ["GPT", "BERT_CASED", "BERT_SENTIMENT"]:
            model = "GPT"
        
        # context çek
        retrieved = self.retrieve(text, top_k=3)
        contexts = [r['context'] for r in retrieved]
        scores = [r['score'] for r in retrieved]
        
        if not contexts:
            return {
                "answer": "Önce veri yükleyin",
                "context": "",
                "all_contexts": [],
                "retrieval_scores": [],
                "confidence": 0.0,
                "model_used": model.replace("_", "-"),
                "response_time_ms": round((time.time() - start) * 1000, 2)
            }
        
        # modele sor
        if model == "GPT":
            answer, conf = self._ask_gpt(text, contexts)
        elif model == "BERT_CASED":
            answer, conf = self._ask_bert_cased(text, contexts)
        else:
            answer, conf = self._ask_bert_sentiment(text, contexts)
        
        return {
            "answer": answer,
            "context": contexts[0] if contexts else "",
            "all_contexts": contexts,
            "retrieval_scores": scores,
            "confidence": round(conf, 4),
            "model_used": model.replace("_", "-"),
            "response_time_ms": round((time.time() - start) * 1000, 2)
        }
    
    def reload_index(self, index_name: str = None):
        # index değiştirmek için
        if index_name:
            self.index_name = index_name
            self.index_path = INDEX_DIR / index_name
        self._load_vectorstore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", help="Soru")
    parser.add_argument("--model", "-m", default="GPT", 
                       choices=["GPT", "BERT-CASED", "BERT-SENTIMENT"])
    parser.add_argument("--index", "-i", default="default")
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    engine = RAGEngine(args.index)
    
    if args.interactive:
        print("\nRAG Chat - 'q' ile çık, 'model X' ile model değiştir\n")
        current = "GPT"
        
        while True:
            try:
                inp = input(f"[{current}] > ").strip()
                if not inp:
                    continue
                if inp.lower() in ['q', 'quit', 'exit']:
                    break
                if inp.startswith('model '):
                    m = inp.split()[1].upper().replace("-", "_")
                    if m in ["GPT", "BERT_CASED", "BERT_SENTIMENT"]:
                        current = m.replace("_", "-")
                        print(f"Model: {current}")
                    continue
                
                res = engine.query(inp, current)
                print(f"\n{res['answer']}")
                print(f"  güven: {res['confidence']:.0%} | süre: {res['response_time_ms']}ms\n")
                
            except KeyboardInterrupt:
                break
    
    elif args.query:
        result = engine.query(args.query, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
