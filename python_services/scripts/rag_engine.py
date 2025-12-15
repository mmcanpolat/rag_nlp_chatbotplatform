#!/usr/bin/env python3
# RAG engine - soru gelince FAISS'ten ilgili dökümanları bulup modele veriyorum
# GPT için Hugging Face GPT-2 kullanıyorum, BERT'ler için basit kelime eşleştirmesi yapıyorum

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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Dizin ayarları - FAISS index'leri burada tutuluyor
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# Default embedding modeli - index metadata'sından da alabilirim ama şimdilik bu
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGEngine:
    # RAG motoru - soru gelince FAISS'ten context çekip modele veriyorum
    # GPT için Hugging Face GPT-2 kullanıyorum, BERT modelleri için basit kelime eşleştirmesi yapıyorum
    
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.vectorstore = None  # FAISS index'i buraya yüklenecek
        self._embeddings = None  # Lazy load yapıyorum, gerektiğinde yükleniyor
        self._gpt_model = None  # Hugging Face GPT modeli
        self._gpt_tokenizer = None  # GPT tokenizer
        
        self._setup()
    
    @property
    def embeddings(self):
        # Embedding modelini lazy load yapıyorum - ilk kullanımda yükleniyor
        # GPU varsa otomatik kullanıyor, yoksa CPU'ya düşüyor
        if self._embeddings is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Index metadata'sından hangi embedding model kullanıldığını alıyorum
            embedding_model = self._get_embedding_model_from_index()
            print(f"[*] Embedding modeli yükleniyor: {embedding_model} ({device})")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def _get_embedding_model_from_index(self) -> str:
        # Index oluşturulurken hangi embedding model kullanıldıysa onu alıyorum
        # metadata.json dosyasında saklanıyor
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
        # Başlangıçta FAISS'i yüklüyorum ve Hugging Face GPT modelini hazırlıyorum
        print("[*] RAG engine başlatılıyor...")
        
        # FAISS index'ini yüklüyorum
        self._load_vectorstore()
        
        # Hugging Face GPT modeli lazy load yapılacak - ilk kullanımda yüklenecek
        print("[+] GPT modeli hazır (lazy load)")
    
    def _load_vectorstore(self) -> bool:
        # FAISS index'ini yüklüyorum - daha önce oluşturulmuş olmalı
        index_file = self.index_path / "index.faiss"
        
        if not index_file.exists():
            print(f"[!] Index bulunamadı: {self.index_path}")
            return False
        
        try:
            # FAISS'i yüklüyorum - embeddings lazım çünkü index'i decode ediyor
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
        # Sorguya en benzer chunk'ları getiriyorum - top_k kadar
        # FAISS similarity search yapıyorum
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Score'u benzerliğe çeviriyorum (mesafe -> benzerlik)
        return [{
            'context': doc.page_content,
            'score': float(1 - score),  # Mesafe ne kadar azsa o kadar benzer
            'metadata': doc.metadata
        } for doc, score in results]
    
    def _load_gpt_model(self):
        # Hugging Face GPT-2 modelini lazy load yapıyorum - ilk kullanımda yükleniyor
        if self._gpt_model is None:
            print("[*] Hugging Face GPT-2 modeli yükleniyor...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Türkçe GPT-2 modeli kullanıyorum
            model_name = "dbmdz/gpt2-turkish-cased"
            try:
                self._gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self._gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
                self._gpt_model.to(device)
                self._gpt_model.eval()  # Inference modu
                # Pad token yoksa eos token kullan
                if self._gpt_tokenizer.pad_token is None:
                    self._gpt_tokenizer.pad_token = self._gpt_tokenizer.eos_token
                print(f"[+] GPT-2 modeli yüklendi ({device})")
            except Exception as e:
                print(f"[!] GPT-2 yükleme hatası: {e}")
                # Fallback: İngilizce GPT-2
                print("[*] İngilizce GPT-2 deneniyor...")
                self._gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self._gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
                self._gpt_model.to(device)
                self._gpt_model.eval()
                if self._gpt_tokenizer.pad_token is None:
                    self._gpt_tokenizer.pad_token = self._gpt_tokenizer.eos_token
    
    def _ask_gpt(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # Hugging Face GPT-2 modeline soruyorum - context'leri de veriyorum
        try:
            # Modeli yükle (lazy load)
            self._load_gpt_model()
            
            if self._gpt_model is None:
                return "GPT modeli yüklenemedi", 0.0
            
            # Context'leri birleştirip prompt'a ekliyorum
            ctx_text = "\n\n".join([f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts)])  # Her context max 200 karakter
            prompt = f"Bilgiler: {ctx_text}\n\nSoru: {query}\nCevap:"
            
            # Tokenize et
            device = next(self._gpt_model.parameters()).device
            inputs = self._gpt_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            # Generate - kısa cevaplar için max_length düşük
            with torch.no_grad():
                outputs = self._gpt_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,  # 100 token daha üret
                    min_length=inputs.shape[1] + 10,   # En az 10 token
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._gpt_tokenizer.eos_token_id,
                    eos_token_id=self._gpt_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Decode et - sadece yeni üretilen kısmı al
            generated = outputs[0][inputs.shape[1]:]
            answer = self._gpt_tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # Eğer cevap çok kısa veya boşsa, context'ten bir cümle al
            if len(answer) < 10:
                if contexts:
                    answer = contexts[0][:200] + "..."
                    conf = 0.6
                else:
                    answer = "Bu konuda yeterli bilgi bulunamadı."
                    conf = 0.3
            else:
                # Güven skoru - cevap uzunluğuna göre
                conf = min(0.85, 0.5 + len(answer) / 200)
            
            return answer, conf
        except Exception as e:
            return f"GPT hatası: {str(e)}", 0.0
    
    def _ask_bert_cased(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # BERT için basit yaklaşım kullanıyorum - gerçek QA modeli değil ama idare ediyor
        # Context'i cümlelere ayırıp sorguyla en çok kelime eşleşen cümleyi seçiyorum
        text = " ".join(contexts)
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        if not sentences:
            return "Context boş", 0.0
        
        query_words = set(query.lower().split())
        
        # Her cümleyi skorluyorum - kaç kelime eşleşiyor
        best = ("", 0)
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)  # Ortak kelimeler
            if overlap > best[1]:
                best = (sent, overlap)
        
        if best[1] > 0:
            # Güven skorunu hesaplıyorum - ne kadar çok kelime eşleşirse o kadar yüksek
            conf = min(0.9, best[1] / max(len(query_words), 1))
            return best[0] + ".", conf
        
        # Hiç eşleşme yoksa ilk cümleyi döndürüyorum ama düşük güvenle
        return sentences[0] + ".", 0.3
    
    def _ask_bert_sentiment(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        # BERT sentiment için de aynı mantığı kullanıyorum
        # İleride gerçek sentiment modeli eklenebilir ama şimdilik bu yeterli
        return self._ask_bert_cased(query, contexts)
    
    def query(self, text: str, model_type: str = "GPT") -> Dict:
        # Ana sorgu fonksiyonu - kullanıcı sorusunu alıp cevap döndürüyorum
        start = time.time()
        
        # Model tipini normalize ediyorum - frontend'den farklı formatlar gelebilir
        model = model_type.upper().replace("-", "_")
        if model not in ["GPT", "BERT_CASED", "BERT_SENTIMENT"]:
            model = "GPT"  # Geçersizse default GPT
        
        # FAISS'ten ilgili context'leri çekiyorum - top 3
        retrieved = self.retrieve(text, top_k=3)
        contexts = [r['context'] for r in retrieved]
        scores = [r['score'] for r in retrieved]
        
        if not contexts:
            # Context yoksa veri yüklenmemiş demektir
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
