# RAG SaaS Platform - Colab Hücreleri

Aşağıdaki her bölümü Colab'te **AYRI BİR HÜCRE** olarak ekleyin ve sırayla çalıştırın.

---

## HÜCRE 1: Bağımlılık Kontrolü ve Import'lar

```python
# ============================================
# HÜCRE 1: Bağımlılık Kontrolü ve Import'lar
# ============================================

import sys

print("=" * 60)
print("RAG SaaS Platform - Başlatma")
print("=" * 60)
print("\n[1/7] Bağımlılıklar kontrol ediliyor...")

required_packages = [
    "fastapi", "uvicorn[standard]", "gradio>=4.0.0", "langchain", "langchain-community",
    "langchain-huggingface", "langchain-text-splitters", "langchain-core", "transformers", 
    "torch", "sentence-transformers", "faiss-cpu", "pypdf", "docx2txt", "beautifulsoup4", 
    "requests", "python-dotenv" 
]

missing = []
for pkg in required_packages:
    try:
        pkg_clean = pkg.split("[")[0].split(">=")[0]
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
    print(f"❌ {len(missing)} paket eksik!")
    print("Lütfen önce COLAB_INSTALL.py dosyasını çalıştırın")
    sys.exit(1)
else:
    print("✅ Tüm bağımlılıklar mevcut")

print("\n[2/7] Modüller yükleniyor...")

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

# FastAPI ve Gradio
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

# LangChain text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("✅ Modüller yüklendi")
```

---

## HÜCRE 2: Config ve Helper Functions

```python
# ============================================
# HÜCRE 2: Config ve Helper Functions
# ============================================

print("\n[3/7] Yapılandırma...")

BASE_DIR = Path.cwd()
if "rag_nlp_chatbotplatform" in str(BASE_DIR):
    BASE_DIR = BASE_DIR / "rag_nlp_chatbotplatform"

DATA_DIR = BASE_DIR / "python_services" / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
UPLOADS_DIR = DATA_DIR / "uploads"

# Dizinleri oluştur
for d in [DATA_DIR, INDEX_DIR, UPLOADS_DIR, BASE_DIR / "frontend_gradio" / "assets" / "plots"]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory data
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

# Helper functions
def gen_id() -> str:
    return f"{int(time.time() * 1000)}_{secrets.token_hex(4)}"

def generate_strong_password() -> str:
    return secrets.token_urlsafe(24)

print("✅ Yapılandırma tamamlandı")
```

---

## HÜCRE 3: DocumentIngestor Class

```python
# ============================================
# HÜCRE 3: DocumentIngestor Class
# ============================================

print("\n[4/7] DocumentIngestor yükleniyor...")

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
                batch_size = 100
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                first_batch = chunks[:batch_size]
                percent = int((len(first_batch) / total_chunks) * 100) if total_chunks > 0 else 0
                if progress_callback:
                    progress_callback(f"İlk batch oluşturuluyor ({len(first_batch)} parça) - %{percent}", len(first_batch), total_chunks)
                
                vectorstore = FAISS.from_documents(first_batch, self.embeddings)
                
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

print("✅ DocumentIngestor hazır")
```

---

## HÜCRE 4: RAG Engine Class

```python
# ============================================
# HÜCRE 4: RAG Engine Class
# ============================================

print("\n[5/7] RAG Engine yükleniyor...")

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
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except:
                pass
    
    def _load_gpt_model(self):
        if self._gpt_model is None:
            print("[*] Türkçe GPT-2 yükleniyor...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_name = "dbmdz/gpt2-turkish-cased"
            self._gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self._gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
            self._gpt_model.to(device)
            self._gpt_model.eval()
            if self._gpt_tokenizer.pad_token is None:
                self._gpt_tokenizer.pad_token = self._gpt_tokenizer.eos_token
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [{'context': doc.page_content, 'score': float(1 - score)} 
                for doc, score in results]
    
    def _ask_gpt(self, query: str, contexts: List[str]) -> Tuple[str, float]:
        try:
            self._load_gpt_model()
            if self._gpt_model is None:
                return "Model yüklenemedi", 0.0
            
            ctx_text = "\n\n".join([f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts)])
            prompt = f"Bilgiler: {ctx_text}\n\nSoru: {query}\nCevap:"
            
            device = next(self._gpt_model.parameters()).device
            inputs = self._gpt_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = self._gpt_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    min_length=inputs.shape[1] + 10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._gpt_tokenizer.eos_token_id,
                    eos_token_id=self._gpt_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            generated = outputs[0][inputs.shape[1]:]
            answer = self._gpt_tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            if len(answer) < 10:
                answer = contexts[0][:200] + "..." if contexts else "Bilgi bulunamadı."
                conf = 0.6
            else:
                conf = min(0.85, 0.5 + len(answer) / 200)
            
            return answer, conf
        except Exception as e:
            return f"Hata: {str(e)}", 0.0
    
    def query(self, text: str, model_type: str = "GPT") -> Dict:
        start = time.time()
        retrieved = self.retrieve(text, top_k=3)
        contexts = [r['context'] for r in retrieved]
        
        if not contexts:
            return {
                "answer": "Önce veri yükleyin. Agent oluştururken dosya veya URL ekleyin.",
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

print("✅ RAG Engine hazır")
```

---

## HÜCRE 5: FastAPI Backend

```python
# ============================================
# HÜCRE 5: FastAPI Backend
# ============================================

print("\n[6/7] Backend oluşturuluyor...")

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
    
    ingestion_info = {"chunks": 0, "batches": 0, "status": "başlatılıyor"}
    if agent.data_source:
        try:
            def progress_log(message, current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    batch_info = ""
                    if "Batch" in message:
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
async def chat(req: ChatRequest, user: dict = Depends(require_auth)):
    if req.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[req.agent_id]
    if not user.get("isSuperAdmin") and agent.get("companyId") != user.get("companyId"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    rag = SimpleRAGEngine(index_name=agent.get("indexName", f"agent_{req.agent_id}"))
    result = rag.query(req.query, model=req.model)
    
    return {"success": True, "data": result}

@backend_app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    file_path = UPLOADS_DIR / f"upload_{int(time.time() * 1000)}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"success": True, "data": {"filePath": str(file_path), "fileName": file.filename}}

print("✅ Backend hazır")
```

---

## HÜCRE 6: Gradio Frontend (Bölüm 1 - UI Builder)

```python
# ============================================
# HÜCRE 6: Gradio Frontend - UI Builder
# ============================================

print("\n[7/7] Frontend oluşturuluyor...")

def build_gradio_ui():
    current_user = {"username": "Giriş yapılmadı"}
    current_token = None
    current_agents = []
    
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
    
    def login_fn(username, password):
        if not username or not password:
            return (
                "Kullanıcı adı ve şifre gerekli", 
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update()
            )
        
        try:
            import requests
            time.sleep(1)
            resp = requests.post(
                "http://localhost:3000/api/auth/login",
                json={"username": username, "password": password},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                nonlocal current_user, current_token, current_agents
                current_user = data["data"]
                current_token = data["token"]
                try:
                    agent_resp = requests.get(
                        "http://localhost:3000/api/agents",
                        headers={"Authorization": f"Bearer {current_token}"},
                        timeout=5
                    )
                    if agent_resp.status_code == 200:
                        current_agents = agent_resp.json()["data"]
                        agent_choices = [f"{a['name']}" for a in current_agents]
                    else:
                        current_agents = []
                        agent_choices = []
                except Exception as e:
                    print(f"[!] Agent listesi alınamadı: {e}")
                    current_agents = []
                    agent_choices = []
                
                return (
                    f"✅ Giriş başarılı: {current_user.get('username', '')}",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=current_user.get("isSuperAdmin", False)),
                    gr.update(visible=True),
                    gr.update(choices=agent_choices, value=agent_choices[0] if agent_choices else None)
                )
            else:
                error_msg = resp.json().get("detail", "Giriş başarısız")
                return (
                    f"❌ {error_msg}", 
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update()
                )
        except requests.exceptions.ConnectionError:
            return (
                "❌ Backend'e bağlanılamadı. Lütfen birkaç saniye bekleyip tekrar deneyin.", 
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update()
            )
        except Exception as e:
            return (
                f"❌ Hata: {str(e)}", 
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update()
            )
    
    def chat_fn(message, history, agent_name, model):
        if not current_token:
            return history or [], "Önce giriş yapın"
        
        if not agent_name or not current_agents:
            return history or [], "Agent seçin veya oluşturun"
        
        agent_id = None
        for a in current_agents:
            if a['name'] == agent_name:
                agent_id = a['id']
                break
        
        if not agent_id:
            return history or [], "Agent bulunamadı"
        
        model_short = "GPT"
        if "gpt2-turkish" in model.lower() or "gpt-2" in model.lower() or "gpt2" in model.lower():
            model_short = "GPT"
        elif "bert-base-turkish" in model.lower() and "sentiment" not in model.lower():
            model_short = "BERT-CASED"
        elif "sentiment" in model.lower():
            model_short = "BERT-SENTIMENT"
        
        try:
            import requests
            resp = requests.post(
                "http://localhost:3000/api/chat",
                json={"agent_id": agent_id, "query": message, "model": model_short},
                headers={"Authorization": f"Bearer {current_token}"},
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()["data"]
                if history is None:
                    history = []
                history.append([message, data["answer"]])
                return history, ""
            return history or [], f"Hata: {resp.status_code}"
        except Exception as e:
            return history or [], f"Hata: {str(e)}"
    
    def create_agent_fn(name, embedding_model, data_source_type, data_source, uploaded_file, progress_output):
        print(f"[DEBUG] create_agent_fn çağrıldı: name={name}, uploaded_file={uploaded_file}")
        try:
            if not current_token:
                return (
                    "Önce giriş yapın", 
                    gr.update(visible=False),
                    gr.update()
                )
            
            if not name or not name.strip():
                return (
                    "Agent adı gerekli", 
                    gr.update(visible=False),
                    gr.update()
                )
            
            progress_msg = "📤 İşlem başlatılıyor..."
            progress_update = gr.update(visible=True, value=progress_msg)
            
            final_data_source = data_source
            
            if uploaded_file is not None:
                try:
                    import requests
                    if isinstance(uploaded_file, dict):
                        file_path = uploaded_file.get('name') or uploaded_file.get('path')
                    else:
                        file_path = str(uploaded_file)
                    
                    file_name = Path(file_path).name
                    file_size = Path(file_path).stat().st_size / (1024 * 1024)
                    progress_msg = f"📤 **Dosya yükleniyor:** {file_name} ({file_size:.1f} MB)"
                    progress_update = gr.update(visible=True, value=progress_msg)
                    
                    with open(file_path, "rb") as f:
                        files = {"file": (file_name, f, "application/octet-stream")}
                        headers = {"Authorization": f"Bearer {current_token}"}
                        
                        upload_resp = requests.post(
                            "http://localhost:3000/api/upload",
                            files=files,
                            headers=headers,
                            timeout=300
                        )
                        
                        if upload_resp.status_code == 200:
                            upload_data = upload_resp.json()["data"]
                            final_data_source = upload_data["filePath"]
                            progress_msg = f"✅ **Dosya yüklendi:** {file_name}\n\n🔄 **Embedding başlıyor...**\n📝 Dosya parse ediliyor ve parçalara bölünüyor..."
                            progress_update = gr.update(visible=True, value=progress_msg)
                        else:
                            return (
                                f"❌ Dosya yükleme hatası: {upload_resp.json().get('detail', 'Bilinmeyen hata')}", 
                                gr.update(visible=False),
                                gr.update()
                            )
                except Exception as e:
                    import traceback
                    print(f"[!] Upload hatası: {traceback.format_exc()}")
                    return (
                        f"❌ Dosya yükleme hatası: {str(e)}", 
                        gr.update(visible=False),
                        gr.update()
                    )
            elif not data_source or not data_source.strip():
                return (
                    "URL veya dosya gerekli", 
                    gr.update(visible=False),
                    gr.update()
                )
            else:
                progress_msg = "🔄 Agent oluşturuluyor ve veriler işleniyor...\n(Bu işlem dosya boyutuna göre birkaç dakika sürebilir)"
                progress_update = gr.update(visible=True, value=progress_msg)
            
            try:
                import requests
                progress_msg = "🔄 **Agent oluşturuluyor...**\n\n📊 **İşlem Adımları:**\n1. Dosya parse ediliyor...\n2. Metin parçalara bölünüyor...\n3. Embedding yapılıyor (bu en uzun süren adım)...\n   - Batch'ler halinde işleniyor\n   - Her batch'te progress terminal'de görünecek\n4. FAISS index oluşturuluyor...\n\n⏳ Bu işlem dosya boyutuna göre birkaç dakika sürebilir.\n📝 Terminal'de detaylı progress log'ları görebilirsiniz (Batch X/Y, %Yüzde)."
                progress_update = gr.update(visible=True, value=progress_msg)
                
                resp = requests.post(
                    "http://localhost:3000/api/agents",
                    json={
                        "name": name,
                        "embedding_model": embedding_model,
                        "data_source_type": data_source_type if not uploaded_file else "file",
                        "data_source": final_data_source
                    },
                    headers={"Authorization": f"Bearer {current_token}"},
                    timeout=600
                )
                if resp.status_code == 200:
                    data = resp.json()
                    agent_data = data.get("data", {})
                    ingestion_info = data.get("ingestion_info", {})
                    chunks = agent_data.get("chunkCount", ingestion_info.get("chunks", 0))
                    agent_id = agent_data.get("id", "bilinmiyor")
                    
                    agent_choices_updated = []
                    try:
                        agent_resp = requests.get(
                            "http://localhost:3000/api/agents",
                            headers={"Authorization": f"Bearer {current_token}"},
                            timeout=5
                        )
                        if agent_resp.status_code == 200:
                            nonlocal current_agents
                            current_agents = agent_resp.json()["data"]
                            agent_choices_updated = [f"{a['name']}" for a in current_agents]
                    except:
                        pass
                    
                    index_name = agent_data.get("indexName", "N/A")
                    
                    success_msg = f"""✅ **Agent Başarıyla Oluşturuldu!**

📋 **Agent Bilgileri:**
- **Ad:** {name}
- **ID:** `{agent_id}`
- **Index Adı:** `{index_name}`
- **İşlenen Parça Sayısı:** {chunks}
- **Embedding Model:** `{embedding_model}`

📁 **Index Konumu:**
`python_services/data/faiss_index/{index_name}/`

💬 **Kullanım:**
Chat sayfasından agent'ı seçip sorularınızı sorabilirsiniz!

📊 **Terminal'de görebileceğiniz log'lar:**
- Dosya parse işlemi
- Chunk oluşturma
- Batch'ler halinde embedding (örn: Batch 1/5 - %20, Batch 2/5 - %40...)
- Index kaydetme"""
                    
                    return (
                        success_msg, 
                        gr.update(visible=False),
                        gr.update(choices=agent_choices_updated, value=agent_choices_updated[0] if agent_choices_updated else None)
                    )
                else:
                    return (
                        f"❌ Hata: {resp.json().get('detail', 'Bilinmeyen hata')}", 
                        gr.update(visible=False),
                        gr.update()
                    )
            except requests.exceptions.Timeout:
                return (
                    "❌ İşlem zaman aşımına uğradı. Dosya çok büyük olabilir.", 
                    gr.update(visible=False),
                    gr.update()
                )
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"[!] Agent oluşturma hatası: {error_detail}")
                return (
                    f"❌ Hata: {str(e)}", 
                    gr.update(visible=False),
                    gr.update()
                )
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[!] Agent oluşturma genel hatası: {error_detail}")
            return (
                f"❌ Genel hata: {str(e)}", 
                gr.update(visible=False),
                gr.update()
            )
    
    def create_company_fn(name, email):
        if not current_token:
            return "Önce giriş yapın"
        try:
            import requests
            resp = requests.post(
                "http://localhost:3000/api/admin/companies",
                json={"name": name, "email": email},
                headers={"Authorization": f"Bearer {current_token}"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()["data"]
                return f"✅ Şirket oluşturuldu!\nKullanıcı: {data['username']}\nŞifre: {data['password']}"
            else:
                return f"❌ Hata: {resp.json().get('detail', 'Bilinmeyen hata')}"
        except Exception as e:
            return f"❌ Hata: {str(e)}"
    
    try:
        app = gr.Blocks(title="RAG SaaS Platform", css=custom_css)
    except TypeError:
        app = gr.Blocks(title="RAG SaaS Platform")
    
    with app:
        gr.Markdown("# RAG SaaS Platform")
        
        with gr.Tab("Giriş", visible=True) as login_tab:
            with gr.Row():
                with gr.Column():
                    login_user = gr.Textbox(label="Kullanıcı Adı", value="admin@ragplatform.com")
                    login_pass = gr.Textbox(label="Şifre", type="password", value="Admin123!@#")
                    login_btn = gr.Button("Giriş Yap", variant="primary")
                    login_status = gr.Markdown()
        
        with gr.Tab("Chat", visible=False) as chat_tab:
            with gr.Row():
                with gr.Column():
                    agent_dropdown = gr.Dropdown(choices=[], label="Agent Seç", interactive=True)
                    model_radio = gr.Radio(
                        [
                            "dbmdz/gpt2-turkish-cased (GPT-2 Türkçe)",
                            "bert-base-turkish-cased (BERT Türkçe)",
                            "savasy/bert-base-turkish-sentiment-cased (BERT Sentiment)"
                        ],
                        value="dbmdz/gpt2-turkish-cased (GPT-2 Türkçe)",
                        label="Model"
                    )
                    chatbot = gr.Chatbot(label="Chat", height=500, type="messages", allow_tags=False)
                    msg_input = gr.Textbox(label="Mesaj", placeholder="Sorunuzu yazın...")
                    send_btn = gr.Button("Gönder", variant="primary")
                    
                    send_btn.click(
                        chat_fn,
                        inputs=[msg_input, chatbot, agent_dropdown, model_radio],
                        outputs=[chatbot, msg_input]
                    )
                    msg_input.submit(
                        chat_fn,
                        inputs=[msg_input, chatbot, agent_dropdown, model_radio],
                        outputs=[chatbot, msg_input]
                    )
        
        with gr.Tab("Agent Oluştur", visible=False) as agents_tab:
            with gr.Row():
                with gr.Column():
                    agent_name = gr.Textbox(label="Agent Adı", placeholder="Örn: Müşteri Destek Botu")
                    agent_embedding = gr.Dropdown(
                        choices=["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
                        value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        label="Embedding Model"
                    )
                    gr.Markdown("### Veri Kaynağı")
                    agent_file_upload = gr.File(
                        label="Dosya Yükle (PDF, DOCX, TXT, JSON, CSV)",
                        file_types=[".pdf", ".docx", ".txt", ".json", ".csv"]
                    )
                    gr.Markdown("**VEYA**")
                    agent_source = gr.Textbox(
                        label="URL veya Dosya Yolu", 
                        placeholder="https://example.com/article veya /path/to/file.pdf"
                    )
                    create_agent_btn = gr.Button("Agent Oluştur", variant="primary")
                    agent_status = gr.Markdown()
                    agent_progress = gr.Markdown(visible=False, value="")
                    
                    agent_source_type_hidden = gr.State(value="file")
                    
                    def agent_click_wrapper(name, embedding_model, data_source_type, data_source, uploaded_file):
                        print(f"[DEBUG] agent_click_wrapper çağrıldı: name={name}, uploaded_file={uploaded_file}")
                        try:
                            result = create_agent_fn(name, embedding_model, data_source_type, data_source, uploaded_file, None)
                            print(f"[DEBUG] create_agent_fn sonucu: {result}")
                            return result
                        except Exception as e:
                            import traceback
                            error_detail = traceback.format_exc()
                            print(f"[DEBUG] Wrapper hatası: {error_detail}")
                            return (
                                f"❌ Beklenmeyen hata: {str(e)}", 
                                gr.update(visible=False),
                                gr.update()
                            )
                    
                    create_agent_btn.click(
                        agent_click_wrapper,
                        inputs=[agent_name, agent_embedding, agent_source_type_hidden, agent_source, agent_file_upload],
                        outputs=[agent_status, agent_progress, agent_dropdown],
                        show_progress=True
                    )
        
        with gr.Tab("Şirket Yönetimi", visible=False) as companies_tab:
            with gr.Row():
                with gr.Column():
                    comp_name = gr.Textbox(label="Şirket Adı")
                    comp_email = gr.Textbox(label="Email (opsiyonel)")
                    create_comp_btn = gr.Button("Şirket Oluştur", variant="primary")
                    comp_status = gr.Markdown()
                    
                    create_comp_btn.click(
                        create_company_fn,
                        inputs=[comp_name, comp_email],
                        outputs=[comp_status]
                    )
        
        login_btn.click(
            login_fn,
            inputs=[login_user, login_pass],
            outputs=[login_status, login_tab, chat_tab, companies_tab, agents_tab, agent_dropdown]
        )
    
    return app, custom_css

print("✅ Frontend builder hazır")
```

---

## HÜCRE 7: Startup - Backend ve Frontend Başlatma

```python
# ============================================
# HÜCRE 7: Startup - Backend ve Frontend Başlatma
# ============================================

def run_backend():
    uvicorn.run(backend_app, host="0.0.0.0", port=3000, log_level="error")

def run_frontend():
    result = build_gradio_ui()
    if isinstance(result, tuple):
        app, custom_css = result
    else:
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
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value,
            show_error=True
        )
    except Exception as e:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value
        )

# Backend'i başlat
print("\n[STARTUP] Backend başlatılıyor...")
backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()
time.sleep(5)  # Backend'in başlaması için bekle
print("✅ Backend: http://localhost:3000")

# Frontend'i başlat
print("\n[STARTUP] Frontend başlatılıyor...")
print("\n" + "=" * 60)
print("🔑 Giriş: admin@ragplatform.com / Admin123!@#")
print("=" * 60)
print("\n⏳ Gradio public URL oluşturuluyor...")
print("   (Bu işlem 10-20 saniye sürebilir)\n")

# Frontend'i ana thread'de çalıştır (blocking)
run_frontend()
```

---

## Kullanım Talimatları

1. **Önce paketleri kurun** (sadece bir kere):
   - `COLAB_INSTALL.py` dosyasını çalıştırın

2. **Yukarıdaki 7 hücreyi sırayla Colab'e ekleyin:**
   - Her hücreyi ayrı bir Colab hücresine kopyalayın
   - Hücre 1'den başlayıp Hücre 7'ye kadar sırayla çalıştırın
   - Her hücreyi Shift+Enter ile çalıştırın

3. **Son hücre (Hücre 7) çalıştıktan sonra:**
   - Terminal çıktısında Gradio public URL'i görünecek
   - URL'i tarayıcıda açın
   - Giriş yapın: `admin@ragplatform.com` / `Admin123!@#`

