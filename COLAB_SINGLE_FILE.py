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

import subprocess
import sys

# Bağımlılıkları kontrol et ve kur
print("=" * 60)
print("RAG SaaS Platform - Tek Dosya Başlatma")
print("=" * 60)
print("\n[1/5] Bağımlılıklar kontrol ediliyor...")

required_packages = [
    "fastapi", "uvicorn[standard]", "gradio>=4.0.0", "langchain", "langchain-community",
    "langchain-huggingface", "langchain-text-splitters", "transformers", "torch", 
    "sentence-transformers", "faiss-cpu", "pypdf", "docx2txt", "beautifulsoup4", 
    "requests", "python-dotenv"
]

missing = []
for pkg in required_packages:
    try:
        pkg_import = pkg.split("[")[0].split(">=")[0].replace("-", "_")
        __import__(pkg_import)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"   {len(missing)} paket eksik, kuruluyor (5-10 dakika)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing, check=False)
    print("✅ Bağımlılıklar kuruldu")
else:
    print("✅ Tüm bağımlılıklar mevcut")

print("\n[2/5] Modüller yükleniyor...")

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
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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
    
    def ingest(self, source: str) -> dict:
        try:
            docs = self.load_document(source)
            if not docs:
                return {"success": False, "error": "Döküman yüklenemedi"}
            
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                return {"success": False, "error": "Döküman bölünemedi"}
            
            # Mevcut index varsa merge et, yoksa yeni oluştur
            if (self.index_path / "index.faiss").exists():
                vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            vectorstore.save_local(str(self.index_path))
            
            # Metadata kaydet
            with open(self.index_path / "metadata.json", "w") as f:
                json.dump({
                    "embedding_model": self.embedding_model_name,
                    "created_at": datetime.now().isoformat(),
                    "chunk_count": len(chunks)
                }, f)
            
            return {"success": True, "chunks": len(chunks)}
        except Exception as e:
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
    
    # Ingestion yap
    if agent.data_source:
        try:
            ingestor = DocumentIngestor(index_name=index_name, embedding_model=agent.embedding_model)
            result = ingestor.ingest(agent.data_source)
            if not result.get("success"):
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": result.get("error", "Ingestion failed")}
                )
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": str(e)}
            )
    
    agents[agent_id] = new_agent
    return {"success": True, "data": new_agent}

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

# ==================== GRADIO FRONTEND ====================
def build_gradio_ui():
    current_user = {"username": "Giriş yapılmadı"}
    current_token = None
    current_agents = []
    
    def login_fn(username, password):
        try:
            import requests
            resp = requests.post(
                "http://localhost:3000/api/auth/login",
                json={"username": username, "password": password},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                nonlocal current_user, current_token, current_agents
                current_user = data["data"]
                current_token = data["token"]
                # Agent listesini güncelle
                try:
                    agent_resp = requests.get(
                        "http://localhost:3000/api/agents",
                        headers={"Authorization": f"Bearer {current_token}"},
                        timeout=5
                    )
                    if agent_resp.status_code == 200:
                        current_agents = agent_resp.json()["data"]
                        agent_choices = [f"{a['name']}" for a in current_agents]
                        agent_values = [a['id'] for a in current_agents]
                    else:
                        agent_choices = []
                        agent_values = []
                except:
                    agent_choices = []
                    agent_values = []
                
                return (
                    f"✅ Giriş başarılı: {current_user.get('username', '')}",
                    gr.update(visible=True),
                    gr.update(visible=current_user.get("isSuperAdmin", False)),
                    gr.update(choices=agent_choices, value=agent_choices[0] if agent_choices else None)
                )
            return "❌ Giriş başarısız", gr.update(visible=False), gr.update(visible=False), gr.update()
        except Exception as e:
            return f"❌ Hata: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update()
    
    def chat_fn(message, history, agent_name, model):
        if not current_token:
            return history or [], "Önce giriş yapın"
        
        if not agent_name or not current_agents:
            return history or [], "Agent seçin veya oluşturun"
        
        # Agent ID'yi bul
        agent_id = None
        for a in current_agents:
            if a['name'] == agent_name:
                agent_id = a['id']
                break
        
        if not agent_id:
            return history or [], "Agent bulunamadı"
        
        try:
            import requests
            resp = requests.post(
                "http://localhost:3000/api/chat",
                json={"agent_id": agent_id, "query": message, "model": model},
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
    
    def create_agent_fn(name, embedding_model, data_source_type, data_source):
        if not current_token:
            return "Önce giriş yapın"
        try:
            import requests
            resp = requests.post(
                "http://localhost:3000/api/agents",
                json={
                    "name": name,
                    "embedding_model": embedding_model,
                    "data_source_type": data_source_type,
                    "data_source": data_source
                },
                headers={"Authorization": f"Bearer {current_token}"},
                timeout=120
            )
            if resp.status_code == 200:
                return "✅ Agent oluşturuldu! Chat sayfasından kullanabilirsiniz."
            else:
                return f"❌ Hata: {resp.json().get('detail', 'Bilinmeyen hata')}"
        except Exception as e:
            return f"❌ Hata: {str(e)}"
    
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
    
    # Custom CSS - Snow White Theme
    custom_css = """
    .gradio-container {
        background: #fafafa !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .gr-button {
        background: #2c3e50 !important;
        color: white !important;
    }
    .gr-button:hover {
        background: #34495e !important;
    }
    """
    
    with gr.Blocks(title="RAG SaaS Platform", css=custom_css) as app:
        gr.Markdown("# RAG SaaS Platform")
        
        with gr.Tab("Giriş"):
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
                    model_radio = gr.Radio(["GPT", "BERT-CASED", "BERT-SENTIMENT"], value="GPT", label="Model")
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
                    agent_name = gr.Textbox(label="Agent Adı")
                    agent_embedding = gr.Dropdown(
                        choices=["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
                        value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        label="Embedding Model"
                    )
                    agent_source_type = gr.Radio(
                        ["file", "url"],
                        value="url",
                        label="Veri Kaynağı Tipi"
                    )
                    agent_source = gr.Textbox(label="URL veya Dosya Yolu", placeholder="https://example.com veya /path/to/file.pdf")
                    create_agent_btn = gr.Button("Agent Oluştur", variant="primary")
                    agent_status = gr.Markdown()
                    
                    create_agent_btn.click(
                        create_agent_fn,
                        inputs=[agent_name, agent_embedding, agent_source_type, agent_source],
                        outputs=[agent_status]
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
            outputs=[login_status, chat_tab, companies_tab, agent_dropdown]
        )
        login_btn.click(
            lambda: gr.update(visible=True),
            outputs=[agents_tab]
        )
    
    return app

# ==================== STARTUP ====================
def run_backend():
    uvicorn.run(backend_app, host="0.0.0.0", port=3000, log_level="error")

def run_frontend():
    app = build_gradio_ui()
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except:
        pass
    
    share_value = is_colab or os.getenv("GRADIO_SHARE", "").lower() == "true"
    print(f"\n[*] Gradio başlatılıyor (share={share_value})...")
    
    # Gradio 5.0+ için theme ve css parametreleri launch'a taşındı
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value,
            show_error=True
        )
    except TypeError:
        # Eski Gradio versiyonları için
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share_value
        )

if __name__ == "__main__":
    print("\n[4/5] Backend başlatılıyor...")
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    time.sleep(5)  # Backend'in başlaması için bekle
    print("✅ Backend: http://localhost:3000")
    
    print("\n[5/5] Frontend başlatılıyor...")
    print("\n" + "=" * 60)
    print("🔑 Giriş: admin@ragplatform.com / Admin123!@#")
    print("=" * 60)
    print("\n⏳ Gradio public URL oluşturuluyor...")
    print("   (Bu işlem 10-20 saniye sürebilir)\n")
    
    # Frontend'i ana thread'de çalıştır (blocking)
    run_frontend()
