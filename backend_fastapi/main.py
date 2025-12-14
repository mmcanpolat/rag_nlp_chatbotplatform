#!/usr/bin/env python3
"""
FastAPI Backend - RAG SaaS Platform
Express.js backend'inin Python'a çevrilmiş hali
"""

import os
import sys
import json
import secrets
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Python servislerini import et
python_services_path = Path(__file__).parent.parent / "python_services" / "scripts"
sys.path.insert(0, str(python_services_path))

from dotenv import load_dotenv
load_dotenv()

# Python servislerini import et
try:
    from rag_engine import RAGEngine
    from ingestor import DocumentIngestor
    from evaluator import Evaluator
except ImportError as e:
    # Alternatif import yolu
    import importlib.util
    python_services_path = Path(__file__).parent.parent / "python_services" / "scripts"
    
    spec = importlib.util.spec_from_file_location("rag_engine", python_services_path / "rag_engine.py")
    rag_engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_engine_module)
    RAGEngine = rag_engine_module.RAGEngine
    
    spec = importlib.util.spec_from_file_location("ingestor", python_services_path / "ingestor.py")
    ingestor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingestor_module)
    DocumentIngestor = ingestor_module.DocumentIngestor
    
    spec = importlib.util.spec_from_file_location("evaluator", python_services_path / "evaluator.py")
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    Evaluator = evaluator_module.Evaluator

# ==================== CONFIG ====================

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "python_services" / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600000"))  # 10 dakika

# ==================== FASTAPI APP ====================

app = FastAPI(title="RAG SaaS Platform API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit için
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== IN-MEMORY DATA ====================

companies: Dict[str, dict] = {}
agents: Dict[str, dict] = {}
sessions: Dict[str, dict] = {}

# SuperAdmin default
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
    """Benzersiz ID üret"""
    return f"{int(time.time() * 1000)}_{secrets.token_hex(4)}"

def generate_strong_password() -> str:
    """24 karakter güçlü şifre üret"""
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*'
    return ''.join(secrets.choice(chars) for _ in range(24))

def get_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Session'dan user al"""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    return sessions.get(token)

def require_auth(user: Optional[dict] = Depends(get_user)) -> dict:
    """Auth gerektiren endpoint'ler için"""
    if not user:
        raise HTTPException(status_code=401, detail="Giriş yapmanız gerekiyor")
    return user

def require_superadmin(user: dict = Depends(require_auth)) -> dict:
    """SuperAdmin gerektiren endpoint'ler için"""
    if not user.get("isSuperAdmin"):
        raise HTTPException(status_code=403, detail="Yetki yok")
    return user

# ==================== MODELS ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class CompanyCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""

class AgentCreate(BaseModel):
    name: str
    embedding_model: str
    data_source_type: str  # 'file' or 'url'
    data_source: str  # file path or URL
    
    class Config:
        populate_by_name = True

class ChatRequest(BaseModel):
    query: str
    agent_id: str
    model: str = "gpt"  # 'gpt', 'bert-turkish', 'bert-sentiment'
    
    class Config:
        # camelCase desteği
        populate_by_name = True

# ==================== AUTH ENDPOINTS ====================

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Kullanıcı girişi"""
    # SuperAdmin kontrol
    if request.username == SUPER_ADMIN["username"] and request.password == SUPER_ADMIN["password"]:
        token = gen_id()
        sessions[token] = {"isSuperAdmin": True, "username": request.username}
        return {
            "success": True,
            "data": {
                "userId": "admin",
                "username": request.username,
                "isSuperAdmin": True,
                "sessionToken": token
            }
        }
    
    # Şirket kontrol
    company = next((c for c in companies.values() if c["username"] == request.username), None)
    if company and company["password"] == request.password:
        token = gen_id()
        sessions[token] = {
            "companyId": company["id"],
            "username": request.username,
            "companyName": company["name"]
        }
        return {
            "success": True,
            "data": {
                "userId": company["id"],
                "username": request.username,
                "companyId": company["id"],
                "companyName": company["name"],
                "isSuperAdmin": False,
                "sessionToken": token
            }
        }
    
    raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

@app.post("/api/auth/logout")
async def logout(user: dict = Depends(require_auth)):
    """Çıkış"""
    # Token'ı sil (header'dan al)
    return {"success": True}

# ==================== PUBLIC ENDPOINTS ====================

@app.get("/api/companies/count")
async def get_companies_count():
    """Public - şirket sayısı"""
    return {"success": True, "count": len(companies)}

@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ==================== ADMIN ENDPOINTS ====================

@app.post("/api/admin/companies")
async def create_company(company_data: CompanyCreate, user: dict = Depends(require_superadmin)):
    """Şirket oluştur"""
    if not company_data.name.strip():
        raise HTTPException(status_code=400, detail="Şirket adı gerekli")
    
    company_id = gen_id()
    
    # Username oluştur
    if company_data.email and company_data.email.strip():
        username = company_data.email.strip().lower()
    else:
        base_email = company_data.name.lower().replace(" ", "").replace("-", "") + "@company.com"
        username = base_email
    
    password = generate_strong_password()
    
    company = {
        "id": company_id,
        "name": company_data.name.strip(),
        "description": company_data.description or "",
        "phone": company_data.phone or "",
        "email": company_data.email or "",
        "username": username,
        "password": password,
        "createdAt": datetime.now().isoformat()
    }
    
    companies[company_id] = company
    
    # Password'ü response'da döndür
    return {
        "success": True,
        "data": company  # Password dahil
    }

@app.get("/api/admin/companies")
async def list_companies(user: dict = Depends(require_superadmin)):
    """Şirket listesi"""
    company_list = [
        {
            "id": c["id"],
            "name": c["name"],
            "description": c["description"],
            "phone": c["phone"],
            "email": c["email"],
            "username": c["username"],
            "createdAt": c["createdAt"]
        }
        for c in companies.values()
    ]
    return {"success": True, "data": company_list}

@app.delete("/api/admin/companies/{company_id}")
async def delete_company(company_id: str, user: dict = Depends(require_superadmin)):
    """Şirket sil"""
    if company_id not in companies:
        raise HTTPException(status_code=404, detail="Şirket bulunamadı")
    
    del companies[company_id]
    return {"success": True}

# ==================== AGENT ENDPOINTS ====================

@app.post("/api/agents")
async def create_agent(agent_data: AgentCreate, user: dict = Depends(require_auth)):
    """Agent oluştur"""
    agent_id = gen_id()
    index_name = f"agent_{agent_id}"
    
    agent = {
        "id": agent_id,
        "name": agent_data.name,
        "companyId": user.get("companyId"),
        "embeddingModel": agent_data.embedding_model,
        "dataSourceType": agent_data.data_source_type,
        "dataSource": agent_data.data_source,
        "indexName": index_name,
        "createdAt": datetime.now().isoformat()
    }
    
    # Eğer veri kaynağı varsa, ingestion yap
    if agent_data.data_source:
        try:
            ingestor = DocumentIngestor(
                index_name=index_name,
                embedding_model=agent_data.embedding_model
            )
            
            if agent_data.data_source_type == "file":
                # Dosya yolu
                file_path = Path(agent_data.data_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"Dosya bulunamadı: {agent_data.data_source}")
                ingestor.ingest(str(file_path))
            elif agent_data.data_source_type == "url":
                # URL
                ingestor.ingest(agent_data.data_source)
            else:
                raise ValueError("Geçersiz veri kaynağı tipi")
            
        except Exception as e:
            return {"success": False, "error": f"Ingestion hatası: {str(e)}"}
    
    agents[agent_id] = agent
    return {"success": True, "data": agent}

@app.get("/api/agents")
async def list_agents(user: dict = Depends(require_auth)):
    """Agent listesi"""
    user_company_id = user.get("companyId")
    
    if user.get("isSuperAdmin"):
        agent_list = list(agents.values())
    else:
        agent_list = [a for a in agents.values() if a.get("companyId") == user_company_id]
    
    return {"success": True, "data": agent_list}

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, user: dict = Depends(require_auth)):
    """Agent sil"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent bulunamadı")
    
    agent = agents[agent_id]
    
    # Yetki kontrolü
    if not user.get("isSuperAdmin") and agent.get("companyId") != user.get("companyId"):
        raise HTTPException(status_code=403, detail="Yetki yok")
    
    del agents[agent_id]
    return {"success": True}

# ==================== CHAT ENDPOINT ====================

@app.post("/api/chat")
async def chat(request: ChatRequest, user: dict = Depends(require_auth)):
    """Chat endpoint - RAG sorgusu"""
    agent_id = request.agent_id
    
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent bulunamadı")
    
    agent = agents[agent_id]
    
    # Yetki kontrolü
    if not user.get("isSuperAdmin") and agent.get("companyId") != user.get("companyId"):
        raise HTTPException(status_code=403, detail="Yetki yok")
    
    index_name = agent.get("indexName", f"agent_{agent_id}")
    
    try:
        rag = RAGEngine(index_name=index_name)
        result = rag.query(request.query, model=request.model)
        
        return {
            "success": True,
            "data": {
                "answer": result["answer"],
                "context": result["context"],
                "confidence": result["confidence"],
                "model_used": result["model_used"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat hatası: {str(e)}")

# ==================== UPLOAD ENDPOINT ====================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: dict = Depends(require_auth)
):
    """Dosya yükleme"""
    allowed_extensions = ['.json', '.txt', '.md', '.pdf', '.docx', '.doc', '.csv']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz dosya tipi: {file_ext}. İzin verilen: {', '.join(allowed_extensions)}"
        )
    
    # Dosyayı kaydet
    file_path = UPLOAD_DIR / f"upload_{int(time.time() * 1000)}{file_ext}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "success": True,
        "data": {
            "filePath": str(file_path),
            "fileName": file.filename
        }
    }

# ==================== BENCHMARK ENDPOINT ====================

@app.post("/api/benchmark")
async def run_benchmark(request: dict, user: dict = Depends(require_auth)):
    """Benchmark çalıştır"""
    agent_id = request.get("agent_id")
    if not agent_id or agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent bulunamadı")
    
    agent = agents[agent_id]
    index_name = agent.get("indexName", f"agent_{agent_id}")
    
    try:
        evaluator = Evaluator(index_name=index_name)
        results = evaluator.evaluate_all()
        
        return {
            "success": True,
            "data": results,
            "plots": ["confusion_matrix.png", "metrics_comparison_bar.png"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark hatası: {str(e)}")

# ==================== MAIN ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

