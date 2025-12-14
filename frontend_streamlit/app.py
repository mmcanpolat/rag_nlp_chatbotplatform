#!/usr/bin/env python3
# Streamlit Frontend - RAG SaaS Platform
# Angular yerine Streamlit kullanıyorum, daha basit ve Python-only

import streamlit as st
import requests
import json
from typing import Optional, Dict, List
import os

# ==================== CONFIG ====================

# API base URL - backend FastAPI'ye bağlanıyorum
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")
API_URL = f"{API_BASE_URL}/api"

# ==================== SESSION STATE ====================

# Streamlit session state - sayfa yenilense bile veriler kalıyor
# Kullanıcı bilgisi, token, agent'lar vs. burada tutuluyor
if "user" not in st.session_state:
    st.session_state.user = None
if "session_token" not in st.session_state:
    st.session_state.session_token = None
if "agents" not in st.session_state:
    st.session_state.agents = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # Her agent için ayrı chat history

# ==================== HELPER FUNCTIONS ====================

def get_headers() -> Dict[str, str]:
    # API istekleri için header'lar hazırlıyorum
    # Token varsa Authorization header'ına ekliyorum
    headers = {"Content-Type": "application/json"}
    if st.session_state.session_token:
        headers["Authorization"] = f"Bearer {st.session_state.session_token}"
    return headers

def api_request(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
    # API isteği yapıyorum - GET, POST, DELETE destekliyorum
    # Hata olursa try-except ile yakalıyorum
    url = f"{API_URL}{endpoint}"
    headers = get_headers()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            return {"success": False, "error": "Geçersiz method"}
        
        return response.json()
    except Exception as e:
        # Hata olursa detaylı mesaj döndürüyorum
        return {"success": False, "error": str(e)}

# ==================== LOGIN PAGE ====================

def login_page():
    """Giriş sayfası"""
    st.title("🔐 Giriş Yap")
    
    with st.form("login_form"):
        username = st.text_input("Kullanıcı Adı / Email")
        password = st.text_input("Şifre", type="password")
        submit = st.form_submit_button("Giriş Yap")
        
        if submit:
            result = api_request("POST", "/auth/login", {
                "username": username,
                "password": password
            })
            
            if result.get("success"):
                st.session_state.user = result["data"]
                st.session_state.session_token = result["data"]["sessionToken"]
                st.success("Giriş başarılı!")
                st.rerun()
            else:
                st.error(result.get("error", "Giriş başarısız"))
    
    # SuperAdmin bilgileri
    st.markdown("---")
    st.markdown("### SuperAdmin Girişi")
    st.info("**Kullanıcı:** admin@ragplatform.com  \n**Şifre:** Admin123!@#")

# ==================== MAIN APP ====================

def main_app():
    """Ana uygulama"""
    user = st.session_state.user
    
    # Sidebar
    with st.sidebar:
        st.title("RAG Platform")
        
        # Kullanıcı bilgisi
        st.markdown(f"**Kullanıcı:** {user.get('username', 'N/A')}")
        if user.get("isSuperAdmin"):
            st.markdown("**Rol:** SuperAdmin")
        else:
            st.markdown(f"**Şirket:** {user.get('companyName', 'N/A')}")
        
        st.markdown("---")
        
        # Menü
        page = st.radio(
            "Menü",
            ["💬 Chat", "📊 Analytics", "🤖 Agents", "👥 Şirket Yönetimi" if user.get("isSuperAdmin") else None, "📤 Veri Yükle"],
            filter(lambda x: x is not None, ["💬 Chat", "📊 Analytics", "🤖 Agents", "👥 Şirket Yönetimi" if user.get("isSuperAdmin") else None, "📤 Veri Yükle"])
        )
        
        if st.button("🚪 Çıkış"):
            st.session_state.user = None
            st.session_state.session_token = None
            st.rerun()
    
    # Sayfa içeriği
    if page == "💬 Chat":
        chat_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "🤖 Agents":
        agents_page()
    elif page == "👥 Şirket Yönetimi":
        companies_page()
    elif page == "📤 Veri Yükle":
        upload_page()

# ==================== CHAT PAGE ====================

def chat_page():
    """Chat sayfası"""
    st.title("💬 Chat")
    
    # Agent seçimi
    agents_result = api_request("GET", "/agents")
    if not agents_result.get("success"):
        st.error("Agent'lar yüklenemedi")
        return
    
    agents_list = agents_result.get("data", [])
    if not agents_list:
        st.warning("Henüz agent oluşturulmamış. Önce 'Agents' sayfasından agent oluşturun.")
        return
    
    agent_names = [f"{a['name']} ({a['id']})" for a in agents_list]
    selected_agent_idx = st.selectbox("Agent Seçin", range(len(agent_names)), format_func=lambda x: agent_names[x])
    selected_agent = agents_list[selected_agent_idx]
    
    st.session_state.active_agent = selected_agent["id"]
    
    # Model seçimi
    model = st.radio("Model", ["gpt", "bert-turkish", "bert-sentiment"], horizontal=True)
    
    st.markdown("---")
    
    # Chat history
    agent_id = selected_agent["id"]
    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []
    
    # Chat mesajlarını göster
    for msg in st.session_state.chat_history[agent_id]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if "context" in msg:
                    with st.expander("📄 Retrieved Context"):
                        st.text(msg["context"])
                if "confidence" in msg:
                    st.caption(f"Güven: {msg['confidence']:.2%}")
    
    # Chat input
    query = st.chat_input("Sorunuzu yazın...")
    
    if query:
        # Kullanıcı mesajını göster
        with st.chat_message("user"):
            st.write(query)
        
        st.session_state.chat_history[agent_id].append({"role": "user", "content": query})
        
        # API'ye istek gönder
        with st.spinner("Yanıt oluşturuluyor..."):
            result = api_request("POST", "/chat", {
                "query": query,
                "agent_id": agent_id,  # snake_case
                "model": model
            })
        
        if result.get("success"):
            data = result["data"]
            
            # Asistan yanıtını göster
            with st.chat_message("assistant"):
                st.write(data["answer"])
                
                if data.get("context"):
                    with st.expander("📄 Retrieved Context"):
                        st.text(data["context"])
                
                if data.get("confidence"):
                    st.caption(f"Güven: {data['confidence']:.2%}")
            
            st.session_state.chat_history[agent_id].append({
                "role": "assistant",
                "content": data["answer"],
                "context": data.get("context"),
                "confidence": data.get("confidence")
            })
        else:
            st.error(result.get("error", "Chat hatası"))

# ==================== ANALYTICS PAGE ====================

def analytics_page():
    """Analytics sayfası"""
    st.title("📊 Analytics")
    
    # Agent seçimi
    agents_result = api_request("GET", "/agents")
    if not agents_result.get("success"):
        st.error("Agent'lar yüklenemedi")
        return
    
    agents_list = agents_result.get("data", [])
    if not agents_list:
        st.warning("Henüz agent oluşturulmamış.")
        return
    
    agent_names = [f"{a['name']} ({a['id']})" for a in agents_list]
    selected_agent_idx = st.selectbox("Agent Seçin", range(len(agent_names)), format_func=lambda x: agent_names[x])
    selected_agent = agents_list[selected_agent_idx]
    
    if st.button("🔄 Benchmark Çalıştır"):
        with st.spinner("Benchmark çalıştırılıyor (bu biraz zaman alabilir)..."):
            result = api_request("POST", "/benchmark", {"agent_id": selected_agent["id"]})
        
        if result.get("success"):
            data = result["data"]
            
            # Metrikleri göster
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Ortalama Accuracy", f"{data.get('avg_accuracy', 0):.2%}")
            with col2:
                st.metric("Ortalama BLEU", f"{data.get('avg_bleu', 0):.4f}")
            with col3:
                st.metric("Ortalama ROUGE-L", f"{data.get('avg_rouge', 0):.4f}")
            with col4:
                st.metric("Ortalama F1", f"{data.get('avg_f1', 0):.4f}")
            
            # Plot'ları göster - frontend_streamlit/assets/plots/ klasöründen
            plots = result.get("plots", [])
            for plot_name in plots:
                # Önce frontend klasöründe ara, yoksa python_services'te ara
                plot_path = f"frontend_streamlit/assets/plots/{plot_name}"
                if not os.path.exists(plot_path):
                    plot_path = f"python_services/data/plots/{plot_name}"
                if os.path.exists(plot_path):
                    st.image(plot_path)
                else:
                    st.warning(f"Plot bulunamadı: {plot_name}")
        else:
            st.error(result.get("error", "Benchmark hatası"))

# ==================== AGENTS PAGE ====================

def agents_page():
    """Agents sayfası"""
    st.title("🤖 Agents")
    
    # Agent listesi
    agents_result = api_request("GET", "/agents")
    if agents_result.get("success"):
        agents_list = agents_result.get("data", [])
        
        if agents_list:
            st.subheader("Mevcut Agents")
            for agent in agents_list:
                with st.expander(f"🤖 {agent['name']}"):
                    st.write(f"**ID:** {agent['id']}")
                    st.write(f"**Embedding Model:** {agent.get('embeddingModel', 'N/A')}")
                    st.write(f"**Oluşturulma:** {agent.get('createdAt', 'N/A')}")
                    
                    if st.button(f"🗑️ Sil", key=f"delete_{agent['id']}"):
                        result = api_request("DELETE", f"/agents/{agent['id']}")
                        if result.get("success"):
                            st.success("Agent silindi")
                            st.rerun()
                        else:
                            st.error(result.get("error"))
        else:
            st.info("Henüz agent oluşturulmamış.")
    
    st.markdown("---")
    
    # Yeni agent oluştur
    st.subheader("Yeni Agent Oluştur")
    
    with st.form("create_agent"):
        agent_name = st.text_input("Agent Adı *")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["paraphrase-multilingual-MiniLM-L12-v2", "text-embedding-3-large"]
        )
        data_source_type = st.radio("Veri Kaynağı", ["file", "url"])
        
        if data_source_type == "file":
            data_source = st.file_uploader("Dosya Yükle", type=["pdf", "docx", "txt", "csv", "json"])
        else:
            data_source = st.text_input("URL")
        
        submit = st.form_submit_button("Agent Oluştur")
        
        if submit:
            if not agent_name.strip():
                st.error("Agent adı gerekli")
            else:
                # Dosya yüklendiyse önce upload et
                file_path = None
                if data_source_type == "file" and data_source:
                    # Streamlit file uploader'dan gelen dosyayı API'ye gönder
                    files = {"file": (data_source.name, data_source.getvalue(), data_source.type)}
                    headers = get_headers()
                    headers.pop("Content-Type", None)  # multipart/form-data için kaldır
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            headers=headers,
                            timeout=600
                        )
                        upload_result = response.json()
                        
                        if upload_result.get("success"):
                            file_path = upload_result["data"]["filePath"]
                        else:
                            st.error(upload_result.get("error"))
                            return
                    except Exception as e:
                        st.error(f"Upload hatası: {str(e)}")
                        return
                
                # Agent oluştur
                agent_data = {
                    "name": agent_name,
                    "embedding_model": embedding_model,  # snake_case
                    "data_source_type": data_source_type,  # snake_case
                    "data_source": file_path if file_path else data_source  # snake_case
                }
                
                result = api_request("POST", "/agents", agent_data)
                
                if result.get("success"):
                    st.success("Agent oluşturuldu!")
                    st.rerun()
                else:
                    st.error(result.get("error", "Agent oluşturulamadı"))

# ==================== COMPANIES PAGE ====================

def companies_page():
    """Şirket yönetimi sayfası (SuperAdmin)"""
    st.title("👥 Şirket Yönetimi")
    
    # Şirket listesi
    companies_result = api_request("GET", "/admin/companies")
    if companies_result.get("success"):
        companies_list = companies_result.get("data", [])
        
        if companies_list:
            st.subheader("Mevcut Şirketler")
            for company in companies_list:
                with st.expander(f"🏢 {company['name']}"):
                    st.write(f"**Username:** {company['username']}")
                    st.write(f"**Email:** {company.get('email', 'N/A')}")
                    st.write(f"**Telefon:** {company.get('phone', 'N/A')}")
                    
                    if st.button(f"🗑️ Sil", key=f"delete_{company['id']}"):
                        result = api_request("DELETE", f"/admin/companies/{company['id']}")
                        if result.get("success"):
                            st.success("Şirket silindi")
                            st.rerun()
                        else:
                            st.error(result.get("error"))
        else:
            st.info("Henüz şirket oluşturulmamış.")
    
    st.markdown("---")
    
    # Yeni şirket oluştur
    st.subheader("Yeni Şirket Oluştur")
    
    with st.form("create_company"):
        company_name = st.text_input("Şirket Adı *")
        company_description = st.text_area("Açıklama")
        company_phone = st.text_input("Telefon")
        company_email = st.text_input("Email")
        
        submit = st.form_submit_button("Şirket Oluştur")
        
        if submit:
            if not company_name.strip():
                st.error("Şirket adı gerekli")
            else:
                result = api_request("POST", "/admin/companies", {
                    "name": company_name,
                    "description": company_description,
                    "phone": company_phone,
                    "email": company_email
                })
                
                if result.get("success"):
                    data = result["data"]
                    st.success("Şirket oluşturuldu!")
                    st.info(f"**Username:** {data['username']}  \n**Password:** {data['password']}")
                    st.warning("Bu bilgileri kopyalayın! Tekrar gösterilmeyecek.")
                    st.rerun()
                else:
                    st.error(result.get("error", "Şirket oluşturulamadı"))

# ==================== UPLOAD PAGE ====================

def upload_page():
    """Veri yükleme sayfası"""
    st.title("📤 Veri Yükle")
    
    st.info("Dosya yüklemek için Agents sayfasını kullanın.")

# ==================== MAIN ====================

def main():
    """Ana fonksiyon"""
    st.set_page_config(
        page_title="RAG SaaS Platform",
        page_icon="🤖",
        layout="wide"
    )
    
    # Giriş kontrolü
    if not st.session_state.user:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()

