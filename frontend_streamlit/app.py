#!/usr/bin/env python3
# Streamlit Frontend - RAG SaaS Platform
# Profesyonel tasarım - Snow White Theme

import streamlit as st
import requests
import json
from typing import Optional, Dict, List
import os

# ==================== CUSTOM CSS ====================

CUSTOM_CSS = """
<style>
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Sidebar User Info */
    .sidebar-user-info {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-user-info p {
        margin: 0.25rem 0;
        font-size: 0.875rem;
        color: #475569;
    }
    
    /* Radio Buttons - Custom Styling */
    [data-testid="stRadio"] label {
        font-size: 0.875rem;
        color: #475569;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s;
    }
    
    [data-testid="stRadio"] label:hover {
        background: #f1f5f9;
    }
    
    [data-testid="stRadio"] input[type="radio"]:checked + label {
        background: #f1f5f9;
        color: #1e293b;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1e293b;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background: #334155;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Button */
    .btn-secondary {
        background: #f1f5f9;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }
    
    .btn-secondary:hover {
        background: #e2e8f0;
    }
    
    /* Danger Button */
    .btn-danger {
        background: #ef4444;
        color: white;
    }
    
    .btn-danger:hover {
        background: #dc2626;
    }
    
    /* Cards */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06);
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    
    .card-content {
        color: #64748b;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.625rem 0.875rem;
        font-size: 0.875rem;
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #1e293b;
        box-shadow: 0 0 0 3px rgba(30, 41, 59, 0.1);
        outline: none;
    }
    
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stChatMessageUser"] {
        background: #f1f5f9;
        border-left: 3px solid #1e293b;
    }
    
    [data-testid="stChatMessageAssistant"] {
        background: white;
        border: 1px solid #e2e8f0;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 0.5rem;
        color: #166534;
    }
    
    .stError {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 0.5rem;
        color: #991b1b;
    }
    
    .stWarning {
        background: #fffbeb;
        border: 1px solid #fde047;
        border-radius: 0.5rem;
        color: #854d0e;
    }
    
    .stInfo {
        background: #eff6ff;
        border: 1px solid #93c5fd;
        border-radius: 0.5rem;
        color: #1e40af;
    }
    
    /* Titles */
    h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 2rem 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #1e293b transparent transparent transparent;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #e2e8f0;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1e293b;
        background: #f8fafc;
    }
    
    /* Login Form */
    .login-container {
        max-width: 400px;
        margin: 4rem auto;
        padding: 2rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Table */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

# ==================== CONFIG ====================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")
API_URL = f"{API_BASE_URL}/api"

# ==================== SESSION STATE ====================

if "user" not in st.session_state:
    st.session_state.user = None
if "session_token" not in st.session_state:
    st.session_state.session_token = None
if "agents" not in st.session_state:
    st.session_state.agents = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# ==================== HELPER FUNCTIONS ====================

def get_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if st.session_state.session_token:
        headers["Authorization"] = f"Bearer {st.session_state.session_token}"
    return headers

def api_request(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
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
        return {"success": False, "error": str(e)}

# ==================== LOGIN PAGE ====================

def login_page():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <h1 style="text-align: center; margin-bottom: 2rem;">RAG Platform</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Giriş Yap")
        
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı / Email", placeholder="admin@ragplatform.com")
            password = st.text_input("Şifre", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Giriş Yap", use_container_width=True)
            
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
        
        st.markdown("---")
        with st.expander("SuperAdmin Giriş Bilgileri", expanded=False):
            st.markdown("""
            **Kullanıcı:** admin@ragplatform.com  
            **Şifre:** Admin123!@#
            """)

# ==================== MAIN APP ====================

def main_app():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    user = st.session_state.user
    
    # Sidebar
    with st.sidebar:
        st.title("RAG Platform")
        
        # User Info Card
        st.markdown("""
        <div class="sidebar-user-info">
            <p><strong>Kullanıcı:</strong> {}</p>
            <p><strong>Rol:</strong> {}</p>
        </div>
        """.format(
            user.get('username', 'N/A'),
            'SuperAdmin' if user.get("isSuperAdmin") else user.get('companyName', 'N/A')
        ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Menu
        menu_options = ["Chat", "Analytics", "Agents", "Veri Yükle"]
        if user.get("isSuperAdmin"):
            menu_options.insert(3, "Şirket Yönetimi")
        
        page = st.radio("Menü", menu_options, label_visibility="collapsed")
        
        st.markdown("---")
        
        if st.button("Çıkış", use_container_width=True):
            st.session_state.user = None
            st.session_state.session_token = None
            st.rerun()
    
    # Page Content
    if page == "Chat":
        chat_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Agents":
        agents_page()
    elif page == "Şirket Yönetimi":
        companies_page()
    elif page == "Veri Yükle":
        upload_page()

# ==================== CHAT PAGE ====================

def chat_page():
    st.title("Chat")
    
    # Agent Selection
    agents_result = api_request("GET", "/agents")
    if not agents_result.get("success"):
        st.error("Agent'lar yüklenemedi")
        return
    
    agents_list = agents_result.get("data", [])
    if not agents_list:
        st.warning("Henüz agent oluşturulmamış. Önce 'Agents' sayfasından agent oluşturun.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        agent_names = [f"{a['name']}" for a in agents_list]
        selected_agent_idx = st.selectbox("Agent Seçin", range(len(agent_names)), format_func=lambda x: agent_names[x])
        selected_agent = agents_list[selected_agent_idx]
    
    with col2:
        model = st.radio("Model", ["gpt", "bert-turkish", "bert-sentiment"], horizontal=True, label_visibility="collapsed")
    
    st.session_state.active_agent = selected_agent["id"]
    
    st.markdown("---")
    
    # Chat Container
    chat_container = st.container()
    
    agent_id = selected_agent["id"]
    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []
    
    # Display Chat History
    with chat_container:
        for msg in st.session_state.chat_history[agent_id]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if "context" in msg and msg["context"]:
                        with st.expander("Retrieved Context", expanded=False):
                            st.text(msg["context"])
                    if "confidence" in msg:
                        st.caption(f"Güven: {msg['confidence']:.2%}")
    
    # Chat Input
    query = st.chat_input("Sorunuzu yazın...")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        
        st.session_state.chat_history[agent_id].append({"role": "user", "content": query})
        
        with st.spinner("Yanıt oluşturuluyor..."):
            result = api_request("POST", "/chat", {
                "query": query,
                "agent_id": agent_id,
                "model": model
            })
        
        if result.get("success"):
            data = result["data"]
            
            with st.chat_message("assistant"):
                st.write(data["answer"])
                
                if data.get("context"):
                    with st.expander("Retrieved Context", expanded=False):
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
    st.title("Analytics")
    
    agents_result = api_request("GET", "/agents")
    if not agents_result.get("success"):
        st.error("Agent'lar yüklenemedi")
        return
    
    agents_list = agents_result.get("data", [])
    if not agents_list:
        st.warning("Henüz agent oluşturulmamış.")
        return
    
    agent_names = [f"{a['name']}" for a in agents_list]
    selected_agent_idx = st.selectbox("Agent Seçin", range(len(agent_names)), format_func=lambda x: agent_names[x])
    selected_agent = agents_list[selected_agent_idx]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Benchmark Çalıştır", use_container_width=True):
            with st.spinner("Benchmark çalıştırılıyor (bu biraz zaman alabilir)..."):
                result = api_request("POST", "/benchmark", {"agent_id": selected_agent["id"]})
            
            if result.get("success"):
                data = result["data"]
                
                # Metrics Grid
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Ortalama Accuracy", f"{data.get('avg_accuracy', 0):.2%}")
                with col2:
                    st.metric("Ortalama BLEU", f"{data.get('avg_bleu', 0):.4f}")
                with col3:
                    st.metric("Ortalama ROUGE-L", f"{data.get('avg_rouge', 0):.4f}")
                with col4:
                    st.metric("Ortalama F1", f"{data.get('avg_f1', 0):.4f}")
                
                # Plots
                plots = result.get("plots", [])
                for plot_name in plots:
                    plot_path = f"frontend_streamlit/assets/plots/{plot_name}"
                    if not os.path.exists(plot_path):
                        plot_path = f"python_services/data/plots/{plot_name}"
                    if os.path.exists(plot_path):
                        st.image(plot_path, use_container_width=True)
                    else:
                        st.warning(f"Plot bulunamadı: {plot_name}")
            else:
                st.error(result.get("error", "Benchmark hatası"))

# ==================== AGENTS PAGE ====================

def agents_page():
    st.title("Agents")
    
    # Existing Agents
    agents_result = api_request("GET", "/agents")
    if agents_result.get("success"):
        agents_list = agents_result.get("data", [])
        
        if agents_list:
            st.subheader("Mevcut Agents")
            for agent in agents_list:
                with st.expander(f"{agent['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**ID:** {agent['id']}")
                        st.write(f"**Embedding Model:** {agent.get('embeddingModel', 'N/A')}")
                    with col2:
                        st.write(f"**Oluşturulma:** {agent.get('createdAt', 'N/A')}")
                    
                    if st.button("Sil", key=f"delete_{agent['id']}", type="secondary"):
                        result = api_request("DELETE", f"/agents/{agent['id']}")
                        if result.get("success"):
                            st.success("Agent silindi")
                            st.rerun()
                        else:
                            st.error(result.get("error"))
        else:
            st.info("Henüz agent oluşturulmamış.")
    
    st.markdown("---")
    
    # Create New Agent
    st.subheader("Yeni Agent Oluştur")
    
    with st.form("create_agent"):
        agent_name = st.text_input("Agent Adı *", placeholder="Örn: Müşteri Destek Botu")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["paraphrase-multilingual-MiniLM-L12-v2", "text-embedding-3-large"]
        )
        data_source_type = st.radio("Veri Kaynağı", ["file", "url"], horizontal=True)
        
        if data_source_type == "file":
            data_source = st.file_uploader("Dosya Yükle", type=["pdf", "docx", "txt", "csv", "json"])
        else:
            data_source = st.text_input("URL", placeholder="https://example.com")
        
        submit = st.form_submit_button("Agent Oluştur", use_container_width=True)
        
        if submit:
            if not agent_name.strip():
                st.error("Agent adı gerekli")
            else:
                file_path = None
                if data_source_type == "file" and data_source:
                    files = {"file": (data_source.name, data_source.getvalue(), data_source.type)}
                    headers = get_headers()
                    headers.pop("Content-Type", None)
                    
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
                
                agent_data = {
                    "name": agent_name,
                    "embedding_model": embedding_model,
                    "data_source_type": data_source_type,
                    "data_source": file_path if file_path else data_source
                }
                
                result = api_request("POST", "/agents", agent_data)
                
                if result.get("success"):
                    st.success("Agent oluşturuldu!")
                    st.rerun()
                else:
                    st.error(result.get("error", "Agent oluşturulamadı"))

# ==================== COMPANIES PAGE ====================

def companies_page():
    st.title("Şirket Yönetimi")
    
    companies_result = api_request("GET", "/admin/companies")
    if companies_result.get("success"):
        companies_list = companies_result.get("data", [])
        
        if companies_list:
            st.subheader("Mevcut Şirketler")
            for company in companies_list:
                with st.expander(f"{company['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Username:** {company['username']}")
                        st.write(f"**Email:** {company.get('email', 'N/A')}")
                    with col2:
                        st.write(f"**Telefon:** {company.get('phone', 'N/A')}")
                    
                    if st.button("Sil", key=f"delete_{company['id']}", type="secondary"):
                        result = api_request("DELETE", f"/admin/companies/{company['id']}")
                        if result.get("success"):
                            st.success("Şirket silindi")
                            st.rerun()
                        else:
                            st.error(result.get("error"))
        else:
            st.info("Henüz şirket oluşturulmamış.")
    
    st.markdown("---")
    
    # Create Company
    st.subheader("Yeni Şirket Oluştur")
    
    with st.form("create_company"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Şirket Adı *", placeholder="Örn: ABC Teknoloji")
            company_email = st.text_input("Email", placeholder="info@example.com")
        with col2:
            company_phone = st.text_input("Telefon", placeholder="+90 555 123 4567")
            company_description = st.text_area("Açıklama", placeholder="Şirket hakkında kısa bilgi")
        
        submit = st.form_submit_button("Şirket Oluştur", use_container_width=True)
        
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
    st.title("Veri Yükle")
    st.info("Dosya yüklemek için Agents sayfasını kullanın.")

# ==================== MAIN ====================

def main():
    st.set_page_config(
        page_title="RAG SaaS Platform",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if not st.session_state.user:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
