#!/usr/bin/env python3
# Gradio Frontend - RAG SaaS Platform
# Profesyonel tasarım - Snow White Theme

import gradio as gr
import requests
import json
import os
import time
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# ==================== CONFIG ====================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")
API_URL = f"{API_BASE_URL}/api"

# ==================== GLOBAL STATE ====================

user_state = {"user": None, "token": None}

# ==================== HELPER FUNCTIONS ====================

def get_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if user_state["token"]:
        headers["Authorization"] = f"Bearer {user_state['token']}"
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

def get_agents_dict():
    result = api_request("GET", "/agents")
    if result.get("success"):
        agents = result["data"]
        return {f"{a['name']}": a['id'] for a in agents}
    return {}

# ==================== LOGIN ====================

def login_handler(username: str, password: str):
    if not username or not password:
        return gr.update(visible=True), gr.update(visible=False), "Kullanıcı adı ve şifre gerekli"
    
    result = api_request("POST", "/auth/login", {
        "username": username,
        "password": password
    })
    
    if result.get("success"):
        user_state["user"] = result["data"]
        user_state["token"] = result["data"]["sessionToken"]
        return gr.update(visible=False), gr.update(visible=True), "Giriş başarılı!"
    else:
        return gr.update(visible=True), gr.update(visible=False), f"Hata: {result.get('error', 'Giriş başarısız')}"

def logout_handler():
    user_state["user"] = None
    user_state["token"] = None
    return gr.update(visible=True), gr.update(visible=False), ""

# ==================== CHAT ====================

def chat_send_handler(message: str, agent_name: str, model: str, history: List):
    if not message.strip():
        return history, ""
    
    if not agent_name or agent_name == "Agent yok":
        return history, "Lütfen agent seçin"
    
    agents_dict = get_agents_dict()
    agent_id = agents_dict.get(agent_name)
    
    if not agent_id:
        return history, "Agent bulunamadı"
    
    # Kullanıcı mesajını ekle
    if history is None:
        history = []
    history.append([message, None])
    
    # API'ye istek gönder
    result = api_request("POST", "/chat", {
        "query": message,
        "agent_id": agent_id,
        "model": model
    })
    
    if result.get("success"):
        data = result["data"]
        answer = data["answer"]
        
        # Context ve confidence bilgisi ekle
        context_info = ""
        if data.get("context"):
            context_info = f"\n\n**Retrieved Context:**\n{data['context']}"
        if data.get("confidence"):
            context_info += f"\n**Güven:** {data['confidence']:.2%}"
        
        history[-1][1] = answer + context_info
        return history, ""
    else:
        history[-1][1] = f"Hata: {result.get('error', 'Chat hatası')}"
        return history, ""

def update_agent_dropdown():
    agents_dict = get_agents_dict()
    choices = list(agents_dict.keys()) if agents_dict else ["Agent yok"]
    return gr.update(choices=choices, value=choices[0] if choices and choices[0] != "Agent yok" else None)

# ==================== ANALYTICS ====================

def benchmark_handler(agent_name: str):
    if not agent_name or agent_name == "Agent yok":
        return "Lütfen agent seçin", None
    
    agents_dict = get_agents_dict()
    agent_id = agents_dict.get(agent_name)
    
    if not agent_id:
        return "Agent bulunamadı", None
    
    result = api_request("POST", "/benchmark", {"agent_id": agent_id})
    
    if result.get("success"):
        data = result["data"]
        metrics = f"""
## Metrikler

**Ortalama Accuracy:** {data.get('avg_accuracy', 0):.2%}

**Ortalama BLEU:** {data.get('avg_bleu', 0):.4f}

**Ortalama ROUGE-L:** {data.get('avg_rouge', 0):.4f}

**Ortalama F1:** {data.get('avg_f1', 0):.4f}

**Ortalama Cosine:** {data.get('avg_cosine', 0):.4f}
        """
        
        # Plot'ları göster
        plot_path = None
        plots = result.get("plots", [])
        for plot_name in plots:
            plot_path_test = f"frontend_gradio/assets/plots/{plot_name}"
            if not os.path.exists(plot_path_test):
                plot_path_test = f"python_services/data/plots/{plot_name}"
            if os.path.exists(plot_path_test):
                plot_path = plot_path_test
                break
        
        return metrics, plot_path
    else:
        return f"Hata: {result.get('error', 'Benchmark hatası')}", None

# ==================== AGENTS ====================

def create_agent_handler(name: str, embedding_model: str, data_source_type: str, file, url: str):
    if not name.strip():
        return "Agent adı gerekli", get_agents_list()
    
    file_path = None
    if data_source_type == "file" and file:
        # Gradio file uploader'dan gelen dosyayı işle
        upload_dir = Path("python_services/data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya path'ini al (Gradio file objesi)
        if hasattr(file, 'name'):
            file_name = Path(file.name).name
        else:
            file_name = f"upload_{int(time.time() * 1000)}"
        
        temp_path = upload_dir / file_name
        
        # Dosyayı kaydet
        if hasattr(file, 'read'):
            with open(temp_path, "wb") as f:
                f.write(file.read())
        else:
            # String path ise kopyala
            import shutil
            shutil.copy(file, temp_path)
        
        # Upload endpoint'ine gönder
        try:
            with open(temp_path, "rb") as f:
                files = {"file": (file_name, f, "application/octet-stream")}
                headers = get_headers()
                headers.pop("Content-Type", None)
                
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
                return f"Upload hatası: {upload_result.get('error')}", get_agents_list()
        except Exception as e:
            return f"Upload hatası: {str(e)}", get_agents_list()
    
    agent_data = {
        "name": name,
        "embedding_model": embedding_model,
        "data_source_type": data_source_type,
        "data_source": file_path if file_path else url
    }
    
    result = api_request("POST", "/agents", agent_data)
    
    if result.get("success"):
        return "Agent oluşturuldu!", get_agents_list()
    else:
        return f"Hata: {result.get('error', 'Agent oluşturulamadı')}", get_agents_list()

def get_agents_list():
    result = api_request("GET", "/agents")
    if result.get("success"):
        agents = result["data"]
        if agents:
            return "\n".join([f"**{a['name']}**\n- ID: {a['id']}\n- Model: {a.get('embeddingModel', 'N/A')}\n" for a in agents])
        return "Henüz agent oluşturulmamış."
    return "Agent bulunamadı"

# ==================== COMPANIES ====================

def create_company_handler(name: str, description: str, phone: str, email: str):
    if not name.strip():
        return "Şirket adı gerekli", "", get_companies_list()
    
    result = api_request("POST", "/admin/companies", {
        "name": name,
        "description": description,
        "phone": phone,
        "email": email
    })
    
    if result.get("success"):
        data = result["data"]
        creds = f"**Username:** {data['username']}\n\n**Password:** {data['password']}\n\n⚠️ Bu bilgileri kopyalayın! Tekrar gösterilmeyecek."
        return "Şirket oluşturuldu!", creds, get_companies_list()
    else:
        return f"Hata: {result.get('error', 'Şirket oluşturulamadı')}", "", get_companies_list()

def get_companies_list():
    result = api_request("GET", "/admin/companies")
    if result.get("success"):
        companies = result["data"]
        if companies:
            return "\n".join([f"**{c['name']}**\n- Username: {c['username']}\n- Email: {c.get('email', 'N/A')}\n" for c in companies])
        return "Henüz şirket oluşturulmamış."
    return "Şirket bulunamadı"

# ==================== GRADIO UI ====================

def build_ui():
    # Custom CSS - Snow White Theme
    custom_css = """
    .gradio-container {
        background: #fafafa !important;
    }
    .main-panel {
        background: #fefefe !important;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="slate",
            neutral_hue="slate",
            font=("Inter", "ui-sans-serif", "system-ui", "sans-serif")
        ),
        css=custom_css
    ) as app:
        # Login Screen
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("### Giriş Yap")
            login_username = gr.Textbox(label="Kullanıcı Adı / Email", placeholder="admin@ragplatform.com")
            login_password = gr.Textbox(label="Şifre", type="password", placeholder="••••••••")
            login_btn = gr.Button("Giriş Yap", variant="primary")
            login_status = gr.Markdown("")
            gr.Markdown("---")
            with gr.Accordion("SuperAdmin Giriş Bilgileri", open=False):
                gr.Markdown("**Kullanıcı:** admin@ragplatform.com  \n**Şifre:** Admin123!@#")
        
        # Main App
        with gr.Column(visible=False) as main_app:
            # Top Bar
            with gr.Row():
                user_info = gr.Markdown("")
                logout_btn = gr.Button("Çıkış", variant="secondary", scale=0)
            
            # Tabs
            with gr.Tabs() as tabs:
                # Chat Tab
                with gr.Tab("Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            agent_dropdown = gr.Dropdown(
                                choices=["Agent yok"],
                                label="Agent Seçin",
                                value=None,
                                interactive=True
                            )
                            model_radio = gr.Radio(
                                choices=["gpt", "bert-turkish", "bert-sentiment"],
                                value="gpt",
                                label="Model",
                                interactive=True
                            )
                            chatbot = gr.Chatbot(label="Chat", height=500, show_label=False)
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="",
                                    placeholder="Sorunuzu yazın...",
                                    scale=4,
                                    show_label=False
                                )
                                chat_send = gr.Button("Gönder", variant="primary", scale=1)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Bilgiler")
                            context_info = gr.Markdown("")
                
                # Analytics Tab
                with gr.Tab("Analytics"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            analytics_agent = gr.Dropdown(
                                choices=["Agent yok"],
                                label="Agent Seçin",
                                value=None
                            )
                            benchmark_btn = gr.Button("Benchmark Çalıştır", variant="primary")
                            metrics_display = gr.Markdown("")
                        with gr.Column(scale=1):
                            plot_display = gr.Image(label="Metrikler", show_label=True)
                
                # Agents Tab
                with gr.Tab("Agents"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Mevcut Agents")
                            agents_list = gr.Markdown(get_agents_list())
                            refresh_agents_btn = gr.Button("Yenile", variant="secondary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Yeni Agent Oluştur")
                            agent_name = gr.Textbox(label="Agent Adı *", placeholder="Örn: Müşteri Destek Botu")
                            embedding_model = gr.Dropdown(
                                choices=["paraphrase-multilingual-MiniLM-L12-v2", "text-embedding-3-large"],
                                value="paraphrase-multilingual-MiniLM-L12-v2",
                                label="Embedding Model"
                            )
                            data_source_type = gr.Radio(
                                choices=["file", "url"],
                                value="file",
                                label="Veri Kaynağı"
                            )
                            agent_file = gr.File(label="Dosya Yükle", file_types=[".pdf", ".docx", ".txt", ".csv", ".json"], visible=True)
                            agent_url = gr.Textbox(label="URL", placeholder="https://example.com", visible=False)
                            create_agent_btn = gr.Button("Agent Oluştur", variant="primary")
                            agent_status = gr.Markdown("")
                
                # Companies Tab (conditional)
                companies_tab = gr.Tab("Şirket Yönetimi", visible=False)
                with companies_tab:
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Mevcut Şirketler")
                            companies_list_display = gr.Markdown(get_companies_list())
                            refresh_companies_btn = gr.Button("Yenile", variant="secondary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Yeni Şirket Oluştur")
                            company_name = gr.Textbox(label="Şirket Adı *", placeholder="Örn: ABC Teknoloji")
                            company_description = gr.Textbox(label="Açıklama", lines=3, placeholder="Şirket hakkında kısa bilgi")
                            company_phone = gr.Textbox(label="Telefon", placeholder="+90 555 123 4567")
                            company_email = gr.Textbox(label="Email", placeholder="info@example.com")
                            create_company_btn = gr.Button("Şirket Oluştur", variant="primary")
                            company_status = gr.Markdown("")
                            company_creds = gr.Markdown("")
        
        # Event Handlers
        def login_success(username, password):
            login_screen_vis, main_app_vis, status = login_handler(username, password)
            if main_app_vis.visible:
                # Agent dropdown'ları güncelle
                agents_dict = get_agents_dict()
                choices = list(agents_dict.keys()) if agents_dict else ["Agent yok"]
                agent_update = gr.update(choices=choices, value=choices[0] if choices and choices[0] != "Agent yok" else None)
                analytics_update = gr.update(choices=choices, value=choices[0] if choices and choices[0] != "Agent yok" else None)
                
                # User info güncelle
                user = user_state.get("user", {})
                user_text = f"**Kullanıcı:** {user.get('username', 'N/A')}"
                if user.get("isSuperAdmin"):
                    user_text += " | **Rol:** SuperAdmin"
                else:
                    user_text += f" | **Şirket:** {user.get('companyName', 'N/A')}"
                
                # Companies tab görünürlüğü
                companies_visible = user.get("isSuperAdmin", False)
                
                return (
                    login_screen_vis, main_app_vis, status,
                    user_text, agent_update, analytics_update,
                    gr.update(visible=companies_visible)
                )
            return login_screen_vis, main_app_vis, status, "", gr.update(), gr.update(), gr.update()
        
        login_btn.click(
            login_success,
            inputs=[login_username, login_password],
            outputs=[login_screen, main_app, login_status, user_info, agent_dropdown, analytics_agent, companies_tab]
        )
        
        def logout_success():
            login_vis, main_vis, status = logout_handler()
            return login_vis, main_vis, status, ""
        
        logout_btn.click(
            logout_success,
            outputs=[login_screen, main_app, login_status, user_info]
        )
        
        chat_send.click(
            chat_send_handler,
            inputs=[chat_input, agent_dropdown, model_radio, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            chat_send_handler,
            inputs=[chat_input, agent_dropdown, model_radio, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        benchmark_btn.click(
            benchmark_handler,
            inputs=[analytics_agent],
            outputs=[metrics_display, plot_display]
        )
        
        create_agent_btn.click(
            create_agent_handler,
            inputs=[agent_name, embedding_model, data_source_type, agent_file, agent_url],
            outputs=[agent_status, agents_list]
        ).then(
            update_agent_dropdown,
            outputs=[agent_dropdown, analytics_agent]
        )
        
        refresh_agents_btn.click(
            lambda: get_agents_list(),
            outputs=[agents_list]
        )
        
        create_company_btn.click(
            create_company_handler,
            inputs=[company_name, company_description, company_phone, company_email],
            outputs=[company_status, company_creds, companies_list_display]
        )
        
        refresh_companies_btn.click(
            lambda: get_companies_list(),
            outputs=[companies_list_display]
        )
        
        data_source_type.change(
            lambda x: (gr.update(visible=x == "file"), gr.update(visible=x == "url")),
            inputs=[data_source_type],
            outputs=[agent_file, agent_url]
        )
    
    return app

# ==================== MAIN ====================

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
