# 🤖 BİL482 Intelligent RAG Platform

A SaaS-style Chatbot & Analytics System for comparing AI models using RAG (Retrieval-Augmented Generation).

![Angular](https://img.shields.io/badge/Angular-17+-red?style=flat-square&logo=angular)
![Node.js](https://img.shields.io/badge/Node.js-20+-green?style=flat-square&logo=node.js)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)

## 🎯 Features

- **3 AI Models to Compare:**
  - 🧠 **GPT-4o-mini** (Generative RAG via OpenAI API)
  - 🔤 **BERT Turkish** (Extractive QA via HuggingFace)
  - 📊 **TF-IDF Baseline** (Traditional IR approach)

- **4 Academic Evaluation Metrics:**
  - Cosine Similarity (semantic similarity)
  - ROUGE-L (n-gram overlap)
  - BLEU (generation precision)
  - Accuracy with Confusion Matrix

- **Snow White UI Theme:**
  - Clean, modern SaaS design
  - Responsive Angular 17 frontend
  - Beautiful data visualizations

## 📁 Project Structure

```
bil482-project/
├── backend/                 # Node.js Express API
│   ├── server.js           # Main server file
│   └── config.js           # Configuration
├── python_services/         # Python ML Services
│   ├── scripts/
│   │   ├── data_ingestion.py   # Data & FAISS index builder
│   │   ├── rag_core.py         # RAG chatbot engine
│   │   └── evaluator.py        # Benchmark & metrics
│   ├── data/
│   │   ├── knowledge_base.json # Turkish QA dataset
│   │   └── faiss_index/        # Vector database
│   └── requirements.txt
├── frontend/                # Angular 17 SPA
│   ├── src/
│   │   ├── app/
│   │   │   ├── features/chat/      # Chat interface
│   │   │   ├── features/analytics/ # Metrics dashboard
│   │   │   └── shared/components/  # Reusable UI
│   │   └── assets/plots/   # Generated visualizations
│   └── tailwind.config.js
├── notebooks/
│   └── colab_setup.ipynb   # Google Colab runner
└── package.json
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Upload the project to Google Colab
2. Open `notebooks/colab_setup.ipynb`
3. Follow the step-by-step cells
4. Access via ngrok public URL

### Option 2: Local Development

```bash
# 1. Install Python dependencies
cd python_services
pip install -r requirements.txt

# 2. Initialize data
python scripts/data_ingestion.py

# 3. Install Node.js dependencies
cd ..
npm install
cd frontend && npm install

# 4. Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# 5. Start backend (terminal 1)
npm start

# 6. Start frontend (terminal 2)
cd frontend && ng serve
```

Visit `http://localhost:4200`

## 🔑 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=sk-your-openai-api-key
PORT=3000
NODE_ENV=development
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Query RAG system |
| `/api/benchmark` | POST | Run evaluation |
| `/api/benchmark/results` | GET | Get cached results |
| `/api/stats` | GET | Knowledge base stats |
| `/api/plots` | GET | List generated plots |
| `/api/init` | POST | Initialize data |

## 🧪 Running Benchmarks

```bash
cd python_services
python scripts/evaluator.py
```

This will:
- Evaluate all 50 QA pairs with each model
- Calculate Cosine Similarity, ROUGE-L, BLEU scores
- Generate accuracy metrics and confusion matrices
- Save plots to `frontend/src/assets/plots/`

## 📈 Sample Results

| Model | Accuracy | BLEU | Response Time |
|-------|----------|------|---------------|
| GPT-4o-mini | ~85% | ~0.45 | ~1500ms |
| BERT-Turkish | ~70% | ~0.35 | ~200ms |
| TF-IDF | ~55% | ~0.25 | ~5ms |

*Results vary based on query complexity and API latency.*

## 🎨 Design System

The "Snow White" theme uses:

- **Background:** `#F8FAFC` (slate-50)
- **Cards:** `#FFFFFF` with soft shadows
- **Primary:** `#3B82F6` (Royal Blue)
- **Text:** `#1E293B` / `#64748B`
- **Font:** DM Sans, Sora (display)

## 📚 Dataset

50 Turkish QA pairs covering:
- 🏛️ **History** (17 pairs): Ottoman, Republic, Atatürk
- 💻 **Technology** (17 pairs): AI, Blockchain, Cloud
- 🔬 **Science** (16 pairs): Biology, Physics, Chemistry

## 🛠️ Technologies

**Frontend:**
- Angular 17 (Standalone Components, Signals)
- Tailwind CSS 3.4
- TypeScript 5.4

**Backend:**
- Node.js 20 / Express 4
- Python 3.10+
- PyTorch, Transformers, FAISS

**AI/ML:**
- OpenAI GPT-4o-mini
- HuggingFace Transformers
- Sentence-Transformers
- FAISS (Facebook AI Similarity Search)

## 📄 License

MIT License - University Project 2024

---

**BİL482 - Natural Language Processing Final Project**

