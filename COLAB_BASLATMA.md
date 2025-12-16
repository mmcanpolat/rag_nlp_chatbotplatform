# RAG SaaS Platform - Colab Başlatma Kılavuzu

## 🚀 Hızlı Başlangıç

### 1️⃣ İlk Kurulum (Sadece Bir Kere)

Colab'te yeni bir hücre oluşturun ve şu komutu çalıştırın:

```python
!wget -q -O - https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_INSTALL.py | python3
```

Bu komut tüm gerekli paketleri kurar (5-10 dakika sürebilir).

### 2️⃣ Projeyi Başlatma

Kurulum tamamlandıktan sonra, projeyi başlatmak için:

```python
!wget -q -O - https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_SINGLE_FILE.py | python3
```

Bu komut:
- Backend'i (FastAPI) başlatır
- Frontend'i (Gradio) başlatır
- Public URL'i terminal'de gösterir

## 🔑 Giriş Bilgileri

- **Kullanıcı Adı:** `admin@ragplatform.com`
- **Şifre:** `Admin123!@#`

## 📋 Özellikler

### ✅ Batch Progress Tracking
- Dosya yükleme sırasında terminal'de batch progress görüntülenir
- Her batch için yüzde bilgisi gösterilir (örn: Batch 1/5 - %20)

### ✅ Agent Dropdown Güncelleme
- Agent oluşturulduktan sonra dropdown otomatik güncellenir
- Chat sayfasında yeni agent'lar hemen görünür

### ✅ Tam Model İsimleri
- Model seçiminde tam model isimleri gösterilir:
  - `dbmdz/gpt2-turkish-cased (GPT-2 Türkçe)`
  - `bert-base-turkish-cased (BERT Türkçe)`
  - `savasy/bert-base-turkish-sentiment-cased (BERT Sentiment)`

## 🔄 Tekrar Başlatma

Eğer projeyi tekrar başlatmak isterseniz, sadece **2️⃣ Projeyi Başlatma** adımını tekrarlayın. Paketler zaten kurulu olduğu için hızlıca başlar.

## ⚠️ Notlar

- İlk kurulum 5-10 dakika sürebilir
- Public URL oluşturulması 10-20 saniye sürebilir
- Terminal çıktısında public URL'i görebilirsiniz
- Colab runtime'ı yeniden başlatıldığında sadece başlatma komutunu çalıştırmanız yeterli
