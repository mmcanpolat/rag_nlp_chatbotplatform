# Colab'te Projeyi Başlatma - TEK KOMUT

## Yöntem 1: Direkt GitHub'dan Çalıştır (ÖNERİLEN)

Colab'te yeni bir hücre oluştur ve şunu çalıştır:

```python
!wget -q -O - https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_SINGLE_FILE.py | python3
```

## Yöntem 2: Manuel Kopyala-Yapıştır

1. GitHub'da dosyayı aç: https://github.com/mmcanpolat/rag_nlp_chatbotplatform/blob/main/COLAB_SINGLE_FILE.py
2. **Raw** butonuna tıkla (veya direkt: https://raw.githubusercontent.com/mmcanpolat/rag_nlp_chatbotplatform/main/COLAB_SINGLE_FILE.py)
3. Tüm kodu kopyala (Ctrl+A, Ctrl+C)
4. Colab'te yeni bir hücre oluştur
5. Kodu yapıştır (Ctrl+V)
6. **Shift+Enter** ile çalıştır

## Ne Olacak?

1. ✅ Bağımlılıklar otomatik kurulur (5-10 dakika, ilk seferde)
2. ✅ Backend (FastAPI) arka planda başlar (port 3000)
3. ✅ Frontend (Gradio) başlar ve **public URL oluşturulur**
4. ✅ Terminal çıktısında **public URL görünür**

## Giriş Bilgileri

- **Email:** `admin@ragplatform.com`
- **Şifre:** `Admin123!@#`

## Özellikler

- ✅ Tek dosya - tüm proje tek dosyada
- ✅ Otomatik kurulum - bağımlılıklar otomatik kurulur
- ✅ Public URL - Colab'te otomatik oluşturulur
- ✅ API key gerekmez - Hugging Face modelleri kullanılıyor

## Notlar

- İlk çalıştırmada bağımlılıklar kurulur (5-10 dakika)
- Türkçe GPT-2 modeli ilk kullanımda indirilir (~500MB)
- Public URL terminal çıktısında görünecek
- Backend ve Frontend aynı dosyada, ayrı process'ler olarak çalışır

