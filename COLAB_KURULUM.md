# Google Colab'da RAG SaaS Platform Kurulumu

**Basit 3 Adım:** Kur → Ayarla → Başlat

## 📋 Ön Gereksinimler

1. **Google Colab Hesabı**: [colab.research.google.com](https://colab.research.google.com)
2. **OpenAI API Key**: [platform.openai.com](https://platform.openai.com)

**ngrok ŞART DEĞİL!** Colab'ın kendi port forwarding'i var (sağ üstteki 🔗 ikonu).

## 🚀 Hızlı Başlangıç

1. **Projeyi Yükle:**
   ```python
   !git clone https://github.com/kullaniciadi/bil482-project.git
   %cd bil482-project
   ```

2. **Notebook'u Aç:** `notebooks/colab_setup.ipynb`

3. **Tüm Hücreleri Çalıştır:** Sırayla tüm hücreleri çalıştırın

4. **Public URL:** 
   - **Kolay Yol:** Sağ üstteki 🔗 ikonuna tıklayın → Port 4200 seçin
   - **ngrok (Opsiyonel):** Notebook'taki son hücreyi çalıştırın

## 📝 Detaylı Adımlar

### 1. Node.js Kurulumu
Colab'da Node.js yok, bu yüzden otomatik olarak kurulacak (18.x versiyonu).

### 2. Python Bağımlılıkları
Tüm Python paketleri (`langchain`, `faiss-cpu`, `transformers`, vb.) otomatik kurulacak.

### 3. Backend ve Frontend Bağımlılıkları
- Backend: `express`, `cors`, `multer`, vb.
- Frontend: `Angular 17`, `Tailwind CSS`, vb.

### 4. Servisleri Başlatma
- **Backend**: Port 3000'de çalışır
- **Frontend**: Port 4200'de çalışır
- **ngrok**: Public URL oluşturur

### 5. Public URL

**ngrok ŞART DEĞİL!** İki seçenek var:

**Seçenek A: Colab Port Forwarding (Önerilen)**
- Sağ üstteki 🔗 ikonuna tıklayın
- Port 4200'i seçin
- Otomatik public URL alırsınız

**Seçenek B: ngrok (Opsiyonel)**
- Notebook'taki son hücreyi çalıştırın
- ngrok size URL verecek

## 🔍 Sorun Giderme

### Backend çalışmıyor
```python
# Backend loglarını kontrol edin
!tail -50 /tmp/backend.log
```

### Frontend çalışmıyor
```python
# Frontend loglarını kontrol edin
!tail -50 /tmp/frontend.log
```

Angular build 2-3 dakika sürebilir, bekleyin.

### Port zaten kullanılıyor
```python
# Kullanılan portları kontrol edin
!lsof -i :3000
!lsof -i :4200

# Process'leri sonlandırın
!pkill -f "node.*server.js"
!pkill -f "ng serve"
```

### ngrok hatası
- **ngrok kullanmayın!** Colab'ın kendi port forwarding'i var (🔗 ikonu)

## 📊 Servis Durumunu Kontrol Etme

```python
# Backend health check
import requests
response = requests.get('http://localhost:3000/api/health')
print(response.json())

# Frontend kontrolü
response = requests.get('http://localhost:4200')
print("Frontend durumu:", response.status_code)
```

## 🔄 Yeni Session'da Çalıştırma

Colab session'ı kapandığında:
1. Notebook'u tekrar açın
2. Tüm hücreleri sırayla çalıştırın
3. Yeni bir public URL alacaksınız (🔗 ikonu veya ngrok)

## ⚠️ Önemli Notlar

1. **Session Süresi**: Colab session'ları 12 saat sonra otomatik kapanır
2. **RAM Limiti**: Büyük dosyalar için yeterli RAM olduğundan emin olun
3. **GPU**: GPU kullanmak isterseniz, Colab'da GPU'yu etkinleştirin (Runtime > Change runtime type > GPU)
4. **Timeout**: Büyük dosya yüklemeleri için timeout 10 dakika olarak ayarlanmıştır

## 🎯 Kullanım

1. Public URL'yi açın
2. SuperAdmin ile giriş yapın:
   - **Email**: `admin@ragplatform.com`
   - **Şifre**: `Admin123!@#`
3. Şirket oluşturun
4. Agent (chatbot) oluşturun
5. Veri seti yükleyin (PDF, DOCX, TXT, CSV, Web URL)
6. Chat ile test edin

## 📞 Destek

Sorun yaşarsanız:
- Backend loglarını kontrol edin: `/tmp/backend.log`
- Frontend loglarını kontrol edin: `/tmp/frontend.log`
- Python script hatalarını kontrol edin: Backend response'larında

---

**🎉 Başarılar! Platformunuz hazır!**

