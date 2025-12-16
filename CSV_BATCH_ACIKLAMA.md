# CSV Batch ve Embedding İşlemi Açıklaması

## 📊 CSV İşleme Süreci

### 1️⃣ **CSV Yükleme (load_document)**
```python
loader = CSVLoader(source)
docs = loader.load()
```

**Ne oluyor:**
- CSV dosyasındaki **her satır** bir `Document` objesi olarak yüklenir
- 20001 satır = 20001 Document objesi
- Her Document'in `page_content` alanında o satırın tüm verisi (tüm kolonlar birleştirilmiş) bulunur

**Örnek:**
```
Satır 1: "id,question,answer" → Document 1
Satır 2: "1,Merhaba,Nasılsın?" → Document 2
...
Satır 20001: ... → Document 20001
```

---

### 2️⃣ **Chunk'lara Bölme (split_documents)**
```python
chunks = self.text_splitter.split_documents(docs)
```

**Text Splitter Ayarları:**
- `chunk_size = 750` karakter
- `chunk_overlap = 100` karakter (chunk'lar arası örtüşme)
- `separators = ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]`

**Ne oluyor:**
- Her Document (satır) **750 karakterden uzunsa**, birden fazla chunk'a bölünür
- Her chunk **maksimum 750 karakter** olur
- Chunk'lar arasında **100 karakter overlap** olur (bağlantı için)

**Örnek:**
```
Document 1 (2000 karakter) → Chunk 1 (750), Chunk 2 (750), Chunk 3 (500)
Document 2 (500 karakter) → Chunk 4 (500)
Document 3 (1500 karakter) → Chunk 5 (750), Chunk 6 (750)
...
```

**20001 satır → ~32,700 chunk** (ortalama her satır 1.6 chunk)

---

### 3️⃣ **Batch'lere Bölme ve Embedding**
```python
batch_size = 100  # Her batch'te 100 chunk
total_batches = (len(chunks) + batch_size - 1) // batch_size

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]  # 100 chunk al
    vectorstore.add_documents(batch)  # Bu 100 chunk'ı embed et
```

**Ne oluyor:**
- **32,700 chunk** var
- Her batch'te **100 chunk** işlenir
- **327 batch** = 32,700 / 100

**Embedding İşlemi:**
- Her batch'teki 100 chunk **aynı anda** embedding modeline gönderilir
- Embedding modeli (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) her chunk'ı **384 boyutlu vektöre** çevirir
- Bu vektörler FAISS index'ine eklenir

**Örnek:**
```
Batch 1: Chunk 1-100 → Embedding → FAISS'e ekle
Batch 2: Chunk 101-200 → Embedding → FAISS'e ekle
...
Batch 327: Chunk 32,601-32,700 → Embedding → FAISS'e ekle
```

---

## 🔢 Hesaplama Örneği

**Senin durumun:**
- **20,001 satır** (CSV satır sayısı)
- **32,700 chunk** (ortalama her satır 1.6 chunk)
- **327 batch** (32,700 / 100 = 327)

**Neden 327 batch?**
- Her satır ortalama **1.6 chunk** oluşturuyor (satırlar 750 karakterden uzun)
- 20,001 × 1.6 ≈ **32,000 chunk**
- 32,000 / 100 = **320 batch** (ama tam 327 çıkmış, bazı satırlar daha uzun)

---

## 📝 Özet

1. **CSV Yükleme:** Her satır → 1 Document
2. **Chunk'lara Bölme:** Her Document → 750 karakterlik chunk'lar (overlap ile)
3. **Batch'lere Bölme:** Her 100 chunk → 1 batch
4. **Embedding:** Her batch'teki 100 chunk → 384 boyutlu vektörler
5. **FAISS'e Ekleme:** Vektörler → FAISS index'ine kaydedilir

**Neden batch kullanıyoruz?**
- Tüm chunk'ları tek seferde embed etmek **bellek hatası** verir
- 100'lük batch'ler **bellek kullanımını** optimize eder
- **Progress tracking** için batch numarası gösterilir

