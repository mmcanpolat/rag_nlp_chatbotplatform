/*
 * Backend API sunucusu
 * Express + Python script çağırma
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const crypto = require('crypto');
const config = require('./config');

const app = express();

// dosya yükleme
const uploadDir = path.resolve(__dirname, '../python_services/data/uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `upload_${Date.now()}${ext}`);
  }
});

const upload = multer({ 
  storage,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['.json', '.txt', '.md', '.pdf', '.docx', '.doc', '.csv'];
    const ext = path.extname(file.originalname).toLowerCase();
    
    if (allowed.includes(ext)) {
      console.log(`[Multer] Dosya kabul edildi: ${file.originalname} (${ext})`);
      cb(null, true);
    } else {
      console.error(`[Multer] Geçersiz dosya tipi: ${file.originalname} (${ext})`);
      cb(new Error(`Geçersiz dosya tipi: ${ext}. İzin verilen: ${allowed.join(', ')}`), false);
    }
  }
});

// middleware
app.use(cors({
  origin: config.CORS_ORIGINS,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  credentials: true
}));

// bodyParser - multipart/form-data için JSON ve URL encoded sadece
// multer kendi bodyParser'ını kullanıyor
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// python script yolları
const SCRIPTS_DIR = path.resolve(__dirname, config.PYTHON_SERVICES_PATH);
const INGESTOR_SCRIPT = path.join(SCRIPTS_DIR, 'ingestor.py');
const RAG_ENGINE_SCRIPT = path.join(SCRIPTS_DIR, 'rag_engine.py');
const EVALUATOR_SCRIPT = path.join(SCRIPTS_DIR, 'evaluator.py');

// in-memory veri
const companies = new Map();
const agents = new Map();
const sessions = new Map();

// superadmin - default olarak ekle (migration gibi)
const SUPER_ADMIN = {
  id: 'superadmin',
  username: 'admin@ragplatform.com',
  password: 'Admin123!@#',
  isSuperAdmin: true,
  companyId: 'superadmin',
  companyName: 'SuperAdmin'
};

// superadmin'i sessions'a ekle (default login için)
// sessions.set('default-superadmin', SUPER_ADMIN);

// 24 karakter güçlü şifre üret
function generateStrongPassword() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
  let password = '';
  const bytes = crypto.randomBytes(24);
  for (let i = 0; i < 24; i++) {
    password += chars[bytes[i] % chars.length];
  }
  return password;
}

// benzersiz id
function genId() {
  return `${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
}

// session'dan user al
function getUser(req) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  return sessions.get(token);
}

// python çalıştır - timeout ve hata yönetimi iyileştirildi
function runPython(script, args = []) {
  return new Promise((resolve, reject) => {
    const env = { ...process.env, OPENAI_API_KEY: config.OPENAI_API_KEY, PYTHONIOENCODING: 'utf-8' };
    const proc = spawn(config.PYTHON_EXECUTABLE, [script, ...args], { 
      cwd: path.dirname(script), 
      env,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '', stderr = '';
    
    proc.stdout.on('data', d => {
      const data = d.toString();
      stdout += data;
      // progress log'ları göster
      if (data.includes('[*]') || data.includes('[+]')) {
        console.log(`[Python] ${data.trim()}`);
      }
    });
    
    proc.stderr.on('data', d => {
      const data = d.toString();
      stderr += data;
      // warning'leri log'la ama hata olarak sayma
      if (!data.includes('UserWarning') && !data.includes('Warning')) {
        console.error(`[Python Error] ${data.trim()}`);
      }
    });

    const timeoutId = setTimeout(() => {
      console.error(`[Python Timeout] ${script} - ${config.REQUEST_TIMEOUT}ms aşıldı`);
      proc.kill('SIGTERM');
      // force kill after 5 seconds
      setTimeout(() => {
        if (!proc.killed) {
          proc.kill('SIGKILL');
        }
      }, 5000);
      reject(new Error(`İşlem zaman aşımına uğradı (${Math.round(config.REQUEST_TIMEOUT / 1000)} saniye)`));
    }, config.REQUEST_TIMEOUT);

    proc.on('close', (code, signal) => {
      clearTimeout(timeoutId);
      if (code === 0) {
        resolve(stdout);
      } else {
        const errorMsg = stderr || `Python script hatası: exit code ${code}${signal ? `, signal ${signal}` : ''}`;
        reject(new Error(errorMsg));
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timeoutId);
      reject(new Error(`Python process başlatılamadı: ${err.message}`));
    });
  });
}

// ================== AUTH ==================

app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ success: false, error: 'Kullanıcı adı ve şifre gerekli' });
  }

  // superadmin kontrol
  if (username === SUPER_ADMIN.username && password === SUPER_ADMIN.password) {
    const token = genId();
    sessions.set(token, { isSuperAdmin: true, username });
    return res.json({
      success: true,
      data: {
        userId: 'admin',
        username,
        isSuperAdmin: true,
        sessionToken: token
      }
    });
  }

  // şirket kontrol
  const company = Array.from(companies.values()).find(c => c.username === username);
  if (company && company.password === password) {
    const token = genId();
    sessions.set(token, { companyId: company.id, username, companyName: company.name });
    return res.json({
      success: true,
      data: {
        userId: company.id,
        username,
        companyId: company.id,
        companyName: company.name,
        isSuperAdmin: false,
        sessionToken: token
      }
    });
  }

  res.status(401).json({ success: false, error: 'Kullanıcı adı veya şifre hatalı' });
});

app.post('/api/auth/logout', (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (token) sessions.delete(token);
  res.json({ success: true });
});

// ================== PUBLIC - ŞİRKET KONTROLÜ ==================

// public endpoint - company sayısını döndür (ilk açılış kontrolü için)
app.get('/api/companies/count', (req, res) => {
  const count = companies.size;
  res.json({ success: true, count });
});

// ================== ADMIN - ŞİRKET YÖNETİMİ ==================

app.post('/api/admin/companies', (req, res) => {
  const user = getUser(req);
  if (!user?.isSuperAdmin) {
    return res.status(403).json({ success: false, error: 'Yetki yok' });
  }

  const { name, description, phone, email } = req.body;
  if (!name?.trim()) {
    return res.status(400).json({ success: false, error: 'Şirket adı gerekli' });
  }

  const id = genId();
  // username email formatında oluştur
  // eğer email varsa onu kullan, yoksa şirket adından oluştur
  let username;
  if (email && email.trim()) {
    username = email.trim().toLowerCase();
  } else {
    // email yoksa şirket adından email formatında oluştur
    const baseEmail = name.toLowerCase().replace(/[^a-z0-9]/g, '') + '@company.com';
    username = baseEmail;
  }
  
  const password = generateStrongPassword(); // 24 haneli şifre

  const company = {
    id,
    name: name.trim(),
    description: description || '',
    phone: phone || '',
    email: email || '',
    username,
    password,
    createdAt: new Date().toISOString()
  };

  companies.set(id, company);

  // password'ü response'da döndür (sadece oluşturulurken)
  const responseData = {
    ...company,
    password: company.password  // ilk oluşturmada göster
  };

  res.json({ success: true, data: responseData });
});

app.get('/api/admin/companies', (req, res) => {
  const user = getUser(req);
  if (!user?.isSuperAdmin) {
    return res.status(403).json({ success: false, error: 'Yetki yok' });
  }

  const list = Array.from(companies.values()).map(c => ({
    id: c.id,
    name: c.name,
    description: c.description,
    phone: c.phone,
    email: c.email,
    username: c.username,
    createdAt: c.createdAt
    // password dönme
  }));

  res.json({ success: true, data: list });
});

app.delete('/api/admin/companies/:id', (req, res) => {
  const user = getUser(req);
  if (!user?.isSuperAdmin) {
    return res.status(403).json({ success: false, error: 'Yetki yok' });
  }

  if (!companies.has(req.params.id)) {
    return res.status(404).json({ success: false, error: 'Şirket bulunamadı' });
  }

  companies.delete(req.params.id);
  res.json({ success: true });
});

// ================== AGENT ==================

app.post('/api/agents', async (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ success: false, error: 'Giriş yapın' });

  const { name, description, indexName, embeddingModel, dataSourceType, dataSource } = req.body;
  if (!name?.trim()) {
    return res.status(400).json({ success: false, error: 'Agent adı gerekli' });
  }

  const id = genId();
  const finalIndexName = indexName || `agent-${id}`;
  const finalEmbeddingModel = embeddingModel || 'paraphrase-multilingual-MiniLM-L12-v2';

  // veri kaynağı varsa işle
  if (dataSourceType && dataSource) {
    try {
      let args = ['--index', finalIndexName, '--embedding', finalEmbeddingModel];
      
      if (dataSourceType === 'file') {
        // dosya yolu backend'de olmalı
        const filePath = path.join(uploadDir, dataSource);
        if (!fs.existsSync(filePath)) {
          return res.status(400).json({ success: false, error: 'Dosya bulunamadı' });
        }
        args = ['--file', filePath, ...args];
      } else if (dataSourceType === 'url') {
        args = ['--url', dataSource, ...args];
      }
      
      const output = await runPython(INGESTOR_SCRIPT, args);
      console.log(`[Agent] Veri işlendi: ${finalIndexName}`);
    } catch (err) {
      console.error('[Agent] Veri işleme hatası:', err);
      return res.status(500).json({ success: false, error: `Veri işleme hatası: ${err.message}` });
    }
  }

  const agent = {
    id,
    companyId: user.companyId || 'admin',
    name: name.trim(),
    description: description || '',
    indexName: finalIndexName,
    embeddingModel: finalEmbeddingModel,
    dataSourceType: dataSourceType || null,
    dataSource: dataSource || null,
    createdAt: new Date().toISOString(),
    status: 'active'
  };

  agents.set(id, agent);
  res.json({ success: true, data: agent });
});

app.get('/api/agents', (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ success: false, error: 'Giriş yapın' });

  const companyId = user.companyId || 'admin';
  const list = Array.from(agents.values()).filter(a => a.companyId === companyId);
  res.json({ success: true, data: list });
});

app.get('/api/agents/:id', (req, res) => {
  const agent = agents.get(req.params.id);
  if (!agent) {
    return res.status(404).json({ success: false, error: 'Agent bulunamadı' });
  }
  res.json({ success: true, data: agent });
});

app.delete('/api/agents/:id', (req, res) => {
  const user = getUser(req);
  if (!user) return res.status(401).json({ success: false, error: 'Giriş yapın' });

  if (!agents.has(req.params.id)) {
    return res.status(404).json({ success: false, error: 'Agent bulunamadı' });
  }

  agents.delete(req.params.id);
  res.json({ success: true });
});

// ================== UPLOAD ==================

// multer hata yakalama middleware
const uploadMiddleware = upload.single('file');

app.post('/api/upload', (req, res, next) => {
  uploadMiddleware(req, res, (err) => {
    if (err) {
      console.error('[Upload Multer Error]', err.message);
      return res.status(400).json({ success: false, error: err.message || 'Dosya yükleme hatası' });
    }
    next();
  });
}, async (req, res) => {
  try {
    // multer hata kontrolü
    if (!req.file) {
      console.error('[Upload] Dosya alınamadı:', {
        body: req.body,
        files: req.files,
        headers: req.headers['content-type'],
        hasFile: !!req.file
      });
      return res.status(400).json({ success: false, error: 'Dosya yok veya geçersiz format' });
    }

    const indexName = req.body.indexName || 'default';
    const embeddingModel = req.body.embeddingModel || 'paraphrase-multilingual-MiniLM-L12-v2';
    
    const fileSizeMB = (req.file.size / (1024 * 1024)).toFixed(2);
    console.log(`[Upload] ${req.file.originalname} (${fileSizeMB} MB) -> ${indexName} (${embeddingModel})`);

    const args = ['--file', req.file.path, '--index', indexName];
    if (embeddingModel) {
      args.push('--embedding', embeddingModel);
    }
    
    console.log(`[Upload] Python script başlatılıyor... (bu biraz zaman alabilir)`);
    const output = await runPython(INGESTOR_SCRIPT, args);

    try {
      const result = JSON.parse(output.trim());
      res.json({ 
        success: result.success, 
        data: { 
          ...result, 
          filePath: req.file.filename  // dosya adını döndür
        } 
      });
    } catch {
      res.json({ 
        success: true, 
        data: { 
          message: 'İşlendi',
          filePath: req.file.filename 
        } 
      });
    }
  } catch (err) {
    console.error('[Upload Error]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/upload/url', async (req, res) => {
  try {
    const { url, indexName = 'default' } = req.body;
    const embeddingModel = req.body.embeddingModel || 'paraphrase-multilingual-MiniLM-L12-v2';

    if (!url) {
      return res.status(400).json({ success: false, error: 'URL gerekli' });
    }

    console.log(`[Upload URL] ${url} -> ${indexName} (${embeddingModel})`);

    const args = ['--url', url, '--index', indexName];
    if (embeddingModel) {
      args.push('--embedding', embeddingModel);
    }
    const output = await runPython(INGESTOR_SCRIPT, args);

    try {
      const result = JSON.parse(output.trim());
      res.json({ success: result.success, data: result });
    } catch {
      res.json({ success: true, data: { message: 'İşlendi' } });
    }
  } catch (err) {
    console.error('[Upload URL Error]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ================== CHAT ==================

app.post('/api/chat', async (req, res) => {
  try {
    const { query, model = 'GPT', indexName = 'default', agentId } = req.body;

    if (!query) {
      return res.status(400).json({ success: false, error: 'Sorgu gerekli' });
    }

    // agentId varsa indexName'i agent'tan al
    let finalIndexName = indexName;
    if (agentId) {
      const agent = agents.get(agentId);
      if (agent) {
        finalIndexName = agent.indexName;
      }
    }

    const args = ['--query', query, '--model', model, '--index', finalIndexName];
    const output = await runPython(RAG_ENGINE_SCRIPT, args);

    try {
      const result = JSON.parse(output.trim());
      res.json({ success: true, data: result });
    } catch {
      res.json({ success: true, data: { answer: output.trim(), model_used: model } });
    }
  } catch (err) {
    console.error('[Chat Error]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ================== BENCHMARK ==================

app.post('/api/benchmark', async (req, res) => {
  try {
    const { indexName = 'default' } = req.body;

    console.log('[Benchmark] Başlatılıyor...');
    const args = ['--index', indexName];
    const output = await runPython(EVALUATOR_SCRIPT, args);

    const match = output.match(/\{[\s\S]*\}$/);
    if (match) {
      res.json({ success: true, data: JSON.parse(match[0]) });
    } else {
      res.json({ success: true, data: { message: 'Tamamlandı' } });
    }
  } catch (err) {
    console.error('[Benchmark Error]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get('/api/benchmark/results', (req, res) => {
  const file = path.resolve(__dirname, '../frontend/src/assets/plots/evaluation_results.json');
  if (fs.existsSync(file)) {
    res.json({ success: true, data: JSON.parse(fs.readFileSync(file, 'utf-8')) });
  } else {
    res.status(404).json({ success: false, error: 'Sonuç yok' });
  }
});

// ================== MISC ==================

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/indices', (req, res) => {
  const dir = path.resolve(__dirname, '../python_services/data/faiss_index');
  if (!fs.existsSync(dir)) {
    return res.json({ success: true, data: { indices: [] } });
  }

  const indices = fs.readdirSync(dir)
    .filter(d => fs.statSync(path.join(dir, d)).isDirectory())
    .map(d => ({ name: d, documentCount: 0 }));

  res.json({ success: true, data: { indices } });
});

app.get('/api/plots', (req, res) => {
  const dir = path.resolve(__dirname, '../frontend/src/assets/plots');
  if (!fs.existsSync(dir)) {
    return res.json({ success: true, data: { plots: [] } });
  }

  const plots = fs.readdirSync(dir)
    .filter(f => f.endsWith('.png'))
    .map(f => ({ name: f, path: `/assets/plots/${f}` }));

  res.json({ success: true, data: { plots } });
});

// ================== START ==================

app.listen(config.PORT, '0.0.0.0', () => {
  console.log('');
  console.log('='.repeat(50));
  console.log('  RAG Platform API');
  console.log('='.repeat(50));
  console.log(`  Port: ${config.PORT}`);
  console.log(`  SuperAdmin: ${SUPER_ADMIN.username}`);
  console.log('='.repeat(50));
  console.log('');
});

module.exports = app;
