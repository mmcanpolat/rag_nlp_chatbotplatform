// sunucu config - env'den okuyor

require('dotenv').config();

module.exports = {
  PORT: process.env.PORT || 3000,
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  CORS_ORIGINS: process.env.CORS_ORIGINS 
    ? process.env.CORS_ORIGINS.split(',') 
    : ['http://localhost:4200', 'http://127.0.0.1:4200'],
  
  OPENAI_API_KEY: process.env.OPENAI_API_KEY || '',
  
  PYTHON_EXECUTABLE: process.env.PYTHON_EXECUTABLE || 'python3',
  PYTHON_SERVICES_PATH: process.env.PYTHON_SERVICES_PATH || '../python_services/scripts',
  
  REQUEST_TIMEOUT: parseInt(process.env.REQUEST_TIMEOUT) || 600000, // 10 dakika - büyük dosyalar için
  MAX_QUERY_LENGTH: parseInt(process.env.MAX_QUERY_LENGTH) || 2000,
  
  LOG_LEVEL: process.env.LOG_LEVEL || 'info'
};
