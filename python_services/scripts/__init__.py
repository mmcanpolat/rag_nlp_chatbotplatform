"""
Python servis modülleri.

ingestor: döküman yükleme ve FAISS'e kaydetme
rag_engine: soru-cevap motoru
evaluator: model karşılaştırma
"""

from scripts.ingestor import DocumentIngestor, ingest_from_file, ingest_from_url
from scripts.rag_engine import RAGEngine
from scripts.evaluator import ModelEvaluator

__all__ = [
    'DocumentIngestor',
    'ingest_from_file', 
    'ingest_from_url',
    'RAGEngine',
    'ModelEvaluator'
]
