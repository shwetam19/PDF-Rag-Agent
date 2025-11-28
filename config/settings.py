"""
Configuration Settings
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # UI Configuration
    PAGE_TITLE = "Multi-Agent PDF Analysis"
    PAGE_ICON = "📚"
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.1
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return True