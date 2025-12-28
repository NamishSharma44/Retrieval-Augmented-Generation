"""
Configuration Management Module
Handles all configuration settings and API keys
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Vector Store Configuration
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./data/chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")
    
    # Document Processing
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    DEFAULT_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    
    # Memory Configuration
    MEMORY_K: int = int(os.getenv("MEMORY_K", "5"))  # Number of previous messages to remember
    
    # Application Settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_FILE_TYPES: list = ['.pdf', '.docx', '.doc', '.txt']
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_api_keys(cls) -> dict:
        """Validate which API keys are available and not placeholder values"""
        google_key = cls.GOOGLE_API_KEY
        openai_key = cls.OPENAI_API_KEY
        
        # Check if keys exist and are not placeholder values
        google_valid = bool(google_key) and not google_key.startswith("YOUR_") and len(google_key) > 20
        openai_valid = bool(openai_key) and not openai_key.startswith("YOUR_") and openai_key.startswith("sk-") and len(openai_key) > 20
        
        return {
            "google": google_valid,
            "openai": openai_valid
        }
    
    @classmethod
    def get_active_llm_config(cls) -> dict:
        """Get configuration for the active LLM provider"""
        if cls.DEFAULT_LLM_PROVIDER == "google" and cls.GOOGLE_API_KEY:
            return {
                "provider": "google",
                "model": cls.GOOGLE_MODEL,
                "api_key": cls.GOOGLE_API_KEY
            }
        elif cls.DEFAULT_LLM_PROVIDER == "openai" and cls.OPENAI_API_KEY:
            return {
                "provider": "openai",
                "model": cls.OPENAI_MODEL,
                "api_key": cls.OPENAI_API_KEY
            }
        else:
            return None
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        Path(cls.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
        Path("./data/uploads").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.ensure_directories()