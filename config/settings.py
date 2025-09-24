# config/settings.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional
import tempfile

# Load environment variables
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    # Model settings
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    
    # Document processing
    chunk_size: int = 2000
    chunk_overlap: int = 400
    
    # Memory settings
    memory_tokens: int = 500
    max_recent_exchanges: int = 3
    
    # Vector store
    collection_name: str = "rag_memory"
    persist_directory: str = "./chroma_db"
    retrieval_k: int = 6
    
    # API keys (from environment)
    google_api_key: Optional[str] = None
    
def __post_init__(self):
    # Load API key from environment if not provided
    if self.google_api_key is None:
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
    
    # Force writable database path for deployment
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    self.persist_directory = os.path.join(temp_dir, "chroma_db")
    
    # Try to create directory, fallback to memory if it fails
    try:
        os.makedirs(self.persist_directory, exist_ok=True)
    except Exception:
        self.persist_directory = ":memory:"
    
    # Validate required settings
    if not self.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required")

def validate_config(api_key: str = None) -> bool:
    """Validate if config can be created with given API key"""
    try:
        RAGConfig(google_api_key=api_key)
        return True
    except ValueError:
        return False

def get_config_or_none(api_key: str = None) -> Optional[RAGConfig]:
    """Get config if valid, otherwise return None"""
    try:
        return RAGConfig(google_api_key=api_key)
    except ValueError:
        return None