
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    GROQ_API_KEY: Optional[str] = None

    # LLM Models
    GEMINI_MODEL_NAME: str = "gemini-3.5-flash"
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # Retrieval Settings (PDF pipeline)
    SIMILARITY_THRESHOLD: float = 1.0
    TOP_K: int = 10
    RERANK_TOP_K: int = 5
    RERANKER_RELEVANCE_THRESHOLD: float = 0.5

    # Embedding Models (PDF pipeline)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MedCPT Models (PubMed pipeline)
    MEDCPT_QUERY_ENCODER_NAME: str = "ncbi/MedCPT-Query-Encoder"
    MEDCPT_ARTICLE_ENCODER_NAME: str = "ncbi/MedCPT-Article-Encoder"
    MEDCPT_MODEL_NAME: str = "ncbi/MedCPT-Query-Encoder"

    # PubMed Dataset Settings
    PUBMED_DATASET_NAME: str = "slinusc/PubMedAbstractsSubsetEmbedded"
    PUBMED_COLLECTION_NAME: str = "pubmed_medcpt_docs"
    PUBMED_BATCH_SIZE: int = 1000
    PUBMED_TOP_K: int = 5
    PUBMED_SIMILARITY_THRESHOLD: float = 0.35
    PUBMED_EMBEDDING_DIM: int = 768

    # PDF Collection Settings
    PDF_COLLECTION_NAME: str = "medical_pdf_docs"

    # Chunking Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    LOGS_DIR: Path = BASE_DIR / "logs"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    MODELS_DIR: Path = BASE_DIR / "models"

    # ChromaDB (backward compat: original collection name now aliased to PDF)
    COLLECTION_NAME: str = "medical_pdf_docs"

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8001

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()

settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

