
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from backend.app.core.config import settings
from loguru import logger
from pathlib import Path

# Get project root and models directory
project_root = Path(__file__).parent.parent.parent.parent
models_dir = project_root / "models"

# Simple fallback embeddings that just return zeros (to avoid failures)
class DummyEmbeddings:
    def __init__(self):
        logger.warning("Using dummy embeddings - RAG functionality will be limited")

    def embed_documents(self, texts):
        import random
        return [[random.random() for _ in range(384)] for _ in texts]

    def embed_query(self, text):
        import random
        return [random.random() for _ in range(384)]


try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    class LocalEmbeddings:
        def __init__(self):
            # Try loading from local models directory first
            local_model_path = models_dir / settings.EMBEDDING_MODEL_NAME
            if local_model_path.exists():
                logger.info(f"Loading embedding model from local path: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path))
            else:
                logger.info(f"Loading embedding model from Hugging Face Hub: {settings.EMBEDDING_MODEL_NAME}")
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

        def embed_documents(self, texts):
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()

        def embed_query(self, text):
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.tolist()

    embedding_model = LocalEmbeddings()
    logger.info(f"Successfully loaded embedding model {settings.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    logger.info("Falling back to dummy embeddings")
    embedding_model = DummyEmbeddings()

