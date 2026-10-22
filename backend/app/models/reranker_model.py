
from backend.app.core.config import settings
from loguru import logger
from pathlib import Path
import traceback

# Get project root and models directory
project_root = Path(__file__).parent.parent.parent.parent
models_dir = project_root / "models"

try:
    from sentence_transformers import CrossEncoder

    class Reranker:
        def __init__(self):
            # Try loading from local models directory first
            # The reranker model name is "cross-encoder/ms-marco-MiniLM-L-6-v2", 
            # we saved it as "ms-marco-MiniLM-L-6-v2"
            local_model_name = settings.RERANKER_MODEL_NAME.split("/")[-1]
            local_model_path = models_dir / local_model_name
            if local_model_path.exists():
                logger.info(f"Loading reranker from local path: {local_model_path}")
                self.model = CrossEncoder(str(local_model_path))
            else:
                logger.info(f"Loading reranker from Hugging Face Hub: {settings.RERANKER_MODEL_NAME}")
                self.model = CrossEncoder(settings.RERANKER_MODEL_NAME)
            logger.info(f"Reranker initialized with model {settings.RERANKER_MODEL_NAME}")
        
        def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
            """Rerank documents using cross-encoder and return top_k with scores."""
            if not documents:
                return []
            
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Combine docs with scores and sort
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return doc_score_pairs[:top_k]

    reranker = Reranker()
except Exception as e:
    logger.error(f"Failed to load reranker: {e}")
    logger.error(traceback.format_exc())
    logger.info("Falling back to no reranking")
    
    class DummyReranker:
        def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    reranker = DummyReranker()
