
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from backend.app.core.config import settings
from loguru import logger
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
models_dir = settings.MODELS_DIR


class MedCPTQueryEncoder:
    """
    Wrapper for the NCBI MedCPT Query Encoder.

    Model ID: ncbi/MedCPT-Query-Encoder
    Output: 768-dimensional embeddings
    Query max length: 64 tokens

    The <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token's last hidden state is used as the query representation,
    matching the official MedCPT usage pattern. The resulting embeddings
    live in the same shared embedding space as the precomputed MedCPT
    article embeddings in the PubMed dataset.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.torch = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.torch = torch

            local_path = models_dir / "MedCPT-Query-Encoder"
            model_name = settings.MEDCPT_QUERY_ENCODER_NAME

            if local_path.exists() and (local_path / "config.json").exists():
                logger.info(f"Loading MedCPT Query Encoder from local: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                self.model = AutoModel.from_pretrained(str(local_path))
            else:
                logger.info(f"Loading MedCPT Query Encoder from Hugging Face: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)

            self.model.eval()
            logger.info(
                f"MedCPT Query Encoder ready. dim=768, vocab_size={self.tokenizer.vocab_size}"
            )
        except Exception as e:
            logger.error(f"Failed to load MedCPT Query Encoder: {e}")
            self.model = None
            self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def embed_query(self, text: str) -> list[float]:
        if not self.is_loaded:
            raise RuntimeError("MedCPT Query Encoder not loaded")
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if not self.is_loaded:
            raise RuntimeError("MedCPT Query Encoder not loaded")

        torch = self.torch
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=64,
            )
            outputs = self.model(**encoded)
            cls_embeds = outputs.last_hidden_state[:, 0, :]
            embeds = cls_embeds.cpu().numpy().astype(np.float32)

        dim = embeds.shape[1]
        if dim != settings.PUBMED_EMBEDDING_DIM:
            raise ValueError(
                f"MedCPT query embedding dim mismatch: got {dim}, "
                f"expected {settings.PUBMED_EMBEDDING_DIM}"
            )

        return embeds.tolist()


class DummyMedCPTQueryEncoder:
    """Fallback encoder used when the real MedCPT model cannot be loaded."""

    def __init__(self):
        logger.warning("Using DUMMY MedCPT query encoder — PubMed retrieval is disabled.")

    @property
    def is_loaded(self) -> bool:
        return False

    def embed_query(self, text: str) -> list[float]:
        import random
        return [random.random() for _ in range(settings.PUBMED_EMBEDDING_DIM)]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


try:
    medcpt_encoder = MedCPTQueryEncoder()
except Exception as e:
    logger.error(f"MedCPT encoder init exception: {e}")
    medcpt_encoder = DummyMedCPTQueryEncoder()
