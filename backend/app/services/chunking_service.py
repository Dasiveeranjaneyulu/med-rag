
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings
from loguru import logger

class MedicalChunker:
    """Splits medical text into semantic chunks."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"MedicalChunker initialized with size={chunk_size}, overlap={chunk_overlap}")

    def chunk_text(self, text: str) -> list:
        """Split text into chunks."""
        chunks = self.splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks from text")
        return chunks
