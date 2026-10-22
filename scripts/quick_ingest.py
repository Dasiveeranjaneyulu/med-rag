
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.pdf_service import PDFProcessor
from backend.app.services.chunking_service import MedicalChunker
from backend.app.services.rag_service import MedicalRAGPipeline
from backend.app.models.embeddings import embedding_model
from loguru import logger
import os
from backend.app.core.config import settings


def quick_ingest():
    logger.info("Starting quick ingest with larger chunks...")

    pipeline = MedicalRAGPipeline()

    # Use much larger chunks to speed up embedding!
    chunker = MedicalChunker(chunk_size=5000, chunk_overlap=1000)
    pdf_processor = PDFProcessor()

    all_chunks = []

    if not os.path.exists(str(settings.UPLOADS_DIR)):
        os.makedirs(str(settings.UPLOADS_DIR))

    for filename in os.listdir(str(settings.UPLOADS_DIR)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(str(settings.UPLOADS_DIR), filename)
            logger.info(f"Processing: {filename}")
            text = pdf_processor.process_pdf(file_path)
            if text:
                chunks = chunker.chunk_text(text)
                all_chunks.extend(chunks)

    if all_chunks:
        logger.info(f"Adding {len(all_chunks)} chunks to Chroma and BM25...")
        embeddings = embedding_model.embed_documents(all_chunks)
        ids = [f"quick_chunk_{i}_{os.urandom(4).hex()}" for i in range(len(all_chunks))]
        pipeline.chroma.add_documents(ids=ids, documents=all_chunks, embeddings=embeddings)
        pipeline.bm25.add_documents(all_chunks)

        logger.info("Done!")
        print(f"Ingested {len(all_chunks)} chunks!")
    else:
        print("No PDFs found to ingest!")


if __name__ == "__main__":
    quick_ingest()

