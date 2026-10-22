
from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger
import os
import shutil
from backend.app.core.config import settings
from backend.app.schemas.chat import UploadResponse
from backend.app.services.rag_service import rag_pipeline

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

# Use the global RAG pipeline from rag_service
pipeline = rag_pipeline


@router.post("", response_model=UploadResponse)
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """Upload a medical PDF to the knowledge base."""
    try:
        if not pipeline:
            return UploadResponse(
                status="error",
                message="RAG pipeline not available, cannot ingest PDFs"
            )

        # Save uploaded file
        file_path = os.path.join(str(settings.UPLOADS_DIR), file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded: {file.filename}. Starting ingestion...")
        report = pipeline.ingest_pdfs(str(settings.UPLOADS_DIR))

        if report["chunks_added"] == 0:
            return UploadResponse(
                status="warning",
                message=f"'{file.filename}' was saved but NO content was indexed. It may be a scanned/image PDF needing OCR, or unparsable text.",
                details=report
            )

        return UploadResponse(
            status="success",
            message=f"File {file.filename} uploaded and ingested ({report['chunks_added']} chunks).",
            details=report
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

