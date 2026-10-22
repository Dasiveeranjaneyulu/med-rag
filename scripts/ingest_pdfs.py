
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.rag_service import MedicalRAGPipeline
from backend.app.core.config import settings

if __name__ == "__main__":
    print("Starting PDF ingestion...")
    pipeline = MedicalRAGPipeline()
    report = pipeline.ingest_pdfs(str(settings.UPLOADS_DIR))
    print(f"Ingestion complete: {report}")

