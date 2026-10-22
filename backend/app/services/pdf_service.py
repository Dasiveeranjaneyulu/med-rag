
from pypdf import PdfReader
from loguru import logger


class PDFProcessor:
    """Handles text extraction from PDF files."""

    def process_pdf(self, file_path: str) -> str:
        """Process a PDF and return extracted text using pypdf."""
        logger.info(f"Processing PDF: {file_path}")
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text with pypdf: {e}")
        return text

