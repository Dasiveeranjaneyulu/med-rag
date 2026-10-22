
import google.generativeai as genai
from loguru import logger
from backend.app.core.config import settings


class GeminiClient:
    """Client for interacting with Google Gemini API."""

    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required.")

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        logger.info(f"Gemini Client initialized with {settings.GEMINI_MODEL_NAME}")

    def generate_response(self, prompt: str) -> str:
        """Generate a response based on the provided prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return f"Error: {str(e)}"

