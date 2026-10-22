
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from backend.app.core.config import settings
from loguru import logger

# Initialize Gemini LLM
try:
    gemini_llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3
    )
    logger.info(f"Successfully initialized Gemini model: {settings.GEMINI_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    gemini_llm = None

# Initialize Groq LLM as fallback
groq_client = None
try:
    if settings.GROQ_API_KEY:
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info(f"Successfully initialized Groq model: {settings.GROQ_MODEL_NAME}")
    else:
        logger.warning("GROQ_API_KEY not set in environment variables")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None


def _call_gemini(query: str) -> str:
    if not gemini_llm:
        raise RuntimeError("Gemini LLM not initialized")
    response = gemini_llm.invoke(query)
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)


def _call_groq(query: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq client not initialized")
    completion = groq_client.chat.completions.create(
        model=settings.GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": query}],
        temperature=0.3
    )
    return completion.choices[0].message.content


def generate_general_response(query):
    """Router that tries Gemini first, then Groq as fallback, with helpful messages!"""
    providers = [
        ("gemini", _call_gemini),
        ("groq", _call_groq),
    ]
    last_error = None
    for name, fn in providers:
        try:
            logger.info(f"Generating general response using {name}...")
            return fn(query)
        except Exception as e:
            logger.error(f"{name} general response error: {e}")
            last_error = e

    # Fallback message if no models work
    return f"⚠️ I'm having trouble connecting to the AI services right now. Here are some things you can check:\n\n1. **API Keys**: Make sure your GOOGLE_API_KEY and GROQ_API_KEY in the .env file are valid and correct\n2. **Quotas**: Check if you've exceeded your free quota or billing limits for either service\n3. **Try Again Later**: Sometimes services are temporarily unavailable"
