
from fastapi import APIRouter, HTTPException
from loguru import logger
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.rag_service import rag_pipeline
from backend.app.services.llm_router import generate_general_response

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Use the global RAG pipeline from rag_service
pipeline = rag_pipeline


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the hybrid RAG system."""
    try:
        logger.info(f"Received question: {request.question}")
        if pipeline:
            response = pipeline.run(request.question)
        else:
            response = generate_general_response(request.question)
        return ChatResponse(answer=response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return a graceful error message instead of raising 500
        try:
            response = generate_general_response(request.question)
            return ChatResponse(answer=response)
        except:
            error_msg = "I'm sorry, I encountered an internal error. Please try again later."
            if "quota" in str(e).lower() or "429" in str(e):
                error_msg = "The AI service is currently at its limit (Quota Exceeded). Please wait a moment."
            return ChatResponse(answer=error_msg)


@router.get("/kb-status")
async def kb_status_endpoint():
    """Get knowledge base status."""
    if not pipeline:
        return {"status": "no pipeline"}
    return {"status": "ok", "total_vectors": pipeline.chroma.count()}

